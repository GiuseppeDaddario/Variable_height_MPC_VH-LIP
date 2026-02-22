import numpy as np
import casadi as cs

class Ismpc:
  def __init__(self, initial, footstep_planner, params):
    # parameters
    self.params = params
    self.N = params['N']
    self.delta = params['world_time_step']
    self.h = params['h']
    self.eta = params['eta']
    self.foot_size = params['foot_size']
    self.initial = initial
    self.footstep_planner = footstep_planner
    self.sigma = lambda t, t0, t1: np.clip((t - t0) / (t1 - t0), 0, 1) # piecewise linear sigmoidal function

    # optimization options
    p_opts = {"expand": True}
    s_opts = {"max_iter": 1000, "verbose": False}

    # QP-z optimization problem
    self.opt_z = cs.Opti('conic')
    self.opt_z.solver("osqp", p_opts, s_opts)

    self.Z = self.opt_z.variable(2, self.N + 1)
    self.Fz = self.opt_z.variable(1, self.N)

    self.z0_param = self.opt_z.parameter(2)
    self.z_ref_param = self.opt_z.parameter(self.N)
    self.zz_param = self.opt_z.parameter(self.N) # z_z = z_mc

    for i in range(self.N):
        self.opt_z.subject_to(self.Z[0, i+1] == self.Z[0, i] + self.delta * self.Z[1, i])
        self.opt_z.subject_to(self.Z[1, i+1] == self.Z[1, i] + self.delta * (self.Fz[0, i]/self.params['m'] - self.params['g']))

    alpha = 1e-3
    cost_z = 100 * cs.sumsqr(self.Z[0, 1:].T - self.z_ref_param) + alpha * cs.sumsqr(self.Fz)

    self.opt_z.minimize(cost_z)

    self.opt_z.subject_to(self.Z[:, 0] == self.z0_param) # initial state constraint
    self.opt_z.subject_to(self.Fz >= 0) # unilateral GRF constraint

    # QP-xy optimization problem
    self.opt = cs.Opti('conic')
    self.opt.solver("osqp", p_opts, s_opts)

    self.X = self.opt.variable(6, self.N + 1)
    self.U = self.opt.variable(2, self.N) # [ẋz, ẏz]
    
    self.x0_param = self.opt.parameter(6)
    self.lambda_param = self.opt.parameter(1, self.N)
    self.zmp_x_mid_param = self.opt.parameter(self.N)
    self.zmp_y_mid_param = self.opt.parameter(self.N)

    for i in range(self.N):
        lam = self.lambda_param[0, i]

        A_xy = cs.vertcat(
          cs.horzcat(0,   1,  0),
          cs.horzcat(lam, 0, -lam),
          cs.horzcat(0,   0,  0)
        )

        A_lip_i = cs.DM.zeros(6, 6)
        B_lip = cs.DM.zeros(6, 2)
        drift = cs.DM.zeros(6, 1)

        # x dynamics block
        A_lip_i[0:3, 0:3] = A_xy
        B_lip[0:3, 0] = cs.vertcat(0, 0, 1)

        # y dynamics block
        A_lip_i[3:6, 3:6] = A_xy
        B_lip[3:6, 1] = cs.vertcat(0, 0, 1)

        # constrain dynamics
        self.opt.subject_to(
           self.X[:, i + 1] == 
           self.X[:, i] + self.delta * (A_lip_i @ self.X[:, i] + B_lip @ self.U[:, i])
        )

    cost = cs.sumsqr(self.U) + \
           100 * cs.sumsqr(self.X[2, 1:].T - self.zmp_x_mid_param) + \
           100 * cs.sumsqr(self.X[5, 1:].T - self.zmp_y_mid_param)

    self.opt.minimize(cost)
    
    # zmp constraints
    self.opt.subject_to(self.X[2, 1:].T <= self.zmp_x_mid_param + self.foot_size / 2.)
    self.opt.subject_to(self.X[2, 1:].T >= self.zmp_x_mid_param - self.foot_size / 2.)
    self.opt.subject_to(self.X[5, 1:].T <= self.zmp_y_mid_param + self.foot_size / 2.)
    self.opt.subject_to(self.X[5, 1:].T >= self.zmp_y_mid_param - self.foot_size / 2.)

    # initial state constraint
    self.opt.subject_to(self.X[:, 0] == self.x0_param)

    # stability constraint with periodic tail
    self.opt.subject_to(self.X[1, 0     ] + self.eta * (self.X[0, 0     ] - self.X[2, 0     ]) == \
                        self.X[1, self.N] + self.eta * (self.X[0, self.N] - self.X[2, self.N]))
    self.opt.subject_to(self.X[4, 0     ] + self.eta * (self.X[3, 0     ] - self.X[5, 0     ]) == \
                        self.X[4, self.N] + self.eta * (self.X[3, self.N] - self.X[5, self.N]))

    # state
    self.x = np.zeros(8)
    self.lip_state = {'com': {'pos': np.zeros(3), 'vel': np.zeros(3), 'acc': np.zeros(3)},
                      'zmp': {'pos': np.zeros(3), 'vel': np.zeros(3)}}

  def solve(self, current, t):
    self.x = np.array([
        current['com']['pos'][0], current['com']['vel'][0], current['zmp']['pos'][0],
        current['com']['pos'][1], current['com']['vel'][1], current['zmp']['pos'][1]
    ])
    
    mc_x, mc_y, mc_z = self.generate_moving_constraint(t)

    # solve QP-z optimization problem
    self.opt_z.set_value(self.z0_param, np.array([current['com']['pos'][2], current['com']['vel'][2]]))
    self.opt_z.set_value(self.z_ref_param, mc_z + self.h) # CoM height reference
    self.opt_z.set_value(self.zz_param, mc_z)

    sol_z = self.opt_z.solve()
    Z_pred = sol_z.value(self.Z)
    Fz_pred = sol_z.value(self.Fz)

    self.opt_z.set_initial(self.Z, Z_pred)
    self.opt_z.set_initial(self.Fz, Fz_pred)

    # compute lambda over horizon (1, N)
    den = (Z_pred[0, 0:self.N] - mc_z)
    lambda_seq = (Fz_pred[0, :] / self.params['m']) / den

    # solve QP-xy optimization problem
    self.opt.set_value(self.x0_param, self.x)
    self.opt.set_value(self.lambda_param, lambda_seq.reshape(1, self.N))
    self.opt.set_value(self.zmp_x_mid_param, mc_x)
    self.opt.set_value(self.zmp_y_mid_param, mc_y)

    self.opt.set_value(self.fz_param, Fz_pred.reshape(self.N))
    self.opt.set_value(self.zc_param, Z_pred[0, :].reshape(self.N + 1))
    self.opt.set_value(self.zc_dot_param, Z_pred[1, :].reshape(self.N + 1))

    sol = self.opt.solve()
    self.x = sol.value(self.X[:,1])
    self.u = sol.value(self.U[:,0])

    self.opt.set_initial(self.U, sol.value(self.U))
    self.opt.set_initial(self.X, sol.value(self.X))

    # create output LIP state
    self.lip_state['com']['pos'] = np.array([self.x[0], self.x[3], Z_pred[0, 1]])
    self.lip_state['com']['vel'] = np.array([self.x[1], self.x[4], Z_pred[1, 1]])
    self.lip_state['zmp']['pos'] = np.array([self.x[2], self.x[5], mc_z[0]])
    self.lip_state['zmp']['vel'] = np.array([self.u[0], self.u[1], 0.0])
    self.lip_state['com']['acc'] = np.array([
      lambda_seq[0] * (self.x[0] - self.x[2]),
      lambda_seq[0] * (self.x[3] - self.x[5]),
      Fz_pred[0, 0] / self.params['m'] - self.params['g']
    ])

    contact = self.footstep_planner.get_phase_at_time(t)
    if contact == 'ss':
      contact = self.footstep_planner.plan[self.footstep_planner.get_step_index_at_time(t)]['foot_id']

    return self.lip_state, contact
  
  def generate_moving_constraint(self, t):
    mc_x = np.full(self.N, (self.initial['lfoot']['pos'][3] + self.initial['rfoot']['pos'][3]) / 2.)
    mc_y = np.full(self.N, (self.initial['lfoot']['pos'][4] + self.initial['rfoot']['pos'][4]) / 2.)
    time_array = np.array(range(t, t + self.N))
    for j in range(len(self.footstep_planner.plan) - 1):
      fs_start_time = self.footstep_planner.get_start_time(j)
      ds_start_time = fs_start_time + self.footstep_planner.plan[j]['ss_duration']
      fs_end_time = ds_start_time + self.footstep_planner.plan[j]['ds_duration']
      fs_current_pos = self.footstep_planner.plan[j]['pos'] if j > 0 else np.array([mc_x[0], mc_y[0]])
      fs_target_pos = self.footstep_planner.plan[j + 1]['pos']
      mc_x += self.sigma(time_array, ds_start_time, fs_end_time) * (fs_target_pos[0] - fs_current_pos[0])
      mc_y += self.sigma(time_array, ds_start_time, fs_end_time) * (fs_target_pos[1] - fs_current_pos[1])

    return mc_x, mc_y, np.zeros(self.N)