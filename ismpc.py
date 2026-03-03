import numpy as np
import casadi as cs

class Ismpc:
  def __init__(self, initial, footstep_planner, params):
    # parameters
    self.params = params
    self.N = params['N']
    self.delta = params['world_time_step']
    self.h = params['h']
    #self.eta = params['eta']
    self.foot_size = params['foot_size']
    self.initial = initial
    self.footstep_planner = footstep_planner
    self.sigma = lambda t, t0, t1: np.clip((t - t0) / (t1 - t0), 0, 1) # piecewise linear sigmoidal function

    self.g = self.params["g"]
    self.m = self.params["m"]
    self.fz_min = self.params["fz_min"]
    self.alpha_z = self.params["alpha_z"]
    self.beta_z = self.params["beta_z"]
    self.alpha_x = self.params["alpha_x"]
    self.alpha_y = self.params["alpha_y"]
    self.beta_x  = self.params["beta_x"]

    # d_g: flat terrain assumption --> Paper "As large as possible" (P_conv)
    self.d_g_x   = self.params.get("d_g_x", 3 * self.foot_size)
    self.d_g_y   = self.params.get("d_g_y", 3 * self.foot_size)

    # N_steps: how many footsteps fit in the MPC horizon
    step_duration = self.params["ss_duration"] + self.params["ds_duration"]
    self.N_steps  = self.params.get("N_steps", max(1, int(np.ceil(self.N / step_duration))))

    # lip model matrices
    #self.A_lip = np.array([[0, 1, 0], [self.eta**2, 0, -self.eta**2], [0, 0, 0]])
    #self.B_lip = np.array([[0], [0], [1]])    

    # dynamics
    #self.f = lambda x, u: cs.vertcat(
    #  self.A_lip @ x[0:3] + self.B_lip @ u[0],
    #  self.A_lip @ x[3:6] + self.B_lip @ u[1],
    #  self.A_lip @ x[6:9] + self.B_lip @ u[2] + np.array([0, - params['g'], 0]),
    #)

    p_opts = {"expand": True}
    s_opts = {"max_iter": 1000, "verbose": False}

    self._init_optZ(p_opts, s_opts)
    self._init_optXY(p_opts, s_opts)

    self.opt = cs.Opti('conic')
    self.opt.solver("osqp", p_opts, s_opts)

    self.U = self.opt.variable(3, self.N)
    self.X = self.opt.variable(9, self.N + 1)

    self.x0_param = self.opt.parameter(9)
    self.zmp_x_mid_param = self.opt.parameter(self.N)
    self.zmp_y_mid_param = self.opt.parameter(self.N)
    self.zmp_z_mid_param = self.opt.parameter(self.N)
    self.eta = self.opt.parameter(self.N)

    #for i in range(self.N):
    #  self.opt.subject_to(self.X[:, i + 1] == self.X[:, i] + self.delta * self.f(self.X[:, i], self.U[:, i]))
    for i in range(self.N):
      eta_k = self.eta[i]
      x_next = cs.vertcat(
        # x - axis
        self.X[0, i] + self.delta * self.X[1, i],
        self.X[1, i] + self.delta * (eta_k**2 * (self.X[0, i] - self.X[2, i])),
        self.X[2, i] + self.delta * self.U[0, i],
        # y - axis
        self.X[3, i] + self.delta * self.X[4, i],
        self.X[4, i] + self.delta * (eta_k**2 * (self.X[3, i] - self.X[5, i])),
        self.X[5, i] + self.delta * self.U[1, i],
        # z - axis
        self.X[6, i] + self.delta * self.X[7, i],
        self.X[7, i] + self.delta * (eta_k**2 * (self.X[6, i] - self.X[8, i]) - params['g']),
        self.X[8, i] + self.delta * self.U[2, i],
      )
      self.opt.subject_to(self.X[:, i + 1] == x_next)

    cost = cs.sumsqr(self.U) + \
           100 * cs.sumsqr(self.X[2, 1:].T - self.zmp_x_mid_param) + \
           100 * cs.sumsqr(self.X[5, 1:].T - self.zmp_y_mid_param) + \
           100 * cs.sumsqr(self.X[8, 1:].T - self.zmp_z_mid_param)

    self.opt.minimize(cost)

    # zmp constraints
    self.opt.subject_to(self.X[2, 1:].T <= self.zmp_x_mid_param + self.foot_size / 2.)
    self.opt.subject_to(self.X[2, 1:].T >= self.zmp_x_mid_param - self.foot_size / 2.)
    self.opt.subject_to(self.X[5, 1:].T <= self.zmp_y_mid_param + self.foot_size / 2.)
    self.opt.subject_to(self.X[5, 1:].T >= self.zmp_y_mid_param - self.foot_size / 2.)
    self.opt.subject_to(self.X[8, 1:].T <= self.zmp_z_mid_param + self.foot_size / 2.)
    self.opt.subject_to(self.X[8, 1:].T >= self.zmp_z_mid_param - self.foot_size / 2.)

    # initial state constraint
    self.opt.subject_to(self.X[:, 0] == self.x0_param)

    # stability constraint with periodic tail
    self.opt.subject_to(self.X[1, 0     ] + self.eta[0] * (self.X[0, 0     ] - self.X[2, 0     ]) == \
                        self.X[1, self.N] + self.eta[self.N-1] * (self.X[0, self.N] - self.X[2, self.N]))
    self.opt.subject_to(self.X[4, 0     ] + self.eta[0] * (self.X[3, 0     ] - self.X[5, 0     ]) == \
                        self.X[4, self.N] + self.eta[self.N-1] * (self.X[3, self.N] - self.X[5, self.N]))
    self.opt.subject_to(self.X[7, 0     ] + self.eta[0] * (self.X[6, 0     ] - self.X[8, 0     ]) == \
                        self.X[7, self.N] + self.eta[self.N-1] * (self.X[6, self.N] - self.X[8, self.N]))

    # state
    self.x = np.zeros(9)
    self.lip_state = {'com': {'pos': np.zeros(3), 'vel': np.zeros(3), 'acc': np.zeros(3)},
                      'zmp': {'pos': np.zeros(3), 'vel': np.zeros(3)}}

  def _init_optZ(self, p_opts, s_opts):
    self.opt_z = cs.Opti('conic')
    self.opt_z.solver("osqp", p_opts, s_opts)

    self.Fz = self.opt_z.variable(self.N)
    self.z0_param = self.opt_z.parameter()
    self.dz0_param = self.opt_z.parameter()
    self.z_ref_param = self.opt_z.parameter(self.N)

    _z = [self.z0_param]
    _dz = [self.dz0_param]
    for k in range(self.N):
      _dz.append(_dz[-1] + self.delta * (self.Fz[k] / self.m - self.g))
      _z.append(_z[-1] + self.delta * _dz[-2])

    _Z   = cs.vertcat(*_z[1:])
    _dZ  = cs.vertcat(*_dz[1:])
    _dFz = cs.diff(self.Fz)

    cost_z = cs.sumsqr(_Z - self.z_ref_param) + self.alpha_z * cs.sumsqr(_dZ) + self.beta_z * cs.sumsqr(_dFz)
    self.opt_z.minimize(cost_z)
    self.opt_z.subject_to(self.Fz >= self.fz_min)

  def _init_optXY(self, p_opts, s_opts):
    self.opt_xy = cs.Opti('conic')
    self.opt_xy.solver("osqp", p_opts, s_opts)


    self.X_xy = self.opt_xy.variable(6, self.N + 1)
    self.U_xy = self.opt_xy.variable(2, self.N)  

    self.x0_xy_param = self.opt_xy.parameter(6)      
    self.lambda_xy_param = self.opt_xy.parameter(self.N) 
    self.xmc_param = self.opt_xy.parameter(self.N) 
    self.ymc_param = self.opt_xy.parameter(self.N) 

    for i in range(self.N):
      lam_i = self.lambda_xy_param[i]
      # x axis
      self.opt_xy.subject_to(self.X_xy[0, i+1] == self.X_xy[0, i] + self.delta * self.X_xy[1, i])
      self.opt_xy.subject_to(self.X_xy[1, i+1] == self.X_xy[1, i] + self.delta * lam_i * (self.X_xy[0, i] - self.X_xy[2, i]))
      self.opt_xy.subject_to(self.X_xy[2, i+1] == self.X_xy[2, i] + self.delta * self.U_xy[0, i])
      # y axis
      self.opt_xy.subject_to(self.X_xy[3, i+1] == self.X_xy[3, i] + self.delta * self.X_xy[4, i])
      self.opt_xy.subject_to(self.X_xy[4, i+1] == self.X_xy[4, i] + self.delta * lam_i * (self.X_xy[3, i] - self.X_xy[5, i]))
      self.opt_xy.subject_to(self.X_xy[5, i+1] == self.X_xy[5, i] + self.delta * self.U_xy[1, i])

    # Cost function (Section 5.3.6):
    # min ||Xz - Xmc||^2 + alpha_x * ||Delta Xz||^2
    _dxz = cs.diff(self.X_xy[2, :])  
    _dyz = cs.diff(self.X_xy[5, :])  
    cost_xy = (
      100 * cs.sumsqr(self.X_xy[2, 1:].T - self.xmc_param) +
      100 * cs.sumsqr(self.X_xy[5, 1:].T - self.ymc_param) +
      self.alpha_x * cs.sumsqr(_dxz) +
      self.alpha_y * cs.sumsqr(_dyz)
    )
    # ZMP constraints:
    self.opt_xy.subject_to(self.X_xy[2, 1:].T <= self.xmc_param + self.foot_size / 2.)
    self.opt_xy.subject_to(self.X_xy[2, 1:].T >= self.xmc_param - self.foot_size / 2.)
    self.opt_xy.subject_to(self.X_xy[5, 1:].T <= self.ymc_param + self.foot_size / 2.)
    self.opt_xy.subject_to(self.X_xy[5, 1:].T >= self.ymc_param - self.foot_size / 2.)

    self.opt_xy.subject_to(self.X_xy[:, 0] == self.x0_xy_param)

    # Ground patch constraint 
    S = self.N_steps  # number of footsteps in the horizon
    self.X_f = self.opt_xy.variable(2, S)
    self.xf_nominal_param = self.opt_xy.parameter(S)
    self.yf_nominal_param = self.opt_xy.parameter(S)

    margin_x = (self.d_g_x - self.foot_size) / 2.
    margin_y = (self.d_g_y - self.foot_size) / 2.
    for j in range(S):
      self.opt_xy.subject_to(self.X_f[0, j] - self.xf_nominal_param[j] <=  margin_x)
      self.opt_xy.subject_to(self.X_f[0, j] - self.xf_nominal_param[j] >= -margin_x)
      self.opt_xy.subject_to(self.X_f[1, j] - self.yf_nominal_param[j] <=  margin_y)
      self.opt_xy.subject_to(self.X_f[1, j] - self.yf_nominal_param[j] >= -margin_y)

    # Kinematic constraint
    
    # cost: penalise deviation of optimised footsteps from nominal (Section 5.3.6)
    #cost_xy += self.beta_x * cs.sumsqr(self.X_f[0, :] - self.xf_nominal_param)
    #cost_xy += self.beta_x * cs.sumsqr(self.X_f[1, :] - self.yf_nominal_param)
    #self.opt_xy.minimize(cost_xy)

    # TODO: Stability constraint (5.3.5)



  def solve(self, current, t, eta_seq):
    self.x = np.array([current['com']['pos'][0], current['com']['vel'][0], current['zmp']['pos'][0],
                       current['com']['pos'][1], current['com']['vel'][1], current['zmp']['pos'][1],
                       current['com']['pos'][2], current['com']['vel'][2], current['zmp']['pos'][2]])
    
    mc_x, mc_y, mc_z = self.generate_moving_constraint(t)

    # ADD OF QP-z AND QP-xy
    eta_seq, z_c, dz_c = self.qp_z(current, t)

    # solve optimization problem
    self.opt.set_value(self.x0_param, self.x)
    self.opt.set_value(self.zmp_x_mid_param, mc_x)
    self.opt.set_value(self.zmp_y_mid_param, mc_y)
    self.opt.set_value(self.zmp_z_mid_param, mc_z)
    self.opt.set_value(self.eta, eta_seq)

    sol = self.opt.solve()
    self.x = sol.value(self.X[:,1])
    self.u = sol.value(self.U[:,0])

    self.opt.set_initial(self.U, sol.value(self.U))
    self.opt.set_initial(self.X, sol.value(self.X))

    # create output LIP state
    self.lip_state['com']['pos'] = np.array([self.x[0], self.x[3], self.x[6]])
    self.lip_state['com']['vel'] = np.array([self.x[1], self.x[4], self.x[7]])
    self.lip_state['zmp']['pos'] = np.array([self.x[2], self.x[5], self.x[8]])
    self.lip_state['zmp']['vel'] = self.u
    #self.lip_state['com']['acc'] = self.eta**2 * (self.lip_state['com']['pos'] - self.lip_state['zmp']['pos']) + np.hstack([0, 0, - self.params['g']])
    self.lip_state['com']['acc'] = self.eta[0]**2 * (self.lip_state['com']['pos'] - self.lip_state['zmp']['pos']) + np.hstack([0, 0, - self.params['g']])

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
  
  def generateHeightRef(self, t):
    if True: 
      return np.full(self.N, self.params['h'])
    height_ref = np.zeros(self.N)
    time_array = np.array(range(t, t + self.N))
    for j in range(len(self.footstep_planner.plan) - 1):
      fs_start_time = self.footstep_planner.get_start_time(j)
      ds_start_time = fs_start_time + self.footstep_planner.plan[j]['ss_duration']
      fs_end_time = ds_start_time + self.footstep_planner.plan[j]['ds_duration']
      height_ref += self.sigma(time_array, ds_start_time, fs_end_time) * (self.params['h'] - self.initial['com']['pos'][2])

    return height_ref

  def generatePzRef(self, t):
    pass

  def qp_z(self, current, t):
    z_ref = self.generateHeightRef(t)
    _, _, pz = self.generate_moving_constraint(t)

    z0  = current['com']['pos'][2]
    dz0 = current['com']['vel'][2]

    self.opt_z.set_value(self.z0_param, z0)
    self.opt_z.set_value(self.dz0_param, dz0)
    self.opt_z.set_value(self.z_ref_param, z_ref)

    sol_z   = self.opt_z.solve()
    Fz_sol  = sol_z.value(self.Fz)

    self.opt_z.set_initial(self.Fz, Fz_sol)  

    z_sol  = np.zeros(self.N + 1)
    dz_sol = np.zeros(self.N + 1)
    z_sol[0] = z0
    dz_sol[0] = dz0
    for k in range(self.N):
      dz_sol[k + 1] = dz_sol[k] + self.delta * (Fz_sol[k] / self.m - self.g)
      z_sol[k + 1] = z_sol[k]  + self.delta * dz_sol[k]

    ddz_sol = Fz_sol / self.m - self.g
    denom = np.maximum(z_sol[:self.N] - pz, 1e-3)
    eta_seq = np.sqrt((self.g + ddz_sol) / denom)

    return eta_seq, z_sol, dz_sol

    

  def qp_xy(self, current, t, eta_seq):
    lambda_seq = eta_seq**2
    mc_x, mc_y, _ = self.generate_moving_constraint(t)

    x0_xy = np.array([
      current['com']['pos'][0], current['com']['vel'][0], current['zmp']['pos'][0],
      current['com']['pos'][1], current['com']['vel'][1], current['zmp']['pos'][1],
    ])

    self.opt_xy.set_value(self.x0_xy_param, x0_xy)
    self.opt_xy.set_value(self.lambda_xy_param, lambda_seq)
    self.opt_xy.set_value(self.xmc_param, mc_x)
    self.opt_xy.set_value(self.ymc_param, mc_y)

    sol_xy = self.opt_xy.solve()

    self.opt_xy.set_initial(self.X_xy, sol_xy.value(self.X_xy))  
    self.opt_xy.set_initial(self.U_xy, sol_xy.value(self.U_xy))

    pass

