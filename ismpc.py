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
    self.mass = params['mass']

    # lip model matrices
    self.A_lip = np.array([[0, 1, 0], [self.eta**2, 0, -self.eta**2], [0, 0, 0]])
    self.B_lip = np.array([[0], [0], [1]])

    # dynamics
    self.f = lambda x, u: cs.vertcat(
      self.A_lip @ x[0:3] + self.B_lip @ u[0],
      self.A_lip @ x[3:6] + self.B_lip @ u[1],
      self.A_lip @ x[6:9] + self.B_lip @ u[2] + np.array([0, - params['g'], 0]),
    )

    # optimization problem
    self._setup_qp_z()
    self._setup_qp_xy()

    # state
    self.x = np.zeros(9)
    self.lip_state = {'com': {'pos': np.zeros(3), 'vel': np.zeros(3), 'acc': np.zeros(3)},
                      'zmp': {'pos': np.zeros(3), 'vel': np.zeros(3)}}

  def _setup_qp_z(self):
    self.opt_z = cs.Opti('conic')
    p_opts = {"expand": True}
    s_opts = {"max_iter": 10000, "verbose": False, "adaptive_rho": True}
    self.opt_z.solver("osqp", p_opts, s_opts)

    # decision variable
    self.f_z = self.opt_z.variable(self.N)
    # predicted state (Z and Z_dot)
    self.z_c = self.opt_z.variable(2, self.N + 1)

    # parameters
    self.z0 = self.opt_z.parameter(2)
    self.z_star = self.opt_z.parameter(self.N)

    # ensure start position = current robot position
    self.opt_z.subject_to(self.z_c[:, 0] == self.z0)

    # dynamic constraint
    for i in range(self.N):
        z_step = self.z_c[0, i] + self.delta * self.z_c[1, i]
        z_dot_step = self.z_c[1, i] + self.delta * ((self.f_z[i] / self.mass) - self.params['g'])
        # ensure velocity and accel are feasible for all the steps
        self.opt_z.subject_to(self.z_c[0, i + 1] == z_step)
        self.opt_z.subject_to(self.z_c[1, i + 1] == z_dot_step)

    # friction limit
    self.opt_z.subject_to(self.f_z >= self.params['fs_min'])

    # cost
    cost_z = cs.sumsqr(self.z_c[0, 1:].T - self.z_star) + \
             self.params['alpha_z'] * cs.sumsqr(self.z_c[1, 1:]) + \
             self.params['beta_z'] * cs.sumsqr(cs.diff(self.f_z))

    self.opt_z.minimize(cost_z)

  def _setup_qp_xy(self):
    self.opt_xy = cs.Opti('conic')
    p_opts = {"expand": True}
    s_opts = {"max_iter": 10000, "verbose": False, "adaptive_rho": True, "polish": True}
    self.opt_xy.solver("osqp", p_opts, s_opts)

    self.f_max = self.params['f_max']

    # Decision variables (x and y)
    self.x_z = self.opt_xy.variable(self.N)
    self.x_f = self.opt_xy.variable(self.f_max)
    self.x_c = self.opt_xy.variable(2, self.N + 1)

    self.y_z = self.opt_xy.variable(self.N)
    self.y_f = self.opt_xy.variable(self.f_max)
    self.y_c = self.opt_xy.variable(2, self.N + 1)

    # parameters
    self.xy0 = self.opt_xy.parameter(4)
    self.lambda_t = self.opt_xy.parameter(self.N)  # Time varing frequency of the lip

    self.phi_tc = self.opt_xy.parameter(2, 2)
    self.b_int = self.opt_xy.parameter(2, self.N)
    self.tail_dcm_x = self.opt_xy.parameter()
    self.tail_dcm_y = self.opt_xy.parameter()

    # candidate footsteps
    self.x_cf = self.opt_xy.parameter(self.f_max)
    self.y_cf = self.opt_xy.parameter(self.f_max)

    # alpha for moving contraint of zmp
    self.alpha_j = self.opt_xy.parameter(self.N, self.f_max)

    self.x_mc = cs.mtimes(self.alpha_j, self.x_f)
    self.y_mc = cs.mtimes(self.alpha_j, self.y_f)

    # ensure start position = current robot position
    self.opt_xy.subject_to(self.x_c[:, 0] == self.xy0[0:2])
    self.opt_xy.subject_to(self.y_c[:, 0] == self.xy0[2:4])

    # dynamic constraint
    for i in range(self.N):
        x_step = self.x_c[0, i] + self.delta * self.x_c[1, i]
        x_dot_step = self.x_c[1, i] + self.delta * (self.lambda_t[i] * (self.x_c[0, i] - self.x_z[i]))
        self.opt_xy.subject_to(self.x_c[0, i + 1] == x_step)
        self.opt_xy.subject_to(self.x_c[1, i + 1] == x_dot_step)

        y_step = self.y_c[0, i] + self.delta * self.y_c[1, i]
        y_dot_step = self.y_c[1, i] + self.delta * (self.lambda_t[i] * (self.y_c[0, i] - self.y_z[i]))
        self.opt_xy.subject_to(self.y_c[0, i + 1] == y_step)
        self.opt_xy.subject_to(self.y_c[1, i + 1] == y_dot_step)

    # ZMP constraint
    self.opt_xy.subject_to(self.x_z <= self.x_mc + self.foot_size / 2)
    self.opt_xy.subject_to(self.x_z >= self.x_mc - self.foot_size / 2)
    self.opt_xy.subject_to(self.y_z <= self.y_mc + self.foot_size / 2)
    self.opt_xy.subject_to(self.y_z >= self.y_mc - self.foot_size / 2)

    # ground patch constraint
    self.patch_x_min = self.opt_xy.parameter(self.f_max)
    self.patch_x_max = self.opt_xy.parameter(self.f_max)
    self.patch_y_min = self.opt_xy.parameter(self.f_max)
    self.patch_y_max = self.opt_xy.parameter(self.f_max)

    self.opt_xy.subject_to(self.x_f >= self.patch_x_min)
    self.opt_xy.subject_to(self.x_f <= self.patch_x_max)
    self.opt_xy.subject_to(self.y_f >= self.patch_y_min)
    self.opt_xy.subject_to(self.y_f <= self.patch_y_max)

    # kinematic constraint
    self.d_ax = self.opt_xy.parameter(self.f_max)
    self.d_ay = self.opt_xy.parameter(self.f_max)

    self.l = self.params["l"]
    self.l_sign = self.opt_xy.parameter(self.f_max)

    self.curr_sup_x = self.opt_xy.parameter()
    self.curr_sup_y = self.opt_xy.parameter()

    for i in range(self.f_max):
        prev_x = self.curr_sup_x if i == 0 else self.x_f[i - 1]
        prev_y = self.curr_sup_y if i == 0 else self.y_f[i - 1]

        dx = self.x_f[i] - prev_x
        dy = self.y_f[i] - prev_y

        # X axis
        self.opt_xy.subject_to(dx <= self.d_ax[i] / 2)
        self.opt_xy.subject_to(dx >= -self.d_ax[i] / 2)

        # Y axis
        self.opt_xy.subject_to(dy - (self.l * self.l_sign[i]) <= self.d_ay[i] / 2)
        self.opt_xy.subject_to(dy - (self.l * self.l_sign[i]) >= -self.d_ay[i] / 2)

    # swing foot constraint
    self.curr_swg_x = self.opt_xy.parameter()
    self.curr_swg_y = self.opt_xy.parameter()
    self.max_reach = self.opt_xy.parameter()

    # X axis
    self.opt_xy.subject_to(self.x_f[0] - self.curr_swg_x <= self.max_reach)
    self.opt_xy.subject_to(self.curr_swg_x - self.x_f[0] <= self.max_reach)

    # Y axis
    self.opt_xy.subject_to(self.y_f[0] - self.curr_swg_y <= self.max_reach)
    self.opt_xy.subject_to(self.curr_swg_y - self.y_f[0] <= self.max_reach)

    # stability constraint
    self.eta_lip_param = self.opt_xy.parameter()
    g_val = cs.horzcat(1, 1 / self.eta_lip_param)

    zmp_integral_x = 0
    for i in range(self.N):
        zmp_integral_x += self.b_int[:, i] * self.x_z[i]
    self.opt_xy.subject_to(g_val @ (self.phi_tc @ self.xy0[0:2] + zmp_integral_x) == self.tail_dcm_x)

    zmp_integral_y = 0
    for i in range(self.N):
        zmp_integral_y += self.b_int[:, i] * self.y_z[i]
    self.opt_xy.subject_to(g_val @ (self.phi_tc @ self.xy0[2:4] + zmp_integral_y) == self.tail_dcm_y)

    # cost function
    alpha_xy = self.params['alpha_xy']
    beta_xy = self.params['beta_xy']

    cost_x = cs.sumsqr(self.x_z - self.x_mc) + \
             alpha_xy * cs.sumsqr(cs.diff(self.x_z)) + \
             beta_xy * cs.sumsqr(self.x_f - self.x_cf)

    cost_y = cs.sumsqr(self.y_z - self.y_mc) + \
             alpha_xy * cs.sumsqr(cs.diff(self.y_z)) + \
             beta_xy * cs.sumsqr(self.y_f - self.y_cf)

    self.opt_xy.minimize(cost_x + cost_y)

  def solve(self, current, t):
    self.xy_init = np.array([current['com']['pos'][0], current['com']['vel'][0],
                             current['com']['pos'][1], current['com']['vel'][1]])

    self.z_init =  np.array([current['com']['pos'][2], current['com']['vel'][2]])

    mc_x, mc_y, mc_z = self.generate_moving_constraint(t)

    idx = self.footstep_planner.get_step_index_at_time(t)
    if idx is None: idx = len(self.footstep_planner.plan) - 1
    current_h_ref = self.footstep_planner.plan[idx]['h_ref']

    z_ref = mc_z + current_h_ref

    # warm start
    if not hasattr(self, 'qp_z_warm_started'):
        self.opt_z.set_initial(self.f_z, self.mass * self.params['g'])
        self.opt_z.set_initial(self.z_c[0, :], self.z_init[0])
        self.opt_z.set_initial(self.z_c[1, :], 0.0)
        self.qp_z_warm_started = True

    # solve qp-z
    self.opt_z.set_value(self.z0, self.z_init)
    self.opt_z.set_value(self.z_star, z_ref)

    sol_z = self.opt_z.solve()
    z_c = sol_z.value(self.z_c)
    f_z = sol_z.value(self.f_z)

    self.opt_z.set_initial(self.z_c, z_c)
    self.opt_z.set_initial(self.f_z, f_z)

    # lambda calc
    denom = z_c[0, :-1] - mc_z
    denom = np.where(np.abs(denom) < 1e-6, 1e-6, denom)
    lambda_val = (f_z / self.mass) / denom
    lambda_val = np.clip(lambda_val, 1e-4, None)

    x_cf, y_cf, p_min_x, p_max_x, p_min_y, p_max_y, d_ax, d_ay, signs = self.generate_step_params(t)
    idx = self.footstep_planner.get_step_index_at_time(t)
    phase = self.footstep_planner.get_phase_at_time(t)

    swing_foot = self.footstep_planner.plan[idx]['foot_id']
    sup_foot = 'lfoot' if swing_foot == 'rfoot' else 'rfoot'
    curr_sup_x = current[sup_foot]['pos'][3]
    curr_sup_y = current[sup_foot]['pos'][4]
    curr_swg_x = current[swing_foot]['pos'][3]
    curr_swg_y = current[swing_foot]['pos'][4]

    if phase == 'ss':
        start_time = self.footstep_planner.get_start_time(idx)
        ss_duration = self.footstep_planner.plan[idx]['ss_duration']

        rem_steps = (start_time + ss_duration) - t
        rem_time = max(0, rem_steps * self.delta) # [s]

        v_sw_max = self.params['v_sw_max']
        max_reach_th = rem_time * v_sw_max
        if rem_steps <= 3:
            max_reach = 1e5
        else:
            max_reach = max(max_reach_th, 0.05)
    else:
        max_reach = 1e5

    # warm start
    if not hasattr(self, 'qp_xy_warm_started'):
        self.opt_xy.set_initial(self.x_c[0, :], self.xy_init[0])
        self.opt_xy.set_initial(self.x_c[1, :], self.xy_init[1])
        self.opt_xy.set_initial(self.y_c[0, :], self.xy_init[2])
        self.opt_xy.set_initial(self.y_c[1, :], self.xy_init[3])
        self.opt_xy.set_initial(self.x_z, self.xy_init[0])
        self.opt_xy.set_initial(self.y_z, self.xy_init[2])
        self.opt_xy.set_initial(self.x_f, x_cf)
        self.opt_xy.set_initial(self.y_f, y_cf)
        self.qp_xy_warm_started = True

    # solve qp-xy
    self.opt_xy.set_value(self.xy0, self.xy_init)
    self.opt_xy.set_value(self.lambda_t, lambda_val)
    self.opt_xy.set_value(self.x_cf, x_cf)
    self.opt_xy.set_value(self.y_cf, y_cf)
    self.opt_xy.set_value(self.patch_x_min, p_min_x)
    self.opt_xy.set_value(self.patch_x_max, p_max_x)
    self.opt_xy.set_value(self.patch_y_min, p_min_y)
    self.opt_xy.set_value(self.patch_y_max, p_max_y)
    self.opt_xy.set_value(self.d_ax, d_ax)
    self.opt_xy.set_value(self.d_ay, d_ay)
    self.opt_xy.set_value(self.l_sign, signs)
    self.opt_xy.set_value(self.curr_sup_x, curr_sup_x)
    self.opt_xy.set_value(self.curr_sup_y, curr_sup_y)
    self.opt_xy.set_value(self.curr_swg_x, curr_swg_x)
    self.opt_xy.set_value(self.curr_swg_y, curr_swg_y)
    self.opt_xy.set_value(self.max_reach, max_reach)

    # alpha_j: maps optimized footstep positions to per-timestep ZMP references
    alpha_val = self.compute_alpha_j(t)
    self.opt_xy.set_value(self.alpha_j, alpha_val)

    # proposition 2 calc
    phis = []
    b_matrices = []
    for val in lambda_val:
        w = np.sqrt(val)
        phi = np.array([[np.cosh(w * self.delta), np.sinh(w * self.delta) / w],
                        [w * np.sinh(w * self.delta), np.cosh(w * self.delta)]])
        b_mtx = np.array([[1 - np.cosh(w * self.delta)],
                          [-w * np.sinh(w * self.delta)]])
        phis.append(phi)
        b_matrices.append(b_mtx)

    phi_total = np.eye(2)
    for phi in phis:
        phi_total = phi @ phi_total

    b_integrated = np.zeros((2, self.N))
    for i in range(self.N):
        term_phi = np.eye(2)
        for j in range(i + 1, self.N):
            term_phi = phis[j] @ term_phi
        b_integrated[:, i] = (term_phi @ b_matrices[i]).flatten()

    target_h_ref = self.footstep_planner.plan[min(idx + self.N, len(self.footstep_planner.plan) - 1)]['h_ref']
    eta_lip = np.sqrt(self.params['g'] / (mc_z[-1] + target_h_ref))

    tail_dcm_x_val, tail_dcm_y_val = self._compute_tail_integral(t, eta_lip)

    # set values to opt_xy
    self.opt_xy.set_value(self.phi_tc, phi_total)
    self.opt_xy.set_value(self.b_int, b_integrated)
    self.opt_xy.set_value(self.eta_lip_param, eta_lip)
    self.opt_xy.set_value(self.tail_dcm_x, tail_dcm_x_val)
    self.opt_xy.set_value(self.tail_dcm_y, tail_dcm_y_val)

    try:
        sol_xy = self.opt_xy.solve()
    except RuntimeError as e:
        current_step_idx = self.footstep_planner.get_step_index_at_time(t)
        total_steps = len(self.footstep_planner.plan)
        if current_step_idx is None or current_step_idx >= total_steps - 2:
            print("\n" + "=" * 60)
            print(f"[INFO] SUCCESSFUL RUN: Simulation ended at t={t:.2f}s.")
            print("The QP solver stopped because the MPC preview horizon")
            print("reached the end of the planned footstep array.")
            print("=" * 60 + "\n")
            import sys
            sys.exit(0)

        else:
            print("\n" + "=" * 60)
            print(f"[FATAL ERROR] QP Solver failed mid-walk at t={t:.2f}s!")
            print(f"Current footstep index: {current_step_idx} out of {total_steps}.")
            print("This is a mathematical infeasibility (conflicting constraints).")
            print("=" * 60 + "\n")
            raise e

    x_c_opt = sol_xy.value(self.x_c)
    y_c_opt = sol_xy.value(self.y_c)
    x_z_opt = sol_xy.value(self.x_z)
    y_z_opt = sol_xy.value(self.y_z)

    x_f_opt = sol_xy.value(self.x_f)
    y_f_opt = sol_xy.value(self.y_f)
    self.opt_xy.set_initial(self.x_c, x_c_opt)
    self.opt_xy.set_initial(self.y_c, y_c_opt)
    self.opt_xy.set_initial(self.x_z, x_z_opt)
    self.opt_xy.set_initial(self.y_z, y_z_opt)
    self.opt_xy.set_initial(self.x_f, x_f_opt)
    self.opt_xy.set_initial(self.y_f, y_f_opt)

    # create output LIP state
    self.lip_state['com']['pos'] = np.array([x_c_opt[0, 1], y_c_opt[0, 1], z_c[0, 1]])
    self.lip_state['com']['vel'] = np.array([x_c_opt[1, 1], y_c_opt[1, 1], z_c[1, 1]])
    self.lip_state['zmp']['pos'] = np.array([x_z_opt[0], y_z_opt[0], mc_z[0]])
    self.lip_state['zmp']['vel'] = (self.lip_state['zmp']['pos'] - current['zmp']['pos']) / self.delta
    self.lip_state['com']['acc'] = np.array([
        lambda_val[0] * (self.lip_state['com']['pos'][0] - self.lip_state['zmp']['pos'][0]),
        lambda_val[0] * (self.lip_state['com']['pos'][1] - self.lip_state['zmp']['pos'][1]),
        (f_z[0] / self.mass) - self.params['g']
    ])

    contact = self.footstep_planner.get_phase_at_time(t)
    if contact == 'ss':
      contact = self.footstep_planner.plan[self.footstep_planner.get_step_index_at_time(t)]['foot_id']

    return self.lip_state, contact
  
  def generate_moving_constraint(self, t):
    mc_x = np.full(self.N, (self.initial['lfoot']['pos'][3] + self.initial['rfoot']['pos'][3]) / 2.)
    mc_y = np.full(self.N, (self.initial['lfoot']['pos'][4] + self.initial['rfoot']['pos'][4]) / 2.)
    mc_z = np.full(self.N, (self.initial['lfoot']['pos'][5] + self.initial['rfoot']['pos'][5]) / 2.)
    time_array = np.array(range(t, t + self.N))
    for j in range(len(self.footstep_planner.plan) - 1):
      fs_start_time = self.footstep_planner.get_start_time(j)
      ds_start_time = fs_start_time + self.footstep_planner.plan[j]['ss_duration']
      fs_end_time = ds_start_time + self.footstep_planner.plan[j]['ds_duration']
      fs_current_pos = self.footstep_planner.plan[j]['pos'] if j > 0 else np.array([mc_x[0], mc_y[0], mc_z[0]])
      fs_target_pos = self.footstep_planner.plan[j + 1]['pos']
      mc_x += self.sigma(time_array, ds_start_time, fs_end_time) * (fs_target_pos[0] - fs_current_pos[0])
      mc_y += self.sigma(time_array, ds_start_time, fs_end_time) * (fs_target_pos[1] - fs_current_pos[1])
      mc_z += self.sigma(time_array, ds_start_time, fs_end_time) * (fs_target_pos[2] - fs_current_pos[2])

    return mc_x, mc_y, mc_z

  def generate_step_params(self, t):
    idx = self.footstep_planner.get_step_index_at_time(t)
    phase = self.footstep_planner.get_phase_at_time(t)

    # init arrays
    x_cf = np.zeros(self.f_max)
    y_cf = np.zeros(self.f_max)
    p_min_x, p_max_x = np.zeros(self.f_max), np.zeros(self.f_max)
    p_min_y, p_max_y = np.zeros(self.f_max), np.zeros(self.f_max)
    d_ax, d_ay = np.zeros(self.f_max), np.zeros(self.f_max)
    signs = np.zeros(self.f_max)

    # admissible region on planar ground
    d_ax0 = self.params['d_ax']
    d_ay0 = self.params['d_ay']
    sigma = self.params['sigma']
    dz_max = self.params['dz_max']

    for i in range(self.f_max):
        step_idx = min(idx + i, len(self.footstep_planner.plan) - 1)
        prev_idx = max(0, step_idx - 1)

        step = self.footstep_planner.plan[step_idx]
        prev_step = self.footstep_planner.plan[prev_idx]

        x_cf[i] = step['pos'][0]
        y_cf[i] = step['pos'][1]
        signs[i] = 1.0 if step['foot_id'] == 'lfoot' else -1.0

        if phase == 'ds':
            p_min_x[i] = -1e5
            p_max_x[i] = 1e5
            p_min_y[i] = -1e5
            p_max_y[i] = 1e5
            d_ax[i] = 1e5
            d_ay[i] = 1e5
        else:
            # Minkowski patch
            margin = self.foot_size / 2
            p_min_x[i] = x_cf[i] - 0.1 + margin
            p_max_x[i] = x_cf[i] + 0.1 - margin
            p_min_y[i] = y_cf[i] - 0.1 + margin
            p_max_y[i] = y_cf[i] + 0.1 - margin

            dz = abs(step['pos'][2] - prev_step['pos'][2])
            scaling = (1 - sigma * (dz / dz_max))
            d_ax[i] = scaling * d_ax0
            d_ay[i] = scaling * d_ay0

    return x_cf, y_cf, p_min_x, p_max_x, p_min_y, p_max_y, d_ax, d_ay, signs

  def compute_alpha_j(self, t):
    idx = self.footstep_planner.get_step_index_at_time(t)
    alpha = np.zeros((self.N, self.f_max))
    for k in range(self.N):
        abs_time = t + k
        step_idx = self.footstep_planner.get_step_index_at_time(abs_time)
        if step_idx is None:
            step_idx = len(self.footstep_planner.plan) - 1
        local = step_idx - idx
        local = max(0, min(local, self.f_max - 1))
        phase = self.footstep_planner.get_phase_at_time(abs_time)
        if phase == 'ds':
            fs_start = self.footstep_planner.get_start_time(step_idx)
            ds_start = fs_start + self.footstep_planner.plan[step_idx]['ss_duration']
            fs_end = ds_start + self.footstep_planner.plan[step_idx]['ds_duration']
            if fs_end > ds_start:
                sigma_val = np.clip((abs_time - ds_start) / (fs_end - ds_start), 0, 1)
            else:
                sigma_val = 1.0
            next_local = min(local + 1, self.f_max - 1)
            if local == next_local:
                alpha[k, local] = 1.0
            else:
                alpha[k, local] = 1.0 - sigma_val
                alpha[k, next_local] = sigma_val
        else:
            alpha[k, local] = 1.0
    return alpha

  def _compute_tail_integral(self, t, eta_lip):
    tail_mc_x, tail_mc_y = self.generate_tail_moving_constraint(t)
    n_tail = len(tail_mc_x)
    time_steps = np.arange(1, n_tail + 1) * self.delta
    weights = eta_lip * np.exp(-eta_lip * time_steps) * self.delta
    x_tail = np.sum(weights * tail_mc_x)
    y_tail = np.sum(weights * tail_mc_y)
    terminal_weight = np.exp(-eta_lip * n_tail * self.delta)
    x_tail += terminal_weight * tail_mc_x[-1]
    y_tail += terminal_weight * tail_mc_y[-1]
    return x_tail, y_tail

  def generate_tail_moving_constraint(self, t):
    t_tail_start = t + self.N
    P = self.params.get('P', 200)
    mid_x = (self.initial['lfoot']['pos'][3] + self.initial['rfoot']['pos'][3]) / 2.
    mid_y = (self.initial['lfoot']['pos'][4] + self.initial['rfoot']['pos'][4]) / 2.
    tail_mc_x = np.full(P - self.N, mid_x)
    tail_mc_y = np.full(P - self.N, mid_y)
    time_array = np.arange(t_tail_start, t + P)
    for j in range(len(self.footstep_planner.plan) - 1):
        fs_start_time = self.footstep_planner.get_start_time(j)
        ds_start_time = fs_start_time + self.footstep_planner.plan[j]['ss_duration']
        fs_end_time = ds_start_time + self.footstep_planner.plan[j]['ds_duration']
        fs_curr = self.footstep_planner.plan[j]['pos'] if j > 0 else np.array([mid_x, mid_y, 0.])
        fs_targ = self.footstep_planner.plan[j + 1]['pos']
        sigma_val = self.sigma(time_array, ds_start_time, fs_end_time)
        tail_mc_x += sigma_val * (fs_targ[0] - fs_curr[0])
        tail_mc_y += sigma_val * (fs_targ[1] - fs_curr[1])
    return tail_mc_x, tail_mc_y