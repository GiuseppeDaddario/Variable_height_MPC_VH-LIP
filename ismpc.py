import numpy as np
import casadi as cs

class Ismpc:
  def __init__(self, initial, footstep_planner, params):
    # parameters
    self.params = params
    self.N = params['N'] # C!
    self.delta = params['world_time_step']
    self.h = params['h']
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
    self.beta_y  = self.params.get("beta_y", self.beta_x)

    # d_g: flat terrain assumption --> Paper "As large as possible" (P_conv)
    self.d_g_x   = self.params.get("d_g_x", 3 * self.foot_size)
    self.d_g_y   = self.params.get("d_g_y", 3 * self.foot_size)

    # N_steps: how many footsteps fit in the MPC horizon
    step_duration = self.params["ss_duration"] + self.params["ds_duration"]
    self.N_steps  = self.params.get("N_steps", max(1, int(np.ceil(self.N / step_duration))))

    self.P = params.get('P', self.N)  # default: P = N

    p_opts = {"expand": True}
    s_opts = {"max_iter": 4000, "verbose": False,
              "eps_abs": 1e-6, "eps_rel": 1e-6,
              "polish": True, "adaptive_rho": True}

    self._init_optZ(p_opts, s_opts)
    self._init_optXY(p_opts, s_opts)

    # LIP matrices for Kalman filter (nominal constant-height model)
    eta_nom = np.sqrt(self.g / self.h)
    self.A_lip = np.array([[0, 1, 0], [eta_nom**2, 0, -eta_nom**2], [0, 0, 0]])
    self.B_lip = np.array([[0], [0], [1]])
    self.current_lambda = eta_nom**2  

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

    # Flight mask: 0 = stance (Fz >= fz_min), 1 = flight (Fz == 0)
    self.flight_mask_param = self.opt_z.parameter(self.N)
    self.fz_max = 10.0 * self.m * self.g  # generous upper bound
    self.opt_z.subject_to(self.Fz >= self.fz_min * (1 - self.flight_mask_param))
    self.opt_z.subject_to(self.Fz <= self.fz_max * (1 - self.flight_mask_param))

  def _init_optXY(self, p_opts, s_opts):
    self.opt_xy = cs.Opti('conic')
    self.opt_xy.solver("osqp", p_opts, s_opts)

    self.X_xy = self.opt_xy.variable(4, self.N + 1)  # stato [xc, dxc, yc, dyc]
    self.p_zmp = self.opt_xy.variable(2, self.N)     # ZMP [xz, yz] piecewise-constant
    S = self.S = self.N_steps
    self.X_f = self.opt_xy.variable(2, S)            # footstep positions adattabili

    # Parametri (fissati a runtime da qp_xy)
    self.x0_xy_param = self.opt_xy.parameter(4)
    self.lambda_xy_param = self.opt_xy.parameter(self.N)

    self.xmc_param = self.opt_xy.parameter(self.N)
    self.ymc_param = self.opt_xy.parameter(self.N)
    
    self.flight_mask_param_xy = self.opt_xy.parameter(self.N) # eq (14)

    self.xf_nominal_param = self.opt_xy.parameter(S) # Nominal position of the next S footsteps
    self.yf_nominal_param = self.opt_xy.parameter(S)
    
    # Kinematic limits 
    self.da_x_param = self.opt_xy.parameter(S) # eq (15)
    self.da_y_param = self.opt_xy.parameter(S)

    self.sign_param = self.opt_xy.parameter(S) # Just say if the j-th footstep is right (+) or left (-)
    self.stance_xy_param = self.opt_xy.parameter(2) # p_0

    self.swing_xy_param = self.opt_xy.parameter(2) # eq (16)

    self.rem_ss_param = self.opt_xy.parameter() # T_rem

    # Proposition 2 of Stability!
    self.tail_x_param = self.opt_xy.parameter() 
    self.tail_y_param = self.opt_xy.parameter() 

    # Condizione iniziale
    self.opt_xy.subject_to(self.X_xy[:, 0] == self.x0_xy_param)

    for i in range(self.N):
        lam_i = self.lambda_xy_param[i]
        eta = cs.sqrt(lam_i)
        f = self.flight_mask_param_xy[i]  # 0 = stance, 1 = flight

        ch = cs.cosh(eta * self.delta)
        sh = cs.sinh(eta * self.delta)
        sh_over_eta = (1 - f) * sh / (eta + 1e-8) + f * self.delta

        # Standing dynamics
        xc_next_stance = ch * self.X_xy[0, i] + sh_over_eta * self.X_xy[1, i] + (1 - ch) * self.p_zmp[0, i]
        dxc_next_stance = eta * sh * self.X_xy[0, i] + ch * self.X_xy[1, i] - eta * sh * self.p_zmp[0, i]
        
        yc_next_stance = ch * self.X_xy[2, i] + sh_over_eta * self.X_xy[3, i] + (1 - ch) * self.p_zmp[1, i]
        dyc_next_stance = eta * sh * self.X_xy[2, i] + ch * self.X_xy[3, i] - eta * sh * self.p_zmp[1, i]

        # Flight dynamics (free-flying) 
        xc_next_flight = self.X_xy[0, i] + self.delta * self.X_xy[1, i]
        dxc_next_flight = self.X_xy[1, i]

        yc_next_flight = self.X_xy[2, i] + self.delta * self.X_xy[3, i]
        dyc_next_flight = self.X_xy[3, i]

        # The constraint depends if it is flying or not
        self.opt_xy.subject_to(self.X_xy[0, i+1] == (1 - f) * xc_next_stance + f * xc_next_flight)
        self.opt_xy.subject_to(self.X_xy[1, i+1] == (1 - f) * dxc_next_stance + f * dxc_next_flight)

        self.opt_xy.subject_to(self.X_xy[2, i+1] == (1 - f) * yc_next_stance + f * yc_next_flight)
        self.opt_xy.subject_to(self.X_xy[3, i+1] == (1 - f) * dyc_next_stance + f * dyc_next_flight)

    # ZMP Constraint
    M_val = 10.0 # Used to relax ZMP constraints during flight phase (big-M method)
    self.opt_xy.subject_to(self.p_zmp[0, :].T <= self.xmc_param + self.foot_size / 2. + M_val * self.flight_mask_param_xy)
    self.opt_xy.subject_to(self.p_zmp[0, :].T >= self.xmc_param - self.foot_size / 2. - M_val * self.flight_mask_param_xy)
    self.opt_xy.subject_to(self.p_zmp[1, :].T <= self.ymc_param + self.foot_size / 2. + M_val * self.flight_mask_param_xy)
    self.opt_xy.subject_to(self.p_zmp[1, :].T >= self.ymc_param - self.foot_size / 2. - M_val * self.flight_mask_param_xy)

    # Ground patch constraint
    margin_x = (self.d_g_x - self.foot_size) / 2.
    margin_y = (self.d_g_y - self.foot_size) / 2.
    for j in range(S):
      self.opt_xy.subject_to(self.X_f[0, j] - self.xf_nominal_param[j] <=  margin_x)
      self.opt_xy.subject_to(self.X_f[0, j] - self.xf_nominal_param[j] >= -margin_x)
      self.opt_xy.subject_to(self.X_f[1, j] - self.yf_nominal_param[j] <=  margin_y)
      self.opt_xy.subject_to(self.X_f[1, j] - self.yf_nominal_param[j] >= -margin_y)

    # Kinematic constraints
    ell = self.params.get('ell', 0.25)

    # j = 0: w.r.t the current stance foot (stance_xy_param)
    self.opt_xy.subject_to(self.X_f[0, 0] - self.stance_xy_param[0] <= self.da_x_param[0] / 2.0)
    self.opt_xy.subject_to(self.X_f[0, 0] - self.stance_xy_param[0] >= -self.da_x_param[0] / 2.0)
    self.opt_xy.subject_to(self.X_f[1, 0] - self.stance_xy_param[1] + self.sign_param[0] * ell <= self.da_y_param[0] / 2.0)
    self.opt_xy.subject_to(self.X_f[1, 0] - self.stance_xy_param[1] + self.sign_param[0] * ell >= -self.da_y_param[0] / 2.0)

    # j > 0: w.r.t the previous footstep
    for j in range(1, S):
        self.opt_xy.subject_to(self.X_f[0, j] - self.X_f[0, j-1] <= self.da_x_param[j] / 2.0)
        self.opt_xy.subject_to(self.X_f[0, j] - self.X_f[0, j-1] >= -self.da_x_param[j] / 2.0)
        self.opt_xy.subject_to(self.X_f[1, j] - self.X_f[1, j-1] + self.sign_param[j] * ell <= self.da_y_param[j] / 2.0)
        self.opt_xy.subject_to(self.X_f[1, j] - self.X_f[1, j-1] + self.sign_param[j] * ell >= -self.da_y_param[j] / 2.0)

    # Swing foot constraint
    v_sw_max = self.params.get('v_sw_max', 3.0)
    swing_bound = self.rem_ss_param * v_sw_max
    self.opt_xy.subject_to(self.X_f[0, 0] - self.swing_xy_param[0] <= swing_bound)
    self.opt_xy.subject_to(self.X_f[0, 0] - self.swing_xy_param[0] >= -swing_bound)
    self.opt_xy.subject_to(self.X_f[1, 0] - self.swing_xy_param[1] <= swing_bound)
    self.opt_xy.subject_to(self.X_f[1, 0] - self.swing_xy_param[1] >= -swing_bound)

    # Stability constraint
    lam_lip_nom = self.g / self.params['h']
    eta_lip = cs.sqrt(lam_lip_nom)
    DCM_x_end = self.X_xy[0, self.N] + (1.0 / eta_lip) * self.X_xy[1, self.N]
    DCM_y_end = self.X_xy[2, self.N] + (1.0 / eta_lip) * self.X_xy[3, self.N]

    self.opt_xy.subject_to(DCM_x_end == self.tail_x_param)
    self.opt_xy.subject_to(DCM_y_end == self.tail_y_param)

    # Minimize
    _dxz = cs.diff(self.p_zmp[0, :])
    _dyz = cs.diff(self.p_zmp[1, :])
    cost_xy = (
      100 * cs.sumsqr(self.p_zmp[0, :].T - self.xmc_param) +
      100 * cs.sumsqr(self.p_zmp[1, :].T - self.ymc_param) +
      self.alpha_x * cs.sumsqr(_dxz) +
      self.alpha_y * cs.sumsqr(_dyz)
    )
    cost_xy += self.beta_x * cs.sumsqr(self.X_f[0, :].T - self.xf_nominal_param)
    cost_xy += self.beta_y * cs.sumsqr(self.X_f[1, :].T - self.yf_nominal_param)
    self.opt_xy.minimize(cost_xy)




  def solve(self, current, t):
    lambda_seq, z_sol, dz_sol = self.qp_z(current, t)

    self.current_lambda = lambda_seq[0]

    self.A_lip[1, 0] = self.current_lambda   
    self.A_lip[1, 2] = -self.current_lambda    

    X_xy_sol, p_zmp_sol = self.qp_xy(current, t, lambda_seq)

    self.lip_state['com']['pos'] = np.array([X_xy_sol[0, 1], X_xy_sol[2, 1], z_sol[1]])
    self.lip_state['com']['vel'] = np.array([X_xy_sol[1, 1], X_xy_sol[3, 1], dz_sol[1]])
    self.lip_state['zmp']['pos'] = np.array([p_zmp_sol[0, 0], p_zmp_sol[1, 0], current['zmp']['pos'][2]])

    # ZMP velocity: approximate rate of change for Kalman filter compatibility
    zmp_vel_x = (p_zmp_sol[0, 0] - current['zmp']['pos'][0]) / self.delta
    zmp_vel_y = (p_zmp_sol[1, 0] - current['zmp']['pos'][1]) / self.delta
    self.lip_state['zmp']['vel'] = np.array([zmp_vel_x, zmp_vel_y, 0.0])

    lam0 = lambda_seq[0]
    self.lip_state['com']['acc'] = np.array([
      lam0 * (X_xy_sol[0, 1] - p_zmp_sol[0, 0]),
      lam0 * (X_xy_sol[2, 1] - p_zmp_sol[1, 0]),
      (dz_sol[2] - dz_sol[1]) / self.delta if len(dz_sol) > 2 else 0.0
    ])

    # Used for Kalman
    self.x = np.array([
      X_xy_sol[0, 1], X_xy_sol[1, 1], p_zmp_sol[0, 0],
      X_xy_sol[2, 1], X_xy_sol[3, 1], p_zmp_sol[1, 0],
      z_sol[1], dz_sol[1], current['zmp']['pos'][2]
    ])

    contact = self.footstep_planner.get_phase_at_time(t)
    if contact == 'ss':
      contact = self.footstep_planner.plan[self.footstep_planner.get_step_index_at_time(t)]['foot_id']

    return self.lip_state, contact
  
  def generate_moving_constraint(self, t, length=None):
    if length is None:
      length = self.N
    mc_x = np.full(length, (self.initial['lfoot']['pos'][3] + self.initial['rfoot']['pos'][3]) / 2.)
    mc_y = np.full(length, (self.initial['lfoot']['pos'][4] + self.initial['rfoot']['pos'][4]) / 2.)
    mc_z = np.full(length, (self.initial['lfoot']['pos'][5] + self.initial['rfoot']['pos'][5]) / 2.)
    time_array = np.array(range(t, t + length))
    for j in range(len(self.footstep_planner.plan) - 1):
      fs_start_time = self.footstep_planner.get_start_time(j)
      ds_start_time = fs_start_time + self.footstep_planner.plan[j]['ss_duration']
      fs_end_time = ds_start_time + self.footstep_planner.plan[j]['ds_duration']
      fs_current_pos = self.footstep_planner.plan[j]['pos'] if j > 0 else np.array([mc_x[0], mc_y[0], mc_z[0]])
      fs_target_pos = self.footstep_planner.plan[j + 1]['pos']
      sig = self.sigma(time_array, ds_start_time, fs_end_time)
      mc_x += sig * (fs_target_pos[0] - fs_current_pos[0])
      mc_y += sig * (fs_target_pos[1] - fs_current_pos[1])
      mc_z += sig * (fs_target_pos[2] - fs_current_pos[2])

    return mc_x, mc_y, mc_z
  
  def _get_flight_mask(self, t):
    # TODO: At the moment it is never in flight
    mask = np.zeros(self.N)
    for k in range(self.N):
      phase = self.footstep_planner.get_phase_at_time(t + k)
      if phase == 'flight':
        mask[k] = 1.0
    return mask

  def generateHeightRef(self, mc_z, t):
    h_ref = np.full(self.N, self.params['h'])
    for k in range(self.N):
      step_idx = self.footstep_planner.get_step_index_at_time(t + k)
      if step_idx is None:
        step_idx = len(self.footstep_planner.plan) - 1
      step = self.footstep_planner.plan[step_idx]
      if 'h' in step:
        h_ref[k] = step['h']
    return mc_z + h_ref

  def qp_z(self, current, t):
    _, _, pz = self.generate_moving_constraint(t)
    z_ref = self.generateHeightRef(pz, t)

    z0  = current['com']['pos'][2]
    dz0 = current['com']['vel'][2]

    self.opt_z.set_value(self.z0_param, z0)
    self.opt_z.set_value(self.dz0_param, dz0)
    self.opt_z.set_value(self.z_ref_param, z_ref)
    self.opt_z.set_value(self.flight_mask_param, self._get_flight_mask(t))

    sol_z   = self.opt_z.solve()
    Fz_sol  = sol_z.value(self.Fz)
    self.Fz_sol = Fz_sol  # store for external access

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
    lambda_seq = np.maximum((self.g + ddz_sol) / denom, 0.0)

    return lambda_seq, z_sol, dz_sol

    

  def qp_xy(self, current, t, lambda_seq):
    mc_x_full, mc_y_full, _ = self.generate_moving_constraint(t, length=self.P)
    mc_x = mc_x_full[:self.N]  # primi C campioni per vincoli & costo
    mc_y = mc_y_full[:self.N]

    x0_xy = np.array([
        current['com']['pos'][0], current['com']['vel'][0],
        current['com']['pos'][1], current['com']['vel'][1],
    ])
    self.opt_xy.set_value(self.x0_xy_param, x0_xy)
    self.opt_xy.set_value(self.lambda_xy_param, lambda_seq)
    self.opt_xy.set_value(self.flight_mask_param_xy, self._get_flight_mask(t))
    self.opt_xy.set_value(self.xmc_param, mc_x)
    self.opt_xy.set_value(self.ymc_param, mc_y)

    # Ground patch constraint
    xf_nom, yf_nom = self._get_nominal_footsteps(t)
    self.opt_xy.set_value(self.xf_nominal_param, xf_nom)
    self.opt_xy.set_value(self.yf_nominal_param, yf_nom)

    # Kinematic consttraint
    da_x, da_y = self.calculate_kinematic_limits(t)
    self.opt_xy.set_value(self.da_x_param, da_x)
    self.opt_xy.set_value(self.da_y_param, da_y)
    signs, stance_xy, swing_xy, rem_ss = self._get_footstep_info(t)
    self.opt_xy.set_value(self.sign_param, signs)
    self.opt_xy.set_value(self.stance_xy_param, stance_xy)

    # Seing foot constraint
    self.opt_xy.set_value(self.swing_xy_param, swing_xy)
    self.opt_xy.set_value(self.rem_ss_param, rem_ss)

    # Stability constraint
    tail_x, tail_y = self.calculate_stability_tail(mc_x_full, mc_y_full)
    self.opt_xy.set_value(self.tail_x_param, tail_x)
    self.opt_xy.set_value(self.tail_y_param, tail_y)

    # Solve
    sol_xy = self.opt_xy.solve()
    self.opt_xy.set_initial(self.X_xy, sol_xy.value(self.X_xy))
    self.opt_xy.set_initial(self.p_zmp, sol_xy.value(self.p_zmp))

    return sol_xy.value(self.X_xy), sol_xy.value(self.p_zmp)

  def _get_footstep_info(self, t):
    # Compute sign +1 left, -1 right, stance foot position, swing foot position, and remaining swing time for the next step.
    step_idx = self.footstep_planner.get_step_index_at_time(t)
    plan = self.footstep_planner.plan
    phase = self.footstep_planner.get_phase_at_time(t)

    signs = np.zeros(self.S)
    for j in range(self.S):
      idx = min(step_idx + j + 1, len(plan) - 1)
      signs[j] = 1.0 if plan[idx]['foot_id'] == 'rfoot' else -1.0

    current_step = plan[step_idx]
    stance_xy = current_step['pos'][:2]

    next_step_idx = min(step_idx + 1, len(plan) - 1)
    next_swing_id = plan[next_step_idx]['foot_id']

    # Find the swing foot's START position (where it was before this swing)
    swing_start = None
    for k in range(step_idx, -1, -1):
      if plan[k]['foot_id'] == next_swing_id:
        swing_start = plan[k]['pos'][:2].copy()
        break
    if swing_start is None:
      swing_start = self.initial[next_swing_id]['pos'][3:5].copy()

    # Remaining time and swing foot interpolation
    start_time = self.footstep_planner.get_start_time(step_idx)
    if phase == 'ds':
      ds_start = start_time + current_step['ss_duration']
      ds_end = ds_start + current_step['ds_duration']
      rem_ds = (ds_end - t) * self.delta
      next_ss = plan[next_step_idx]['ss_duration'] * self.delta
      rem_ss = max(rem_ds + next_ss, 1e-3)
      swing_xy = swing_start  # foot hasn't started moving yet
    else:
      ss_duration = current_step['ss_duration']
      time_in_step = t - start_time
      rem_ss = max((ss_duration - time_in_step) * self.delta, 1e-3)

      # Interpolate swing foot along nominal trajectory (Eq. 16).
      if ss_duration > 0:
        alpha = min(time_in_step / ss_duration, 1.0)
        target_xy = plan[next_step_idx]['pos'][:2]
        swing_xy = swing_start + alpha * (target_xy - swing_start)
      else:
        swing_xy = swing_start

    return signs, stance_xy, swing_xy, rem_ss
  
  def _get_nominal_footsteps(self, t):
    #Get nominal candidate
    step_idx = self.footstep_planner.get_step_index_at_time(t)
    plan = self.footstep_planner.plan
    xf_nom = np.zeros(self.S)
    yf_nom = np.zeros(self.S)
    for j in range(self.S):
      idx = min(step_idx + j + 1, len(plan) - 1)
      xf_nom[j] = plan[idx]['pos'][0]
      yf_nom[j] = plan[idx]['pos'][1]
    return xf_nom, yf_nom



  def calculate_kinematic_limits(self, t):
    da_x = np.zeros(self.S)
    da_y = np.zeros(self.S)
    d0_x = self.params.get('da_x', 1.0) 
    d0_y = self.params.get('da_y', 0.12)
    sigma_scale = 0.5 # sigma (from the paper)
    dz_max = 0.5     # Δz_max (from the paper)

    curr_idx = self.footstep_planner.get_step_index_at_time(t)
    for j in range(self.S):
        idx = min(curr_idx + j, len(self.footstep_planner.plan)-1)
        prev_idx = max(0, idx - 1)
            
        dz = abs(self.footstep_planner.plan[idx]['pos'][2] - 
                 self.footstep_planner.plan[prev_idx]['pos'][2])
            
        scale = 1.0 - sigma_scale * (dz / dz_max)
        da_x[j] = scale * d0_x
        da_y[j] = scale * d0_y
    return da_x, da_y

  def calculate_stability_tail(self, mc_x_full, mc_y_full):  # Prop. 2
    eta = np.sqrt(self.g / self.h)  
    C = self.N   
    P = self.P   
    L = P - C    # number of tail intervals

    if L <= 0:
        return mc_x_full[C - 1], mc_y_full[C - 1]

    j = np.arange(L)
    exp_j  = np.exp(-eta * j * self.delta)
    exp_j1 = np.exp(-eta * (j + 1) * self.delta)
    w = exp_j - exp_j1                  
    w_tail = exp_j1[-1]                 

    mc_x_preview = mc_x_full[C:P]
    mc_y_preview = mc_y_full[C:P]

    mc_x_end = mc_x_full[P - 1]
    mc_y_end = mc_y_full[P - 1]

    tail_x = np.dot(w, mc_x_preview) + w_tail * mc_x_end
    tail_y = np.dot(w, mc_y_preview) + w_tail * mc_y_end

    return tail_x, tail_y