import numpy as np
import sys
sys.path.insert(0, '.')
import ismpc, footstep_planner
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ---------------------------------------------------------------------------
# Parameters (Section 7 paper values)
# ---------------------------------------------------------------------------
params = {
    'g': 9.81, 'h': 0.70, 'foot_size': 0.08,
    'ss_duration': 50, 'ds_duration': 20,
    'world_time_step': 0.01, 'first_swing': 'rfoot',
    'N': 70,           # control horizon C  (T_c = C*delta = 0.7 s)
    'P': 180,          # preview horizon P  (T_p = P*delta = 1.8 s)
    'm': 39.0,
    'fz_min': 114.0,
    'alpha_z': 1e-5,
    'beta_z':  1e-5,
    'alpha_x': 1.0,
    'alpha_y': 1.0,
    'beta_x':  1e4,
    'd_a_x_0': 1.0,
    'd_a_y_0': 1.0,
    'ell':     0.1,      # realistic flat-ground lateral half-stride (~0.1 m for HRP-4)
    'beta_stab': 1e6,    # strong enough to keep CoM on periodic orbit vs. ZMP-tracking cost
    'N_steps': 2,        # footsteps in MPC horizon (allows adaptation with N=70, step_dur=70)
}

ell  = params['ell']
half = ell / 2      # feet sit ell/2 from sagittal centre-line
dt   = params['world_time_step']

# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------
initial = {
    'lfoot': {'pos': np.array([0., 0., 0.,  0.,  half, 0.])},
    'rfoot': {'pos': np.array([0., 0., 0.,  0., -half, 0.])},
}

# Enough reference steps to cover T_total seconds:
#   step 0: ds=20 ticks (shortened init DS); remaining steps: ss+ds=70 ticks each
# → N_ref=17 covers 20 + 16*70 = 1140 ticks = 11.4 s
vx = 0.1   # m/s forward reference
N_ref = 17
reference = [(vx, 0., 0.)] * N_ref

fp  = footstep_planner.FootstepPlanner(reference, initial['lfoot']['pos'],
                                        initial['rfoot']['pos'], params)

# --- Variable height schedule (Paper Sec. 7.1) ---
# Steps 0-3:  h = 0.70 m
# Steps 4-7:  h = 0.77 m
# Steps 8-12: linearly decrease from 0.77 to 0.58 m
for j, step in enumerate(fp.plan):
    if j <= 3:
        step['h'] = 0.70
    elif j <= 7:
        step['h'] = 0.77
    elif j <= 12:
        step['h'] = 0.77 - (j - 7) * (0.77 - 0.58) / 5.0
    else:
        step['h'] = 0.58

mpc = ismpc.Ismpc(initial, fp, params)

current = {
    # Initial DS phase: both feet on ground, ZMP transitions from centre
    # to the first support foot.  Start CoM at midpoint, near-zero velocity.
    'com':   {'pos': np.array([0., 0., params['h']]), # before: [0., -0.052, params['h']]
              'vel': np.array([0., 0.,  0.])},        # before: [0., 0., 0.]
    'zmp':   {'pos': np.zeros(3)},
    'lfoot': {'pos': initial['lfoot']['pos'].copy()},
    'rfoot': {'pos': initial['rfoot']['pos'].copy()},
}
lfoot_xy = current['lfoot']['pos'][3:5].copy()
rfoot_xy = current['rfoot']['pos'][3:5].copy()

# ---------------------------------------------------------------------------
# Simulation loop
# ---------------------------------------------------------------------------
step_dur = params['ss_duration'] + params['ds_duration']
init_ds  = params['ds_duration']                            # shortened first DS
T_total  = min(init_ds + (N_ref - 2) * step_dur, 1000)    # ticks, cap at 10 s

log_com_x = []
log_com_y = []
log_com_z = []
log_zmp_x = []
log_zmp_y = []
log_fz    = []
foot_landings = []   # (x, y, foot_id) recorded when each foot lands

lfoot_xy = current['lfoot']['pos'][3:5].copy()
rfoot_xy = current['rfoot']['pos'][3:5].copy()

prev_phase     = fp.get_phase_at_time(0)
prev_step_idx  = fp.get_step_index_at_time(0)

print(f"Running {T_total} ticks ({T_total*dt:.1f} s) ...")

for t in range(T_total):
    # ---- phase / step tracking ----
    cur_step_idx = fp.get_step_index_at_time(t)
    if cur_step_idx is None:
        break
    cur_phase = fp.get_phase_at_time(t)

    # detect SS→DS transition: swing foot just landed
    if prev_phase == 'ss' and cur_phase == 'ds':
        swing_id   = fp.plan[prev_step_idx]['foot_id']
        landed_xy  = fp.plan[prev_step_idx]['pos'][:2]
        foot_landings.append((landed_xy[0], landed_xy[1], swing_id))
        if swing_id == 'lfoot':
            lfoot_xy = landed_xy.copy()
        else:
            rfoot_xy = landed_xy.copy()

    current['lfoot']['pos'][3:5] = lfoot_xy
    current['rfoot']['pos'][3:5] = rfoot_xy

    # ---- MPC solve ----
    try:
        lip_state, _ = mpc.solve(current, t)
    except Exception as e:
        print(f"  Solver failed at tick {t}: {e}")
        break

    # ---- log ----
    log_com_x.append(lip_state['com']['pos'][0])
    log_com_y.append(lip_state['com']['pos'][1])
    log_com_z.append(lip_state['com']['pos'][2])
    log_zmp_x.append(lip_state['zmp']['pos'][0])
    log_zmp_y.append(lip_state['zmp']['pos'][1])
    log_fz.append(float(mpc.Fz_sol[0]))

    # ---- advance state ----
    current['com']['pos'] = lip_state['com']['pos'].copy()
    current['com']['vel'] = lip_state['com']['vel'].copy()
    current['zmp']['pos'] = lip_state['zmp']['pos'].copy()

    prev_phase    = cur_phase
    prev_step_idx = cur_step_idx

print(f"Done. Simulated {len(log_com_x)} ticks.")

# ---------------------------------------------------------------------------
# Build time / trajectory arrays
# ---------------------------------------------------------------------------
T_sim  = len(log_com_x)
time   = np.arange(T_sim) * dt

com_x = np.array(log_com_x)
com_y = np.array(log_com_y)
zmp_x = np.array(log_zmp_x)
zmp_y = np.array(log_zmp_y)
fz    = np.array(log_fz)

# ---------------------------------------------------------------------------
# Plot (Figure-10 style + CoM height)
# ---------------------------------------------------------------------------
fig, (ax_traj, ax_fz, ax_hz) = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Walking on flat ground – Variable CoM height (Sec. 7.1)', fontsize=12)

# ---- Left: CoM/ZMP x-y trajectory + foot rectangles ----
ax_traj.plot(com_x, com_y, color='red',  linewidth=1.5, label='CoM')
ax_traj.plot(zmp_x, zmp_y, color='blue', linewidth=1.2, label='ZMP')

fs = params['foot_size']
for (fx, fy, fid) in foot_landings:
    rect = patches.Rectangle(
        (fx - fs/2, fy - fs/2), fs, fs,
        linewidth=1.5, edgecolor='magenta', facecolor='none')
    ax_traj.add_patch(rect)

# Initial feet
for xy in [lfoot_xy, rfoot_xy]:
    pass  # already added via foot_landings if any

# draw initial foot positions (before any landing event)
for xy, label in [(initial['lfoot']['pos'][3:5], 'lfoot'),
                  (initial['rfoot']['pos'][3:5], 'rfoot')]:
    rect = patches.Rectangle(
        (xy[0] - fs/2, xy[1] - fs/2), fs, fs,
        linewidth=1.5, edgecolor='magenta', facecolor='none')
    ax_traj.add_patch(rect)

ax_traj.legend(loc='upper left', fontsize=9)
ax_traj.set_xlabel('x [m]')
ax_traj.set_ylabel('y [m]')
ax_traj.set_title('CoM/ZMP trajectories (top view)')
ax_traj.set_aspect('equal', adjustable='datalim')
ax_traj.grid(True, alpha=0.4)

# ---- Right: vertical GRF ----
ax_fz.plot(time, fz, color='magenta', linewidth=1.2, label='$f_z$')
ax_fz.axhline(params['fz_min'], color='gray', linestyle='--', linewidth=0.8,
              label=f"$f_z^{{min}}$ = {params['fz_min']} N")
ax_fz.set_xlabel('t [s]')
ax_fz.set_ylabel('$f_z$ [N]')
ax_fz.set_title('Vertical ground reaction force')
ax_fz.legend(fontsize=9)
ax_fz.grid(True, alpha=0.4)

# ---- Right: CoM height ----
com_z = np.array(log_com_z)
# Build per-tick height reference for plotting
h_ref_plot = []
for tt in range(T_sim):
    si = fp.get_step_index_at_time(tt)
    if si is None: si = len(fp.plan) - 1
    h_val = fp.plan[si].get('h', params['h'])
    h_ref_plot.append(h_val)
h_ref_plot = np.array(h_ref_plot)

ax_hz.plot(time, com_z,      color='red',   linewidth=1.5, label='$z_c$ (actual)')
ax_hz.plot(time, h_ref_plot,  color='green', linewidth=1.2, linestyle='--', label='$h^*$ (reference)')
ax_hz.set_xlabel('t [s]')
ax_hz.set_ylabel('height [m]')
ax_hz.set_title('CoM height tracking')
ax_hz.legend(fontsize=9)
ax_hz.grid(True, alpha=0.4)

plt.tight_layout()
plt.savefig('myTests/full_walk_test.png', dpi=150, bbox_inches='tight')
print("Plot saved to myTests/full_walk_test.png")
plt.show()
