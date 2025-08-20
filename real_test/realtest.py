import os, math, time, csv, threading, argparse
from datetime import datetime
from contextlib import contextmanager

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx  # for MST

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.syncLogger import SyncLogger
from cflib.crazyflie.log import LogConfig
from cflib.positioning.position_hl_commander import PositionHlCommander

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# USER CONFIG
URI = 'radio://0/80/2M/E7E7E7E7E7'

CRUISE_Z     = 0.35
TAKEOFF_Z    = 0.35
SPEED_XY     = 0.4
TRAVEL_SPEED = 0.5
HOVER_TIME   = 4.0

DEFAULT_ALGO = 'mst' # 'greedy' | 'twoopt' | 'energy2opt' | 'mst'
DEFAULT_POINTSET = 5     # 1-5


def _build_rel_points_from_seed(seed: int):
    rng = np.random.default_rng(seed)
    pts = []
    while len(pts) < 10:
        x, y = rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5)
        if abs(x) < 1e-6 and abs(y) < 1e-6:
            continue  # skip exact origin
        pts.append((round(float(x), 3), round(float(y), 3)))
    return pts


_POINTSET_SEEDS = {
    1: 101,
    2: 202,
    3: 303,
    4: 404,
    5: 505,
}

def build_rel_points(pointset: int = DEFAULT_POINTSET, seed: int | None = None):
    """
    If `seed` is given, use that exact seed (repeatable custom set).
    Otherwise choose one of the 5 predefined sets with `pointset` in {1..5}.
    """
    if seed is not None:
        return _build_rel_points_from_seed(int(seed))
    if pointset not in _POINTSET_SEEDS:
        raise ValueError("pointset must be in {1,2,3,4,5}")
    return _build_rel_points_from_seed(_POINTSET_SEEDS[pointset])


FLOWERS_REL = None

# RF predictor "thrust labels" (for prediction only)
HOVER_THRUST  = 40000
TRAVEL_THRUST = 35000

#Sag / Recovery model
STARTUP_SAG_BASE_V     = 0.14
STARTUP_SAG_PER_TNORM  = 1.00
SAG_SOC_REF_V          = 4.00
SAG_SOC_GAIN           = 1.60

SAG_RISE_TAU_S         = 0.6
SETTLE_REBOUND_FRAC    = 0.05
SETTLE_REBOUND_TAU_S   = 1.5

LAND_REBOUND_FRAC      = 0.98
LAND_REBOUND_CAP_V     = 0.8
LAND_REBOUND_TAU_S     = 4.0

#Battery / safety
VBAT_HARD_MIN      = 3.20
HARD_GRACE_SEC     = 6.0
HARD_DEBOUNCE_SEC  = 2.0
VBAT_EMA_ALPHA     = 0.10
LOG_HZ             = 10

#RF model training
def train_rf_from_csv(csv_path='combined_data.csv'):
    df = pd.read_csv(csv_path)
    df = df[df['thrust'] > 0].copy()
    df['time_diff'] = df['timestamp (s)'].diff().fillna(0)
    df = df[df['time_diff'] > 0]
    df['prev_voltage'] = df['battery_voltage (V)'].shift(1).bfill()
    df['voltage_drop'] = df['prev_voltage'] - df['battery_voltage (V)']
    df = df[(df['voltage_drop'] >= 0) & (df['voltage_drop'] < 0.05)]
    X = df[['time_diff', 'thrust', 'prev_voltage']]
    y = df['voltage_drop']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def predict_vdrop(model, t_sec, thrust, prev_v):
    X = pd.DataFrame([[t_sec, thrust, prev_v]], columns=['time_diff', 'thrust', 'prev_voltage'])
    return float(model.predict(X)[0])


from typing import ContextManager

@contextmanager
def log_estimator(scf, period_ms=100):
    lc = LogConfig(name='est', period_in_ms=period_ms)
    lc.add_variable('kalman.varPX', 'float')
    lc.add_variable('kalman.varPY', 'float')
    lc.add_variable('kalman.varPZ', 'float')
    with SyncLogger(scf, lc) as lg:
        yield lg

@contextmanager
def log_state(scf, period_ms=max(10, int(1000//LOG_HZ))):
    lc = LogConfig(name='state', period_in_ms=period_ms)
    lc.add_variable('stateEstimate.x', 'float')
    lc.add_variable('stateEstimate.y', 'float')
    lc.add_variable('stateEstimate.z', 'float')
    lc.add_variable('pm.vbat', 'float')
    with SyncLogger(scf, lc) as lg:
        yield lg

def wait_for_estimator(scf, max_wait=15.0, settle=2.0):
    print('[INFO] Waiting for estimator to settle...')
    t0 = time.time(); stable_since = None
    with log_estimator(scf, 100) as lg:
        for _ts, d, _ in lg:
            ok = (d.get('kalman.varPX', 1.0) < 0.003 and
                  d.get('kalman.varPY', 1.0) < 0.003 and
                  d.get('kalman.varPZ', 1.0) < 0.003)
            if ok:
                stable_since = stable_since or time.time()
                if time.time() - stable_since >= settle: return True
            else:
                stable_since = None
            if time.time() - t0 > max_wait: return False

def read_state_once(scf):
    with log_state(scf, 50) as lg:
        for _ts, d, _ in lg:
            return (d.get('stateEstimate.x', 0.0),
                    d.get('stateEstimate.y', 0.0),
                    d.get('stateEstimate.z', 0.0),
                    d.get('pm.vbat', 0.0))
    return 0.0, 0.0, 0.0, 0.0


class CsvLogger:
    def __init__(self, scf, prefix='cf_energy_compare', period_ms=max(10, int(1000//LOG_HZ))):
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_path = f'{prefix}_{ts}.csv'
        self.period_ms = period_ms
        self._stop = threading.Event()
        self._thr = None
        self.scf = scf
        self.t0 = None

    def start(self):
        self.t0 = time.time()
        self._thr = threading.Thread(target=self._run, daemon=True)
        self._thr.start()
        print(f'[INFO] Logging flight to {self.csv_path}')

    def stop(self):
        self._stop.set()
        if self._thr: self._thr.join(timeout=2.0)

    def _run(self):
        try:
            with open(self.csv_path, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['t','x','y','z','vbat'])
                with log_state(self.scf, self.period_ms) as lg:
                    for _ts, d, _ in lg:
                        if self._stop.is_set(): break
                        t = time.time() - self.t0
                        w.writerow([f'{t:.3f}',
                                    f"{d.get('stateEstimate.x', float('nan')):.4f}",
                                    f"{d.get('stateEstimate.y', float('nan')):.4f}",
                                    f"{d.get('stateEstimate.z', float('nan')):.4f}",
                                    f"{d.get('pm.vbat', float('nan')):.3f}"])
        except Exception as e:
            print('[WARN] CSV logger error:', e)


def euclidean(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def greedy_order(coords):
    path, unv = [0], set(range(1, len(coords)))
    while unv:
        last = path[-1]
        nxt = min(unv, key=lambda j: euclidean(coords[j], coords[last]))
        path.append(nxt); unv.remove(nxt)
    path.append(0)
    return path

def path_length(order, coords):
    return sum(euclidean(coords[order[i]], coords[order[i+1]]) for i in range(len(order)-1))

def two_opt(order, coords):
    best = order[:]
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best)-2):
            for j in range(i+1, len(best)-1):
                if j - i == 1: 
                    continue
                new = best[:i] + best[i:j][::-1] + best[j:]
                if path_length(new, coords) < path_length(best, coords):
                    best = new
                    improved = True
                    break
            if improved:
                break
    return best

def mst_dfs_order(coords):
    G = nx.Graph()
    for i in range(len(coords)):
        G.add_node(i)
    for i in range(len(coords)):
        for j in range(i+1, len(coords)):
            G.add_edge(i, j, weight=euclidean(coords[i], coords[j]))
    mst = nx.minimum_spanning_tree(G)
    order = list(nx.dfs_preorder_nodes(mst, source=0))
    order.append(0)
    return order


def mission_segments_from_order(order, coords, speed=TRAVEL_SPEED, hover_time=HOVER_TIME):
    segs = []
    for i in range(1, len(order)):
        a = coords[order[i-1]]; b = coords[order[i]]
        d = euclidean(a, b)
        t_travel = d / speed
        include_hover = (i < len(order)-1)
        segs.append({'from': a, 'to': b, 't_travel': t_travel, 'hover': hover_time if include_hover else 0.0})
    return segs

def energy_cost_of_order(order, coords, rf_model, v0):
    v = float(v0)
    cost = 0.0
    segs = mission_segments_from_order(order, coords)
    tnorm = TRAVEL_THRUST/65535.0
    sag_v = STARTUP_SAG_BASE_V + STARTUP_SAG_PER_TNORM*tnorm + SAG_SOC_GAIN*max(0.0, SAG_SOC_REF_V - v0)
    v -= sag_v * (1.0 - math.exp(-SAG_RISE_TAU_S / max(SAG_RISE_TAU_S, 1e-6)))
    v += (SETTLE_REBOUND_FRAC*sag_v) * (1.0 - math.exp(-SETTLE_REBOUND_TAU_S / max(SETTLE_REBOUND_TAU_S,1e-6)))

    for seg in segs:
        if seg['t_travel'] > 0:
            dv = predict_vdrop(rf_model, seg['t_travel'], TRAVEL_THRUST, v)
            cost += dv; v = max(v - dv, 0.0)
        if seg['hover'] > 0:
            dv = predict_vdrop(rf_model, seg['hover'], HOVER_THRUST, v)
            cost += dv; v = max(v - dv, 0.0)
    return cost

def energy_aware_2opt(coords, rf_model, v0, start_order=None):
    base = start_order if start_order is not None else greedy_order(coords)
    best = base[:]
    best_cost = energy_cost_of_order(best, coords, rf_model, v0)
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best)-2):
            for j in range(i+1, len(best)-1):
                if j - i == 1:
                    continue
                candidate = best[:i] + best[i:j][::-1] + best[j:]
                cand_cost = energy_cost_of_order(candidate, coords, rf_model, v0)
                if cand_cost < best_cost:
                    best = candidate
                    best_cost = cand_cost
                    improved = True
                    break
            if improved:
                break
    return best


def _startup_sag_amplitude(first_thrust, v0):
    tnorm = max(0.0, min(1.0, first_thrust / 65535.0))
    thrust_part = STARTUP_SAG_PER_TNORM * tnorm
    soc_part = SAG_SOC_GAIN * max(0.0, SAG_SOC_REF_V - float(v0))
    return STARTUP_SAG_BASE_V + thrust_part + soc_part

def build_predicted_voltage_trace_with_sag(rf_model, v0, segments):
    t, v = 0.0, float(v0)
    t_list = [t]; v_list = [v]

  
    first_thrust = TRAVEL_THRUST if (segments and segments[0]['t_travel'] > 0) else HOVER_THRUST
    sag_v = _startup_sag_amplitude(first_thrust, v0)
    settle_v = SETTLE_REBOUND_FRAC * sag_v

    if SAG_RISE_TAU_S > 0.0 and sag_v > 0.0:
        dt = SAG_RISE_TAU_S
        drop = sag_v * (1.0 - math.exp(-dt / SAG_RISE_TAU_S))
        v = max(v - drop, 0.0); t += dt
        t_list.append(t); v_list.append(v)

    if SETTLE_REBOUND_FRAC > 0.0 and SETTLE_REBOUND_TAU_S > 0.0:
        dt = SETTLE_REBOUND_TAU_S
        add = settle_v * (1.0 - math.exp(-dt / SETTLE_REBOUND_TAU_S))
        v = v + add; t += dt
        t_list.append(t); v_list.append(v)

    for seg in segments:
        if seg['t_travel'] > 0:
            dv_travel = predict_vdrop(rf_model, seg['t_travel'], TRAVEL_THRUST, v)
            v = max(v - dv_travel, 0.0); t += seg['t_travel']
            t_list.append(t); v_list.append(v)
        if seg['hover'] > 0:
            dv_hover = predict_vdrop(rf_model, seg['hover'], HOVER_THRUST, v)
            v = max(v - dv_hover, 0.0); t += seg['hover']
            t_list.append(t); v_list.append(v)

    
    if LAND_REBOUND_FRAC > 0.0:
        target_rebound = min(LAND_REBOUND_FRAC * sag_v, LAND_REBOUND_CAP_V)
        if target_rebound > 0.0 and LAND_REBOUND_TAU_S > 0.0:
            v_target = v + target_rebound
            steps = 20
            total = 4.0 * LAND_REBOUND_TAU_S
            dt = total / steps
            for _ in range(steps):
                v = v_target - (v_target - v) * math.exp(-dt / LAND_REBOUND_TAU_S)
                t += dt
                t_list.append(t); v_list.append(v)

    return np.array(t_list), np.array(v_list)

def interp_pred_to_actual(t_pred, v_pred, t_actual):
    return np.interp(t_actual, t_pred, v_pred, left=v_pred[0], right=v_pred[-1])

#Sag-aware low-battery guard
class HardLowGuard:
    def __init__(self, vmin=VBAT_HARD_MIN, grace=HARD_GRACE_SEC, debounce=HARD_DEBOUNCE_SEC, alpha=VBAT_EMA_ALPHA):
        self.vmin=vmin; self.grace=grace; self.debounce=debounce; self.alpha=alpha
        self.ema=None; self.below_since=None; self.t_takeoff=None
    def start(self):
        self.ema=None; self.below_since=None; self.t_takeoff=time.time()
    def update(self, v_now):
        self.ema = float(v_now) if self.ema is None else self.alpha*float(v_now) + (1-self.alpha)*self.ema
        t_since = time.time() - (self.t_takeoff or time.time())
        if t_since < self.grace:
            self.below_since=None; return False
        if self.ema < self.vmin:
            if self.below_since is None: self.below_since=time.time()
            elif time.time() - self.below_since >= self.debounce: return True
        else:
            self.below_since=None
        return False

#Planning wrapper
def build_order(coords, algo, rf_model, v0):
    algo = algo.lower()
    if algo == 'greedy':
        return greedy_order(coords)
    elif algo == 'twoopt':
        return two_opt(greedy_order(coords), coords)
    elif algo == 'energy2opt':
        start = greedy_order(coords)
        return energy_aware_2opt(coords, rf_model, v0, start_order=start)
    elif algo == 'mst':
        return mst_dfs_order(coords)
    else:
        raise ValueError(f'Unknown algo: {algo}')

def plan_segments_from_algo(home, flowers_world, algo, rf_model, v0, speed=TRAVEL_SPEED, hover_time=HOVER_TIME):
    coords = [home] + flowers_world
    order = build_order(coords, algo, rf_model, v0)
    segs = mission_segments_from_order(order, coords, speed, hover_time)
    return order, segs

#Mission
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', default=DEFAULT_ALGO, choices=['greedy','twoopt','energy2opt','mst'],
                        help='Path planning algorithm to use')
    parser.add_argument('--pointset', type=int, default=DEFAULT_POINTSET, choices=[1,2,3,4,5],
                        help='Choose which 10-point set (1..5) to use')
    parser.add_argument('--seed', type=int, default=None,
                        help='Optional custom seed (overrides --pointset)')
    args = parser.parse_args()
    algo = args.algo.lower()

    # Build the selected points (relative to launch)
    rel_pts = build_rel_points(pointset=args.pointset, seed=args.seed)
    global FLOWERS_REL
    FLOWERS_REL = rel_pts

    print('[INFO] Training RF voltage-drop model from combined_data.csv …')
    rf_model = train_rf_from_csv('combined_data.csv')
    print('[INFO] RF model ready.')
    print(f'[INFO] Using planner: {algo}')
    print(f'[INFO] Using point set: seed={args.seed if args.seed is not None else _POINTSET_SEEDS[args.pointset]} ({len(FLOWERS_REL)} points)')

    cflib.crtp.init_drivers(enable_debug_driver=False)
    print('[INFO] Connecting to', URI)
    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
        scf.cf.param.set_value('stabilizer.estimator', '2'); time.sleep(0.1)
        if not wait_for_estimator(scf):
            print('[ERROR] Estimator did not settle.'); return

        x0, y0, z0, v0 = read_state_once(scf)
        if v0 <= 0: print('[ERROR] Invalid initial vbat.'); return
        print(f'[INFO] Start: x={x0:.3f} y={y0:.3f} z={z0:.3f} vbat={v0:.2f} V')

        flowers_world = [(x0+dx, y0+dy) for (dx,dy) in FLOWERS_REL]
        home = (x0, y0)
        order, segments = plan_segments_from_algo(home, flowers_world, algo, rf_model, v0,
                                                  speed=TRAVEL_SPEED, hover_time=HOVER_TIME)

        # Predicted trace with sag + rebound
        t_pred, v_pred = build_predicted_voltage_trace_with_sag(rf_model, v0, segments)

    
        tag = f"{algo}_seed{args.seed}" if args.seed is not None else f"{algo}_set{args.pointset}"
        logger = CsvLogger(scf, prefix=f'cf_energy_compare_{tag}', period_ms=max(10, int(1000//LOG_HZ)))
        logger.start()

        visited = 0
        guard = HardLowGuard()
        try:
            with PositionHlCommander(scf, default_velocity=SPEED_XY, default_height=TAKEOFF_Z, x=x0, y=y0, z=0.0) as pc:
                print(f'[INFO] PositionHL engaged (auto‑takeoff to ~{TAKEOFF_Z:.2f} m).')
                time.sleep(1.0); guard.start()

                coords_full = [home] + flowers_world
                for i in range(1, len(order)):
                    target = coords_full[order[i]]
                    _x, _y, _z, vnow = read_state_once(scf)
                    if guard.update(vnow):
                        print('[WARN] <=3.20 V (sag-aware). RTB.'); break

                    print(f'[INFO] -> go_to({target[0]:.2f}, {target[1]:.2f}, {CRUISE_Z:.2f})')
                    pc.go_to(target[0], target[1], CRUISE_Z)

                    is_final_leg = (i == len(order)-1)
                    if not is_final_leg:
                        t0 = time.time()
                        while time.time()-t0 < HOVER_TIME:
                            time.sleep(0.2)
                            _x, _y, _z, vnow = read_state_once(scf)
                            if guard.update(vnow):
                                print('[WARN] <=3.20 V during hover. RTB.'); is_final_leg=True; break
                        if is_final_leg: break
                        visited += 1

                print('[INFO] Return home & land…')
                pc.go_to(home[0], home[1], CRUISE_Z); time.sleep(0.3)
                pc.land(velocity=0.4); time.sleep(0.4)
        finally:
            logger.stop()

        #Compare predicted vs actual (percent error series)
        df_actual = pd.read_csv(logger.csv_path)
        t_actual = df_actual['t'].to_numpy(dtype=float)
        v_actual = df_actual['vbat'].to_numpy(dtype=float)
        v_pred_on_actual = np.interp(t_actual, t_pred, v_pred, left=v_pred[0], right=v_pred[-1])

        err = v_pred_on_actual - v_actual
        abs_err = np.abs(err)
        pct_err = np.where(v_actual!=0, 100.0*err/v_actual, np.nan)
        pct_abs_err = np.abs(pct_err)

        rmse = float(np.sqrt(np.mean(err**2))) if len(v_actual) else float('nan')
        final_diff = float(v_actual[-1] - v_pred_on_actual[-1]) if len(v_actual) else float('nan')
        mean_abs = float(np.mean(abs_err)) if len(v_actual) else float('nan')
        mean_pct_abs = float(np.nanmean(pct_abs_err)) if len(v_actual) else float('nan')
        max_pct_abs = float(np.nanmax(pct_abs_err)) if len(v_actual) else float('nan')

        base = os.path.splitext(logger.csv_path)[0]
        comp_csv = base + '_compare.csv'
        comp_png = base + '_compare.png'
        pct_png  = base + '_pcterr.png'

        pd.DataFrame({'t':t_actual,'v_actual':v_actual,'v_pred':v_pred_on_actual,
                      'err_V':err,'abs_err_V':abs_err,'pct_err':pct_err,'pct_abs_err':pct_abs_err}).to_csv(comp_csv, index=False)

        plt.figure()
        plt.plot(t_actual, v_actual, label='Actual vbat (V)', linewidth=2)
        plt.plot(t_actual, v_pred_on_actual, '--', label='Predicted vbat (V)', linewidth=2)
        plt.xlabel('Time (s)'); plt.ylabel('Voltage (V)')
        plt.title(f'{algo.capitalize()} | points: {tag}\nRMSE={rmse:.3f} V | |Δ_final|={abs(final_diff):.3f} V')
        plt.grid(True); plt.legend(); plt.tight_layout()
        plt.savefig(comp_png, dpi=150)

        plt.figure()
        plt.plot(t_actual, pct_err, linewidth=1.5)
        plt.xlabel('Time (s)'); plt.ylabel('Percent error (%)')
        plt.title(f'{algo.capitalize()} | points: {tag}\nMean |%err|={mean_pct_abs:.1f}% | Max |%err|={max_pct_abs:.1f}%')
        plt.grid(True); plt.tight_layout()
        plt.savefig(pct_png, dpi=150)

        final_v = float(v_actual[-1]) if len(v_actual) else float('nan')
        print(f'[RESULT] Algorithm: {algo}')
        print(f'[RESULT] Points: {tag}')
        print(f'[RESULT] Visited flowers (actual): {visited}/{len(FLOWERS_REL)}')
        print(f'[RESULT] Final voltage: {final_v:.2f} V')
        print(f'[RESULT] Comparison saved: {comp_csv}')
        print(f'[RESULT] Plots saved:      {comp_png}, {pct_png}')
        print(f'[RESULT] RMSE={rmse:.3f} V | Mean |err|={mean_abs:.3f} V | Mean |%err|={mean_pct_abs:.1f}% | Max |%err|={max_pct_abs:.1f}%')

if __name__ == '__main__':
    main()
