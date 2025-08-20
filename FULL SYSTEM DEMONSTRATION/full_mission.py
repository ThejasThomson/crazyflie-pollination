import os, math, time, csv, threading
from datetime import datetime
from contextlib import contextmanager

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

from ultralytics import YOLO

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.syncLogger import SyncLogger
from cflib.crazyflie.log import LogConfig
from cflib.positioning.position_hl_commander import PositionHlCommander

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

#USER CONFIG
URI = 'radio://0/80/2M/E7E7E7E7E7'


IMAGE_PATH = r"E:\Downloads\new8.jpg"        
YOLO_WEIGHTS = 'runs/detect/train13/weights/best.pt'  
YOLO_CONF = 0.8                        
YOLO_CLASS_NAME = 'strawberry'         

CRUISE_Z     = 0.35
TAKEOFF_Z    = 0.35
SPEED_XY     = 0.4
TRAVEL_SPEED = 0.5
HOVER_TIME   = 4.0


HOVER_THRUST  = 40000
TRAVEL_THRUST = 35000

#Sag / Recovery model
STARTUP_SAG_BASE_V     = 0.14
STARTUP_SAG_PER_TNORM  = 1.00
SAG_RISE_TAU_S         = 0.6
SETTLE_REBOUND_FRAC    = 0.05
SETTLE_REBOUND_TAU_S   = 1.5
LAND_REBOUND_FRAC      = 0.98
LAND_REBOUND_CAP_V     = 0.80

VBAT_HARD_MIN      = 3.20
HARD_GRACE_SEC     = 6.0
HARD_DEBOUNCE_SEC  = 2.0
VBAT_EMA_ALPHA     = 0.10
LOG_HZ             = 10

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
    model = RandomForestRegressor(n_estimators=120, random_state=42)
    model.fit(X_train, y_train)
    return model

def predict_vdrop(model, t_sec, thrust, prev_v):
    X = pd.DataFrame([[t_sec, thrust, prev_v]], columns=['time_diff', 'thrust', 'prev_voltage'])
    return float(model.predict(X)[0])

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


def greedy_order(coords):
    path, unv = [0], set(range(1, len(coords)))
    while unv:
        last = path[-1]
        nxt = min(unv, key=lambda j: math.hypot(coords[j][0]-coords[last][0],
                                                coords[j][1]-coords[last][1]))
        path.append(nxt); unv.remove(nxt)
    path.append(0)
    return path

def plan_segments(home, flowers_world, speed=TRAVEL_SPEED, hover_time=HOVER_TIME):
    coords = [home] + flowers_world
    order = greedy_order(coords)
    segments = []
    for i in range(1, len(order)):
        a = coords[order[i-1]]; b = coords[order[i]]
        d = math.hypot(b[0]-a[0], b[1]-a[1]); t_travel = d / speed if speed > 0 else 0.0
        include_hover = (i < len(order)-1)
        segments.append({'from':a, 'to':b, 't_travel':t_travel, 'hover': hover_time if include_hover else 0.0})
    return order, segments


def _startup_sag_amplitude(first_thrust):
    tnorm = max(0.0, min(1.0, first_thrust / 65535.0))
    return STARTUP_SAG_BASE_V + STARTUP_SAG_PER_TNORM * tnorm

def build_predicted_voltage_trace_with_sag(rf_model, v0, segments):
    first_thrust = TRAVEL_THRUST if (segments and segments[0]['t_travel'] > 0) else HOVER_THRUST
    sag_v = _startup_sag_amplitude(first_thrust)
    settle_v = SETTLE_REBOUND_FRAC * sag_v

    t = 0.0
    v = float(v0)
    t_list = [t]; v_list = [v]

    if SAG_RISE_TAU_S > 0.0 and sag_v > 0.0:
        dt = SAG_RISE_TAU_S
        drop = sag_v * (1.0 - math.exp(-dt / SAG_RISE_TAU_S))
        v = max(v - drop, 0.0); t += dt
        t_list.append(t); v_list.append(v)

    if SETTLE_REBOUND_FRAC > 0.0 and SETTLE_REBOUND_TAU_S > 0.0:
        dt = SETTLE_REBOUND_TAU_S
        add = settle_v * (1.0 - math.exp(-dt / SETTLE_REBOUND_TAU_S))
        v = v + add
        t += dt
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
        rebound = min(LAND_REBOUND_FRAC * sag_v, LAND_REBOUND_CAP_V)
        if rebound > 0.0:
            t += 1.0
            v = v + rebound
            t_list.append(t); v_list.append(v)

    return np.array(t_list), np.array(v_list)

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

def get_aruco_detector():
  
    aruco = cv2.aruco
    try:
        dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        parameters = aruco.DetectorParameters()
        detector = aruco.ArucoDetector(dictionary, parameters)
        return detector, dictionary
    except AttributeError:
   
        dictionary = aruco.Dictionary_get(aruco.DICT_4X4_50)
        parameters = aruco.DetectorParameters_create()
        return (dictionary, parameters), dictionary  

def detect_aruco_corners(image_bgr):
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    detector, dictionary = get_aruco_detector()
    if isinstance(detector, tuple):
     
        aruco = cv2.aruco
        corners, ids, _ = aruco.detectMarkers(gray, detector[0], parameters=detector[1])
    else:
        corners, ids, _ = detector.detectMarkers(gray)

    if ids is None or len(ids) < 4:
        raise ValueError("Did not find enough ArUco markers.")

    ids = ids.flatten().tolist()
    id_to_corner = {}
    for i, mid in enumerate(ids):
        id_to_corner[mid] = corners[i].reshape(-1, 2)  # 4x2

    need = [0,1,2,3]
    if not all(mid in id_to_corner for mid in need):
        raise ValueError(f"Missing required corner IDs. Found: {sorted(id_to_corner.keys())}")

    def center(pts):
        return np.mean(pts, axis=0)

    pts_img = np.array([
        center(id_to_corner[0]),  
        center(id_to_corner[1]),  
        center(id_to_corner[2]), 
        center(id_to_corner[3])  
    ], dtype=np.float32)

    return pts_img, id_to_corner

def build_homography(pts_img, width_m, height_m):

    world_pts = np.array([
        [-width_m/2,  height_m/2],
        [ width_m/2,  height_m/2],
        [ width_m/2, -height_m/2],
        [-width_m/2, -height_m/2],
    ], dtype=np.float32)


    H, _ = cv2.findHomography(pts_img, world_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if H is None:
        raise RuntimeError("Homography estimation failed.")
    return H

def image_points_to_world(H, pts_img):
    
    pts = np.hstack([pts_img, np.ones((pts_img.shape[0], 1))])
    wp = (H @ pts.T).T
    wp = wp[:, :2] / wp[:, 2:3]
    return wp 

def load_yolo(weights_path):
    model = YOLO(weights_path)
    return model

def detect_flowers(model, image_bgr, conf=0.4, class_name='strawberry'):
    results = model.predict(source=image_bgr, conf=conf, verbose=False)
    if len(results) == 0:
        return np.zeros((0,2), dtype=np.float32), []

    r = results[0]
    boxes = r.boxes
    if boxes is None or boxes.xyxy is None or boxes.xyxy.shape[0] == 0:
        return np.zeros((0,2), dtype=np.float32), []

   
    names = r.names if hasattr(r, 'names') else model.names
    target_cls = None
    if isinstance(names, dict):
        for k, v in names.items():
            if str(v).lower() == class_name.lower():
                target_cls = int(k)
                break

    flower_pts_img = []
    used = []
    for i in range(boxes.xyxy.shape[0]):
        cls = int(boxes.cls[i].item()) if boxes.cls is not None else None
        if (target_cls is None) or (cls == target_cls):
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().tolist()
            cx = 0.5*(x1+x2); cy = 0.5*(y1+y2)
            flower_pts_img.append([cx, cy])
            used.append((i, cls, float(boxes.conf[i].item())))
    return np.array(flower_pts_img, dtype=np.float32), used

def annotate_and_save(image_bgr, id_to_corner, flower_pts_img, out_path='annotated.png'):
    vis = image_bgr.copy()
    
    for mid, pts in id_to_corner.items():
        pts = pts.astype(int)
        cv2.polylines(vis, [pts], True, (0,255,0), 2)
        c = np.mean(pts, axis=0).astype(int)
        cv2.circle(vis, tuple(c), 4, (0,255,0), -1)
        cv2.putText(vis, f'ID {mid}', tuple(c+np.array([5,-5])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    
    for p in flower_pts_img:
        cv2.circle(vis, (int(p[0]), int(p[1])), 5, (255,0,0), -1)

    cv2.imwrite(out_path, vis)
    print(f'[INFO] Saved annotation: {out_path}')


def main():
    
    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"IMAGE_PATH not found: {IMAGE_PATH}")
    image_bgr = cv2.imread(IMAGE_PATH)
    if image_bgr is None:
        raise RuntimeError(f"Failed to read image: {IMAGE_PATH}")

    try:
        width_m  = float(input("Enter RECTANGLE width in metres (left→right, X): ").strip())
        height_m = float(input("Enter RECTANGLE height in metres (top→bottom, Y): ").strip())
    except Exception:
        raise ValueError("Please enter numeric width/height in metres.")

    print('[INFO] Detecting ArUco 0/1/2/3…')
    pts_img, id_to_corner = detect_aruco_corners(image_bgr)

    print('[INFO] Building image→world homography (origin at rectangle centre)…')
    H = build_homography(pts_img, width_m, height_m)

    print('[INFO] Loading YOLOv8 model…')
    yolo = load_yolo(YOLO_WEIGHTS)

    print('[INFO] Detecting strawberries…')
    flower_pts_img, used = detect_flowers(yolo, image_bgr, conf=YOLO_CONF, class_name=YOLO_CLASS_NAME)

    if flower_pts_img.shape[0] == 0:
        print('[WARN] No flowers detected. Exiting before flight.')
        return

    flowers_world = image_points_to_world(H, flower_pts_img) 
    
    flowers_world = flowers_world.tolist()

    
    annotate_and_save(image_bgr, id_to_corner, flower_pts_img, out_path='annotated_search_area.png')

    print('\n[INFO] Flower world coordinates (metres, origin at rectangle centre):')
    for i, (x, y) in enumerate(flowers_world, 1):
        print(f'  Flower {i:02d}: x={x:+.3f}, y={y:+.3f}')

   
    print('\n[INFO] Training RF voltage-drop model from combined_data.csv …')
    rf_model = train_rf_from_csv('combined_data.csv')
    print('[INFO] RF model ready.')

    
    cflib.crtp.init_drivers(enable_debug_driver=False)
    print('[INFO] Connecting to', URI)
    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
       
        scf.cf.param.set_value('stabilizer.estimator', '2'); time.sleep(0.1)
        if not wait_for_estimator(scf):
            print('[ERROR] Estimator did not settle.'); return

      
        x0, y0, z0, v0 = read_state_once(scf)
        if v0 <= 0:
            print('[ERROR] Invalid initial vbat.'); return

        print(f'[INFO] Start: x={x0:.3f} y={y0:.3f} z={z0:.3f} vbat={v0:.2f} V')
       
        home = (0.0, 0.0)

        
        order, segments = plan_segments(home, flowers_world, TRAVEL_SPEED, HOVER_TIME)

        
        t_pred, v_pred = build_predicted_voltage_trace_with_sag(rf_model, v0, segments)

        logger = CsvLogger(scf, prefix='cf_energy_compare', period_ms=max(10, int(1000//LOG_HZ)))
        logger.start()

        visited = 0
        guard = HardLowGuard()
        try:
            with PositionHlCommander(scf, default_velocity=SPEED_XY, default_height=TAKEOFF_Z, x=home[0], y=home[1], z=0.0) as pc:
                print(f'[INFO] PositionHL engaged (auto‑takeoff to ~{TAKEOFF_Z:.2f} m).')
                time.sleep(1.0); guard.start()

                coords_full = [home] + flowers_world
                for i in range(1, len(order)):
                    target = coords_full[order[i]]
                    _x, _y, _z, vnow = read_state_once(scf)
                    if guard.update(vnow):
                        print('[WARN] <=3.20 V (sag-aware). RTB.'); break

                    print(f'[INFO] -> go_to({target[0]:+.2f}, {target[1]:+.2f}, {CRUISE_Z:.2f})')
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

        
        df_actual = pd.read_csv(logger.csv_path)
        if len(df_actual) == 0:
            print('[WARN] No flight samples recorded.')
            return
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
        plt.title(f'Greedy: Predicted vs Actual Voltage\nRMSE={rmse:.3f} V | |Δ_final|={abs(final_diff):.3f} V')
        plt.grid(True); plt.legend(); plt.tight_layout()
        plt.savefig(comp_png, dpi=150)

        plt.figure()
        plt.plot(t_actual, pct_err, linewidth=1.5)
        plt.xlabel('Time (s)'); plt.ylabel('Percent error (%)')
        plt.title(f'Prediction Percent Error Over Time\nMean |%err|={mean_pct_abs:.1f}% | Max |%err|={max_pct_abs:.1f}%')
        plt.grid(True); plt.tight_layout()
        plt.savefig(pct_png, dpi=150)

        final_v = float(v_actual[-1]) if len(v_actual) else float('nan')
        print(f'\n[RESULT] Flowers visited (actual hover counts): {visited}/{len(flowers_world)}')
        print(f'[RESULT] Final voltage: {final_v:.2f} V')
        print(f'[RESULT] Comparison saved: {comp_csv}')
        print(f'[RESULT] Plots saved:      {comp_png}, {pct_png}')
        print(f'[RESULT] RMSE={rmse:.3f} V | Mean |err|={mean_abs:.3f} V | Mean |%err|={mean_pct_abs:.1f}% | Max |%err|={max_pct_abs:.1f}%')

if __name__ == '__main__':
    main()
