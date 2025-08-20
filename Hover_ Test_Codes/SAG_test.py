import os, time, csv
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.syncLogger import SyncLogger
from cflib.crazyflie.log import LogConfig
from cflib.positioning.motion_commander import MotionCommander

#USER CONFIG
URI = 'radio://0/80/2M/E7E7E7E7E7'

CYCLES = 5
HOVER_HEIGHT = 0.35     
TAKEOFF_VEL = 0.4       
LAND_VEL = 0.4          
HOVER_TIME = 1.0         
BASELINE_PRE_SEC = 0.8   
REBOUND_WINDOW_SEC = 3.0 
REST_BETWEEN_SEC = 1.0   

LOG_HZ = 50              
PLOT = True


def wait_for_estimator(scf, max_wait=15.0, settle=2.0):
    print("[INFO] Waiting for estimator to settle...")
    t0 = time.time(); stable_since = None
    lc = LogConfig('est', period_in_ms=100)
    lc.add_variable('kalman.varPX','float'); lc.add_variable('kalman.varPY','float'); lc.add_variable('kalman.varPZ','float')
    with SyncLogger(scf, lc) as lg:
        for _ts, d, _ in lg:
            ok = (d.get('kalman.varPX',1.0)<0.003 and d.get('kalman.varPY',1.0)<0.003 and d.get('kalman.varPZ',1.0)<0.003)
            if ok:
                stable_since = stable_since or time.time()
                if time.time()-stable_since >= settle:
                    print("[INFO] Estimator settled.")
                    return True
            else:
                stable_since = None
            if time.time()-t0 > max_wait:
                print("[WARN] Estimator did not fully settle; continuing.")
                return False

def run():
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    sample_csv = f'cf_sag_rebound_samples_{ts}.csv'
    cycle_csv  = f'cf_sag_rebound_cycles_{ts}.csv'
    fig_path   = f'cf_sag_rebound_{ts}.png'

    cflib.crtp.init_drivers()

    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
        scf.cf.param.set_value('stabilizer.estimator','2')
        wait_for_estimator(scf)

       
        logconf = LogConfig('battery', period_in_ms=max(5, int(1000/LOG_HZ)))
        logconf.add_variable('pm.vbat','float')

        os.makedirs('.', exist_ok=True)
        with open(sample_csv, 'w', newline='') as sfile, open(cycle_csv, 'w', newline='') as cfile:
            sw = csv.writer(sfile); cw = csv.writer(cfile)
            sw.writerow(['t','cycle','phase','vbat'])
            cw.writerow(['cycle','V_baseline_pre','V_min_hover','V_peak_post_land',
                         'sag_V','rebound_from_min_V','rebound_vs_baseline_V'])

            t0 = time.time()

            def log_sample(cycle, phase, logger):
                _, data, _ = next(logger)
                v = float(data.get('pm.vbat', float('nan')))
                sw.writerow([f'{time.time()-t0:.3f}', cycle, phase, f'{v:.3f}'])
                return v

            with SyncLogger(scf, logconf) as logger:
             
                mc = MotionCommander(scf) 
                try:
                    print("[INFO] Starting sag & rebound test…")
                    time.sleep(0.5)

                    for cycle in range(1, CYCLES+1):
                        print(f"\n[INFO] Cycle {cycle}/{CYCLES}")

                    
                        baseline = []
                        t_start = time.time()
                        while time.time()-t_start < BASELINE_PRE_SEC:
                            baseline.append(log_sample(cycle,'baseline',logger))
                        V_baseline = float(np.median(baseline)) if baseline else np.nan

                        
                        print("   [STEP] Takeoff & 1s hover")
                        mc.take_off(height=HOVER_HEIGHT, velocity=TAKEOFF_VEL)
                        v_min = +1e9
                        hover_start = time.time()
                        while time.time()-hover_start < HOVER_TIME:
                            v = log_sample(cycle,'hover',logger)
                            if v < v_min: v_min = v
                        V_min_hover = float(v_min) if v_min < 1e9 else np.nan

                    
                        print("   [STEP] Land & rebound window")
                        mc.land(velocity=LAND_VEL)
                        v_peak = -1e9
                        rebound_start = time.time()
                        while time.time()-rebound_start < REBOUND_WINDOW_SEC:
                            v = log_sample(cycle,'rebound',logger)
                            if v > v_peak: v_peak = v
                        V_peak_post = float(v_peak) if v_peak > -1e9 else np.nan

                      
                        sag_V = V_baseline - V_min_hover
                        rebound_from_min_V = V_peak_post - V_min_hover
                        rebound_vs_baseline_V = V_peak_post - V_baseline
                        cw.writerow([cycle, f'{V_baseline:.3f}', f'{V_min_hover:.3f}', f'{V_peak_post:.3f}',
                                     f'{sag_V:.3f}', f'{rebound_from_min_V:.3f}', f'{rebound_vs_baseline_V:.3f}'])
                        print(f"   [RESULT] baseline={V_baseline:.3f} V | min_hover={V_min_hover:.3f} V | peak_post={V_peak_post:.3f} V")
                        print(f"            sag={sag_V:.3f} V | rebound_from_min={rebound_from_min_V:.3f} V | rebound_vs_baseline={rebound_vs_baseline_V:.3f} V")

                      
                        rest_start = time.time()
                        while time.time()-rest_start < REST_BETWEEN_SEC:
                            log_sample(cycle,'rest',logger)

                finally:
             
                    try:
                        mc.land(velocity=LAND_VEL)
                    except Exception:
                        pass
                    mc.stop()

   
    if PLOT:
        import pandas as pd
        df = pd.read_csv(sample_csv)
        plt.figure(figsize=(10,6))
        for cyc in sorted(df['cycle'].unique()):
            sub = df[df['cycle']==cyc]
            plt.plot(sub['t'], sub['vbat'], label=f'Cycle {cyc}')
        plt.xlabel('Time (s)'); plt.ylabel('Voltage (V)')
        plt.title('Battery Sag & Rebound (5 cycles)')
        plt.grid(True); plt.legend(ncol=2, fontsize=9)
        plt.tight_layout(); plt.savefig(fig_path, dpi=150)
        print(f"[INFO] Plot saved to {fig_path}")


    import pandas as pd
    cs = pd.read_csv(cycle_csv)
    print("\n========== SUMMARY ==========")
    for _, r in cs.iterrows():
        print(f"Cycle {int(r['cycle'])}: sag={r['sag_V']:.3f} V | rebound_from_min={r['rebound_from_min_V']:.3f} V | rebound_vs_baseline={r['rebound_vs_baseline_V']:.3f} V")
    print(f"Average sag: {cs['sag_V'].mean():.3f} V")
    print(f"Average rebound_from_min: {cs['rebound_from_min_V'].mean():.3f} V")
    print(f"Average rebound_vs_baseline: {cs['rebound_vs_baseline_V'].mean():.3f} V")
    print(f"[INFO] Sample log: {sample_csv}")
    print(f"[INFO] Cycle metrics: {cycle_csv}")

if __name__ == "__main__":
    run()
