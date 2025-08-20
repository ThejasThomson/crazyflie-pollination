import time
import threading
import csv
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.log import LogConfig
from cflib.positioning.motion_commander import MotionCommander


URI = 'radio://0/80/2M/E7E7E7E7E7'
TARGET_ALTITUDES = [0.2, 0.3, 0.4, 0.5, 0.6]
HOVER_DURATION = 4
LOGGING_FREQ_HZ = 10

log_data = []
log_lock = threading.Lock()
logging_active = False


def log_callback(timestamp, data, logconf):
    if not logging_active:
        return
    entry = {
        'timestamp': timestamp / 1000.0,
        'z': data['stateEstimate.z'],
        'vbat': data['pm.vbat'],
        'thrust': data['stabilizer.thrust']
    }
    with log_lock:
        log_data.append(entry)


def log_thread_func(duration):
    global logging_active
    logging_active = True
    time.sleep(duration)
    logging_active = False


def save_and_plot():
    filename = f"cf_hover_thrust_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['timestamp', 'z', 'vbat', 'thrust'])
        writer.writeheader()
        with log_lock:
            writer.writerows(log_data)
    print(f"[INFO] Log saved to {filename}")


    df = pd.read_csv(filename)
    df['z_rounded'] = df['z'].round(2)

    summary_df = df.groupby('z_rounded').agg(
        mean_thrust=('thrust', 'mean'),
        std_thrust=('thrust', 'std'),
        count=('thrust', 'count')
    ).reset_index()


    plt.figure(figsize=(8, 5))
    sns.lineplot(data=summary_df, x='z_rounded', y='mean_thrust', marker='o')
    plt.fill_between(summary_df['z_rounded'],
                     summary_df['mean_thrust'] - summary_df['std_thrust'],
                     summary_df['mean_thrust'] + summary_df['std_thrust'],
                     alpha=0.3)
    plt.title("Mean Hover Thrust vs Altitude")
    plt.xlabel("Altitude (m)")
    plt.ylabel("Mean Thrust (PWM units)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def get_latest_height():
    with log_lock:
        if log_data:
            return log_data[-1]['z']
        else:
            return 0.0

def hover_thrust_test(scf):
    cf = scf.cf

    log_config = LogConfig(name='HoverLog', period_in_ms=int(1000 / LOGGING_FREQ_HZ))
    log_config.add_variable('stateEstimate.z', 'float')
    log_config.add_variable('pm.vbat', 'float')
    log_config.add_variable('stabilizer.thrust', 'uint16_t')

    cf.log.add_config(log_config)
    log_config.data_received_cb.add_callback(log_callback)
    log_config.start()

    total_duration = len(TARGET_ALTITUDES) * HOVER_DURATION + 5
    thread = threading.Thread(target=log_thread_func, args=(total_duration,))
    thread.start()

    with MotionCommander(scf, default_height=0.2) as mc:
        print("[INFO] Takeoff and stabilize...")
        time.sleep(3)

        for target in TARGET_ALTITUDES:
            current = get_latest_height()
            delta = target - current

            print(f"[INFO] Moving to height: {target:.2f} m (Δ={delta:.2f})")

            if delta > 0:
                mc.up(abs(delta))
            else:
                mc.down(abs(delta))

            time.sleep(0.5)
            print(f"[INFO] Hovering at {target:.2f} m...")
            mc.start_linear_motion(0, 0, 0)
            time.sleep(HOVER_DURATION)
            mc.stop()
            time.sleep(1)

        print("[INFO] Test complete. Landing...")

    log_config.stop()
    thread.join()

    save_and_plot()

if __name__ == '__main__':
    import cflib.crtp
    cflib.crtp.init_drivers(enable_debug_driver=False)

    try:
        with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
            hover_thrust_test(scf)
    except Exception as e:
        print(f"[ERROR] {e}")
