import numpy as np
import pandas as pd
import random
import math
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

#CONFIGURATION
SEARCH_AREA = 2          # meters
HOVER_TIME = 4           # seconds per flower
TRAVEL_SPEED = 0.5       # meters/second
HOVER_THRUST = 40000
TRAVEL_THRUST = 35000
V_FULL = 4.2
V_EMPTY = 3.2
BATTERY_WH = 0.888
SAFETY_MARGIN = 0.02     # Wh

# SAG model (from live test code; NO rebound)
# amplitude = BASE + PER*thrust_norm + SOC_GAIN * max(0, SOC_REF - v_now)
STARTUP_SAG_BASE_V     = 0.14
STARTUP_SAG_PER_TNORM  = 1.00
SAG_SOC_REF_V          = 4.00
SAG_SOC_GAIN           = 1.60

SAG_RISE_TAU_S         = 0.6   
SETTLE_REBOUND_FRAC    = 0.05 
SETTLE_REBOUND_TAU_S   = 1.5

#LOAD & TRAIN ENERGY (RF voltage-drop) MODEL 
df = pd.read_csv("combined_data.csv")
df = df[df['thrust'] > 0].copy()
df['time_diff'] = df['timestamp (s)'].diff().fillna(0)
df = df[df['time_diff'] > 0]
df['prev_voltage'] = df['battery_voltage (V)'].shift(1).bfill()
df['voltage_drop'] = df['prev_voltage'] - df['battery_voltage (V)']
df = df[(df['voltage_drop'] >= 0) & (df['voltage_drop'] < 0.05)]

X = df[['time_diff', 'thrust', 'prev_voltage']]
y = df['voltage_drop']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

energy_model = RandomForestRegressor(n_estimators=100, random_state=42)
energy_model.fit(X_train, y_train)


def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def predict_energy(t, thrust, voltage):
    
    return energy_model.predict(pd.DataFrame([[t, thrust, voltage]],
                           columns=['time_diff', 'thrust', 'prev_voltage']))[0]

def voltage_from_energy(energy_used):
    percent = min(energy_used / BATTERY_WH, 1.0)
    return max(V_FULL - percent * (V_FULL - V_EMPTY), V_EMPTY)

def greedy_tsp(coords):
    path = [0]
    unvisited = set(range(1, len(coords)))
    while unvisited:
        last = path[-1]
        next_node = min(unvisited, key=lambda x: euclidean(coords[last], coords[x]))
        path.append(next_node)
        unvisited.remove(next_node)
    path.append(0)
    return path

def two_opt(path, coords):
    best = path
    improved = True
    while improved:
        improved = False
        for i in range(1, len(path) - 2):
            for j in range(i + 1, len(path)):
                if j - i == 1:
                    continue
                new_path = best[:i] + best[i:j][::-1] + best[j:]
                if path_length(new_path, coords) < path_length(best, coords):
                    best = new_path
                    improved = True
                    break
            if improved:
                break
    return best

def path_length(path, coords):
    return sum(euclidean(coords[path[i]], coords[path[i + 1]]) for i in range(len(path) - 1))

def mst_dfs(coords):
    G = nx.complete_graph(len(coords))
    for i, j in G.edges:
        G[i][j]['weight'] = euclidean(coords[i], coords[j])
    mst = nx.minimum_spanning_tree(G)
    return list(nx.dfs_preorder_nodes(mst, source=0)) + [0]

# === SAG utilities (added) ===
def _thrust_norm(thrust):
    return max(0.0, min(1.0, float(thrust)/65535.0))

def _startup_sag_amplitude(v_now, thrust_for_takeoff):
    tnorm = _thrust_norm(thrust_for_takeoff)
    thrust_part = STARTUP_SAG_PER_TNORM * tnorm
    soc_part = SAG_SOC_GAIN * max(0.0, SAG_SOC_REF_V - float(v_now))
    return max(0.0, STARTUP_SAG_BASE_V + thrust_part + soc_part)

def _apply_sag_and_settle(v_now):
    """
    Note: We do NOT model rebound because the battery is swapped on depletion in this sim.
    """
    extra_time = 0.0
   
    sag_v = _startup_sag_amplitude(v_now, HOVER_THRUST)
    if sag_v > 0.0 and SAG_RISE_TAU_S > 0.0:
        drop = sag_v * (1.0 - math.exp(-SAG_RISE_TAU_S / max(SAG_RISE_TAU_S, 1e-6)))
        v_now = max(v_now - drop, V_EMPTY)
        extra_time += SAG_RISE_TAU_S

    if SETTLE_REBOUND_FRAC > 0.0 and SETTLE_REBOUND_TAU_S > 0.0:
        add = (SETTLE_REBOUND_FRAC * sag_v) * (1.0 - math.exp(-SETTLE_REBOUND_TAU_S / max(SETTLE_REBOUND_TAU_S,1e-6)))
        v_now = v_now + add
        extra_time += SETTLE_REBOUND_TAU_S
    return v_now, extra_time


def simulate_path(path, coords):

    voltage = V_FULL
    energy_used = 0.0
    total_energy = 0.0
    time_spent = 0.0
    distance = 0.0
    battery_cycles = 1

    voltage, dt_extra = _apply_sag_and_settle(voltage)
    time_spent += dt_extra

    for i in range(1, len(path)):
        d = euclidean(coords[path[i - 1]], coords[path[i]])
        t_travel = d / TRAVEL_SPEED

        e_hover = predict_energy(HOVER_TIME, HOVER_THRUST, voltage)
        e_travel = predict_energy(t_travel, TRAVEL_THRUST, voltage)
        segment_energy = e_hover + e_travel

        if energy_used + segment_energy + SAFETY_MARGIN > BATTERY_WH:
 
            return_time = euclidean(coords[path[i - 1]], coords[0]) / TRAVEL_SPEED
            return_energy = predict_energy(return_time, TRAVEL_THRUST, voltage)

            energy_used += return_energy
            total_energy += return_energy
            time_spent += return_time
            distance += euclidean(coords[path[i - 1]], coords[0])

      
            energy_used = 0.0
            battery_cycles += 1
            voltage = V_FULL
            voltage, dt_extra = _apply_sag_and_settle(voltage)
            time_spent += dt_extra

            continue

        # Execute leg normally
        energy_used += segment_energy
        total_energy += segment_energy
        voltage = voltage_from_energy(energy_used)  
        time_spent += HOVER_TIME + t_travel
        distance += d

    return time_spent, total_energy, distance, battery_cycles


def run_trials(n_trials=10, flower_counts=[10, 50, 100]):
    results = []
    for n in flower_counts:
        for trial in range(n_trials):
            print(f"Running trial {trial+1}/{n_trials} for {n} flowers...")
            coords = [(0, 0)] + [(random.uniform(0, SEARCH_AREA), random.uniform(0, SEARCH_AREA)) for _ in range(n)]
            greedy = greedy_tsp(coords)
            opt = two_opt(greedy.copy(), coords)
            mst = mst_dfs(coords)

            for name, path in [('Greedy', greedy), ('2-Opt', opt), ('MST', mst)]:
                t_mis, energy, dist, cycles = simulate_path(path, coords)
                results.append({
                    'algorithm': name,
                    'flowers': n,
                    'mission_time': t_mis,
                    'energy': energy,
                    'distance': dist,
                    'battery_cycles': cycles
                })
    return pd.DataFrame(results)

df_results = run_trials()
df_results.to_csv("simulation_results_energy_sag_only1.csv", index=False)
print(" Simulation complete. Saved to simulation_results_energy_sag_only1.csv")
