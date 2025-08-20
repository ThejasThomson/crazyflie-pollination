import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load simulation results
df = pd.read_csv("simulation_results_energy_sag_only1.csv")

# Set seaborn style for clean visuals
sns.set(style="whitegrid")

# Plot 1: Mission Time
plt.figure(figsize=(10, 6))
sns.boxplot(x='flowers', y='mission_time', hue='algorithm', data=df)
plt.title("Mission Time vs Number of Flowers")
plt.ylabel("Mission Time (s)")
plt.xlabel("Number of Flowers")
plt.legend(title='Algorithm')
plt.tight_layout()
plt.show()

# Plot 2: Energy Consumption 
plt.figure(figsize=(10, 6))
sns.boxplot(x='flowers', y='energy', hue='algorithm', data=df)
plt.title("Energy Consumption vs Number of Flowers")
plt.ylabel("Energy Consumed (Wh approx.)")
plt.xlabel("Number of Flowers")
plt.legend(title='Algorithm')
plt.tight_layout()
plt.show()

# Plot 3: Total Distance
plt.figure(figsize=(10, 6))
sns.boxplot(x='flowers', y='distance', hue='algorithm', data=df)
plt.title("Total Travel Distance vs Number of Flowers")
plt.ylabel("Distance (meters)")
plt.xlabel("Number of Flowers")
plt.legend(title='Algorithm')
plt.tight_layout()
plt.show()

# Plot 4: Battery Cycles Used
plt.figure(figsize=(10, 6))
sns.boxplot(x='flowers', y='battery_cycles', hue='algorithm', data=df)
plt.title("Battery Cycles vs Number of Flowers")
plt.ylabel("Battery Cycles Used")
plt.xlabel("Number of Flowers")
plt.legend(title='Algorithm')
plt.tight_layout()
plt.show()
