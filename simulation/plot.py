import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

#Load and preprocess dataset
df = pd.read_csv("combined_data.csv")
df = df[df['thrust'] > 0].copy()
df['time_diff'] = df['timestamp (s)'].diff().fillna(0)
df = df[df['time_diff'] > 0]
df['prev_voltage'] = df['battery_voltage (V)'].shift(1).bfill()
df['voltage_drop'] = df['prev_voltage'] - df['battery_voltage (V)']
df = df[(df['voltage_drop'] >= 0) & (df['voltage_drop'] < 0.05)]

# Prepare features and target 
X = df[['time_diff', 'thrust', 'prev_voltage']]
y = df['voltage_drop']

#Train/test split and model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

#Predictions and evaluation
y_pred = model.predict(X_test)
residuals = y_test - y_pred
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

#Residuals Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=residuals, alpha=0.6)
plt.axhline(0, color='r', linestyle='--')
plt.xlabel("Actual Voltage Drop (V)")
plt.ylabel("Residual (Actual - Predicted)")
plt.title(f"Residuals Plot\n$R^2$ = {r2:.3f}, MSE = {mse:.6f}")
plt.xlim(0, 0.015)
plt.ylim(-0.01, 0.01)
plt.grid(True)
plt.tight_layout()
plt.savefig("voltage_residuals_plot.png", dpi=300)
plt.show()
