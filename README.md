# crazyflie-pollination
Energy-aware path planning and vision-in-the-loop control for a Crazyflie 2.1 micro-UAV, developed for autonomous pollination in vertical farms. Includes battery voltage drop modelling, sag-aware mission simulation, and a full system demonstration with YOLOv8 flower detection.
# Energy-Aware Path Planning for Micro-UAV Pollination

This repository contains the code, datasets, and results for my MSc dissertation project at the University of Bristol.  
The project investigates **energy-aware mission planning** and **vision-in-the-loop autonomy** for the Crazyflie 2.1 nano-quadcopter in the context of vertical farm pollination.  

---

##  Project Overview
Micro-UAVs such as the [Crazyflie 2.1](https://www.bitcraze.io/products/crazyflie-2-1/) have extreme endurance constraints (flight times only a few minutes).  
For tasks like flower pollination in vertical farms, naïve path planning may lead to **premature battery depletion**.  

This project develops and validates an **Energy-Aware Path Planning Framework**:
- A **Random Forest voltage-drop predictor** trained on real hover and sag–rebound flight data.
- **Sag-aware energy logic** that models transient voltage drops at takeoff.
- Classical path planners (Greedy, 2-Opt, MST-DFS) integrated with energy checks.
- A **vision-in-the-loop demonstration** combining ArUco calibration and YOLOv8 flower detection.

  
## Repository Layout
├── FULL SYSTEM DEMONSTRATION/ # Vision-in-the-loop mission scripts
│ ├── full_mission.py
│ ├── roboflow_dataset/ # YOLO training dataset
│ ├── simple_check.py
│
├── Hover_Test_Codes/ # Raw hover and sag experiments
│ ├── Hover_test1.py
│ ├── SAG_test.py
│
├── real_test/ # Real Crazyflie mission flight scripts
│ ├── realtest.py
│
├── simulation/ # Path planning & energy-aware simulation
│ ├── sim.py
│ ├── plot.py
│ ├── combined_data.csv # Training dataset from hover flights
│
├── results/ # Processed outputs & plots
│ ├── all_runs.txt
│ ├── cf_energy_compare_* # CSVs, comparison plots, error metrics
│ ├── simulation_results_*.csv
│
├── LICENSE # Open source license
├── README.md # This file
