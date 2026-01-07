# Traveling Salesman Problem – Heuristics & Metaheuristics

This repository contains a Python implementation of multiple **heuristic and metaheuristic algorithms**
for solving the **Traveling Salesman Problem (TSP)** on Euclidean instances.

The project was developed as part of an Operations Research / Optimization coursework and focuses on
**constructive heuristics**, **local search**, and **metaheuristic improvement strategies**.

---

## Problem Description

The **Traveling Salesman Problem (TSP)** aims to find the shortest possible tour that:

- Visits each city exactly once
- Returns to the starting city
- Minimizes the total travel distance

This implementation assumes **Euclidean TSP instances**, where distances are computed using
2D coordinates.

---

## Implemented Methods

### Constructive Heuristics
- **Nearest Neighbor (NN)**
- **Outlier Insertion (Deterministic)**

### Metaheuristics
- **GRASP (Greedy Randomized Adaptive Search Procedure)**  
  - Insertion-based construction
  - Restricted Candidate Lists (RCL)
  - Both quality-based (α) and top-k RCL variants

### Local Search
- **2-opt**
  - First-improvement
  - Best-improvement
  - Sampled and k-nearest-neighbor accelerated variants

### Hybrid Approaches
- **GRASP + 2-opt**
- **Iterated Local Search (ILS)**

> Simulated Annealing (SA) was also tested experimentally but is not used in the final workflow.

---

## Repository Structure

```text
.
├── main.py
├── README.md
├── Instances/
│   └── Small/
│       └── berlin52.tsp
│       └── kroA100.tsp
│       └── lin105.tsp
│       └── pr76.tsp
│       └── ulysses22.tsp
│   └── Medium/
│       └── a280.tsp
│       └── rat195.tsp
│       └── st70.tsp
│   └── Large/
│       └── d1291.tsp
│       └── pr1002.tsp
├── outputs/
│   ├── before_vs_after_grasp+2opt_costs.png
│   └── grasp+2opt_tour.png
│   └── outlier_insertion_tour.png
└── report/
    └── tsp_report.pdf
```
## Requirements
- Python 3.x
- numpy
- matplotlib

## How to Run
1.Clone the repository or download it as a ZIP file
2. Place the desired .tsp instance file under the Instances/ directory
3.Run: 
```
python main.py
```
The instance file is specified directly in the script: 
```
instFilename = "Instances/Small/berlin52.tsp"
```
## Output
The script produces:
- Console output with feasibility checks and tour costs
- Visual plots of:
    - Constructed tours
    - GRASP vs 2-opt improvement comparison
- Summary statistics for Nearest Neighbor runs from all starting cities
