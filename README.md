# Mars Habitat Crack-Sealing — PA3 Project
## Learning from Demonstration for Emergency Repair

### Scenario
A micrometeorite has struck the Mars habitat wall, creating a crack.
Atmosphere is leaking. A teleoperated repair robot must trace the crack
to apply emergency sealant before pressure drops critically.

### Research Question
**Does haptic feedback during kinesthetic demonstration improve the quality
of trajectories learned via Gaussian Process regression for a crack-sealing task?**

### Hypothesis
Demonstrations recorded with haptic guidance (groove + wall feedback) will
produce more accurate GP reproductions than those recorded with the Haply
device alone (no haptics) or with a standard mouse.

---

## Three Experimental Conditions

| Condition | Input Device | Haptic Feedback | Key to Enable |
|-----------|-------------|-----------------|---------------|
| **Mouse** | Standard mouse | None (pseudo-haptic) | Default when no Haply connected |
| **Haply (no haptic)** | Haply pantograph | Walls=OFF, Groove=OFF | W=off, H=off |
| **Haply (haptic)** | Haply pantograph | Walls=ON, Groove=ON | W=on, H=on |

---

## Setup

### Requirements
```
pip install numpy scipy pygame
```

### Files
```
mars_main.py          — Main application (run this!)
mars_renderer.py      — Mars environment visual renderer
crack_trajectory.py   — Procedural crack pattern generation
crack_haptics.py      — Haptic force computation (walls + groove)
gp_trajectory.py      — Gaussian Process trajectory learning
metrics.py            — Evaluation metrics (MND, Hausdorff, smoothness)
nasa_tlx.py           — NASA-TLX questionnaire
Physics.py            — Haply device interface
HaplyHAPI.py          — Haply hardware API
```

### Required Assets (from your existing project)
- `robot.png` — window icon
- `handle.png` — haptic handle sprite

### Running
```bash
python mars_main.py
```

---

## Controls

| Key | Action |
|-----|--------|
| `SPACE` | Start/stop recording a demonstration |
| `ENTER` | Confirm demo / go back to IDLE |
| `D` | Delete last demo |
| `G` | Train GP on all demos |
| `P` | Replay GP trajectory |
| `A` | Auto-play (PD controller follows GP) |
| `N` | NASA-TLX questionnaire |
| `W` | Toggle virtual wall haptics |
| `H` | Toggle groove guidance haptics |
| `T` | Cycle crack shape (idle only) |
| `C` | Clear all demos and reset |
| `R` | Toggle linkage display |
| `F` | Toggle debug overlay |
| `Q` | Quit and save |

---

## Experiment Protocol

### Per participant (within-subjects design):
1. **Practice** — 1-2 practice runs (discard data)
2. **Condition A** — Record 3-5 demonstrations
   - Press `G` to train GP
   - Press `N` for NASA-TLX
   - Press `Q` to save and quit
3. **Condition B** — Repeat with different feedback setting
4. **Condition C** — Repeat with third condition

### Counterbalance the order of conditions across participants!

### Measured Metrics
- **Accuracy**: Mean Nearest Distance (MND) from GP trajectory to crack centerline
- **Precision**: Hausdorff distance
- **Smoothness**: Mean absolute jerk
- **Completion time**: Per-demo recording time
- **Seal coverage**: Percentage of crack sealed
- **Pressure remaining**: Final habitat pressure
- **Subjective**: NASA-TLX workload scores
- **Wall hits**: Number of times the cursor hit the crack wall

---

## Output Data

Results are saved to `results/session_<timestamp>/`:
- `metrics.json` — All computed metrics
- `summary.json` — Session metadata
- `demo_1.npy`, `demo_2.npy`, ... — Raw trajectories
- `gp_trajectory.npy` — Learned GP mean
- `gp_std.npy` — GP uncertainty

---

## Crack Shapes Available
- `diagonal` — Diagonal crack across the wall
- `s_curve` — S-shaped crack (default)
- `zigzag` — Zigzag pattern
- `vertical` — Nearly vertical crack
