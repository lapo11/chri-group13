# PA3 — Kinesthetic Teaching & Haptic Guidance
### RO47013 Control in Human-Robot Interaction — TU Delft

A haptic learning-from-demonstration system where a user navigates a curved
tube with a Haply pantograph device, and a Gaussian Process learns the
trajectory from repeated demonstrations. Different haptic guidance modes
can be enabled to study their effect on learning quality.

---

## Project Structure

| File | Description |
|---|---|
| `PA3_main.py` | Main application loop, rendering, state machine |
| `haptics.py` | Force computation: walls, groove, GP groove, fading groove |
| `gp_trajectory.py` | Gaussian Process trajectory learning from demonstrations |
| `targets.py` | Tube definitions and geometry (centerline, walls, normals) |
| `metrics.py` | MND, Hausdorff, Fréchet, DTW trajectory evaluation |
| `nasa_tlx.py` | NASA-TLX workload questionnaire |
| `Physics.py` | Haply device interface |
| `Graphics.py` | Pygame rendering and coordinate conversion |

---

## How It Works

1. **Record** — press `SPACE` to start/stop a demonstration through the tube
2. **Review** — accept (`ENTER`) or delete (`D`) the demo
3. **Train** — press `G` to train a GP on all recorded demos
4. **Replay** — the GP mean trajectory is shown with uncertainty bands
5. **Evaluate** — metrics are computed and saved automatically on quit

The left panel shows the haptic view (force feedback + pantograph linkages).
The right panel shows the VR view (all demos + GP trajectory + uncertainty).

---

## Haptic Guidance Modes

| Key | Mode | Effect |
|---|---|---|
| `H` | Groove | Constant spring pull toward tube centerline |
| `K` | GP Groove | Pull toward GP trajectory, grows with demo confidence |
| `J` | Fading Groove | Centerline pull that fades as GP confidence grows |
| `W` | Walls on/off | Toggle virtual wall force feedback |

Only one guidance mode can be active at a time (H, K, J are mutually exclusive).

---

## Controls

| Key | Action |
|---|---|
| `SPACE` | Start / stop recording |
| `ENTER` | Confirm demo / add more demos (from DONE) |
| `D` | Delete last demo (in REVIEW) |
| `G` | Train GP on all demos |
| `P` | Replay GP trajectory |
| `A` | Autonomous GP playback (PD controller) |
| `N` | NASA-TLX questionnaire |
| `1–4` | Select s-curve orientation (0°, 90°, 180°, 270°) |
| `T` | Cycle tube shape |
| `C` | Clear all demos and GP |
| `R` | Toggle linkage rendering |
| `Q` | Quit and save |

---

## Output

Results are saved to `results/session_<timestamp>/` on quit:

```
session_<timestamp>/
  summary.json        # tube, condition, n_demos
  metrics.json        # per-training-step trajectory metrics
  demo_1.npy          # raw trajectory arrays (physical coords, meters)
  demo_2.npy
  ...
  gp_trajectory.npy   # final GP mean trajectory
  gp_std.npy          # final GP uncertainty
```

### Loading a session in Python

```python
import numpy as np, json

folder = "results/session_1234567890"
metrics = json.load(open(f"{folder}/metrics.json"))
demos   = [np.load(f"{folder}/demo_{i+1}.npy")
           for i in range(metrics[-1]["n_demos"])]
gp_traj = np.load(f"{folder}/gp_trajectory.npy")
gp_std  = np.load(f"{folder}/gp_std.npy")
```

---

## Confidence Model

GP groove confidence combines two factors:

- **std_factor**: how much the demos agree at each point (low GP std = high confidence)
- **demo_factor**: how many demos have been recorded (grows with N, zero at N=1)

```
confidence = std_factor * demo_factor
demo_factor = 1 - exp(-(N_demos - 1) / 3)
```

The confidence bar (top-right of VR panel) shows current local confidence
as a green/red bar alongside the raw σ value in mm.

---

## Dependencies

```
pip install pygame numpy scipy scikit-learn tomlkit
```

Runs in simulation mode (mouse control) if no Haply device is connected.
