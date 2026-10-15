# Context-Aware Smart Home Surveillance

A modular surveillance pipeline combining video anomaly detection (MNAD),
object detection and tracking (YOLO + ByteTrack), rule-based context
reasoning, configurable risk scoring, and deterministic explainability —
with email alerting for high-risk events.
video → MNAD (anomaly score) → YOLO + ByteTrack → Context Engine
→ Risk Engine → Explainability Engine → Annotated Video + Alerts
## Overview

MNAD is used strictly as a black box (frame sequence in, anomaly score
out) — its architecture is never modified. All context reasoning, risk
scoring, and explanation logic is implemented independently.

## Features

- Object detection & tracking: person, knife, scissors, backpack, handbag,
  suitcase, dog, cat, car, motorcycle, bicycle
- Context detection: loitering, multiple people, abandoned/possibly-taken
  objects, entry-point approach, possible altercation, movement direction,
  speed category, erratic movement, group convergence/dispersal
- Configurable, weighted, non-linear risk fusion (0–100 score)
- Deterministic, rule-based explanations (no LLM)
- Real-time-style clock overlay on annotated output video
- Email alerts on high-risk events (with cooldown to avoid spam)
- Dockerized for easy setup — no manual Python/dependency install needed

## Project Structure


SmartHomeSurveillance/
Dockerfile
requirements.txt
run_pipeline.py         <- main entry point
app/
mnad_detector.py       <- MNAD black-box wrapper
object_tracker.py      <- YOLO + ByteTrack wrapper
context_risk_explain.py <- Context/Risk/Explainability engines
config/
config.yaml
notebooks/                <- original Colab notebooks (01–05)
.env.example               <- template for email credentials
checkpoints/                <- NOT in repo, see Setup below
datasets/                    <- NOT in repo, put your input video here
outputs/                      <- generated results land here

## Requirements

- [Docker](https://docs.docker.com/get-docker/) installed
- (Optional) NVIDIA GPU + [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  for faster processing. **Works on CPU too — just slower.**

## Setup

1. Clone this repository:
```bash
   git clone https://github.com/<your-username>/SmartHomeSurveillance.git
   cd SmartHomeSurveillance
```

2. Get the MNAD checkpoint files (`model.pth`, `m_items.pt`) — these are
   NOT in this repo (too large for Git). Ask the project owner for them,
   or train your own using `notebooks/02_MNAD.ipynb`. Place them at:
checkpoints/ped2/model.pth
checkpoints/ped2/m_items.pt

3. Set up email alert credentials:
```bash
   cp .env.example .env
```
   Edit `.env` with a text editor and fill in:
   - `ALERT_SENDER_EMAIL` — a Gmail address
   - `ALERT_SENDER_APP_PASSWORD` — a Gmail **App Password** (not your real
     password — generate one at myaccount.google.com/apppasswords, requires
     2-Step Verification enabled first)
   - `ALERT_OWNER_EMAIL` — who should receive alerts

4. Place your test video in `datasets/input_videos/`.

5. Build the Docker image (only needed once, or after code changes):
```bash
   docker build -t smart-home-surveillance .
```

## Running

### With GPU
```bash
docker run --gpus all \
  -v $(pwd)/datasets:/app/datasets \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -v $(pwd)/outputs:/app/outputs \
  --env-file .env \
  smart-home-surveillance \
  --video /app/datasets/input_videos/YOUR_VIDEO.mp4 \
  --occupancy vacant --time 23:30 --email-alerts
```

### Without GPU (CPU only — just remove `--gpus all`)
```bash
docker run \
  -v $(pwd)/datasets:/app/datasets \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -v $(pwd)/outputs:/app/outputs \
  --env-file .env \
  smart-home-surveillance \
  --video /app/datasets/input_videos/YOUR_VIDEO.mp4 \
  --occupancy vacant --time 23:30 --email-alerts
```
CPU mode works identically, just noticeably slower — expect several times
longer runtime depending on video length and CPU speed.

### No Docker / no local GPU at all — use Google Colab instead

If you don't want to install Docker, or don't have a capable machine at
all, run the pipeline in Google Colab (free GPU access):

1. Open a new Colab notebook, set **Runtime → Change runtime type → GPU**.
2. Mount Google Drive and upload this project's `app/` folder, `config/`,
   your checkpoint files, and your test video into your Drive.
3. Use `notebooks/05_Final_Pipeline.ipynb` from this repo — it runs the
   exact same logic interactively, cell by cell, without needing Docker
   or a local GPU at all.
4. If Colab's free GPU quota runs out, it automatically falls back to CPU
   (slower, but still works) — no code changes needed.

### CLI options reference

| Flag | Default | Meaning |
|---|---|---|
| `--video` | *(required)* | Path to input video |
| `--occupancy` | `vacant` | `vacant` or `occupied` |
| `--time` | `23:30` | Simulated start time, HH:MM |
| `--entry-point` | none | `name,x,y` e.g. `front_door,450,300` |
| `--conf` | `0.55` | YOLO confidence threshold |
| `--multiple-people-thresh` | `2` | People count considered "crowded" |
| `--high-risk-thresh` | `70.0` | Risk score that counts as a high-risk event |
| `--email-alerts` | off | Enable email notifications |
| `--email-cooldown-sec` | `120.0` | Minimum seconds between alert emails |

## Outputs

After a run, check `outputs/`:
- `<video>_annotated.mp4` — video with bounding boxes, risk score, MNAD
  score, real-time clock, and context labels overlaid
- `<video>_risk_scores.csv` — full per-frame breakdown
- `<video>_events.json` — logged high-risk moments with reasons
- `<video>_risk_plot.png` — risk score over time graph

## Known Limitations

- The MNAD checkpoint was trained on UCSD Ped2 (outdoor pedestrian
  walkway); its anomaly score is a noisier signal on indoor/home footage
  due to domain mismatch. The Risk Engine dampens its influence at
  low/moderate scores to compensate.
- ByteTrack can fragment identities in dense-crowd footage; features
  relying on stable track identity (e.g. possible-object-taken) should be
  treated as candidates for review in such scenes.
- Zone-based reasoning (e.g. "front yard" vs "driveway") is not yet
  implemented — only single-point entry-approach detection is available.

## License / Academic Use

This project uses:
- MNAD (Park et al., CVPR 2020) — used unmodified as a black box
- Ultralytics YOLO
- ByteTrack (Zhang et al., ECCV 2022)


