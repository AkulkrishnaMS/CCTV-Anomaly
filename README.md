# CCTV-Anomaly: Context-Aware Explainable Anomaly Detection and Risk Assessment for Smart Home Surveillance

A modular surveillance pipeline that adds context-aware risk scoring and plain-language explanations on top of an existing video anomaly detector.

Project team: SentinalAI (PES University, CDSAML & CCBD Internship, June to July 2026)

## What it does

```
Video -> MNAD (anomaly score) -> YOLO + ByteTrack -> Context Engine
      -> Risk Engine (0-100) -> Explainability Engine -> Annotated video + alerts
```

- **MNAD** ([Park et al., CVPR 2020](https://github.com/cvlab-yonsei/MNAD)) is used as a black box: frames in, anomaly score out. The model architecture is not modified.
- **YOLO + ByteTrack** detect and track 11 classes: person, knife, scissors, backpack, handbag, suitcase, dog, cat, car, motorcycle, bicycle.
- **Context Engine** derives situational labels from tracking data: person present, multiple people, loitering, unknown object, abandoned object, possible object taken, approaching entry point, possible altercation, movement direction, speed category (walking/jogging/running), frame-edge entry, erratic movement, group converging/dispersing, night activity.
- **Risk Engine** fuses the MNAD score and context labels into a 0-100 score using configurable weights. A non-linear function dampens low or moderate MNAD scores (domain-mismatch noise) and boosts high ones.
- **Explainability Engine** produces rule-based reasons for every score. No LLM is used.
- Optional **email alerts** when risk crosses a threshold, with a cooldown.

## Repository layout

```
CCTV-Anomaly/
  run_pipeline.py          main entry point
  app/
    mnad_detector.py       MNAD black-box wrapper
    object_tracker.py      YOLO + ByteTrack wrapper and false-positive filters
    context_risk_explain.py  context, risk and explainability engines
  MNAD/                    MNAD source code (see "Changes to MNAD")
  config/config.yaml
  Notebooks/               development notebooks (Colab)
  checkpoints/             NOT in repo - trained model goes here
  datasets/                NOT in repo - input videos go here
  outputs/                 generated results
```

## Setup

1. Python 3.10 or 3.11 is recommended.
2. Install packages:
   ```
   pip install torch torchvision opencv-python numpy pandas matplotlib pyyaml scikit-learn ultralytics lap
   ```
3. Download the trained MNAD checkpoint (`model.pth`, `m_items.pt`) from: **[ADD YOUR DRIVE LINK HERE]** and place them in:
   ```
   checkpoints/ped2/model.pth
   checkpoints/ped2/m_items.pt
   ```
   Without a checkpoint the pipeline still runs, with the MNAD score set to 0.
4. Create `config/config.yaml`:
   ```yaml
   project_root: .
   device: cuda    # use "cpu" if there is no NVIDIA GPU
   ```
5. Put a test video in `datasets/input_videos/`. Use a short clip (20 to 60 seconds) on CPU.

YOLO weights (`yolo11s.pt`) download automatically on first run.

## Run

```
python run_pipeline.py --video datasets/input_videos/demo.mp4 --project-root . --occupancy vacant --time 23:30
```

| Flag | Default | Meaning |
|---|---|---|
| `--video` | required | Input video path |
| `--project-root` | `/app` | Use `.` when running locally |
| `--occupancy` | `vacant` | `vacant` or `occupied` |
| `--time` | `23:30` | Simulated start time (HH:MM) |
| `--entry-point` | none | `name,x,y` e.g. `front_door,450,300` |
| `--conf` | `0.55` | YOLO confidence threshold |
| `--multiple-people-thresh` | `2` | People count for the "multiple people" label |
| `--high-risk-thresh` | `70` | Risk score counted as a high-risk event |
| `--email-alerts` | off | Send email on high-risk events |
| `--email-cooldown-sec` | `120` | Minimum gap between alert emails |

### Email alerts (optional)

Credentials are read from environment variables, never from code:

```
ALERT_SENDER_EMAIL=your@gmail.com
ALERT_SENDER_APP_PASSWORD=<Gmail App Password>
ALERT_OWNER_EMAIL=recipient@example.com
```

Use a Gmail **App Password** (requires 2-Step Verification), not your normal password. Never commit these values.

## Outputs

Results are written to `outputs/`:

- `<video>_annotated.mp4`: boxes, track IDs, timestamp, risk score, MNAD score, context labels
- `<video>_risk_scores.csv`: per-frame data
- `<video>_events.json`: high-risk events with reasons
- `<video>_risk_plot.png`: risk score over time

## Changes to MNAD

The MNAD model architecture is unchanged. Only data loading and the entry point were modified so training runs on Windows:

- `model/utils.py`: path handling uses `os.path` instead of splitting on `/`.
- `Train.py`: code wrapped in `main()` with `if __name__ == '__main__'` (needed for multiprocessing on Windows).

Frame filenames must be zero-padded (`000.jpg`, `001.jpg`, ...). MNAD sorts filenames as strings, so unpadded names corrupt the frame order.

## Results so far

| Metric | Value |
|---|---|
| MNAD trained on UCSD Ped2, 60 epochs, AUC | 96.87% |
| Original paper (Park et al.) | 97.0% |

The pipeline runs end to end on custom footage. A labeled evaluation of the context labels (precision/recall per label, comparison against MNAD score alone) is in progress and has not been completed yet.

## Known limitations

- The checkpoint above was trained on UCSD Ped2 (outdoor walkway). Its score is noisy on indoor home footage, so the Risk Engine dampens it. Training on a household dataset is in progress.
- In crowded footage ByteTrack can split one person into several IDs, which affects labels that depend on stable IDs (possible object taken, possible altercation).
- All thresholds and weights are hand-set defaults, not learned values.
- Zone-based reasoning (front yard, driveway) is not implemented. Only single-point entry detection exists.

## References

1. W. Park, J. Kim, I. S. Kweon, "Learning Memory-Guided Normality for Anomaly Detection," CVPR 2020.
2. G. Jocher et al., Ultralytics YOLO, 2023.
3. Y. Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box," ECCV 2022.
4. V. Mahadevan et al., "Anomaly Detection in Crowded Scenes," CVPR 2010 (UCSD dataset).
