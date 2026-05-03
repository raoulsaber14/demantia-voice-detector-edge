# Edge AI Dementia Voice Early Detector

A speech-based dementia screening-support tool that runs on a Raspberry Pi 4. It records audio from a USB microphone (or accepts uploaded WAV files), runs a two-stage wav2vec2 ONNX inference pipeline, and serves results through a browser-based UI accessible from any device on the same network.

**This is a screening-support tool only, not a medical diagnosis.**

## Repository Layout

```text
.
├── app/
│   └── flask/
│       ├── web_server.py        # Flask server (entry point)
│       └── templates/
│           └── index.html       # Browser UI
├── configs/
│   └── edge_inference.yaml      # Model paths, labels, risk thresholds
├── edge_inference/
│   ├── backends.py              # ONNX / TFLite / two-stage backends
│   ├── config.py                # EdgeConfig dataclass + YAML loader
│   ├── preprocessing.py         # Waveform → tensor (resample, pad, truncate)
│   └── wrapper.py               # DementiaScreener public API
├── models/                      # NOT in git — copy manually (see below)
├── src/
│   └── audio_pipeline.py        # Mic capture, VAD cleaning, WAV loading
├── requirements.txt
├── Makefile
└── Dockerfile
```

## Models

**Model files are not committed to this repository** (too large for git). After cloning, copy the `models/` directory to the project root manually:

```bash
# Example: copy from the deployment machine via SCP
scp -r user@pi-ip:~/project/models ./models
```

The config expects the Phase D wav2vec2 two-stage deployment layout:

```
models/
└── phaseD_frozen_ensemble_deploy/
    ├── wav2vec2_base/
    │   └── wav2vec2_base_chunk8_accumulator_fp32_external/
    │       ├── model.onnx
    │       └── model.*.weight   (external weight files — copy the whole directory)
    └── wav2vec2_component_head_fp32.onnx
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires PortAudio for microphone capture (`sudo apt install portaudio19-dev` on Raspberry Pi OS).

## Running

```bash
python3 -m app.flask.web_server
```

Then open `http://<pi-ip>:5000` in any browser on the same network.

**Important:** run as a module (`python3 -m app.flask.web_server`) from the project root, not as a script (`python3 app/flask/web_server.py`).

## How It Works

The Pi acts as a local web server. The browser UI lets a clinician either record directly from the Pi's USB microphone or upload a WAV file. The audio is cleaned (VAD, normalization), then fed through a two-stage ONNX pipeline:

1. **Backbone** — wav2vec2 accumulator processes audio in 8-second chunks and accumulates frame-level statistics.
2. **Head** — a classification head takes the mean+std embedding and outputs P(dementia).

Results are shown as a risk band (LOW / REQUIRES REVIEW / HIGH RISK INDICATOR) with a 0–100 risk score.

## Known Limits

- One active session at a time (single-user, single-mic design).
- Microphone capture uses the Pi's connected USB mic; browser mic is not supported.
- This is a screening-support demo, not a clinically validated deployment.
