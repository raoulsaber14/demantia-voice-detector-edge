# Edge AI Dementia Voice Detector

This project is an edge-deployed dementia voice screening prototype built around a Raspberry Pi workflow. It captures or uploads speech audio, cleans the signal, runs local inference through a model wrapper, and presents a screening-oriented result through CLI and web interfaces.

## Important Notice

This system is a **screening support tool, not a diagnosis system**. Any output produced by this repository is a risk indicator only and must not be treated as a medical diagnosis or as the sole basis for a clinical decision.

## What The System Does

- Records speech from a microphone or accepts uploaded WAV audio.
- Cleans the waveform with bandpass filtering, spectral subtraction, and normalization.
- Prepares model-ready audio at 16 kHz mono.
- Runs local inference through a backend wrapper that supports dummy, ONNX, TFLite, and a two-stage wav2vec2 path.
- Returns a structured result with a risk score, risk band, probabilities, and a non-diagnostic disclaimer.

## Repository Structure

```text
.
├── README.md
├── audio_pipeline.py
├── audio_pipeline_documentation.docx
├── record_and_screen.py
├── test_pipeline.py
├── test_edge_inference.py
├── verify_two_stage.py
├── web_server.py
├── generate_test_audio.py
├── Project_tree_structure.txt
├── edge_inference/
│   ├── __init__.py
│   ├── backends.py
│   ├── benchmark.py
│   ├── cli.py
│   ├── config.py
│   ├── make_dummy_model.py
│   ├── model_config.yaml
│   ├── preprocessing.py
│   └── wrapper.py
├── templates/
│   └── index.html
├── models/
│   └── demantia_wav2vec_int8.onnx.txt
├── Audio_test_samples_dementia/
└── non_dementia_Audio_test_samples/
```

## Main Components

- `audio_pipeline.py`: signal-processing pipeline for capture, denoising, filtering, VAD support, normalization, and WAV I/O.
- `edge_inference/`: runtime inference layer, backend abstraction, preprocessing, benchmarking, and config loading.
- `record_and_screen.py`: CLI entry point for end-to-end screening from mic or WAV.
- `web_server.py`: Flask web app for browser-based screening sessions.
- `test_pipeline.py`: standalone pipeline validation and recording utility.
- `test_edge_inference.py`: smoke tests for the inference wrapper and benchmark path.
- `verify_two_stage.py`: checks that the expected two-stage ONNX backbone and head files match the wrapper contract.

## Installation

### 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Python dependencies

Recommended base install:

```bash
pip install numpy scipy flask sounddevice onnxruntime onnx requests pyyaml soundfile
```

Notes:

- `numpy` and `scipy` are required for the audio pipeline.
- `flask` is required for the web UI.
- `sounddevice` is required for live microphone capture.
- `onnxruntime` is required for the ONNX inference path.
- `onnx` is used by the dummy ONNX model generator and some smoke-test paths.
- `requests` is only needed for optional result-posting helpers in `record_and_screen.py`.
- `pyyaml` is optional because the project has a minimal YAML fallback parser, but installing it is recommended.
- `soundfile` is optional because the CLI has a stdlib WAV fallback, but installing it is recommended.

### 3. Install system dependency for microphone capture

On Raspberry Pi OS / Debian-based Linux:

```bash
sudo apt update
sudo apt install libportaudio2
```

Optional utilities:

```bash
sudo apt install ffmpeg sox libsox-fmt-all
```

- `ffmpeg` helps if you want to generate TTS-based test audio.
- `sox` is useful for local audio inspection and playback.

## Model Setup

The default config in `edge_inference/model_config.yaml` expects two ONNX files that are **not included in this snapshot**:

- a wav2vec2 accumulator backbone
- a classification head

Before running the default end-to-end app, do one of the following:

1. Place the real ONNX model files at the paths referenced in `edge_inference/model_config.yaml`.
2. Or edit `edge_inference/model_config.yaml` for a temporary smoke-test setup:
   - set `model_path: dummy`
   - set `backend: dummy`
   - remove or clear `head_model_path`

Without one of those steps, `web_server.py` and the default end-to-end CLI path will fail at startup because the configured model files are missing.

## How To Run

### Run the audio pipeline only

Record from microphone:

```bash
python3 test_pipeline.py
```

Process an existing WAV file:

```bash
python3 test_pipeline.py --input path/to/file.wav
```

List available audio devices:

```bash
python3 test_pipeline.py --list-devices
```

### Run end-to-end screening from the command line

With a configured backend:

```bash
python3 record_and_screen.py --input path/to/file.wav --config edge_inference/model_config.yaml
```

Record directly from the microphone:

```bash
python3 record_and_screen.py --duration 30 --config edge_inference/model_config.yaml
```

### Run the web interface

After the model config points to a valid backend:

```bash
python3 web_server.py
```

Then open:

- `http://127.0.0.1:5000` on the same machine, or
- `http://<pi-ip>:5000` from another device on the same network

### Run the edge inference CLI directly

```bash
python3 -m edge_inference.cli --config edge_inference/model_config.yaml --audio path/to/file.wav
```

Run the benchmark mode:

```bash
python3 -m edge_inference.cli --config edge_inference/model_config.yaml --audio path/to/file.wav --benchmark --runs 30
```

## How To Test

### Inference wrapper smoke tests

```bash
python3 test_edge_inference.py
```

This exercises:

- dummy backend behavior
- padding, truncation, and resampling paths
- YAML config loading
- benchmark code
- ONNX path if the runtime and model-generation dependencies are available

### Two-stage ONNX contract verification

```bash
python3 verify_two_stage.py
```

Use this only after the expected backbone and head ONNX files exist at the configured paths.

### Pipeline validation

```bash
python3 test_pipeline.py --input path/to/file.wav
```

This produces cleaned audio and a processing summary in `pipeline_output/`.

## Known Limitations

- The real two-stage ONNX model artifacts referenced by the default config are not included in this repo snapshot.
- The web app can only handle one active session at a time.
- The upload flow only accepts WAV files.
- Clinical validity depends on the external trained model and proper validation outside this repository.
- The repository contains edge deployment and pipeline code, but the larger nested training/reporting project referenced in `Project_tree_structure.txt` is not present in this attached snapshot.
- The model output is a screening indicator only and should not be interpreted as a diagnosis.

## Sample Data

The repository includes example audio under:

- `Audio_test_samples_dementia/`
- `non_dementia_Audio_test_samples/`

These are useful for manual pipeline and inference checks if your model backend is configured.

## Additional Documentation

- `audio_pipeline_documentation.docx`: detailed technical write-up for the audio preprocessing pipeline.
- `Project_tree_structure.txt`: archived tree view from a broader project layout.
