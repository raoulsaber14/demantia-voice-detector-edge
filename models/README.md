# External Model Artifacts

This repository now runs out of the box with the committed dummy backend in
`configs/edge_inference.yaml`, so no binary model file is required for the
default smoke-test path.

Real ONNX deployment artifacts are not included in this submission snapshot.
If you want to run the edge prototype with external models, place them in this
folder and start from `configs/edge_inference.onnx.example.yaml`.

The example config assumes a two-stage export:

- `models/wav2vec2_backbone.onnx`
- `models/wav2vec2_head.onnx`

If you have a single-file ONNX model instead, set `head_model_path: null` and
change the backend to `onnx`.
