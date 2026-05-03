# External Model Artifacts

Real ONNX deployment artifacts are not included in this submission snapshot.

The committed runtime config in `configs/edge_inference.yaml` expects the
two-stage Phase D wav2vec2 export at:

- `models/phaseD_frozen_ensemble_deploy/wav2vec2_base/wav2vec2_base_chunk8_accumulator_fp32_external/model.onnx`
- `models/phaseD_frozen_ensemble_deploy/wav2vec2_component_head_fp32.onnx`

For the committed Docker path, the simplest option is to mount the local
`models/` folder into the container at `/app/models` if you want to supply
external artifacts at runtime.

If you want to run the edge prototype with a different ONNX export layout,
start from `configs/edge_inference.onnx.example.yaml` and update the paths.
