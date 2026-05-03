PYTHON ?= python3
MPLCONFIGDIR ?= .cache/matplotlib
MPLBACKEND ?= Agg
export MPLCONFIGDIR
export MPLBACKEND

.PHONY: setup metadata preprocess features baseline metadata-enhanced preprocess-enhanced features-enhanced baseline-enhanced phase1-enhanced phase3-preprocess phase3-features phase3-governance phase3-prune phase3-baseline phase3-validate phase3-closeout phase3 phase4-baseline dataset-audit segments phase6-prepare phase6-pilot phase6-train wav2vec demo compile test clean-generated

setup:
	$(PYTHON) -m venv .venv
	. .venv/bin/activate && python -m pip install --upgrade pip
	. .venv/bin/activate && python -m pip install -r requirements.txt

metadata:
	$(PYTHON) -m src.metadata_utils --metadata-path data/metadata/raw_metadata.xlsx --raw-audio-dir data/raw_audio

preprocess:
	$(PYTHON) -m src.preprocess --metadata-csv data/metadata/metadata_clean.csv --raw-audio-root data/raw_audio --processed-root data/processed_audio --trim-silence --reuse-existing-valid

features:
	$(PYTHON) -m src.features --metadata-csv data/metadata/metadata_clean.csv --processed-root data/processed_audio --output-csv data/features/file_features.csv

baseline:
	$(PYTHON) -m src.train_baseline --feature-csv data/features/file_features.csv --model-out-dir models/baseline

metadata-enhanced:
	$(PYTHON) -m src.metadata_utils --metadata-path data/metadata/raw_metadata.xlsx --raw-audio-dir data/raw_audio

preprocess-enhanced:
	$(PYTHON) -m src.preprocess --metadata-csv data/metadata/metadata_fixed_splits.csv --raw-audio-root data/raw_audio --processed-root data/processed_audio --trim-silence --reuse-existing-valid

features-enhanced:
	$(PYTHON) -m src.features --metadata-csv data/metadata/metadata_fixed_splits.csv --processed-root data/processed_audio --output-csv data/features/features_enhanced.csv

baseline-enhanced:
	$(PYTHON) -m src.train_baseline --feature-csv data/features/features_enhanced.csv --model-out-dir models/baseline_enhanced_v2 --enhanced --calibrated-model-out-dir models/calibrated_v2

phase1-enhanced: metadata-enhanced preprocess-enhanced features-enhanced baseline-enhanced

phase3-preprocess: preprocess-enhanced

phase3-features: features-enhanced

phase3-governance:
	$(PYTHON) scripts/phase3_closeout.py

phase3-prune: phase3-governance

phase3-baseline: phase3-prune
	$(PYTHON) -m src.train_baseline --feature-csv data/features/features_phase3_governed_pruned.csv --model-out-dir models/baseline_phase3_governed --enhanced --calibrated-model-out-dir models/calibrated_phase3_governed

phase3-validate:
	$(PYTHON) scripts/phase3_closeout.py

phase3-closeout: phase3

phase3: phase3-preprocess phase3-features phase3-baseline phase3-validate

phase4-baseline:
	$(PYTHON) scripts/phase4_baseline_experiments.py

dataset-audit:
	$(PYTHON) scripts/dataset_audit.py

segments:
	$(PYTHON) -m src.vad_segment --metadata-csv data/metadata/metadata_clean.csv --processed-root data/processed_audio --segments-root data/segments

phase6-prepare:
	$(PYTHON) -m src.train_wav2vec --config configs/wav2vec_config.yaml --mode prepare

phase6-pilot: phase6-prepare
	$(PYTHON) -m src.train_wav2vec --config configs/wav2vec_config.yaml --mode pilot

phase6-train:
	$(PYTHON) -m src.train_wav2vec --config configs/wav2vec_config.yaml --mode train

wav2vec: phase6-pilot

demo:
	streamlit run app/demo_app.py

compile:
	$(PYTHON) -m py_compile src/*.py app/demo_app.py

test:
	$(PYTHON) -m unittest discover -s tests -p "test_*.py"

clean-generated:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name .pytest_cache -prune -exec rm -rf {} +
