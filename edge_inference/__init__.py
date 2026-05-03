"""Edge inference wrapper for the dementia voice screening model.

Public API:
    from edge_inference import DementiaScreener, load_config

    cfg = load_config("configs/edge_inference.yaml")
    screener = DementiaScreener(cfg)
    result = screener.predict(audio, sample_rate=16000)
"""

from edge_inference.config import EdgeConfig, load_config
from edge_inference.wrapper import DementiaScreener, ScreeningResult

__all__ = ["DementiaScreener", "ScreeningResult", "EdgeConfig", "load_config"]
