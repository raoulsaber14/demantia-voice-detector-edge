"""
Flask web UI for the edge dementia screener.

Usage:
    pip install flask
    python -m app.flask.web_server
    # Then open http://<pi-ip>:5000 from any browser on the same network

The model is loaded once at startup. Each POST /start-session spawns a
background thread that records, cleans, and runs inference; the browser
polls GET /status/<session_id> every second until done.
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import threading
import uuid
from datetime import datetime
from pathlib import Path

# Ensure the project root is on sys.path when the file is run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from flask import Flask, Response, jsonify, render_template, request

from src.audio_pipeline import (
    MicDisconnectedError,
    PipelineConfig,
    SilenceOnlyError,
    capture_audio,
    load_wav,
    run_pipeline,
)
from edge_inference import DementiaScreener, load_config
from edge_inference.config import EdgeConfig

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("web_server")

# ---------------------------------------------------------------------------
# Model — loaded once at startup; kept alive across sessions
# ---------------------------------------------------------------------------

CONFIG_PATH = Path("configs/edge_inference.yaml")

try:
    _edge_cfg: EdgeConfig = load_config(CONFIG_PATH)
    _screener: DementiaScreener = DementiaScreener(_edge_cfg, num_threads=4)
    _screener.warmup(n=2)
    logger.info("DementiaScreener ready (backend=%s)", _edge_cfg.resolved_backend())
except FileNotFoundError as _e:
    logger.error("Could not load config or model: %s", _e)
    raise SystemExit(1) from _e

# ---------------------------------------------------------------------------
# Session state  (one session at a time)
# ---------------------------------------------------------------------------

_sessions: dict = {}
_active_session_id: str | None = None
_active_lock = threading.Lock()


def _init_session(session_id: str, label: str, initial_state: str = "recording") -> None:
    _sessions[session_id] = {
        "state": initial_state,
        "label": label,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "result": None,
        "error_msg": None,
    }


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

def _run_session(session_id: str, duration: float, label: str) -> None:
    global _active_session_id
    try:
        pipeline_cfg = PipelineConfig(
            sample_rate=_edge_cfg.sample_rate,
            record_duration_s=duration,
        )

        raw_audio = capture_audio(pipeline_cfg)

        _sessions[session_id]["state"] = "processing"

        pipeline_result = run_pipeline(raw_audio, pipeline_cfg)

        result = _screener.predict(
            pipeline_result.cleaned_audio,
            sample_rate=pipeline_cfg.sample_rate,
        )

        _sessions[session_id]["result"] = result.to_dict()
        _sessions[session_id]["state"] = "done"
        logger.info(
            "Session %s done: band=%s score=%d",
            session_id[:8], result.risk_band, result.risk_score,
        )

    except MicDisconnectedError as exc:
        logger.error("Mic error in session %s: %s", session_id[:8], exc)
        _sessions[session_id]["state"] = "error"
        _sessions[session_id]["error_msg"] = (
            f"Microphone disconnected or unavailable: {exc}. "
            "Check the USB microphone connection and try again."
        )

    except SilenceOnlyError as exc:
        logger.warning("No speech detected in session %s: %s", session_id[:8], exc)
        _sessions[session_id]["state"] = "error"
        _sessions[session_id]["error_msg"] = (
            "No speech was detected in the recording. "
            "Ensure the microphone is working and the participant speaks clearly "
            "during the session window."
        )

    except Exception as exc:
        logger.exception("Unexpected error in session %s", session_id[:8])
        _sessions[session_id]["state"] = "error"
        _sessions[session_id]["error_msg"] = (
            f"An unexpected error occurred: {exc}. See server logs for details."
        )

    finally:
        with _active_lock:
            _active_session_id = None


# ---------------------------------------------------------------------------
# Upload worker
# ---------------------------------------------------------------------------

def _run_upload_session(session_id: str, audio_bytes: bytes, filename: str, label: str) -> None:
    global _active_session_id
    try:
        # Write to a temp file so load_wav can read it normally
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        try:
            raw_audio, sr = load_wav(tmp_path)
        finally:
            os.unlink(tmp_path)

        pipeline_cfg = PipelineConfig(sample_rate=_edge_cfg.sample_rate)

        if sr != _edge_cfg.sample_rate:
            logger.warning(
                "Upload session %s: file %r is %d Hz, expected %d Hz. "
                "Skipping cleaning pipeline; the wrapper will resample internally.",
                session_id[:8], filename, sr, _edge_cfg.sample_rate,
            )
            cleaned_audio = raw_audio
            cleaned_sr = sr
        else:
            pipeline_result = run_pipeline(raw_audio, pipeline_cfg)
            cleaned_audio = pipeline_result.cleaned_audio
            cleaned_sr = pipeline_cfg.sample_rate

        result = _screener.predict(cleaned_audio, sample_rate=cleaned_sr)
        _sessions[session_id]["result"] = result.to_dict()
        _sessions[session_id]["state"] = "done"
        logger.info(
            "Upload session %s done: band=%s score=%d",
            session_id[:8], result.risk_band, result.risk_score,
        )

    except SilenceOnlyError as exc:
        logger.warning("No speech in upload session %s: %s", session_id[:8], exc)
        _sessions[session_id]["state"] = "error"
        _sessions[session_id]["error_msg"] = (
            "No speech was detected in the uploaded file. "
            "Ensure the file contains clear speech audio."
        )

    except Exception as exc:
        logger.exception("Unexpected error in upload session %s", session_id[:8])
        _sessions[session_id]["state"] = "error"
        _sessions[session_id]["error_msg"] = (
            f"An unexpected error occurred: {exc}. See server logs for details."
        )

    finally:
        with _active_lock:
            _active_session_id = None


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB upload limit


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/start-session", methods=["POST"])
def start_session():
    global _active_session_id

    data = request.get_json(force=True) or {}
    try:
        duration = int(data.get("duration", 30))
    except (TypeError, ValueError):
        return jsonify({"error": "duration must be an integer"}), 400

    if duration not in (10, 20, 30, 60):
        return jsonify({"error": "duration must be one of: 10, 20, 30, 60"}), 400

    label = str(data.get("label", "")).strip()[:80]

    with _active_lock:
        if _active_session_id is not None:
            return jsonify({"error": "A session is already running. Wait for it to finish."}), 409
        session_id = str(uuid.uuid4())
        _active_session_id = session_id
        _init_session(session_id, label)

    thread = threading.Thread(
        target=_run_session,
        args=(session_id, float(duration), label),
        daemon=True,
    )
    thread.start()
    logger.info("Started session %s (duration=%ds label=%r)", session_id[:8], duration, label)
    return jsonify({"session_id": session_id})


@app.route("/upload-session", methods=["POST"])
def upload_session():
    global _active_session_id

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "No file selected"}), 400

    if not f.filename.lower().endswith(".wav"):
        return jsonify({"error": "Only WAV files are supported"}), 400

    label = request.form.get("label", "").strip()[:80]

    audio_bytes = f.read()
    if len(audio_bytes) == 0:
        return jsonify({"error": "Uploaded file is empty"}), 400

    with _active_lock:
        if _active_session_id is not None:
            return jsonify({"error": "A session is already running. Wait for it to finish."}), 409
        session_id = str(uuid.uuid4())
        _active_session_id = session_id
        _init_session(session_id, label, initial_state="processing")

    thread = threading.Thread(
        target=_run_upload_session,
        args=(session_id, audio_bytes, f.filename, label),
        daemon=True,
    )
    thread.start()
    logger.info("Upload session %s started (file=%r label=%r)", session_id[:8], f.filename, label)
    return jsonify({"session_id": session_id})


@app.route("/status/<session_id>")
def session_status(session_id: str):
    session = _sessions.get(session_id)
    if session is None:
        return jsonify({"error": "Unknown session ID"}), 404
    return jsonify({
        "state":     session["state"],
        "result":    session["result"],
        "error_msg": session["error_msg"],
    })


@app.route("/report/<session_id>")
def session_report(session_id: str):
    session = _sessions.get(session_id)
    if session is None or session["state"] != "done":
        return jsonify({"error": "Report not available — session not complete"}), 404

    r = session["result"]
    label = session.get("label", "")
    timestamp = session.get("timestamp", "")

    lines = [
        "DEMENTIA VOICE SCREENING — SESSION REPORT",
        "=" * 52,
        f"Timestamp:           {timestamp}",
    ]
    if label:
        lines.append(f"Session label:       {label}")
    lines += [
        "",
        "SCREENING RESULT",
        "-" * 52,
        f"Risk band:           {r['risk_band'].upper()}",
        f"Risk score:          {r['risk_score']} / 100",
        f"Screening indicator: {r['predicted_label']}",
        f"Confidence:          {r['confidence']:.3f}",
        "",
        "Per-class screening probabilities:",
    ]
    for name, prob in r["class_probabilities"].items():
        bar = "█" * int(prob * 28)
        lines.append(f"  {name:<22}  {prob:.3f}  {bar}")

    if r.get("markers"):
        lines += ["", "Markers:"]
        for name, val in r["markers"].items():
            lines.append(f"  {name:<22}  {val:.3f}")

    lines += [
        "",
        f"Preprocessing time:  {r['preprocess_ms']:.1f} ms",
        f"Inference time:      {r['inference_ms']:.1f} ms",
        "",
        "=" * 52,
        "IMPORTANT NOTICE",
        "=" * 52,
        r["disclaimer"],
        "",
        "This report is produced by an automated screening tool.",
        "It is not a medical diagnosis. Refer to a qualified clinician",
        "for interpretation and any health decisions.",
        "=" * 52,
    ]

    report_text = "\n".join(lines)
    return Response(
        report_text,
        mimetype="text/plain",
        headers={
            "Content-Disposition": (
                f'attachment; filename="screening_report_{session_id[:8]}.txt"'
            )
        },
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
