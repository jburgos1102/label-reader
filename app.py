import json
import os
import uuid

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

import config
from logger import log
from pipeline import LLM_POLICIES, build_extraction_result, run
from selection import selection_provenance
import storage

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"

# Reject request bodies larger than this before they reach the pipeline.
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB

try:
    storage.init_db()
except Exception:
    log.warning("Storage initialization failed; label results will not be persisted")


@app.errorhandler(413)
def request_too_large(error):
    message = "The uploaded image is too large (limit is 20 MB)."
    if request.path.startswith("/api/"):
        return jsonify({"error": message}), 413
    return render_template("index.html", label_data=None, error_message=message), 413


def _save_upload(uploaded_file):
    filename = secure_filename(uploaded_file.filename)
    if not filename:
        raise ValueError("invalid filename")
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    # Store under a unique name: concurrent uploads sharing an original
    # filename would otherwise overwrite each other mid-processing.
    extension = os.path.splitext(filename)[1].lower()
    path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}{extension}")
    uploaded_file.save(path)
    return path


def _run_and_store(image_path, llm_policy="auto", original_filename=None):
    internal = run(image_path, llm_policy=llm_policy)
    label_id = str(uuid.uuid4())
    result = build_extraction_result(internal, label_id)
    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        selection_reasons, candidates = selection_provenance(
            internal.get("_selections") or {}
        )
        telemetry = result.llm_telemetry or {}
        storage.store(
            result,
            ocr_text=internal.get("_ocr_text", ""),
            image_bytes=image_bytes,
            original_filename=original_filename,
            barcode_raw=internal.get("_barcode_raw", ""),
            ocr_confidence=internal.get("_ocr_confidence"),
            ocr_rotations=internal.get("_ocr_rotations_tried"),
            parser_used=internal.get("parser_used", "") or "",
            selection_reasons=selection_reasons,
            candidates=candidates,
            llm_requested_mode=telemetry.get("requested_mode"),
            llm_model=telemetry.get("model"),
            llm_latency_ms=telemetry.get("latency_ms"),
            llm_trigger_reasons=telemetry.get("trigger_reasons"),
        )
    except Exception:
        log.warning("Failed to persist label %s to storage", label_id)
    return result


@app.route("/")
def home():
    return render_template("index.html", label_data=None, error_message="")


@app.route("/upload", methods=["POST"])
def upload():
    if "label_image" not in request.files:
        return render_template(
            "index.html",
            label_data=None,
            error_message="Please select a shipping label image before clicking Upload.",
        )

    uploaded_file = request.files["label_image"]

    if secure_filename(uploaded_file.filename) == "":
        return render_template(
            "index.html",
            label_data=None,
            error_message="Please upload a file with a valid filename.",
        )

    try:
        image_path = _save_upload(uploaded_file)
    except Exception:
        log.exception("Unable to save uploaded file")
        return render_template(
            "index.html",
            label_data=None,
            error_message="Unable to save the uploaded image. Please try again.",
        )

    try:
        result = _run_and_store(image_path, original_filename=uploaded_file.filename)
    except Exception:
        log.exception("Unable to extract label data from uploaded file")
        return render_template(
            "index.html",
            label_data=None,
            error_message="Unable to read that shipping label image. Please try another file.",
        )

    return render_template(
        "index.html",
        label_data=json.dumps(result.to_dict(), indent=4),
        error_message="",
    )


@app.route("/api/scan", methods=["POST"])
def api_scan():
    if "label_image" not in request.files:
        return (
            jsonify(
                {
                    "error": "No image provided. Send a multipart/form-data POST with a 'label_image' field."
                }
            ),
            400,
        )

    uploaded_file = request.files["label_image"]

    if secure_filename(uploaded_file.filename) == "":
        return jsonify({"error": "Invalid filename."}), 400

    # LLM policy: strict "off" unless the caller explicitly requests a mode
    # AND the server allows it (config.API_LLM_MODES_ALLOWED kill switch).
    requested_mode = (
        request.form.get("llm") or request.args.get("llm") or "off"
    ).strip().lower()
    if requested_mode not in LLM_POLICIES:
        return (
            jsonify({"error": f"llm must be one of: {', '.join(LLM_POLICIES)}."}),
            400,
        )
    if requested_mode not in config.API_LLM_MODES_ALLOWED:
        return (
            jsonify({"error": f"llm mode '{requested_mode}' is not enabled on this server."}),
            400,
        )

    try:
        image_path = _save_upload(uploaded_file)
    except Exception:
        log.exception("Unable to save uploaded file for /api/scan")
        return jsonify({"error": "Unable to save the image."}), 500

    try:
        result = _run_and_store(
            image_path,
            llm_policy=requested_mode,
            original_filename=uploaded_file.filename,
        )
    except Exception:
        log.exception("Unable to extract label data for /api/scan")
        return jsonify({"error": "Unable to process the image."}), 500

    return jsonify(result.to_dict())


_CORRECTABLE_FIELDS = {
    "recipient_name", "street_address", "city", "state",
    "zip_code", "tracking_number", "carrier",
}


@app.route("/labels", methods=["GET"])
def list_labels():
    return jsonify(storage.list_labels())


@app.route("/labels/<label_id>", methods=["GET"])
def get_label(label_id):
    label = storage.get_label(label_id)
    if label is None:
        return jsonify({"error": "Label not found."}), 404
    return jsonify(label)


@app.route("/labels/<label_id>/correct", methods=["POST"])
def correct_label(label_id):
    if storage.get_label(label_id) is None:
        return jsonify({"error": "Label not found."}), 404

    body = request.get_json(silent=True) or {}
    fields = {k: v for k, v in body.items() if k in _CORRECTABLE_FIELDS}

    updated = storage.update_ground_truth(label_id, fields)
    return jsonify(updated)


if __name__ == "__main__":
    app.run()
