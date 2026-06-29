import json
import os
import uuid

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from logger import log
from pipeline import build_extraction_result, run
import storage

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"

try:
    storage.init_db()
except Exception:
    log.warning("Storage initialization failed; label results will not be persisted")


def _save_upload(uploaded_file):
    filename = secure_filename(uploaded_file.filename)
    if not filename:
        raise ValueError("invalid filename")
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    path = os.path.join(UPLOAD_FOLDER, filename)
    uploaded_file.save(path)
    return path


def _run_and_store(image_path):
    internal = run(image_path)
    label_id = str(uuid.uuid4())
    result = build_extraction_result(internal, label_id)
    try:
        storage.store(result, ocr_text=internal.get("_ocr_text", ""))
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
        result = _run_and_store(image_path)
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
        return jsonify({"error": "No image provided. Send a multipart/form-data POST with a 'label_image' field."}), 400

    uploaded_file = request.files["label_image"]

    if secure_filename(uploaded_file.filename) == "":
        return jsonify({"error": "Invalid filename."}), 400

    try:
        image_path = _save_upload(uploaded_file)
    except Exception:
        log.exception("Unable to save uploaded file for /api/scan")
        return jsonify({"error": "Unable to save the image."}), 500

    try:
        result = _run_and_store(image_path)
    except Exception:
        log.exception("Unable to extract label data for /api/scan")
        return jsonify({"error": "Unable to process the image."}), 500

    return jsonify(result.to_dict())


if __name__ == "__main__":
    app.run()
