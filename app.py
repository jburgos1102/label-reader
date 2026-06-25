from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from label_reader import extract_label_data
from logger import log
import os
import json

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"


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

    filename = secure_filename(uploaded_file.filename)

    if filename == "":
        return render_template(
            "index.html",
            label_data=None,
            error_message="Please upload a file with a valid filename.",
        )

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    image_path = os.path.join(UPLOAD_FOLDER, filename)

    try:
        uploaded_file.save(image_path)
    except Exception:
        log.exception("Unable to save uploaded file")
        return render_template(
            "index.html",
            label_data=None,
            error_message="Unable to save the uploaded image. Please try again.",
        )

    try:
        label_data = extract_label_data(image_path)
    except Exception:
        log.exception("Unable to extract label data from uploaded file")
        return render_template(
            "index.html",
            label_data=None,
            error_message="Unable to read that shipping label image. Please try another file.",
        )

    label_data_json = json.dumps(label_data, indent=4)

    return render_template("index.html", label_data=label_data_json, error_message="")


if __name__ == "__main__":
    app.run()
