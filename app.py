from flask import Flask, render_template, request
from label_reader import extract_label_data
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

    if uploaded_file.filename == "":
        return render_template(
            "index.html",
            label_data=None,
            error_message="Please select a shipping label image before clicking Upload.",
        )

    image_path = os.path.join(UPLOAD_FOLDER, uploaded_file.filename)

    uploaded_file.save(image_path)

    label_data = extract_label_data(image_path)

    label_data_json = json.dumps(label_data, indent=4)

    return render_template("index.html", label_data=label_data_json, error_message="")


if __name__ == "__main__":
    app.run()
