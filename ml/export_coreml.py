"""
export_coreml.py
----------------
Converts the trained ONNX model to a CoreML .mlpackage for iOS deployment.

Input:  ml/models/label_reader.onnx     (produced by train.py)
Output: ml/models/LabelReader.mlpackage  (drag into Xcode)

Requirements:
  - macOS (coremltools only runs on macOS)
  - pip install coremltools onnx onnxruntime

Usage:
  python ml/export_coreml.py

The resulting LabelReader.mlpackage:
  - Inputs:  input_ids (Int32, shape [1, seq_len])
             attention_mask (Int32, shape [1, seq_len])
  - Output:  logits (Float32, shape [1, seq_len, num_labels])
  - Runs on: Neural Engine + GPU + CPU (ComputeUnit.ALL)
  - Target:  iOS 15+

iOS usage (Swift):
  let model = try LabelReader(configuration: MLModelConfiguration())
  let inputIds = MLMultiArray(...)       // tokenized input
  let attnMask = MLMultiArray(...)
  let output = try model.prediction(input_ids: inputIds, attention_mask: attnMask)
  let logits = output.logits             // [1, seq_len, 12]
  // argmax per token → label index → ID2LABEL lookup
"""

import json
import sys
from pathlib import Path

ONNX_PATH    = Path(__file__).parent / "models" / "label_reader.onnx"
COREML_PATH  = Path(__file__).parent / "models" / "LabelReader.mlpackage"
MODEL_DIR    = Path(__file__).parent / "models" / "label_reader"

LABELS = [
    "O",
    "B-NAME",    "I-NAME",
    "B-STREET",  "I-STREET",
    "B-CITY",    "I-CITY",
    "B-STATE",
    "B-ZIP",
    "B-TRACKING", "I-TRACKING",
    "B-CARRIER",
]


def verify_onnx(onnx_path: Path):
    """Basic ONNX graph verification before conversion."""
    try:
        import onnx
        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)
        print(f"  ONNX model verified ({onnx_path.stat().st_size / 1e6:.1f} MB)")
    except ImportError:
        print("  onnx not installed — skipping verification")
    except Exception as e:
        print(f"  ONNX verification warning: {e}")


def convert_to_coreml(onnx_path: Path, output_path: Path):
    try:
        import coremltools as ct
    except ImportError:
        print("ERROR: coremltools is not installed.")
        print("  pip install coremltools")
        print("  Note: coremltools requires macOS.")
        sys.exit(1)

    print(f"Converting {onnx_path.name} → CoreML ...")

    mlmodel = ct.convert(
        str(onnx_path),
        minimum_deployment_target=ct.target.iOS15,
        compute_units=ct.ComputeUnit.ALL,
        # Provide input type hints so the iOS developer knows the shape
        inputs=[
            ct.TensorType(name="input_ids",      shape=(1, ct.RangeDim(1, 256)), dtype=int),
            ct.TensorType(name="attention_mask",  shape=(1, ct.RangeDim(1, 256)), dtype=int),
        ],
    )

    # Embed label metadata so iOS can decode without a separate lookup file
    mlmodel.user_defined_metadata["label_schema"] = json.dumps({
        "id2label": {str(i): l for i, l in enumerate(LABELS)},
        "label2id": {l: i for i, l in enumerate(LABELS)},
        "version": "1.0",
        "fields": ["recipient_name", "street_address", "city", "state",
                   "zip_code", "tracking_number", "carrier"],
    })

    mlmodel.short_description = "Shipping label NER — extracts recipient, address, tracking, carrier"
    mlmodel.author            = "Brynka"

    mlmodel.save(str(output_path))
    size_mb = sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file()) / 1e6
    print(f"  Saved → {output_path}  ({size_mb:.1f} MB)")
    return mlmodel


def print_ios_usage():
    print("""
iOS Integration (Swift):
──────────────────────────────────────────────────────────────
import CoreML

// 1. Load model (once, at app startup)
let labelReader = try LabelReader(configuration: MLModelConfiguration())

// 2. Tokenize OCR text (use the same DistilBERT tokenizer logic)
//    See: https://github.com/huggingface/swift-transformers

// 3. Run inference
let inputIds   = MLMultiArray(shape: [1, seqLen], dataType: .int32)
let attnMask   = MLMultiArray(shape: [1, seqLen], dataType: .int32)
// ... fill arrays with token ids ...
let output     = try labelReader.prediction(input_ids: inputIds,
                                             attention_mask: attnMask)
let logits     = output.logits  // shape: [1, seqLen, 12]

// 4. Decode — argmax per position → entity label
//    Read id2label from model.userDefinedMetadata["label_schema"]
──────────────────────────────────────────────────────────────
""")


def main():
    print("CoreML Export Pipeline")
    print("=" * 40)

    if not ONNX_PATH.exists():
        print(f"ONNX model not found at {ONNX_PATH}")
        print("Run:  python ml/train.py  first.")
        sys.exit(1)

    print(f"\nStep 1: Verify ONNX model")
    verify_onnx(ONNX_PATH)

    print(f"\nStep 2: Convert to CoreML")
    convert_to_coreml(ONNX_PATH, COREML_PATH)

    print(f"\nStep 3: Usage instructions")
    print_ios_usage()

    print("Drag LabelReader.mlpackage into your Xcode project to use it.")


if __name__ == "__main__":
    main()
