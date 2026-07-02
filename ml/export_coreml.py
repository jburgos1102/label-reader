"""
export_coreml.py
----------------
Converts the trained HuggingFace model to a CoreML .mlpackage for iOS deployment.
Uses PyTorch → TorchScript → CoreML (coremltools 7+ dropped ONNX support).

Input:  ml/models/label_reader/          (HuggingFace checkpoint from train.py)
Output: ml/models/LabelReader.mlpackage  (drag into Xcode)

Requirements:
  - macOS (coremltools only runs on macOS)
  - pip install coremltools torch transformers

Usage:
  python ml/export_coreml.py

The resulting LabelReader.mlpackage:
  - Inputs:  input_ids      (Int32, shape [1, 64])
             attention_mask (Int32, shape [1, 64])
  - Output:  logits (Float32, shape [1, 64, num_labels])
  - Runs on: Neural Engine + GPU + CPU (ComputeUnit.ALL)
  - Target:  iOS 15+

── KNOWN ISSUE: CoreML conversion currently fails on this stack ──
torch 2.12.1 + coremltools 9.0 (Oct 2025, coremltools last verified against
torch 2.7.0 — this pair is untested upstream).

torch.jit.trace() bakes in a `new_ones` op that coremltools can't convert, so
this script was switched to torch.export.export(..., strict=False) instead.
That gets past the tracer issue but the exported graph is left in the
TRAINING dialect, which coremltools also rejects
  ("NotImplementedError: ... Provided Dialect: TRAINING ...").
Calling exported.run_decompositions({}) fixes *that* error, but the
decomposition reintroduces the same unsupported `new_ones` fx node we were
trying to avoid:
  ("NotImplementedError: Unsupported fx node new_ones, kind new_ones").

Net result: no combination of export/decomposition on this torch +
coremltools pairing gets a DistilBERT token-classification head through
ct.convert() today. Re-attempt this path if/when coremltools ships a build
tested against a newer torch, or if the model is re-exported from an older
torch (2.7.x) environment.

── FALLBACK: ship ONNX + onnxruntime-objc instead of CoreML ──
ml/models/label_reader.onnx (from ml/train.py's ONNX export step) already
converts and validates cleanly — it's the artifact to ship for iOS 15+ until
CoreML conversion is unblocked.

Swift integration (ONNX Runtime Mobile):

    # Podfile
    pod 'onnxruntime-objc', '~> 1.20'

    ```swift
    import onnxruntime_objc

    final class LabelReaderModel {
        private let env: ORTEnv
        private let session: ORTSession
        let id2label: [String: String]

        init() throws {
            env = try ORTEnv(loggingLevel: .warning)
            let modelPath = Bundle.main.path(forResource: "label_reader", ofType: "onnx")!
            let options = try ORTSessionOptions()
            // Optional: try .coreML execution provider first, CPU is the
            // guaranteed-available fallback on every device/OS combo.
            session = try ORTSession(env: env, modelPath: modelPath, sessionOptions: options)

            // ml/label_schema.json ships alongside the .onnx in the app bundle —
            // ONNX has no user_defined_metadata slot like CoreML does, so the
            // label schema travels as a separate JSON sidecar instead.
            let schemaURL = Bundle.main.url(forResource: "label_schema", withExtension: "json")!
            let schemaData = try Data(contentsOf: schemaURL)
            let schema = try JSONSerialization.jsonObject(with: schemaData) as! [String: Any]
            id2label = schema["id2label"] as! [String: String]
        }

        // inputIds / attentionMask: [Int32] of length 64 (SEQ_LEN), produced by
        // a DistilBERT tokenizer — see https://github.com/huggingface/swift-transformers
        func predict(inputIds: [Int32], attentionMask: [Int32]) throws -> [Float] {
            let shape: [NSNumber] = [1, 64]

            let inputIdsData = NSMutableData(bytes: inputIds, length: inputIds.count * 4)
            let inputIdsTensor = try ORTValue(
                tensorData: inputIdsData, elementType: .int32, shape: shape)

            let attnMaskData = NSMutableData(bytes: attentionMask, length: attentionMask.count * 4)
            let attnMaskTensor = try ORTValue(
                tensorData: attnMaskData, elementType: .int32, shape: shape)

            let outputs = try session.run(
                withInputs: ["input_ids": inputIdsTensor, "attention_mask": attnMaskTensor],
                outputNames: ["logits"],
                runOptions: nil)

            let logitsTensor = outputs["logits"]!
            let logitsData = try logitsTensor.tensorData() as Data
            return logitsData.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        }
    }
    ```

    // Decode: argmax over the last dim (num_labels) per token position, then
    // map indices to tag names via id2label (loaded from label_schema.json
    // above). label_schema.json is generated from the same LABELS list
    // defined below in this file — see ml/label_schema.json.
"""

import json
import sys
from pathlib import Path

COREML_PATH = Path(__file__).parent / "models" / "LabelReader.mlpackage"
MODEL_DIR   = Path(__file__).parent / "models" / "label_reader"
SEQ_LEN     = 64   # must match max_length used during training

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


def convert_to_coreml(model_dir: Path, output_path: Path):
    try:
        import coremltools as ct
    except ImportError:
        print("ERROR: coremltools not installed — pip install coremltools")
        sys.exit(1)

    try:
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer
    except ImportError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print(f"  Loading model from {model_dir} ...")
    model = AutoModelForTokenClassification.from_pretrained(
        str(model_dir),
        attn_implementation="eager",   # avoid SDPA ops coremltools can't convert
    )
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model.eval()
    model = model.to("cpu")

    # Wrap to return plain logits tensor (coremltools needs simple tensors, not dataclasses)
    class LabelReaderWrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, input_ids, attention_mask):
            return self.inner(input_ids=input_ids, attention_mask=attention_mask).logits

    wrapper = LabelReaderWrapper(model)
    wrapper.eval()

    # Trace with fixed-length dummy inputs
    dummy = tokenizer(
        ["JOHN SMITH 123 MAIN ST CHICAGO IL 60601"],
        return_tensors="pt",
        truncation=True,
        max_length=SEQ_LEN,
        padding="max_length",
    )
    input_ids     = dummy["input_ids"]
    attention_mask = dummy["attention_mask"]

    print("  Exporting with torch.export (dynamo) ...")
    with torch.no_grad():
        exported = torch.export.export(
            wrapper,
            args=(input_ids, attention_mask),
            strict=False,   # allow data-dependent ops common in transformers
        )
        exported = exported.run_decompositions({})

    print("  Converting ExportedProgram → CoreML ...")
    mlmodel = ct.convert(
        exported,
        inputs=[
            ct.TensorType(name="input_ids",      shape=(1, SEQ_LEN), dtype=int),
            ct.TensorType(name="attention_mask",  shape=(1, SEQ_LEN), dtype=int),
        ],
        minimum_deployment_target=ct.target.iOS15,
        compute_units=ct.ComputeUnit.ALL,
    )

    # Embed label metadata so iOS can decode without a separate lookup file
    mlmodel.user_defined_metadata["label_schema"] = json.dumps({
        "id2label":  {str(i): l for i, l in enumerate(LABELS)},
        "label2id":  {l: i for i, l in enumerate(LABELS)},
        "version":   "1.0",
        "seq_len":   SEQ_LEN,
        "fields":    ["recipient_name", "street_address", "city", "state",
                      "zip_code", "tracking_number", "carrier"],
    })
    mlmodel.short_description = "Shipping label NER — extracts recipient, address, tracking, carrier"
    mlmodel.author            = "Brynka"

    mlmodel.save(str(output_path))
    size_mb = sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file()) / 1e6
    print(f"  Saved → {output_path}  ({size_mb:.1f} MB)")


def print_ios_usage():
    print("""
iOS Integration (Swift):
──────────────────────────────────────────────────────────────
import CoreML

// 1. Load model (once, at app startup)
let labelReader = try LabelReader(configuration: MLModelConfiguration())

// 2. Tokenize OCR text with a DistilBERT tokenizer (fixed length: 64)
//    See: https://github.com/huggingface/swift-transformers

// 3. Run inference
let inputIds    = MLMultiArray(shape: [1, 64], dataType: .int32)
let attnMask    = MLMultiArray(shape: [1, 64], dataType: .int32)
// ... fill arrays with token ids and mask values ...
let output      = try labelReader.prediction(input_ids: inputIds,
                                              attention_mask: attnMask)
let logits      = output.logits   // shape: [1, 64, 12]

// 4. Decode — argmax per position → entity label index
//    Read id2label from model.userDefinedMetadata["label_schema"]
──────────────────────────────────────────────────────────────
""")


def main():
    print("CoreML Export Pipeline")
    print("=" * 40)

    if not MODEL_DIR.exists():
        print(f"ERROR: HuggingFace model not found at {MODEL_DIR}")
        print("Run:  python ml/train.py  first.")
        sys.exit(1)

    print(f"\nStep 1: Convert PyTorch → CoreML")
    convert_to_coreml(MODEL_DIR, COREML_PATH)

    print(f"\nStep 2: Usage instructions")
    print_ios_usage()

    print("Done — drag LabelReader.mlpackage into your Xcode project.")


if __name__ == "__main__":
    main()
