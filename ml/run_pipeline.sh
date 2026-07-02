#!/bin/bash
# run_pipeline.sh
# ---------------
# Full training pipeline:
#   1. Import datasets/ into label_storage.db  (safe to re-run, skips duplicates)
#   2. Export annotated labels as training data
#   3. Fine-tune DistilBERT NER model
#   4. Export CoreML model for iOS (macOS only)
#
# Run from the root of the label reader project:
#   bash ml/run_pipeline.sh
#
# First-time setup:
#   pip install -r ml/requirements_ml.txt

set -e

echo "========================================"
echo " Brynka Label Reader — Training Pipeline"
echo "========================================"
echo ""

# Step 1: Import pre-labeled images from datasets/
echo "[1/4] Importing datasets/ into label_storage.db ..."
python ml/import_datasets.py
echo ""

# Step 2: Export training data from label_storage.db
echo "[2/4] Exporting annotated labels ..."
python ml/export_training_data.py
echo ""

# Step 3: Fine-tune DistilBERT (pass extra args through, e.g. --epochs 12)
echo "[3/4] Training NER model ..."
python ml/train.py "$@"
echo ""

# Step 4: Convert to CoreML (macOS only)
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "[4/4] Exporting CoreML model for iOS ..."
    python ml/export_coreml.py
    echo ""
    echo "✅  Pipeline complete."
    echo "    Model:   ml/models/label_reader/"
    echo "    CoreML:  ml/models/LabelReader.mlpackage  ← drag into Xcode"
else
    echo "[4/4] Skipping CoreML export (not macOS)."
    echo "      Copy ml/models/label_reader.onnx to a Mac and run:"
    echo "      python ml/export_coreml.py"
    echo ""
    echo "✅  Training complete."
    echo "    Model: ml/models/label_reader/"
fi
