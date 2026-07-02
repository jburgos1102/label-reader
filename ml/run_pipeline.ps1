# run_pipeline.ps1
# ----------------
# Full training pipeline:
#   1. Import datasets/ into label_storage.db  (safe to re-run, skips duplicates)
#   2. Export annotated labels as training data
#   3. Fine-tune DistilBERT NER model
#   4. CoreML export reminder (macOS only)
#
# Run from the root of the label reader project:
#   powershell -ExecutionPolicy Bypass -File ml\run_pipeline.ps1
#
# First-time setup:
#   pip install -r ml\requirements_ml.txt

$ErrorActionPreference = "Stop"

Write-Host "========================================"
Write-Host " Brynka Label Reader - Training Pipeline"
Write-Host "========================================"
Write-Host ""

# Step 1: Import pre-labeled images from datasets/
Write-Host "[1/4] Importing datasets/ into label_storage.db ..."
python ml\import_datasets.py
if ($LASTEXITCODE -ne 0) { Write-Error "Import failed."; exit 1 }
Write-Host ""

# Step 2: Export training data from label_storage.db
Write-Host "[2/4] Exporting annotated labels ..."
python ml\export_training_data.py
if ($LASTEXITCODE -ne 0) { Write-Error "Export failed."; exit 1 }
Write-Host ""

# Step 3: Fine-tune DistilBERT (pass extra args through, e.g. --epochs 12)
Write-Host "[3/4] Training NER model ..."
python ml\train.py $args
if ($LASTEXITCODE -ne 0) { Write-Error "Training failed."; exit 1 }
Write-Host ""

# Step 4: CoreML requires macOS
Write-Host "[4/4] CoreML export requires macOS - skipping."
Write-Host "      To produce LabelReader.mlpackage for iOS:"
Write-Host "      1. Copy ml\models\label_reader.onnx to your Mac"
Write-Host "      2. Run: python ml\export_coreml.py"
Write-Host ""
Write-Host "Done."
Write-Host "  Model: ml\models\label_reader\"
Write-Host "  ONNX:  ml\models\label_reader.onnx"
