#!/bin/bash

# Example usage of the DICOM redaction script with Google Cloud Storage
# This script assumes you're running on a GCP Compute Engine instance

# Set your GCS bucket and paths
INPUT_BUCKET="gs://your-dicom-input-bucket/input-folder/"
OUTPUT_BUCKET="gs://your-dicom-output-bucket/output-folder/"
HEADERS_BUCKET="gs://your-dicom-headers-bucket/headers-folder/"

echo "=== GCS Streaming Mode (Recommended) ==="
echo "Processes files one at a time, minimal local disk usage"
echo ""

# Example 1: GCS Streaming Mode - Process DICOM files from GCS, output to GCS
# This uses the new streaming approach that processes one file at a time
echo "Example 1: GCS Streaming Mode (all GCS paths)"
python redact_dicom_image_phi.py \
    --input "$INPUT_BUCKET" \
    --output "$OUTPUT_BUCKET" \
    --headers "$HEADERS_BUCKET" \
    --redact-metadata

echo ""
echo "=== Traditional Mode (Fallback) ==="
echo "Downloads all files locally, processes, then uploads"
echo ""

# Example 2: Traditional Mode - Process DICOM files from GCS, output to local directory
echo "Example 2: Mixed GCS/Local (downloads from GCS, saves locally)"
python redact_dicom_image_phi.py \
    --input "$INPUT_BUCKET" \
    --output "/tmp/dicom_output" \
    --headers "/tmp/dicom_headers" \
    --redact-metadata

# Example 3: Traditional Mode - Process local files, output to GCS
echo "Example 3: Mixed Local/GCS (processes locally, uploads to GCS)"
python redact_dicom_image_phi.py \
    --input "/path/to/local/dicom/files" \
    --output "$OUTPUT_BUCKET" \
    --headers "$HEADERS_BUCKET" \
    --redact-metadata

# Example 4: GCS Streaming Mode with additional options
echo "Example 4: GCS Streaming with additional options"
python redact_dicom_image_phi.py \
    --input "$INPUT_BUCKET" \
    --output "$OUTPUT_BUCKET" \
    --headers "$HEADERS_BUCKET" \
    --redact-metadata \
    --random-text

echo ""
echo "=== Performance Notes ==="
echo "- GCS Streaming Mode: Minimal local disk usage, processes one file at a time"
echo "- Traditional Mode: Downloads all files first, uses more local disk space"
echo "- Use GCS Streaming Mode when all paths are GCS for best efficiency"
