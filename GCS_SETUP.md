# Google Cloud Storage Integration

This script now supports reading from and writing to Google Cloud Storage buckets when running on GCP Compute Engine instances.

## Prerequisites

1. **GCP Compute Engine Instance**: The script should run on a Compute Engine instance with appropriate permissions.

2. **Authentication**: The Compute Engine instance should have the necessary service account permissions to access GCS buckets. The default service account typically has these permissions, but you may need to grant additional roles:

   - `Storage Object Viewer` (for reading from input buckets)
   - `Storage Object Creator` (for writing to output buckets)

3. **Dependencies**: Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### GCS Path Format

Use `gs://bucket-name/path/to/files/` format for GCS paths:

- Input: `gs://my-dicom-bucket/input-folder/`
- Output: `gs://my-dicom-bucket/output-folder/`
- Headers: `gs://my-dicom-bucket/headers-folder/`

### Examples

#### Process DICOM files from GCS, output to GCS

```bash
python redact_dicom_image_phi.py \
    --input "gs://my-dicom-bucket/input/" \
    --output "gs://my-dicom-bucket/output/" \
    --headers "gs://my-dicom-bucket/headers/" \
    --redact-metadata \
    --recursive
```

#### Mixed local and GCS paths

```bash
# Download from GCS, process locally, upload results to GCS
python redact_dicom_image_phi.py \
    --input "gs://my-dicom-bucket/input/" \
    --output "/tmp/local-output/" \
    --headers "gs://my-dicom-bucket/headers/" \
    --redact-metadata
```

## How It Works

The script supports two modes depending on your path configuration:

### GCS Streaming Mode (Recommended)

**Triggers when**: All three paths (input, output, headers) are GCS paths (`gs://`)

1. **File Discovery**: Lists all DICOM files in the input GCS bucket/prefix
2. **One-by-One Processing**: For each file:
   - Downloads the file to a temporary location
   - Processes the DICOM file (redaction, metadata handling)
   - Uploads the processed file to the output GCS bucket
   - Uploads JSON metadata to the headers GCS bucket
   - **Immediately deletes** the temporary file
3. **Minimal Local Storage**: Only one file exists locally at any time

### Traditional Mode (Fallback)

**Triggers when**: Any path is local or mixed local/GCS

1. **Input Processing**: If the input path is a GCS path (`gs://`), the script:

   - Downloads all files from the specified bucket/prefix to a temporary local directory
   - Maintains the directory structure from the bucket

2. **Processing**: The DICOM redaction runs on the local files using the existing logic

3. **Output Processing**: If the output path is a GCS path, the script:

   - Uploads all processed files to the specified bucket/prefix
   - Maintains the directory structure

4. **Cleanup**: Temporary directories are automatically cleaned up after processing

## When to Use Each Mode

### Use GCS Streaming Mode When:

- All three paths (input, output, headers) are GCS paths
- Processing large datasets that might not fit on local disk
- You want minimal local storage usage
- You're running on a Compute Engine instance with limited disk space

### Use Traditional Mode When:

- You have mixed local and GCS paths
- You need to process files locally for debugging or inspection
- You have sufficient local disk space for the entire dataset
- You're doing development or testing

## Performance Considerations

### GCS Streaming Mode

- **Minimal Disk Usage**: Only one DICOM file exists locally at any time
- **Scalable**: Can process datasets of any size without running out of disk space
- **Network**: Ensure good network connectivity between your Compute Engine instance and GCS
- **Memory**: Each file is processed individually, so memory usage is predictable

### Traditional Mode

- **Large Datasets**: Downloads entire datasets locally - ensure sufficient disk space
- **Network**: Ensure good network connectivity between your Compute Engine instance and GCS
- **Storage**: Temporary files are stored in `/tmp` by default - ensure sufficient disk space

## Error Handling

- The script will fail gracefully if GCS authentication fails
- Temporary directories are cleaned up even if processing fails
- Check GCS bucket permissions if you encounter access errors

## Service Account Permissions

Your Compute Engine instance's service account needs these IAM roles:

- `roles/storage.objectViewer` - to read from input buckets
- `roles/storage.objectCreator` - to write to output buckets
- `roles/storage.objectAdmin` - for full read/write access (optional)

You can grant these permissions in the GCP Console under IAM & Admin > IAM.
