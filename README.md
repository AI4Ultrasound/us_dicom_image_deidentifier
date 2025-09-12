# DICOM Deidentifier

A comprehensive tool for deidentifying DICOM medical images by detecting and redacting personally identifiable information (PHI) from both image pixels and metadata headers.

## Features

- **Advanced PHI Detection**: Uses Microsoft Presidio's image redaction engine with OCR to detect text overlays containing PHI
- **Multi-frame Support**: Efficiently processes multi-frame DICOM studies with intelligent ROI-based detection
- **Metadata Anonymization**: Replaces patient identifiers, dates, and other PHI in DICOM headers
- **Flexible Redaction**: Configurable margin-based detection with customizable ROI ratios
- **Batch Processing**: Process entire directories of DICOM files recursively
- **Visual Reports**: Generate PDF comparison reports showing original vs redacted images
- **Random Text Replacement**: Optionally replace redacted areas with random text for better visual appearance

## Installation

### Option 1: Using uv (Recommended)

1. Install uv if you haven't already:

```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via pip
pip install uv
```

2. Clone this repository:

```bash
git clone <repository-url>
cd dicom_deidentifier
```

3. Create and activate virtual environment with uv:

```bash
# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt
```

4. Download spaCy language models:

```bash
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_lg
```

### Option 2: Using pip

1. Clone this repository:

```bash
git clone <repository-url>
cd dicom_deidentifier
```

2. Create virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Download spaCy language models:

```bash
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_lg
```

## System Dependencies

### Tesseract OCR Installation

Tesseract OCR is required for text detection in DICOM images. Install it based on your operating system:

#### macOS

Using Homebrew (recommended):

```bash
brew install tesseract
```

Using MacPorts:

```bash
sudo port install tesseract4
```

#### Windows

1. Download the installer from the [official Tesseract releases](https://github.com/UB-Mannheim/tesseract/wiki)
2. Run the installer and follow the setup wizard
3. Add Tesseract to your PATH environment variable:
   - The default installation path is usually `C:\Program Files\Tesseract-OCR`
   - Add `C:\Program Files\Tesseract-OCR` to your system PATH

Alternatively, using Chocolatey:

```powershell
choco install tesseract
```

Or using Scoop:

```powershell
scoop install tesseract
```

#### Linux

**Ubuntu/Debian:**

```bash
sudo apt update
sudo apt install tesseract-ocr
```

**CentOS/RHEL/Fedora:**

```bash
# For CentOS/RHEL 7/8
sudo yum install tesseract

# For Fedora
sudo dnf install tesseract

# For newer versions
sudo dnf install tesseract-ocr
```

**Arch Linux:**

```bash
sudo pacman -S tesseract
```

**Alpine Linux:**

```bash
sudo apk add tesseract-ocr
```

### Verify Installation

After installation, verify Tesseract is working:

```bash
tesseract --version
```

You should see output similar to:

```
tesseract 5.3.0
 leptonica-1.83.1
  libgif 5.2.1 : libjpeg 8d (libjpeg-turbo 2.1.5.1) : libpng 1.6.37 : libtiff 4.4.0 : zlib 1.2.13 : libwebp 1.2.4 : libopenjp2 2.4.0
 Found AVX2
 Found AVX
 Found FMA
 Found SSE4.1
 Found libarchive 3.6.2 zlib/1.2.13 liblzma/5.2.5 libzstd/1.5.2
```

## Usage

### Basic Usage

```bash
python redact_dicom_image_phi.py \
    --input /path/to/dicom/files \
    --output /path/to/output/directory \
    --headers /path/to/headers/directory
```

### Advanced Options

```bash
python redact_dicom_image_phi.py \
    --input /path/to/dicom/files \
    --output /path/to/output/directory \
    --headers /path/to/headers/directory \
    --recursive \
    --overwrite \
    --samples 5 \
    --top-ratio 0.25 \
    --bottom-ratio 0.25 \
    --left-ratio 0.25 \
    --right-ratio 0.25 \
    --padding 8 \
    --merge-iou 0.2 \
    --expand 8 \
    --patient-prefix "anon" \
    --random-text \
    --text-bold \
    --pdf-report /path/to/report.pdf
```

### Example: Top-Only Detection

For DICOM images where PHI text only appears in the top margin (e.g., some ultrasound or X-ray images), you can focus detection on just that region:

```bash
python redact_dicom_image_phi.py \
    --input /path/to/dicom/files \
    --output /path/to/output/directory \
    --headers /path/to/headers/directory \
    --top-ratio 0.08 \
    --bottom-ratio 0.0 \
    --left-ratio 0.0 \
    --right-ratio 0.0 \
    --samples 3 \
    --padding 4 \
    --merge-iou 0.3 \
    --expand 4
```

This configuration:

- Only scans the top 8% of the image height for PHI text
- Disables detection in bottom, left, and right regions (set to 0.0)
- Uses fewer samples (3) for faster processing
- Reduces padding and expansion for more precise detection
- Increases merge IoU threshold (0.3) to be more selective about combining boxes

### Command Line Arguments

| Argument           | Description                                          | Default  |
| ------------------ | ---------------------------------------------------- | -------- |
| `--input`          | Path to input DICOM file or directory                | Required |
| `--output`         | Output directory for redacted DICOM files            | Required |
| `--headers`        | Directory for storing bbox JSON metadata             | Required |
| `--recursive`      | Process subdirectories recursively                   | True     |
| `--overwrite`      | Overwrite existing output files                      | False    |
| `--samples`        | Number of frames to sample for multi-frame detection | 5        |
| `--top-ratio`      | Ratio of image height for top margin detection       | 0.25     |
| `--bottom-ratio`   | Ratio of image height for bottom margin detection    | 0.25     |
| `--left-ratio`     | Ratio of image width for left margin detection       | 0.25     |
| `--right-ratio`    | Ratio of image width for right margin detection      | 0.25     |
| `--padding`        | Padding width around detected text                   | 8        |
| `--merge-iou`      | IoU threshold for merging overlapping boxes          | 0.2      |
| `--expand`         | Margin expansion around merged boxes                 | 8        |
| `--patient-prefix` | Prefix for anonymized patient names                  | "anon"   |
| `--random-text`    | Add random text to redacted areas                    | False    |
| `--text-bold`      | Make replacement text bold                           | False    |
| `--pdf-report`     | Generate PDF comparison report                       | None     |

## How It Works

### 1. PHI Detection

- Uses Microsoft Presidio's image redaction engine
- Performs OCR on image regions to detect text overlays
- Identifies PHI entities like names, dates, and patient IDs
- Focuses detection on configurable margin regions (top, bottom, left, right)

### 2. Multi-frame Processing

- For multi-frame DICOM studies, samples a subset of frames for detection
- Applies detected redaction boxes to all frames in the study
- Groups files by transducer signature for consistent processing

### 3. Metadata Anonymization

- Replaces patient identifiers with hashed values
- Shifts dates by random offsets while maintaining consistency
- Removes or anonymizes physician names and accession numbers
- Preserves clinical information while removing PHI

### 4. Output Generation

- Creates anonymized DICOM files with redacted pixel data
- Generates JSON metadata files containing redaction information
- Optionally creates PDF comparison reports

## Output Structure

```
output_directory/
├── anonymized_dicom_files.dcm
└── subdirectories/
    └── more_files.dcm

headers_directory/
├── anonymized_dicom_files.dcm_bboxes.json
└── subdirectories/
    └── more_files.dcm_bboxes.json
```

### JSON Metadata Format

Each JSON file contains:

```json
{
  "source": "/path/to/original/file.dcm",
  "output": "/path/to/redacted/file.dcm",
  "num_frames": 1,
  "boxes": [
    {
      "left": 100,
      "top": 50,
      "width": 200,
      "height": 30
    }
  ],
  "leaf_union": true,
  "leaf_key": "subdirectory",
  "transducer_signature": "probe_type_info",
  "phi": {
    "PatientName": "Original Name",
    "PatientID": "Original ID",
    "StudyDate": "20230101"
  }
}
```

## Configuration

### ROI Detection Ratios

Adjust the margin detection ratios based on your DICOM image characteristics:

- **Ultrasound images**: Often have text overlays in top/bottom margins
- **X-ray images**: May have text in corners or along edges
- **CT/MRI**: Text overlays vary by manufacturer and modality

### OCR Configuration

The tool uses Tesseract OCR with optimized settings for medical text:

- Character whitelist: `ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789:/.-`
- Page segmentation mode 6 (uniform block of text)
- Optimized for medical text recognition

## Performance Considerations

- **Memory Usage**: Large multi-frame studies are processed efficiently with streaming
- **Processing Time**: Detection time scales with image size and number of frames
- **Storage**: Original files are not modified; all output is written to new locations

## Dependencies

Key dependencies include:

- `pydicom`: DICOM file handling
- `presidio-image-redactor`: PHI detection and redaction
- `opencv-python`: Image processing
- `pytesseract`: OCR functionality
- `matplotlib`: PDF report generation
- `spacy`: Natural language processing for PHI detection

## License

This project is licensed under the terms specified in the LICENSE file.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

For issues and questions, please create an issue in the repository or contact the maintainers.

## Disclaimer

This tool is designed to help with DICOM deidentification but should be used in accordance with applicable privacy regulations (HIPAA, GDPR, etc.). Always verify that the deidentification meets your specific compliance requirements before using in production environments.
