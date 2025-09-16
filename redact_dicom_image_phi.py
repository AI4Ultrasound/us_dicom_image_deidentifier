from __future__ import annotations

import os
import gc
import json
import random
import string
import argparse
import logging
import hashlib
import datetime
import statistics as stats
from pathlib import Path
from copy import deepcopy
from time import perf_counter
from typing import Tuple, List, Dict, Optional, Union, Sequence
from collections import defaultdict
from itertools import groupby

import numpy as np
import cv2
import pydicom
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian
from pydicom.multival import MultiValue
from pydicom.valuerep import PersonName
from pydicom.tag import Tag
from PIL import Image
from presidio_analyzer import PatternRecognizer, Pattern
from presidio_image_redactor.dicom_image_redactor_engine import DicomImageRedactorEngine
from presidio_image_redactor.image_analyzer_engine import ImageRecognizerResult
from google.cloud import storage
import tempfile
import shutil

def _get_analyzer_results_patched(
    self,
    image: Image.Image,
    instance: pydicom.dataset.FileDataset,
    use_metadata: bool,
    ocr_kwargs: Optional[dict],
    ad_hoc_recognizers: Optional[List[PatternRecognizer]],
    **text_analyzer_kwargs,
) -> List[ImageRecognizerResult]:
    """Analyze image with selected redaction approach.

    :param image: DICOM pixel data as PIL image.
    :param instance: DICOM instance (with metadata).
    :param use_metadata: Whether to redact text in the image that
    are present in the metadata.
    :param ocr_kwargs: Additional params for OCR methods.
    :param ad_hoc_recognizers: List of PatternRecognizer objects to use
    for ad-hoc recognizer.
    :param text_analyzer_kwargs: Additional values for the analyze method
    in AnalyzerEngine (e.g., allow_list). Can include 'dicom_tags' list to
    specify which DICOM tags to extract when use_metadata=True.

    :return: Analyzer results.
    """
    # Check the ad-hoc recognizers list
    self._check_ad_hoc_recognizer_list(ad_hoc_recognizers)

    # Extract and filter out DICOM-specific parameters from text_analyzer_kwargs
    specific_tags = None
    if "dicom_tags" in text_analyzer_kwargs:
        specific_tags = text_analyzer_kwargs["dicom_tags"]
        # Filter out dicom_tags from text_analyzer_kwargs before passing to AnalyzerEngine
        text_analyzer_kwargs = {k: v for k, v in text_analyzer_kwargs.items() if k != "dicom_tags"}

    # Create custom recognizer using DICOM metadata
    if use_metadata:
        original_metadata, is_name, is_patient = self._get_text_metadata(instance, specific_tags)
        phi_list = self._make_phi_list(original_metadata, is_name, is_patient)
        deny_list_recognizer = PatternRecognizer(
            supported_entity="PERSON", deny_list=phi_list
        )

        if ad_hoc_recognizers is None:
            ad_hoc_recognizers = [deny_list_recognizer]
        elif isinstance(ad_hoc_recognizers, list):
            ad_hoc_recognizers.append(deny_list_recognizer)

    # Detect PII
    if ad_hoc_recognizers is None:
        analyzer_results = self.image_analyzer_engine.analyze(
            image,
            ocr_kwargs=ocr_kwargs,
            **text_analyzer_kwargs,
        )
    else:
        analyzer_results = self.image_analyzer_engine.analyze(
            image,
            ocr_kwargs=ocr_kwargs,
            ad_hoc_recognizers=ad_hoc_recognizers,
            **text_analyzer_kwargs,
        )

    return analyzer_results

def _get_text_metadata_patched(
    instance: pydicom.dataset.FileDataset,
    specific_tags: Optional[List[str]] = None,
) -> Tuple[list, list, list]:
    """Retrieve text metadata from the DICOM image.

    :param instance: Loaded DICOM instance.
    :param specific_tags: Optional list of DICOM tag names to extract.
                            If None, extracts all tags (excluding pixel data).

    :return: List of the instance's element values (excluding pixel data),
    bool for if the element is specified as being a name,
    bool for if the element is specified as being related to the patient.
    """
    metadata_text = list()
    is_name = list()
    is_patient = list()

    # If specific tags are provided, only process those tags
    if specific_tags is not None:
        for tag_name in specific_tags:
            if hasattr(instance, tag_name):
                element = getattr(instance, tag_name)
                metadata_text.append(element)

                # Track whether this particular element is a name
                if "name" in tag_name.lower():
                    is_name.append(True)
                else:
                    is_name.append(False)

                # Track whether this particular element is directly tied to the patient
                if "patient" in tag_name.lower():
                    is_patient.append(True)
                else:
                    is_patient.append(False)
    else:
        # Process all elements (original behavior)
        for element in instance:
            # Save all metadata except the DICOM image itself
            if element.name != "Pixel Data":
                # Save the metadata
                metadata_text.append(element.value)

                # Track whether this particular element is a name
                if "name" in element.name.lower():
                    is_name.append(True)
                else:
                    is_name.append(False)

                # Track whether this particular element is directly tied to the patient
                if "patient" in element.name.lower():
                    is_patient.append(True)
                else:
                    is_patient.append(False)
            else:
                metadata_text.append("")
                is_name.append(False)
                is_patient.append(False)

    return metadata_text, is_name, is_patient

# --- Presidio DICOM engine monkey-patch: fix _make_phi_list ---
def _make_phi_list_patched(
    cls,
    original_metadata: List[Union[MultiValue, list, tuple]],
    is_name: List[bool],
    is_patient: List[bool],
) -> list:
    """
    Safe, non-mutating PHI list builder:
    - combine names + patient-related names
    - add generic PHI
    - flatten MultiValue/list/tuple
    - stringify, strip empties
    - de-duplicate (stable)
    """
    # Process names and patient-related names, then combine
    phi = []
    phi.extend(cls._process_names(original_metadata, is_name))
    phi.extend(cls._process_names(original_metadata, is_patient))
    phi = cls._add_known_generic_phi(phi)

    # Flatten without mutating during iteration
    flat = []
    for v in phi:
        if isinstance(v, (MultiValue, list, tuple)):
            for it in v:
                if it is None:
                    continue
                s = str(it).strip()
                if s:
                    flat.append(s)
        else:
            if v is None:
                continue
            s = str(v).strip()
            if s:
                flat.append(s)

    # Stable de-dup
    seen = set()
    out = []
    for s in flat:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out

def _process_names_patched(cls, text_metadata: list, is_name: list) -> list:
    """Process names to have multiple iterations in our PHI list.

    :param text_metadata: List of all the instance's element values
    (excluding pixel data).
    :param is_name: True if the element is specified as being a name.

    :return: Metadata text with additional name iterations appended.
    """
    phi_list = []

    for i in range(0, len(text_metadata)):
        if is_name[i] is True:
            metadata_item = text_metadata[i]

            # Handle MultiValue objects by processing each element individually
            if isinstance(metadata_item, (MultiValue, list, tuple)):
                for item in metadata_item:
                    if item is None:
                        continue
                    original_text = str(item).strip()
                    if original_text:
                        phi_list.extend(cls.augment_word(original_text))
            else:
                # Handle single values
                original_text = str(metadata_item).strip()
                if original_text:
                    phi_list.extend(cls.augment_word(original_text))

    return phi_list

# Apply the patch
DicomImageRedactorEngine._get_analyzer_results = _get_analyzer_results_patched
DicomImageRedactorEngine._get_text_metadata = staticmethod(_get_text_metadata_patched)
DicomImageRedactorEngine._make_phi_list = classmethod(_make_phi_list_patched)
DicomImageRedactorEngine._process_names = classmethod(_process_names_patched)
# --- end monkey-patch ---

# Set environment variables for deterministic behavior
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TESSERACT_NUM_THREADS", "1")

# Make absolutely sure pydicom doesn't emit WARNING logs
pdcm = logging.getLogger("pydicom")
pdcm.setLevel(logging.ERROR)
pdcm.propagate = False
pdcm.handlers[:] = [logging.NullHandler()]

PATIENT_ID_HASH_LENGTH = 10
INSTANCE_ID_HASH_LENGTH = 8
DEFAULT_CONTENT_DATE = '19000101'
DEFAULT_CONTENT_TIME = ''

# GCS helper functions
def list_gcs_files(bucket_name: str, prefix: str, valid_exts: tuple = (".dcm", ".DCM")) -> list:
    """List all DICOM files in a GCS bucket prefix."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    files = []
    blobs = bucket.list_blobs(prefix=prefix)
    for blob in blobs:
        if blob.name.endswith('/'):  # Skip directory markers
            continue

        # Check if file has valid extension
        if any(blob.name.lower().endswith(ext.lower()) for ext in valid_exts):
            files.append(blob)

    return files

def download_gcs_file_to_temp(blob) -> str:
    """Download a single GCS blob to a temporary file and return the path."""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.dcm')
    blob.download_to_filename(temp_file.name)
    return temp_file.name

def upload_file_to_gcs(local_file_path: str, bucket_name: str, blob_name: str) -> None:
    """Upload a single file to GCS."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_file_path)
    print(f"Uploaded: {local_file_path} -> gs://{bucket_name}/{blob_name}")

def upload_string_to_gcs(content: str, bucket_name: str, blob_name: str) -> None:
    """Upload string content to GCS."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(content)
    print(f"Uploaded string content -> gs://{bucket_name}/{blob_name}")

def is_gcs_path(path: str) -> bool:
    """Check if a path is a GCS path (gs://bucket/prefix)."""
    return path.startswith('gs://')

def parse_gcs_path(gcs_path: str) -> tuple[str, str]:
    """Parse a GCS path into bucket name and prefix."""
    if not gcs_path.startswith('gs://'):
        raise ValueError(f"Invalid GCS path: {gcs_path}")

    path_without_gs = gcs_path[5:]  # Remove 'gs://'
    parts = path_without_gs.split('/', 1)
    bucket_name = parts[0]
    prefix = parts[1] if len(parts) > 1 else ''

    return bucket_name, prefix

def force_explicit_vr_le(ds):
    ds.file_meta = ds.file_meta or Dataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    return ds

# ---- small helpers (lean, no external deps) ----
def _force_uncompressed_le(ds: FileDataset) -> FileDataset:
    from pydicom.dataset import FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian
    if not hasattr(ds, "file_meta") or ds.file_meta is None:
        ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    return ds

def _to_uint8(a: np.ndarray) -> np.ndarray:
    if a.dtype == np.uint8: return a
    a = a.astype(np.float32); mn, mx = np.nanmin(a), np.nanmax(a)
    if mx == mn: return np.zeros_like(a, dtype=np.uint8)
    a = (a - mn) / (mx - mn)
    return (a * 255.0 + 0.5).astype(np.uint8)

def _write_pixels_back(ds: FileDataset, arr: np.ndarray) -> FileDataset:
    if arr.ndim == 2:  n,h,w,c = 1,*arr.shape,1
    elif arr.ndim == 3:
        if arr.shape[-1] in (1,3): n,h,w,c = 1,arr.shape[0],arr.shape[1],arr.shape[2]
        else: n,h,w,c = arr.shape[0],arr.shape[1],arr.shape[2],1
    elif arr.ndim == 4: n,h,w,c = arr.shape
    else: raise ValueError(f"shape {arr.shape}")
    arr = _to_uint8(arr)
    ds.Rows, ds.Columns = h, w
    ds.NumberOfFrames = int(n)
    if c == 1:
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = getattr(ds,"PhotometricInterpretation","MONOCHROME2")
        ds.PlanarConfiguration = getattr(ds,"PlanarConfiguration",0)
    else:
        ds.SamplesPerPixel = 3
        ds.PhotometricInterpretation = "RGB"
        ds.PlanarConfiguration = 0
    ds.BitsAllocated = ds.BitsStored = 8; ds.HighBit = 7; ds.PixelRepresentation = 0
    ds.PixelData = arr.tobytes()
    return _force_uncompressed_le(ds)

def _clip_box(l,t,w,h,W,H):
    l=max(0,int(l)); t=max(0,int(t)); w=max(1,int(w)); h=max(1,int(h))
    r=min(W,l+w); b=min(H,t+h)
    if r<=l or b<=t: return None
    return l,t,r-l,b-t

def _iou(a,b):
    ax0, ay0 = a["left"], a["top"]; ax1, ay1 = ax0+a["width"], ay0+a["height"]
    bx0, by0 = b["left"], b["top"]; bx1, by1 = bx0+b["width"], by0+b["height"]
    ix0, iy0 = max(ax0,bx0), max(ay0,by0); ix1, iy1 = min(ax1,bx1), min(ay1,by1)
    iw, ih = max(0, ix1-ix0), max(0, iy1-iy0); inter = iw*ih
    if inter == 0: return 0.0
    ua = a["width"]*a["height"] + b["width"]*b["height"] - inter
    return inter/ua if ua>0 else 0.0

def _merge_boxes(boxes: List[Dict], iou_thresh=0.2) -> List[Dict]:
    boxes = boxes[:]; out=[]
    while boxes:
        base = boxes.pop()
        changed=True
        while changed:
            changed=False
            keep=[]
            for b in boxes:
                if _iou(base,b) >= iou_thresh:
                    x0=min(base["left"],b["left"]); y0=min(base["top"],b["top"])
                    x1=max(base["left"]+base["width"], b["left"]+b["width"])
                    y1=max(base["top"]+base["height"], b["top"]+b["height"])
                    base={"left":x0,"top":y0,"width":x1-x0,"height":y1-y0}
                    changed=True
                else: keep.append(b)
            boxes=keep
        out.append(base)
    return out

def _expand_boxes(boxes: List[Dict], margin: int, W: int, H: int) -> List[Dict]:
    out = []
    for b in boxes:
        if not isinstance(b, dict):
            continue

        # More asymmetric expansion
        left   = b["left"]  - int(0.2 * margin)
        top    = b["top"]   - int(0.2 * margin)
        right  = b["left"] + b["width"]  + int(2.0 * margin)
        bottom = b["top"]  + b["height"] + int(2.5 * margin)

        l = max(0, left)
        t = max(0, top)
        r = min(W, right)
        bot = min(H, bottom)

        w = r - l
        h = bot - t

        if w > 0 and h > 0:
            out.append({"left": l, "top": t, "width": w, "height": h})
    return out

def _apply_boxes_u8(img_u8: np.ndarray, boxes: List[Dict], fill_value: int) -> np.ndarray:
    out = img_u8.copy(); H,W = out.shape[:2]
    for b in boxes:
        c=_clip_box(b["left"],b["top"],b["width"],b["height"],W,H)
        if not c: continue
        l,t,w,h = c
        if out.ndim == 2: out[t:t+h, l:l+w] = fill_value
        else:             out[t:t+h, l:l+w, :] = fill_value
    return out

def _remove_padding_safe(boxes: List[Dict], padding_width: int) -> List[Dict]:
    try:
        from presidio_image_redactor.bbox import BboxProcessor
        return BboxProcessor.remove_bbox_padding(boxes, padding_width)
    except Exception:
        if not padding_width: return boxes
        out=[]
        for b in boxes:
            out.append({
                "left":   int(b["left"])   + padding_width,
                "top":    int(b["top"])    + padding_width,
                "width":  max(0, int(b["width"])  - 2*padding_width),
                "height": max(0, int(b["height"]) - 2*padding_width),
            })
        return out



# -------------------------
# Engine wiring (EDIT THIS)
# -------------------------

def get_engine():
    from presidio_image_redactor import DicomImageRedactorEngine, ImagePreprocessor
    from presidio_image_redactor.image_analyzer_engine import ImageAnalyzerEngine
    from presidio_analyzer import AnalyzerEngine

    # Analyzer: same defaults you used
    analyzer_engine = AnalyzerEngine(
        default_score_threshold=0.7,
        log_decision_process=False
    )

    # Image analyzer with basic preprocessor
    image_analyzer = ImageAnalyzerEngine(
        analyzer_engine=analyzer_engine,
        image_preprocessor=ImagePreprocessor(),
    )

    # Build the DICOM redaction engine
    engine = DicomImageRedactorEngine(image_analyzer_engine=image_analyzer)

    # Optional: if you want deterministic OCR & fewer garbage tokens,
    # pass Tesseract kwargs when you call redact_from_directory:
    #   ocr_kwargs={"config": "--oem 1 --psm 6 -c tessedit_do_invert=0 preserve_interword_spaces=1 -c user_defined_dpi=300"}
    # or keep it None to use defaults.

    return engine


def _merge_inline_neighbors(boxes, *, space_px=12, y_tol_frac=0.30, gap_per_h=0.8):
    """
    Greedy left→right merge of boxes on the same text line when the horizontal
    gap is small. Works without OCR text. Useful to glue DATE + TIME, etc.
    """
    if not boxes:
        return []

    # sort by row then x
    b = sorted(boxes, key=lambda d: (d["top"], d["left"]))
    out = []
    cur = dict(b[0])

    for w in b[1:]:
        # same "line" if y difference small vs height
        max_h = max(cur["height"], w["height"])
        y_tol = int(max_h * y_tol_frac)
        same_line = abs(w["top"] - cur["top"]) <= y_tol

        # horizontally close if gap ≤ adaptive threshold
        gap = w["left"] - (cur["left"] + cur["width"])
        gap_thresh = max(space_px, int(gap_per_h * max_h))
        close = gap <= gap_thresh

        if same_line and close:
            # union the two
            x1 = min(cur["left"], w["left"])
            y1 = min(cur["top"],  w["top"])
            x2 = max(cur["left"] + cur["width"],  w["left"] + w["width"])
            y2 = max(cur["top"]  + cur["height"], w["top"]  + w["height"])
            cur["left"], cur["top"] = x1, y1
            cur["width"], cur["height"] = x2 - x1, y2 - y1
        else:
            out.append(cur)
            cur = dict(w)

    out.append(cur)
    return out

# ---- main: detect in 4 margin ROIs, union, apply to all frames ----
def redact_multiframe_margin_union(
    ds: FileDataset,
    engine,                          # DicomImageRedactorEngine (tuple API)
    samples: int = 5,                # frames to probe (evenly spaced)
    top_ratio: float = 0.25,         # % height for top band
    bottom_ratio: float = 0.25,      # % height for bottom band
    left_ratio: float = 0.25,        # % width for left strip
    right_ratio: float = 0.25,       # % width for right strip
    padding_width: int = 8,
    merge_iou: float = 0.2,
    expand_margin: int = 6,
    fill: str = "contrast",
    ocr_kwargs=None,
    ad_hoc_recognizers=None,
    score_threshold: float = 0.0,
    **text_analyzer_kwargs,
) -> Tuple[FileDataset, List[Dict]]:
    """
    Detect boxes inside 4 margin ROIs (top/bottom/left/right) across a few frames,
    union+expand them, then apply to ALL frames. Returns (new_ds, boxes).
    """
    base = deepcopy(ds)
    try:
        px = base.pixel_array
    except Exception as e:
        print(f"[margin-union] decode failed: {e}")
        _force_uncompressed_le(base)
        return engine.redact_and_return_bbox(base)

    n = getattr(base, "NumberOfFrames", 1)
    H = int(getattr(base, "Rows", px.shape[-2 if px.ndim==4 else -2]))
    W = int(getattr(base, "Columns", px.shape[-1 if px.ndim==4 else -1]))
    photometric = getattr(base, "PhotometricInterpretation", "MONOCHROME2")
    fill_value = 255 if photometric == "MONOCHROME1" else 0

    if n == 1:
        tmp = deepcopy(base); _force_uncompressed_le(tmp)
        red_ds, boxes_raw = engine.redact_and_return_bbox(tmp, fill=fill, padding_width=padding_width, use_metadata=True, ocr_kwargs=ocr_kwargs, ad_hoc_recognizers=ad_hoc_recognizers, score_threshold=score_threshold,
                                                          **text_analyzer_kwargs)

        # Apply ROI filtering to single-frame results
        if isinstance(boxes_raw, list):
            def _filter_boxes_by_roi_single(boxes, top_ratio, bottom_ratio, left_ratio, right_ratio, W, H):
                """Filter boxes to only keep those entirely within areas where the corresponding ratio > 0"""
                filtered = []
                for box in boxes:
                    x, y, w, h = box["left"], box["top"], box["width"], box["height"]
                    x2, y2 = x + w, y + h

                    # Check if box is entirely within any allowed area
                    in_allowed_area = False

                    # Top area (if top_ratio > 0) - box must be entirely within top area
                    if top_ratio > 0:
                        top_h = int(H * top_ratio)
                        if y >= 0 and y2 <= top_h:  # Box entirely within top area
                            in_allowed_area = True

                    # Bottom area (if bottom_ratio > 0) - box must be entirely within bottom area
                    if bottom_ratio > 0:
                        bottom_h = int(H * bottom_ratio)
                        bottom_y = H - bottom_h
                        if y >= bottom_y and y2 <= H:  # Box entirely within bottom area
                            in_allowed_area = True

                    # Left area (if left_ratio > 0) - box must be entirely within left area
                    if left_ratio > 0:
                        left_w = int(W * left_ratio)
                        if x >= 0 and x2 <= left_w:  # Box entirely within left area
                            in_allowed_area = True

                    # Right area (if right_ratio > 0) - box must be entirely within right area
                    if right_ratio > 0:
                        right_w = int(W * right_ratio)
                        right_x = W - right_w
                        if x >= right_x and x2 <= W:  # Box entirely within right area
                            in_allowed_area = True

                    if in_allowed_area:
                        filtered.append(box)

                return filtered

            print(f"Single-frame ROI filtering: top={top_ratio:.2f}, bottom={bottom_ratio:.2f}, left={left_ratio:.2f}, right={right_ratio:.2f}")
            original_count = len(boxes_raw)
            boxes_raw = _filter_boxes_by_roi_single(boxes_raw, top_ratio, bottom_ratio, left_ratio, right_ratio, W, H)
            filtered_count = len(boxes_raw)
            if original_count != filtered_count:
                print(f"Single-frame ROI filtering: {original_count} -> {filtered_count} boxes (removed {original_count - filtered_count} boxes outside allowed areas)")

        return red_ds, boxes_raw

    # Evenly spaced sample indices
    k = min(max(1, samples), n)
    sample_idxs = np.linspace(0, n-1, k, dtype=int).tolist()

    # Define ROIs: (y0, y1, x0, x1)
    t_h = int(H * top_ratio)
    b_h = int(H * bottom_ratio)
    l_w = int(W * left_ratio)
    r_w = int(W * right_ratio)
    rois = []
    if t_h > 0: rois.append(("top",    0,       t_h,    0,      W))
    if b_h > 0: rois.append(("bottom", H-b_h,   H,      0,      W))
    if l_w > 0: rois.append(("left",   0,       H,      0,      l_w))
    if r_w > 0: rois.append(("right",  0,       H,      W-r_w,  W))

    print(f"Multi-frame ROI creation: top={top_ratio:.2f}->{t_h}, bottom={bottom_ratio:.2f}->{b_h}, left={left_ratio:.2f}->{l_w}, right={right_ratio:.2f}->{r_w}")
    print(f"Created {len(rois)} ROIs: {[r[0] for r in rois]}")

    all_boxes: List[Dict] = []

    # Probe each ROI on a few frames
    for idx in sample_idxs:
        full = _to_uint8(px[idx])
        # Convert RGB->mono for OCR robustness; keep 2D
        if full.ndim == 3 and full.shape[2] == 3:
            full_mono = (0.299*full[...,0] + 0.587*full[...,1] + 0.114*full[...,2]).astype(np.uint8)
        elif full.ndim == 3 and full.shape[2] == 1:
            full_mono = full[...,0]
        else:
            full_mono = full

        for name, y0, y1, x0, x1 in rois:
            crop = full_mono[y0:y1, x0:x1]
            if crop.size == 0:
                continue

            # Wrap ROI crop as 1‑frame DICOM
            temp = deepcopy(base); temp.NumberOfFrames = 1
            h, w = crop.shape
            temp.Rows, temp.Columns = h, w
            temp.SamplesPerPixel = 1
            temp.PhotometricInterpretation = "MONOCHROME2"  # working space
            temp.PlanarConfiguration = getattr(temp, "PlanarConfiguration", 0)
            temp.BitsAllocated = temp.BitsStored = 8; temp.HighBit = 7; temp.PixelRepresentation = 0
            temp.PixelData = crop.tobytes()
            _force_uncompressed_le(temp)

            # Ask Presidio for (red_1f, boxes) INSIDE this ROI
            _, boxes_raw = engine.redact_and_return_bbox(
                temp, fill=fill, padding_width=padding_width, use_metadata=True, ocr_kwargs=ocr_kwargs, ad_hoc_recognizers=ad_hoc_recognizers, score_threshold=score_threshold,
                **text_analyzer_kwargs
            )

            # Normalize boxes and map back to full-image coords
            if isinstance(boxes_raw, list):
                roi_boxes = 0
                for b in boxes_raw:
                    l = int(b["left"]) + x0
                    t = int(b["top"])  + y0
                    w = int(b["width"])
                    h = int(b["height"])
                    c = _clip_box(l, t, w, h, W, H)
                    if c:
                        l,t,w,h = c
                        all_boxes.append({"left": l, "top": t, "width": w, "height": h})
                        roi_boxes += 1
                print(f"  ROI {name} (frame {idx}): found {roi_boxes} boxes")

    # If none found, fallback to wiping the 4 margins as conservative rectangles
    if not all_boxes:
        print("No boxes found in ROIs, using fallback mechanism")
        all_boxes = []
        if t_h:
            all_boxes.append({"left":0,    "top":0,     "width":W,   "height":t_h})
            print(f"  Added fallback top box: height={t_h}")
        if b_h:
            all_boxes.append({"left":0,    "top":H-b_h, "width":W,   "height":b_h})
            print(f"  Added fallback bottom box: height={b_h}")
        if l_w:
            all_boxes.append({"left":0,    "top":0,     "width":l_w, "height":H})
            print(f"  Added fallback left box: width={l_w}")
        if r_w:
            all_boxes.append({"left":W-r_w,"top":0,     "width":r_w, "height":H})
            print(f"  Added fallback right box: width={r_w}")
        print(f"Fallback created {len(all_boxes)} boxes")

    # glue neighbors on same line (helps DATE + TIME become one box)
    #all_boxes = _merge_inline_neighbors(all_boxes, space_px=28, y_tol_frac=0.30, gap_per_h=0.9)

    # Filter boxes to only keep those in allowed ROI areas
    def _filter_boxes_by_roi(boxes, top_ratio, bottom_ratio, left_ratio, right_ratio, W, H):
        """Filter boxes to only keep those entirely within areas where the corresponding ratio > 0"""
        filtered = []
        for box in boxes:
            x, y, w, h = box["left"], box["top"], box["width"], box["height"]
            x2, y2 = x + w, y + h

            # Check if box is entirely within any allowed area
            in_allowed_area = False

            # Top area (if top_ratio > 0) - box must be entirely within top area
            if top_ratio > 0:
                top_h = int(H * top_ratio)
                if y >= 0 and y2 <= top_h:  # Box entirely within top area
                    in_allowed_area = True

            # Bottom area (if bottom_ratio > 0) - box must be entirely within bottom area
            if bottom_ratio > 0:
                bottom_h = int(H * bottom_ratio)
                bottom_y = H - bottom_h
                if y >= bottom_y and y2 <= H:  # Box entirely within bottom area
                    in_allowed_area = True

            # Left area (if left_ratio > 0) - box must be entirely within left area
            if left_ratio > 0:
                left_w = int(W * left_ratio)
                if x >= 0 and x2 <= left_w:  # Box entirely within left area
                    in_allowed_area = True

            # Right area (if right_ratio > 0) - box must be entirely within right area
            if right_ratio > 0:
                right_w = int(W * right_ratio)
                right_x = W - right_w
                if x >= right_x and x2 <= W:  # Box entirely within right area
                    in_allowed_area = True

            if in_allowed_area:
                filtered.append(box)

        return filtered

    # Filter boxes to only keep those in allowed ROI areas
    original_count = len(all_boxes)
    print(f"ROI areas: top={top_ratio:.2f}, bottom={bottom_ratio:.2f}, left={left_ratio:.2f}, right={right_ratio:.2f}")
    print(f"ROI dimensions: top_h={int(H * top_ratio)}, bottom_h={int(H * bottom_ratio)}, left_w={int(W * left_ratio)}, right_w={int(W * right_ratio)}")
    all_boxes = _filter_boxes_by_roi(all_boxes, top_ratio, bottom_ratio, left_ratio, right_ratio, W, H)
    filtered_count = len(all_boxes)
    if original_count != filtered_count:
        print(f"ROI filtering: {original_count} -> {filtered_count} boxes (removed {original_count - filtered_count} boxes outside allowed areas)")
    else:
        print(f"ROI filtering: {original_count} boxes (no filtering needed)")

    # Remove padding, merge overlaps, expand a bit, clip
    print(f"Box processing pipeline:")
    print(f"  After ROI filtering: {len(all_boxes)} boxes")

    boxes = _remove_padding_safe(all_boxes, padding_width)
    print(f"  After padding removal: {len(boxes)} boxes")

    boxes = _merge_boxes(boxes, iou_thresh=merge_iou)
    print(f"  After merging (iou_thresh={merge_iou}): {len(boxes)} boxes")

    boxes = _expand_boxes(boxes, expand_margin, W, H)
    print(f"  After expanding (margin={expand_margin}): {len(boxes)} boxes")

    # Apply union boxes to ALL frames
    out = []
    if px.ndim == 3:  # (N,H,W)
        for i in range(n):
            out.append(_apply_boxes_u8(_to_uint8(px[i]), boxes, fill_value))
        stacked = np.stack(out, axis=0)
    elif px.ndim == 4 and px.shape[-1] in (1,3):
        for i in range(n):
            out.append(_apply_boxes_u8(_to_uint8(px[i]), boxes, fill_value))
        stacked = np.stack(out, axis=0)
    else:
        # unexpected layout; return last temp’s redaction
        tmp = deepcopy(base); _force_uncompressed_le(tmp)
        red_1f, boxes_raw = engine.redact_and_return_bbox(tmp)
        return red_1f, boxes_raw if isinstance(boxes_raw, list) else []

    result = deepcopy(base)
    result = _write_pixels_back(result, stacked)
    return result, boxes

def generate_filename_from_dicom_dataset(ds: pydicom.Dataset, hash_patient_id: bool = True) -> tuple[str, str, str]:
    """
    Generate a filename from a DICOM header dictionary.
    Optionally, the name will be a hash of the PatientID and the SOP Instance UID.
    The name will consist of two parts:
    X_Y.dcm
    X is generated by hashing the original patient UID to a 10-digit number.
    Y is generated from the DICOM instance UID, but limited to 8 digits

    :param ds: DICOM dataset
    :param hash_patient_id: If True, hash the patient ID
    :returns: tuple (filename, patientId, instanceId)
    """
    patient_id = ds.PatientID
    instance_id = ds.SOPInstanceUID

    if patient_id is None or patient_id == "":
        logging.error("PatientID not found in DICOM header dict")
        return "", "", ""

    if instance_id is None or instance_id == "":
        logging.error("SOPInstanceUID not found in DICOM header dict")
        return "", "", ""

    if hash_patient_id:
        hash_object = hashlib.sha256()
        hash_object.update(str(patient_id).encode())
        patient_id = int(hash_object.hexdigest(), 16) % 10**10
    else:
        patient_id = patient_id

    hash_object_instance_id = hashlib.sha256()
    hash_object_instance_id.update(str(instance_id).encode())
    instance_id = int(hash_object_instance_id.hexdigest(), 16) % 10**8

    # Add trailing zeros
    patient_id = str(patient_id).zfill(PATIENT_ID_HASH_LENGTH)
    instance_id = str(instance_id).zfill(INSTANCE_ID_HASH_LENGTH)

    return f"{patient_id}_{instance_id}.dcm", patient_id, instance_id

def _shift_date(date_str: str, offset: int) -> str:
    """Shift a single date by the given offset."""
    try:
        date_obj = datetime.datetime.strptime(date_str, "%Y%m%d") + datetime.timedelta(days=offset)
        return date_obj.strftime("%Y%m%d")
    except Exception as e:
        logging.warning(f"Failed to parse date: {date_str}. Using original date. Error: {e}")
        return date_str

def _apply_date_shifting(ds: pydicom.Dataset, source_ds: pydicom.Dataset) -> None:
    """Apply consistent date shifting based on patient ID."""
    patient_id = source_ds.PatientID
    random.seed(patient_id)
    random_offset = random.randint(0, 30)

    # Get dates with defaults
    study_date = getattr(source_ds, 'StudyDate', DEFAULT_CONTENT_DATE)
    series_date = getattr(source_ds, 'SeriesDate', DEFAULT_CONTENT_DATE)
    content_date = getattr(source_ds, 'ContentDate', DEFAULT_CONTENT_DATE)

    # Shift dates
    ds.StudyDate = _shift_date(study_date, random_offset)
    ds.SeriesDate = _shift_date(series_date, random_offset)
    ds.ContentDate = _shift_date(content_date, random_offset)

    # Copy times
    ds.StudyTime = getattr(source_ds, 'StudyTime', DEFAULT_CONTENT_TIME)
    ds.SeriesTime = getattr(source_ds, 'SeriesTime', DEFAULT_CONTENT_TIME)
    ds.ContentTime = getattr(source_ds, 'ContentTime', DEFAULT_CONTENT_TIME)

def _set_conformance_attributes(ds: pydicom.Dataset, source_ds: pydicom.Dataset) -> None:
    """Set required DICOM conformance attributes."""
    # Conditional elements: provide empty defaults if unknown.
    if not hasattr(ds, 'Laterality'):
        ds.Laterality = ''
    if not hasattr(ds, 'InstanceNumber'):
        ds.InstanceNumber = 1
    if not hasattr(ds, 'PatientOrientation'):
        ds.PatientOrientation = []
    if not hasattr(ds, "ImageType"):
        ds.ImageType = r"ORIGINAL\PRIMARY\IMAGE"

    # Multi-frame specific attributes
    if hasattr(ds, 'NumberOfFrames') and int(ds.NumberOfFrames) > 1:
        ds.FrameTime = getattr(source_ds, 'FrameTime', 0.1)
        if hasattr(source_ds, 'FrameIncrementPointer'):
            ds.FrameIncrementPointer = source_ds.FrameIncrementPointer
        else:
            ds.FrameIncrementPointer = Tag(0x0018, 0x1063)

    # For color images, set PlanarConfiguration (Type 1C)
    if hasattr(ds, 'SamplesPerPixel') and ds.SamplesPerPixel == 3:
        ds.PlanarConfiguration = getattr(source_ds, 'PlanarConfiguration', 0)

def _clamp_and_clean_boxes(boxes, *, width: int, height: int):
    cleaned = []
    for b in (boxes or []):  # ← nil-safe
        try:
            x = int(round(b.get("left", 0))); y = int(round(b.get("top", 0)))
            w = int(round(b.get("width", 0))); h = int(round(b.get("height", 0)))
        except Exception:
            continue
        if w <= 1 or h <= 1: continue
        x = max(0, x); y = max(0, y)
        x2 = min(width, x + w); y2 = min(height, y + h)
        if x2 <= x or y2 <= y: continue
        out = {"left": x, "top": y, "width": x2 - x, "height": y2 - y}
        if "entity_type" in b and b["entity_type"]: out["entity_type"] = b["entity_type"]
        cleaned.append(out)
    return cleaned

def _to_norm_boxes(boxes, W, H):
    out = []
    for b in (boxes or []):  # ← nil-safe
        x = b["left"] / max(1, W); y = b["top"] / max(1, H)
        w = b["width"]/ max(1, W); h = b["height"]/max(1, H)
        if w > 0 and h > 0: out.append({"x": x, "y": y, "w": w, "h": h})
    return out

def _from_norm_boxes(nboxes, W, H):
    out = []
    for b in nboxes or []:
        l = int(round(b["x"] * W))
        t = int(round(b["y"] * H))
        w = int(round(b["w"] * W))
        h = int(round(b["h"] * H))
        if w > 0 and h > 0:
            out.append({"left": l, "top": t, "width": w, "height": h})
    return out

def _iou_norm(a, b):
    ax1, ay1, ax2, ay2 = a["x"], a["y"], a["x"]+a["w"], a["y"]+a["h"]
    bx1, by1, bx2, by2 = b["x"], b["y"], b["x"]+b["w"], b["y"]+b["h"]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2-ix1), max(0.0, iy2-iy1)
    inter = iw*ih
    if inter <= 0: return 0.0
    area_a = (ax2-ax1)*(ay2-ay1)
    area_b = (bx2-bx1)*(by2-by1)
    return inter / (area_a + area_b - inter + 1e-9)

def _merge_boxes_norm(nboxes, iou_thresh=0.2):
    """Greedy union in normalized coords: IoU≥thresh → union."""
    if not nboxes: return []
    nboxes = sorted(nboxes, key=lambda b: b["w"]*b["h"], reverse=True)
    out = []
    for b in nboxes:
        merged = False
        for o in out:
            if _iou_norm(b, o) >= iou_thresh:
                x1 = min(o["x"], b["x"]); y1 = min(o["y"], b["y"])
                x2 = max(o["x"]+o["w"], b["x"]+b["w"])
                y2 = max(o["y"]+o["h"], b["y"]+b["h"])
                o.update({"x": x1, "y": y1, "w": x2-x1, "h": y2-y1})
                merged = True
                break
        if not merged:
            out.append(dict(b))
    return out

def _apply_union_and_conform(ds_in: FileDataset, boxes_px: list[dict]) -> FileDataset:
    """
    - Decompress via pixel_array
    - Apply union boxes to every frame (mono or RGB)
    - Force Explicit VR Little Endian
    - Write pixels back with your conformance helper
    Returns a NEW dataset ready to save.
    """
    px = ds_in.pixel_array  # triggers decompression if needed
    mono1 = str(getattr(ds_in, "PhotometricInterpretation", "MONOCHROME2")).upper() == "MONOCHROME1"
    bits = int(getattr(ds_in, "BitsStored", getattr(ds_in, "BitsAllocated", 8)))
    vmax = (1 << bits) - 1
    fill_val = vmax if mono1 else 0

    def _apply_to_frame(fr):
        if fr.ndim == 2:  # mono
            for b in boxes_px:
                y, x = b["top"], b["left"]; h, w = b["height"], b["width"]
                fr[y:y+h, x:x+w] = fill_val
        else:             # color (H,W,3)
            ink = np.array([fill_val, fill_val, fill_val], dtype=fr.dtype)
            for b in boxes_px:
                y, x = b["top"], b["left"]; h, w = b["height"], b["width"]
                fr[y:y+h, x:x+w] = ink
        return fr

    # Apply to all frames (support mono stack or color stack)
    if px.ndim == 2:
        px = _apply_to_frame(px)
    elif px.ndim == 3 and px.shape[-1] in (1, 3):      # (H,W,1) or (H,W,3)
        px = _apply_to_frame(px)
    elif px.ndim == 3:                                 # (N,H,W) mono stack
        for i in range(px.shape[0]): px[i] = _apply_to_frame(px[i])
    elif px.ndim == 4 and px.shape[-1] in (1, 3):      # (N,H,W,C)
        for i in range(px.shape[0]): px[i] = _apply_to_frame(px[i])
    else:
        raise ValueError(f"Unexpected pixel array shape: {px.shape}")

    # Conform tags & transfer syntax exactly as before
    out = deepcopy(ds_in)
    out = force_explicit_vr_le(out)    # ensure uncompressed Explicit VR Little Endian
    out = _write_pixels_back(out, px)  # updates Rows/Cols/Samples/Bits/Frames/PixelData/etc.
    return out

def _transducer_signature(ds) -> str:
    """
    Returns a stable string key for grouping clips by transducer.
    Uses (0018,5010) TransducerData if present (multi-valued), otherwise
    falls back to related US probe fields. Normalizes whitespace/case.
    """
    def norm(s: str) -> str:
        return " ".join(str(s).strip().split()).lower()

    parts: list[str] = []

    # Primary: TransducerData (0018,5010)
    td = getattr(ds, "TransducerData", None)
    if td is not None:
        if isinstance(td, MultiValue) or isinstance(td, (list, tuple)):
            parts.extend([norm(x) for x in td if str(x).strip()])
        else:
            # Sometimes delivered as a single pipe or backslash-separated string
            raw = str(td)
            if "\\" in raw:
                parts.extend([norm(x) for x in raw.split("\\") if x.strip()])
            elif "|" in raw:
                parts.extend([norm(x) for x in raw.split("|") if x.strip()])
            else:
                parts.append(norm(raw))

    # Nice-to-have fallbacks to make the signature more robust if TransducerData is absent/weak
    for tag_name in ("TransducerType", "ProbeType", "Manufacturer", "ManufacturerModelName"):
        val = getattr(ds, tag_name, None)
        if val:
            parts.append(norm(val))

    # As a last resort, include geometry (helps separate radically different overlays)
    try:
        parts.append(f"{int(ds.Rows)}x{int(ds.Columns)}")
    except Exception:
        pass

    sig = " | ".join([p for p in parts if p])
    return sig or "unspecified"


# --- Module-level helper for safe DICOM attribute extraction (JSON-safe) ---
# --- Module-level helper for safe DICOM attribute extraction (JSON-safe) ---
def safe_get_attr(ds, attr_name, default=""):
    """Safely get a DICOM attribute and convert it to a JSON-serializable value.
    Special-cases PN (PersonName) so it does not get split into characters.
    """
    try:
        value = getattr(ds, attr_name, default)
        if value is None:
            return default
        # Treat DICOM PersonName like a scalar string
        try:
            from pydicom.valuerep import PersonName, PersonNameBase
        except Exception:
            PersonName = None
            PersonNameBase = None
        # If it's already a plain scalar (str/bytes) or a PN, return a single string
        if isinstance(value, (str, bytes)) or (PersonName is not None and isinstance(value, (PersonName, PersonNameBase))):
            return str(value)
        # If it's a MultiValue/list/tuple, stringify each element
        if isinstance(value, (MultiValue, list, tuple)):
            out = []
            for it in value:
                if it is None:
                    continue
                # Keep PN elements as single strings too
                if isinstance(it, (str, bytes)) or (PersonName is not None and isinstance(it, (PersonName, PersonNameBase))):
                    s = str(it).strip()
                else:
                    s = str(it).strip()
                if s:
                    out.append(s)
            # Return list if multiple, or single string if only one
            if len(out) == 0:
                return default
            if len(out) == 1:
                return out[0]
            return out
        # Fallback: scalar stringify
        return str(value)
    except Exception:
        return default

# Robust extractor for Other Patient IDs (0010,1000) and the sequence (0010,1002)
def get_other_patient_ids(ds) -> list | str:
    """Return Other Patient IDs as a list when multiple exist, else a single string, or "" if none.
    Handles both (0010,1000) OtherPatientIDs and (0010,1002) OtherPatientIDsSequence.
    """
    out: list[str] = []
    try:
        # Prefer reading by tag to avoid keyword aliasing differences
        elem = ds.get(Tag(0x0010, 0x1000))  # OtherPatientIDs (LO, may be MultiValue)
        if elem is not None:
            val = elem.value
            if isinstance(val, (MultiValue, list, tuple)):
                for it in val:
                    s = str(it).strip()
                    if s:
                        out.append(s)
            else:
                s = str(val).strip()
                if s:
                    # Some datasets embed backslash-separated IDs in a single LO
                    parts = [p.strip() for p in s.split("\\") if p.strip()]
                    out.extend(parts if len(parts) > 1 else [s])
    except Exception:
        pass

    # Also consider the sequence variant
    try:
        seq = getattr(ds, "OtherPatientIDsSequence", None)
        if seq:
            for item in seq:
                pid = str(getattr(item, "PatientID", "")).strip()
                issuer = str(getattr(item, "IssuerOfPatientID", "")).strip()
                if pid or issuer:
                    out.append("|".join([p for p in (pid, issuer) if p]))
    except Exception:
        pass

    if not out:
        return ""
    if len(out) == 1:
        return out[0]
    return out

def redact_from_gcs(
    input_bucket: str,
    input_prefix: str,
    output_bucket: str,
    output_prefix: str,
    headers_bucket: str,
    headers_prefix: str,
    engine,
    valid_exts: Tuple[str, ...] = (".dcm", ".DCM"),
    overwrite: bool = True,
    # margin-union tuning:
    samples: int = 5,
    top_ratio: float = 0.25, bottom_ratio: float = 0.25,
    left_ratio: float = 0.25, right_ratio: float = 0.25,
    padding_width: int = 8,
    merge_iou: float = 0.2,
    expand_margin: int = 6,
    # engine / analyzer:
    ocr_kwargs=None,
    ad_hoc_recognizers=None,
    fill: str = "contrast",
    patient_name_prefix: str = "anon",
    score_threshold: float = 0.0,
    redact_metadata: bool = False,
    **text_analyzer_kwargs,
) -> List[Dict[str, str]]:
    """Process DICOM files directly from GCS without downloading all files locally."""

    # List all DICOM files in the input bucket
    input_files = list_gcs_files(input_bucket, input_prefix, valid_exts)
    print(f"Found {len(input_files)} DICOM files in gs://{input_bucket}/{input_prefix}")

    dicom_pairs: List[Dict[str, str]] = []
    processed, skipped, errored = 0, 0, 0
    temp_files = []  # Track temporary files for cleanup

    try:
        for blob in input_files:
            temp_file_path = None
            try:
                # Download single file to temporary location
                temp_file_path = download_gcs_file_to_temp(blob)
                temp_files.append(temp_file_path)

                # Determine output paths
                relative_path = blob.name[len(input_prefix):].lstrip('/')
                output_blob_name = f"{output_prefix.rstrip('/')}/{relative_path}"
                headers_blob_name = f"{headers_prefix.rstrip('/')}/{relative_path}_bboxes.json"

                # Check if output already exists (if not overwriting)
                if not overwrite:
                    client = storage.Client()
                    output_bucket_obj = client.bucket(output_bucket)
                    if output_bucket_obj.blob(output_blob_name).exists():
                        skipped += 1
                        print(f"[SKIP] {blob.name} -> output already exists")
                        continue

                # Process the single file
                print(f"Processing: {blob.name}")
                result = redact_single_dicom_file(
                    temp_file_path,
                    engine,
                    samples=samples,
                    top_ratio=top_ratio, bottom_ratio=bottom_ratio,
                    left_ratio=left_ratio, right_ratio=right_ratio,
                    padding_width=padding_width, expand_margin=expand_margin,
                    ocr_kwargs=ocr_kwargs, ad_hoc_recognizers=ad_hoc_recognizers,
                    fill=fill, patient_name_prefix=patient_name_prefix,
                    score_threshold=score_threshold, redact_metadata=redact_metadata,
                    **text_analyzer_kwargs
                )

                if result:
                    # Upload processed DICOM file
                    upload_file_to_gcs(result['output_path'], output_bucket, output_blob_name)

                    # Upload JSON metadata if headers bucket is specified
                    if headers_bucket and result.get('json_path'):
                        upload_file_to_gcs(result['json_path'], headers_bucket, headers_blob_name)

                    # Create entry for dicom_pairs
                    entry = {
                        "rel_input": relative_path,
                        "source": f"gs://{input_bucket}/{blob.name}",
                        "output": f"gs://{output_bucket}/{output_blob_name}",
                        "num_frames": result.get('num_frames', 1),
                        "boxes": result.get('boxes', []),
                    }
                    if headers_bucket:
                        entry["boxes_json"] = f"gs://{headers_bucket}/{headers_blob_name}"
                    dicom_pairs.append(entry)

                    processed += 1
                    print(f"[SUCCESS] {blob.name} -> {output_blob_name}")
                else:
                    errored += 1
                    print(f"[ERROR] Failed to process {blob.name}")

            except Exception as e:
                errored += 1
                print(f"[ERROR] {blob.name}: {e}")
            finally:
                # Clean up temporary file immediately
                if temp_file_path and Path(temp_file_path).exists():
                    Path(temp_file_path).unlink()
                    if temp_file_path in temp_files:
                        temp_files.remove(temp_file_path)

    finally:
        # Clean up any remaining temporary files
        for temp_file in temp_files:
            if Path(temp_file).exists():
                Path(temp_file).unlink()

    print(f"\nGCS Processing Summary:")
    print(f"Processed: {processed}, Skipped: {skipped}, Errored: {errored}")
    return dicom_pairs

def redact_single_dicom_file(
    input_file: str,
    engine,
    samples: int = 5,
    top_ratio: float = 0.25, bottom_ratio: float = 0.25,
    left_ratio: float = 0.25, right_ratio: float = 0.25,
    padding_width: int = 8,
    merge_iou: float = 0.2,
    expand_margin: int = 6,
    ocr_kwargs=None,
    ad_hoc_recognizers=None,
    fill: str = "contrast",
    patient_name_prefix: str = "anon",
    score_threshold: float = 0.0,
    redact_metadata: bool = False,
    **text_analyzer_kwargs,
) -> Optional[Dict]:
    """Process a single DICOM file and return results."""

    try:
        # Read DICOM file
        ds: FileDataset = pydicom.dcmread(input_file, stop_before_pixels=False, force=True)

        # Generate filename
        if redact_metadata:
            filename, patient_uid, _ = generate_filename_from_dicom_dataset(ds, True)
            new_patient_name = f"{patient_name_prefix}_{patient_uid}"
        else:
            filename = Path(input_file).name
            patient_uid = ""
            new_patient_name = ""

        # Create temporary output files
        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.dcm')
        temp_json = tempfile.NamedTemporaryFile(delete=False, suffix='.json')

        # Process the DICOM file
        n = int(getattr(ds, "NumberOfFrames", 1))
        if n > 1:
            print(f"    Using ROI processing path ({n} frames)")
            _, boxes = redact_multiframe_margin_union(
                ds, engine,
                samples=samples,
                top_ratio=top_ratio, bottom_ratio=bottom_ratio,
                left_ratio=left_ratio, right_ratio=right_ratio,
                padding_width=padding_width,
                merge_iou=merge_iou,
                expand_margin=expand_margin,
                fill=fill,
                ocr_kwargs=ocr_kwargs,
                ad_hoc_recognizers=ad_hoc_recognizers,
                score_threshold=score_threshold,
                **text_analyzer_kwargs,
            )
        else:
            print(f"    Using fallback path (single frame)")
            _, boxes = engine.redact_and_return_bbox(
                deepcopy(ds), fill=fill, padding_width=padding_width,
                use_metadata=True, ocr_kwargs=ocr_kwargs,
                ad_hoc_recognizers=ad_hoc_recognizers,
                score_threshold=score_threshold,
                **text_analyzer_kwargs
            )

        boxes = boxes or []
        H, W = int(ds.Rows), int(ds.Columns)
        clean_boxes = _clamp_and_clean_boxes(boxes, width=W, height=H)

        # Apply metadata redaction if requested
        if redact_metadata:
            # Apply metadata redaction logic here
            ds.PatientName = new_patient_name if patient_uid else ""
            ds.PatientID = patient_uid or ""
            ds.PatientBirthDate = ""
            ds.ReferringPhysicianName = ""
            ds.AccessionNumber = ""
            # Remove OtherPatientIDs
            from pydicom.tag import Tag
            _tag_opids = Tag(0x0010, 0x1000)
            _tag_opids_seq = Tag(0x0010, 0x1002)
            if _tag_opids in ds:
                del ds[_tag_opids]
            if _tag_opids_seq in ds:
                del ds[_tag_opids_seq]
            _apply_date_shifting(ds, ds)

        _set_conformance_attributes(ds, ds)

        # Save processed DICOM
        pydicom.dcmwrite(temp_output.name, ds, write_like_original=False, little_endian=True, implicit_vr=False)

        # Save JSON metadata
        json_data = {
            "source": input_file,
            "output": temp_output.name,
            "num_frames": n,
            "boxes": clean_boxes,
            "phi": {} if not redact_metadata else {
                "PatientName": getattr(ds, "PatientName", ""),
                "PatientID": getattr(ds, "PatientID", ""),
            }
        }

        with open(temp_json.name, "w") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False, default=str)

        return {
            "output_path": temp_output.name,
            "json_path": temp_json.name,
            "num_frames": n,
            "boxes": clean_boxes
        }

    except Exception as e:
        print(f"Error processing {input_file}: {e}")
        return None

def redact_from_directory(
    input_dicom_path: str,
    output_dir: str,
    headers_dir: str,
    engine,
    recursive: bool = True,
    valid_exts: Tuple[str, ...] = (".dcm", ".DCM"),
    overwrite: bool = True,
    # margin-union tuning:
    samples: int = 5,
    top_ratio: float = 0.25, bottom_ratio: float = 0.25,
    left_ratio: float = 0.25, right_ratio: float = 0.25,
    padding_width: int = 8,
    merge_iou: float = 0.2,
    expand_margin: int = 6,
    # engine / analyzer:
    ocr_kwargs=None,
    ad_hoc_recognizers=None,
    fill: str = "contrast",
    patient_name_prefix: str = "anon",
    score_threshold: float = 0.0,
    redact_metadata: bool = False,
    **text_analyzer_kwargs,
) -> List[Dict[str, str]]:

    in_path = Path(input_dicom_path)
    out_root = Path(output_dir); out_root.mkdir(parents=True, exist_ok=True)
    headers_root = Path(headers_dir); headers_root.mkdir(parents=True, exist_ok=True)
    dicom_pairs: List[Dict[str, str]] = []

    # iterator base
    if in_path.is_dir():
        base = in_path
        it_sorted = sorted(
            (p for p in (in_path.rglob("*") if recursive else in_path.iterdir()) if p.is_file()),
            key=lambda p: (str(p.parent.relative_to(base)) or ".", str(p))
        )
    else:
        base = in_path.parent
        it_sorted = [in_path]

    processed, skipped, errored = 0, 0, 0
    iter_times_ms: List[float] = []
    apply_processed = 0
    apply_errored = 0
    apply_times_ms: List[float] = []
    apply_times_by_group: dict[str, list[float]] = {}

    # ---- STREAM: process one leaf at a time ----
    for leaf_key, leaf_iter in groupby(it_sorted, key=lambda p: str(p.parent.relative_to(base)) or "."):
        # Per-leaf containers (scoped so they can be GC'd after each leaf)
        files_by_group = defaultdict(list)            # (leaf, probe) -> [Path]
        norm_boxes_by_group = defaultdict(list)       # (leaf, probe) -> normalized boxes
        members_by_group = defaultdict(list)          # (leaf, probe) -> member dicts

        # 0) Collect this leaf’s files and subgroup by transducer
        for src in leaf_iter:
            if valid_exts is not None and src.suffix.lower() not in valid_exts:
                skipped += 1
                print(f"[SKIP] {src} -> {src.suffix.lower()} not in {valid_exts}")
                continue
            try:
                ds_hdr = pydicom.dcmread(str(src), stop_before_pixels=True, force=True)
            except Exception:
                ds_hdr = pydicom.dcmread(str(src), stop_before_pixels=False, force=True)
            probe_key = _transducer_signature(ds_hdr)
            files_by_group[(leaf_key, probe_key)].append(src)
            # free header ASAP
            del ds_hdr

        # 1) PASS 1: DETECT for each (leaf, transducer) group IN THIS LEAF
        for (lk, probe_key), src_list in files_by_group.items():
            print(f"  Processing group ({lk}, {probe_key}) with {len(src_list)} files: {[f.name for f in src_list]}")
            for src in src_list:
                t0 = perf_counter()
                rel_parent = src.parent.relative_to(base)
                dst_dir = out_root / rel_parent; dst_dir.mkdir(parents=True, exist_ok=True)
                headers_subdir = headers_root / rel_parent; headers_subdir.mkdir(parents=True, exist_ok=True)

                try:
                    ds: FileDataset = pydicom.dcmread(str(src), stop_before_pixels=False, force=True)
                    if redact_metadata:
                        filename, patient_uid, _ = generate_filename_from_dicom_dataset(ds, True)
                        new_patient_name = f"{patient_name_prefix}_{patient_uid}"
                    else:
                        # Use original filename when metadata redaction is disabled
                        filename = src.name
                        patient_uid = ""
                        new_patient_name = ""
                    dst_dcm = dst_dir / filename
                    dst_json = headers_subdir / (filename + "_bboxes.json")

                    if dst_dcm.exists() and not overwrite:
                        skipped += 1
                        rel_in = str((rel_parent / src.name).as_posix())
                        entry = {
                            "rel_input": rel_in, "source": str(src), "output": str(dst_dcm),
                            "num_frames": int(getattr(ds, "NumberOfFrames", 1)),
                            "boxes": [],
                        }
                        if headers_dir: entry["boxes_json"] = str(dst_json)
                        dicom_pairs.append(entry)
                        members_by_group[(lk, probe_key)].append({
                            "entry": entry, "dst_dcm": str(dst_dcm), "src_path": str(src),
                            "patient_uid": patient_uid, "new_patient_name": new_patient_name,
                            "H": int(getattr(ds, "Rows", 0)), "W": int(getattr(ds, "Columns", 0)),
                        })
                        del ds
                        continue

                    # Detect (no pixel writes)
                    n = int(getattr(ds, "NumberOfFrames", 1))
                    if n > 1:
                        print(f"    Using ROI processing path for {src.name} ({n} frames)")
                        _, boxes = redact_multiframe_margin_union(
                            ds, engine,
                            samples=samples,
                            top_ratio=top_ratio, bottom_ratio=bottom_ratio,
                            left_ratio=left_ratio, right_ratio=right_ratio,
                            padding_width=padding_width,
                            merge_iou=merge_iou,
                            expand_margin=expand_margin,
                            fill=fill,
                            ocr_kwargs=ocr_kwargs,
                            ad_hoc_recognizers=ad_hoc_recognizers,
                            score_threshold=score_threshold,
                            **text_analyzer_kwargs,
                        )
                        # ROI processing path - boxes are already processed and filtered
                        boxes = boxes or []
                        H, W = int(ds.Rows), int(ds.Columns)
                        print(f"  ROI processing result for {src.name}: {len(boxes)} boxes")
                        clean_boxes = _clamp_and_clean_boxes(boxes, width=W, height=H)
                        print(f"  After cleaning: {len(clean_boxes)} boxes")
                        norm_boxes = _to_norm_boxes(clean_boxes, W, H)
                        print(f"  After normalization: {len(norm_boxes)} boxes")
                        norm_boxes_by_group[(lk, probe_key)].extend(norm_boxes)
                    else:
                        print(f"    Using fallback path for {src.name} (single frame)")
                        _, boxes = engine.redact_and_return_bbox(
                            deepcopy(ds), fill=fill, padding_width=padding_width,
                            use_metadata=True, ocr_kwargs=ocr_kwargs,
                            ad_hoc_recognizers=ad_hoc_recognizers,
                            score_threshold=score_threshold,
                            **text_analyzer_kwargs
                        )
                        boxes = boxes or []
                        H, W = int(ds.Rows), int(ds.Columns)
                        print(f"  Fallback detection for {src.name}: found {len(boxes)} boxes")

                        # Apply ROI filtering to fallback boxes
                        def _filter_boxes_by_roi_fallback(boxes, top_ratio, bottom_ratio, left_ratio, right_ratio, W, H):
                            """Filter boxes to only keep those entirely within areas where the corresponding ratio > 0"""
                            filtered = []
                            for box in boxes:
                                x, y, w, h = box["left"], box["top"], box["width"], box["height"]
                                x2, y2 = x + w, y + h

                                # Check if box is entirely within any allowed area
                                in_allowed_area = False

                                # Top area (if top_ratio > 0) - box must be entirely within top area
                                if top_ratio > 0:
                                    top_h = int(H * top_ratio)
                                    if y >= 0 and y2 <= top_h:  # Box entirely within top area
                                        in_allowed_area = True

                                # Bottom area (if bottom_ratio > 0) - box must be entirely within bottom area
                                if bottom_ratio > 0:
                                    bottom_h = int(H * bottom_ratio)
                                    bottom_y = H - bottom_h
                                    if y >= bottom_y and y2 <= H:  # Box entirely within bottom area
                                        in_allowed_area = True

                                # Left area (if left_ratio > 0) - box must be entirely within left area
                                if left_ratio > 0:
                                    left_w = int(W * left_ratio)
                                    if x >= 0 and x2 <= left_w:  # Box entirely within left area
                                        in_allowed_area = True

                                # Right area (if right_ratio > 0) - box must be entirely within right area
                                if right_ratio > 0:
                                    right_w = int(W * right_ratio)
                                    right_x = W - right_w
                                    if x >= right_x and x2 <= W:  # Box entirely within right area
                                        in_allowed_area = True

                                if in_allowed_area:
                                    filtered.append(box)

                            return filtered

                        # Apply ROI filtering to fallback boxes
                        original_count = len(boxes)
                        boxes = _filter_boxes_by_roi_fallback(boxes, top_ratio, bottom_ratio, left_ratio, right_ratio, W, H)
                        filtered_count = len(boxes)
                        if original_count != filtered_count:
                            print(f"  Fallback ROI filtering: {original_count} -> {filtered_count} boxes (removed {original_count - filtered_count} boxes outside allowed areas)")

                        clean_boxes = _clamp_and_clean_boxes(boxes, width=W, height=H)
                        print(f"  After cleaning: {len(clean_boxes)} boxes")
                        norm_boxes = _to_norm_boxes(clean_boxes, W, H)
                        print(f"  After normalization: {len(norm_boxes)} boxes")
                        norm_boxes_by_group[(lk, probe_key)].extend(norm_boxes)

                    # Member to apply later
                    rel_in = str((rel_parent / src.name).as_posix())
                    entry = {
                        "rel_input": rel_in, "source": str(src), "output": str(dst_dcm),
                        "num_frames": n, "boxes": [],
                    }
                    if headers_dir: entry["boxes_json"] = str(dst_json)
                    dicom_pairs.append(entry)
                    members_by_group[(lk, probe_key)].append({
                        "entry": entry, "dst_dcm": str(dst_dcm), "src_path": str(src),
                        "patient_uid": patient_uid, "new_patient_name": new_patient_name,
                        "H": H, "W": W,
                    })

                    dt_ms = (perf_counter() - t0) * 1e3
                    iter_times_ms.append(dt_ms); processed += 1
                    print(f"[DETECT {lk} | {probe_key}] {src.name}  ({len(clean_boxes)} boxes) in {dt_ms:.2f}ms")

                except Exception as e:
                    errored += 1
                    dt_ms = (perf_counter() - t0) * 1e3
                    print(f"[ERR detect {lk} | {probe_key}] {src} after {dt_ms:.1f} ms: {e}")
                finally:
                    # free large objects quickly
                    try:
                        if 'ds' in locals() and ds is not None:
                            del ds
                    except: pass
                    gc.collect()

        # 2) PASS 2: UNION + APPLY for each (leaf, transducer) IN THIS LEAF
        for (lk, probe_key), nboxes in norm_boxes_by_group.items():
            if not nboxes:
                continue
            print(f"  Group ({lk}, {probe_key}): {len(nboxes)} normalized boxes before merging")
            nboxes_union = _merge_boxes_norm(nboxes, iou_thresh=merge_iou)
            print(f"  Group ({lk}, {probe_key}): {len(nboxes_union)} normalized boxes after merging")

            for mem in members_by_group.get((lk, probe_key), []):
                out_path = mem["dst_dcm"]; src_path = mem["src_path"]
                t0_apply = perf_counter()
                try:
                    ds_src = pydicom.dcmread(src_path, force=True)
                    boxes_union_px = _from_norm_boxes(nboxes_union, mem["W"], mem["H"])
                    print(f"  Applying to {Path(src_path).name}: {len(nboxes_union)} normalized boxes -> {len(boxes_union_px)} pixel boxes")
                    if not boxes_union_px:
                        red_ds = force_explicit_vr_le(ds_src)
                    else:
                        red_ds = _apply_union_and_conform(ds_src, boxes_union_px)

                    # Apply metadata redaction only if requested
                    if redact_metadata:
                        # Capture original PHI data before replacing (JSON-safe)
                        original_phi_data = {
                            "PatientName": safe_get_attr(ds_src, "PatientName"),
                            "PatientID": safe_get_attr(ds_src, "PatientID"),
                            "PatientBirthDate": safe_get_attr(ds_src, "PatientBirthDate"),
                            "OtherPatientIDs": get_other_patient_ids(ds_src),
                            "ReferringPhysicianName": safe_get_attr(ds_src, "ReferringPhysicianName"),
                            "AccessionNumber": safe_get_attr(ds_src, "AccessionNumber"),
                            "StudyDate": safe_get_attr(ds_src, "StudyDate"),
                            "SeriesDate": safe_get_attr(ds_src, "SeriesDate"),
                            "ContentDate": safe_get_attr(ds_src, "ContentDate"),
                            "StudyTime": safe_get_attr(ds_src, "StudyTime"),
                            "SeriesTime": safe_get_attr(ds_src, "SeriesTime"),
                            "ContentTime": safe_get_attr(ds_src, "ContentTime"),
                        }

                        # Replace PHI data with generated data
                        red_ds.PatientName = f"{patient_name_prefix}_{mem['patient_uid']}" if mem["patient_uid"] else ""
                        red_ds.PatientID = mem["patient_uid"] or ""
                        red_ds.PatientBirthDate = ""
                        red_ds.ReferringPhysicianName = ""
                        red_ds.AccessionNumber = ""
                        # Remove OtherPatientIDs (0010,1000) and OtherPatientIDsSequence (0010,1002) robustly
                        from pydicom.tag import Tag
                        _tag_opids = Tag(0x0010, 0x1000)
                        _tag_opids_seq = Tag(0x0010, 0x1002)
                        if _tag_opids in red_ds:
                            del red_ds[_tag_opids]
                        if _tag_opids_seq in red_ds:
                            del red_ds[_tag_opids_seq]
                        # Also delete by attribute name if present (covers keyword aliasing)
                        for _attr in ("OtherPatientIDs", "RETIRED_OtherPatientIDs", "OtherPatientIDsSequence"):
                            if hasattr(red_ds, _attr):
                                try:
                                    delattr(red_ds, _attr)
                                except Exception:
                                    pass
                        _apply_date_shifting(red_ds, ds_src)
                    else:
                        # When metadata redaction is disabled, keep original PHI data
                        original_phi_data = {}
                    _set_conformance_attributes(red_ds, ds_src)

                    pydicom.dcmwrite(out_path, red_ds, write_like_original=False, little_endian=True, implicit_vr=False)

                    mem["entry"]["boxes"] = boxes_union_px
                    if headers_dir and "boxes_json" in mem["entry"]:
                        try:
                            with open(mem["entry"]["boxes_json"], "w") as f:
                                json.dump({
                                    "source": mem["src_path"], "output": out_path,
                                    "num_frames": int(getattr(red_ds, "NumberOfFrames", 1)),
                                    "boxes": boxes_union_px,
                                    "leaf_union": True,
                                    "leaf_key": lk,
                                    "transducer_signature": probe_key,
                                    "phi": original_phi_data if redact_metadata else {}
                                }, f, indent=2, ensure_ascii=False, default=str)
                        except Exception as e:
                            print(f"Warning: Failed to write JSON for {mem['src_path']}: {e}")
                            # Write a minimal JSON with error info
                            try:
                                with open(mem["entry"]["boxes_json"], "w") as f:
                                    json.dump({
                                        "source": mem["src_path"], "output": out_path,
                                        "num_frames": int(getattr(red_ds, "NumberOfFrames", 1)),
                                        "boxes": boxes_union_px,
                                        "leaf_union": True,
                                        "leaf_key": lk,
                                        "transducer_signature": probe_key,
                                        "phi": {},
                                        "error": f"Failed to serialize PHI data: {str(e)}"
                                    }, f, indent=2, ensure_ascii=False, default=str)
                            except Exception:
                                pass  # Give up if even the minimal JSON fails

                    dt_ms = (perf_counter() - t0_apply) * 1e3
                    apply_times_ms.append(dt_ms)
                    label = f"{lk} | {probe_key}"
                    apply_times_by_group.setdefault(label, []).append(dt_ms)
                    apply_processed += 1
                    print(f"[APPLY union {lk} | {probe_key}] -> {Path(out_path).name}  ({len(boxes_union_px)} boxes) in {dt_ms:.2f}ms")

                except Exception as e:
                    dt_ms = (perf_counter() - t0_apply) * 1e3
                    apply_times_ms.append(dt_ms)
                    label = f"{lk} | {probe_key}"
                    apply_times_by_group.setdefault(label, []).append(dt_ms)
                    apply_errored += 1
                    print(f"[ERR apply union {lk} | {probe_key}] {out_path} after {dt_ms:.1f} ms: {e}")
                finally:
                    try:
                        if 'ds_src' in locals():
                            del ds_src
                        if 'red_ds' in locals():
                            del red_ds
                    except:
                        pass
                    gc.collect()

        # 3) CLEAR this leaf’s working sets to free memory
        files_by_group.clear()
        norm_boxes_by_group.clear()
        members_by_group.clear()
        gc.collect()

    # ---------- Summaries ----------
    if iter_times_ms:
        total_ms = sum(iter_times_ms); mean_ms = total_ms/len(iter_times_ms)
        median_ms = stats.median(iter_times_ms)
        p95_ms = sorted(iter_times_ms)[max(0, int(0.95*len(iter_times_ms))-1)]
        p99_ms = sorted(iter_times_ms)[max(0, int(0.99*len(iter_times_ms))-1)]
        print("\n=== Detection summary ===")
        print(f"Processed={processed}, Skipped={skipped}, Errored={errored}")
        print(f"Total: {total_ms/1000:.2f} s, Mean: {mean_ms:.2f} ms, Median: {median_ms:.2f} ms")
        print(f"P95: {p95_ms:.2f} ms, P99: {p99_ms:.2f} ms, Min: {min(iter_times_ms):.2f} ms, Max: {max(iter_times_ms):.2f} ms")
    else:
        print(f"\nNo files processed. Skipped: {skipped}, Errors: {errored}")

    if apply_times_ms:
        total_ms2 = sum(apply_times_ms)
        mean_ms2 = total_ms2 / len(apply_times_ms)
        median_ms2 = stats.median(apply_times_ms)
        p95_ms2 = sorted(apply_times_ms)[max(0, int(0.95 * len(apply_times_ms)) - 1)]
        p99_ms2 = sorted(apply_times_ms)[max(0, int(0.99 * len(apply_times_ms)) - 1)]
        print("\n=== Apply (leaf+transducer union) summary ===")
        print(f"Applied: {apply_processed}, Errored: {apply_errored}")
        print(f"Total: {total_ms2/1000:.2f} s, Mean: {mean_ms2:.2f} ms, Median: {median_ms2:.2f} ms")
        print(f"P95: {p95_ms2:.2f} ms, P99: {p99_ms2:.2f} ms, Min: {min(apply_times_ms):.2f} ms, Max: {max(apply_times_ms):.2f} ms")
        for label, times in apply_times_by_group.items():
            t = sum(times); m = t/len(times)
            print(f"  - {label}: {len(times)} files, total {t/1000:.2f}s, mean {m:.2f} ms")
    else:
        print("\n=== Apply (leaf+transducer union) summary ===\nNo apply operations were performed.")

    if iter_times_ms or apply_times_ms:
        total_all = sum(iter_times_ms) + sum(apply_times_ms)
        print(f"\n=== Overall ===\nTotal wall-clock (sum of phases): {total_all/1000:.2f} s")

    print(f"\nDone. output_dir={out_root}\n")
    return dicom_pairs


BBox = Dict[str, int]


def _max_val(ds: FileDataset) -> int:
    # Conservative max based on BitsStored (not Allocated). Default to 8-bit if absent.
    bits_stored = getattr(ds, "BitsStored", getattr(ds, "BitsAllocated", 8))
    bits_stored = int(bits_stored) if bits_stored else 8
    return (1 << bits_stored) - 1


def _mono_is_inverted(ds: FileDataset) -> bool:
    # MONOCHROME1: low pixel values = bright; MONOCHROME2: low pixel values = dark
    return getattr(ds, "PhotometricInterpretation", "MONOCHROME2").upper() == "MONOCHROME1"


def _pick_ink_value(ds: FileDataset, prefer_dark_on_light: bool = True, strong: bool = True) -> int:
    """
    Choose an 'ink' intensity. If strong=True, use pure black/white for max contrast
    (respecting MONOCHROME1 inversion). Otherwise, use softer 20%/80%.
    """
    vmax = _max_val(ds)
    if strong:
        # Pure extremes. In MONOCHROME2 (typical), dark text = 0, light text = vmax.
        dark = 0 if not _mono_is_inverted(ds) else vmax
        light = vmax if not _mono_is_inverted(ds) else 0
    else:
        # Softer (legacy)
        dark = int(0.20 * vmax) if not _mono_is_inverted(ds) else int(0.80 * vmax)
        light = int(0.80 * vmax) if not _mono_is_inverted(ds) else int(0.20 * vmax)
    return dark if prefer_dark_on_light else light


def _rand_text(n: int) -> str:
    # simple alnum placeholder; tweak as needed
    alphabet = string.ascii_uppercase + string.digits
    return "".join(random.choice(alphabet) for _ in range(max(3, n)))


def _estimate_chars_for_box(w: int, h: int) -> int:
    # Heuristic: characters per line ~ box_width / (0.7 * box_height) - more conservative for smaller text
    if h <= 0:
        return 4
    return max(3, int(w / max(10, 0.7 * h)))


def _make_text_mask(w: int, h: int, text: str, thickness: int) -> np.ndarray:
    """
    Build a uint8 mask (H,W) with white pixels where text will be drawn.
    Uses crisp rendering without anti-aliasing for sharp text.
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Calculate scale to fit text in box with some padding
    target_h = max(6, int(0.5 * h))  # Use less of the box height for smaller text
    scale = max(0.3, target_h / 30.0)  # Minimum scale for readability

    # Get text size
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)

    # Center the text
    x = max(0, (w - tw) // 2)
    y = max(th + baseline, (h + th) // 2)

    # Render text with crisp edges (no anti-aliasing)
    cv2.putText(mask, text, (x, y), font, scale, (255,), thickness, lineType=cv2.LINE_8)
    return mask


def _ensure_stack(px: np.ndarray):
    """
    Return (stack, had_batch, is_color)
      mono: (N,H,W)
      color: (N,H,W,3)
    Accepts: (H,W), (N,H,W), (H,W,1), (H,W,3), (N,H,W,3)
    """
    if px.ndim == 2:                 # (H,W) mono
        return px[None, ...], False, False
    if px.ndim == 3:
        if px.shape[-1] == 1:        # (H,W,1) mono
            return px[..., 0][None, ...], False, False
        if px.shape[-1] == 3:        # (H,W,3) color
            return px[None, ...], False, True
        # else assume (N,H,W) mono
        return px, True, False
    if px.ndim == 4 and px.shape[-1] == 3:  # (N,H,W,3) color
        return px, True, True
    raise ValueError(f"Unexpected pixel array shape: {px.shape}")


def _composite_text_into_roi_mono(roi: np.ndarray, ink_val: int, text: str, *, thickness: int, bold: bool) -> None:
    mask = _make_text_mask(roi.shape[1], roi.shape[0], text, thickness=thickness)
    if bold:
        # Use a smaller, more precise dilation kernel for less blur
        k = max(1, min(3, thickness // 4))  # Smaller kernel, fewer iterations
        kernel = np.ones((k, k), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
    roi[mask > 0] = ink_val


def _composite_text_into_roi_color(roi: np.ndarray, ink_val: int, text: str, *, thickness: int, bold: bool) -> None:
    mask = _make_text_mask(roi.shape[1], roi.shape[0], text, thickness=thickness)
    if bold:
        # Use a smaller, more precise dilation kernel for less blur
        k = max(1, min(3, thickness // 4))  # Smaller kernel, fewer iterations
        kernel = np.ones((k, k), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
    # Same intensity on all channels (true black/white look)
    if roi.dtype.kind in ("u", "i"):
        ink_vec = np.array([ink_val, ink_val, ink_val], dtype=roi.dtype)
    else:
        ink_vec = np.array([float(ink_val)] * 3, dtype=roi.dtype)
    roi[mask > 0] = ink_vec


def add_random_text_to_boxes(
    ds: FileDataset,
    boxes: Sequence[Dict[str, int]],
    seed: int | None = None,
    per_frame_boxes: bool = False,
    prefer_dark_ink: bool = True,
    *,
    strong_contrast: bool = True,
    bold: bool = True,
    thickness_denominator: int = 12,
) -> FileDataset:
    if seed is not None:
        random.seed(seed)

    px = ds.pixel_array
    stack, had_batch, is_color = _ensure_stack(px)
    n = stack.shape[0]

    ink_val = _pick_ink_value(ds, prefer_dark_on_light=prefer_dark_ink, strong=strong_contrast)

    # Pre-generate random text per box
    text_for_box: list[str] = []
    for b in boxes or []:
        chars = _estimate_chars_for_box(b.get("width", 0), b.get("height", 0))
        text_for_box.append(_rand_text(chars))

    for f in range(n):
        frame = stack[f]
        for b, text in zip(boxes or [], text_for_box):
            x = max(0, int(b.get("left", 0)))
            y = max(0, int(b.get("top", 0)))
            w = max(0, int(b.get("width", 0)))
            h = max(0, int(b.get("height", 0)))
            if w <= 2 or h <= 2:
                continue
            x2 = min(frame.shape[1], x + w)
            y2 = min(frame.shape[0], y + h)
            if x >= x2 or y >= y2:
                continue

            # Calculate thickness based on box size, ensuring crisp text
            thickness = max(1, min(3, int(h / max(6, thickness_denominator))))
            if is_color:
                _composite_text_into_roi_color(frame[y:y2, x:x2, :], ink_val, text, thickness=thickness, bold=bold)
            else:
                _composite_text_into_roi_mono(frame[y:y2, x:x2], ink_val, text, thickness=thickness, bold=bold)

    out = stack if had_batch else stack[0]
    ds.PixelData = out.tobytes()
    return ds


def _get_frame(ds, idx=0):
    """Extract a single frame from DICOM dataset."""
    px = ds.pixel_array  # requires pylibjpeg/gdcm for compressed
    # (H,W), (H,W,3), (N,H,W), (N,H,W,3/1)
    if px.ndim == 2:
        img = px
    elif px.ndim == 3:
        img = px if px.shape[-1] in (1,3) else px[min(idx, px.shape[0]-1)]
        if img.ndim == 3 and img.shape[-1] == 1:
            img = img[...,0]
    elif px.ndim == 4:
        img = px[min(idx, px.shape[0]-1)]
        if img.ndim == 3 and img.shape[-1] == 1:
            img = img[...,0]
    else:
        raise ValueError(f"Unsupported pixel_array shape {px.shape}")

    # Invert MONOCHROME1 for display
    if getattr(ds, "PhotometricInterpretation", "").upper() == "MONOCHROME1" and img.ndim == 2:
        if np.issubdtype(img.dtype, np.integer):
            info = np.iinfo(img.dtype)
            img = info.max - img
        else:
            img = img.max() - img
    return img


def _normalize_image(img):
    """Normalize image to 0-1 range for display."""
    img = img.astype("float32", copy=False)
    if img.ndim == 2:
        lo, hi = np.percentile(img, [1, 99])
        if hi <= lo:
            lo, hi = float(img.min()), float(img.max())
        img = (img - lo) / max(1e-6, (hi - lo))
    else:
        lo = img.min(axis=(0,1), keepdims=True)
        hi = img.max(axis=(0,1), keepdims=True)
        img = (img - lo) / np.maximum(hi - lo, 1e-6)
    return np.clip(img, 0, 1)


def create_comparison_pdf(dicom_pairs: List[Dict[str, Union[str, List[BBox]]]],
                         output_pdf_path: str,
                         has_random_text: bool = False) -> None:
    """
    Create a PDF report comparing original and redacted DICOM images.

    Args:
        dicom_pairs: List of dictionaries containing source, output paths and boxes
        output_pdf_path: Path where to save the PDF report
        has_random_text: If True, includes a fourth column showing final image with random text
    """
    # Process all images
    pairs_to_process = dicom_pairs

    print(f"Starting PDF generation for {len(pairs_to_process)} images...")

    with PdfPages(output_pdf_path) as pdf:
        for i, item in enumerate(pairs_to_process):
            print(f"Processing image {i+1}/{len(pairs_to_process)}: {Path(item['source']).name}")
            try:
                # Load original image
                original_ds = pydicom.dcmread(item["source"])
                orig_img = _get_frame(original_ds, 0)

                # Load redacted image (this might be the final image with text if random_text was used)
                redacted_ds = pydicom.dcmread(item["output"])
                red_img = _get_frame(redacted_ds, 0)

                # If we have random text, we need to create the redacted image without text for comparison
                if has_random_text:
                    # Create a clean redacted image by applying the boxes to the original
                    red_img_clean = orig_img.copy()
                    boxes = item.get("boxes", [])
                    if isinstance(boxes, list):
                        for box in boxes:
                            if isinstance(box, dict):
                                x = max(0, int(box.get("left", 0)))
                                y = max(0, int(box.get("top", 0)))
                                w = max(0, int(box.get("width", 0)))
                                h = max(0, int(box.get("height", 0)))
                                x2 = min(red_img_clean.shape[1], x + w)
                                y2 = min(red_img_clean.shape[0], y + h)
                                if x < x2 and y < y2:
                                    # Fill with black (assuming MONOCHROME2)
                                    red_img_clean[y:y2, x:x2] = 0
                    red_img_for_diff = red_img_clean
                else:
                    red_img_for_diff = red_img

                # Normalize images for display
                orig_norm = _normalize_image(orig_img)
                red_norm = _normalize_image(red_img)
                red_norm_clean = _normalize_image(red_img_for_diff)

                # Calculate difference between original and clean redacted (not final with text)
                if orig_norm.shape == red_norm_clean.shape:
                    diff_img = np.abs(orig_norm - red_norm_clean)
                else:
                    # Handle size mismatches
                    h = min(orig_norm.shape[0], red_norm_clean.shape[0])
                    w = min(orig_norm.shape[1], red_norm_clean.shape[1])
                    orig_crop = orig_norm[:h, :w] if orig_norm.ndim == 2 else orig_norm[:h, :w, ...]
                    red_crop = red_norm_clean[:h, :w] if red_norm_clean.ndim == 2 else red_norm_clean[:h, :w, ...]
                    diff_img = np.abs(orig_crop - red_crop)

                # Create figure (3 or 4 columns depending on random text)
                ncols = 4 if has_random_text else 3
                figsize = (20, 5) if has_random_text else (15, 5)
                fig, axes = plt.subplots(1, ncols, figsize=figsize)
                fig.suptitle(f'Image {i+1}: {Path(item["source"]).name}', fontsize=14, fontweight='bold')

                # Original image
                if orig_norm.ndim == 2:
                    axes[0].imshow(orig_norm, cmap='gray')
                else:
                    axes[0].imshow(orig_norm)
                axes[0].set_title('Original', fontweight='bold')
                axes[0].axis('off')

                # Redacted image (show clean version if we have random text)
                redacted_display = red_norm_clean if has_random_text else red_norm
                if redacted_display.ndim == 2:
                    axes[1].imshow(redacted_display, cmap='gray')
                else:
                    axes[1].imshow(redacted_display)
                axes[1].set_title('Redacted', fontweight='bold')
                axes[1].axis('off')

                # Difference image
                axes[2].imshow(diff_img, cmap='hot')
                axes[2].set_title('Difference', fontweight='bold')
                axes[2].axis('off')

                # Final image with random text (if applicable)
                if has_random_text:
                    try:
                        # Load the final image with random text
                        final_ds = pydicom.dcmread(item["output"])
                        final_img = _get_frame(final_ds, 0)
                        final_norm = _normalize_image(final_img)

                        if final_norm.ndim == 2:
                            axes[3].imshow(final_norm, cmap='gray')
                        else:
                            axes[3].imshow(final_norm)
                        axes[3].set_title('Final (with text)', fontweight='bold')
                        axes[3].axis('off')

                        # Clean up
                        del final_ds, final_img, final_norm
                    except Exception as e:
                        axes[3].text(0.5, 0.5, f'Error loading final image\n{str(e)}',
                                   ha='center', va='center', fontsize=10,
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.8))
                        axes[3].set_title('Final (with text)', fontweight='bold')
                        axes[3].axis('off')

                # Add redaction boxes overlay to original image (draw last so they appear on top)
                if "boxes" in item and item["boxes"]:
                    boxes = item["boxes"]
                    if isinstance(boxes, list):
                        for box in boxes:
                            if isinstance(box, dict):
                                rect = patches.Rectangle(
                                    (int(box.get("left", 0)), int(box.get("top", 0))),
                                    int(box.get("width", 0)),
                                    int(box.get("height", 0)),
                                    linewidth=1,
                                    edgecolor='red',
                                    facecolor='none',
                                    alpha=0.7
                                )
                                axes[0].add_patch(rect)

                # Add metadata text
                metadata_text = f"""
Source: {Path(item["source"]).name}
Output: {Path(item["output"]).name}
Frames: {item.get("num_frames", 1)}
Boxes: {len(item.get("boxes", []))}
                """.strip()

                fig.text(0.02, 0.02, metadata_text, fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))

                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight', dpi=150)
                plt.close(fig)

                # Clean up memory
                del original_ds, redacted_ds, orig_img, red_img
                print(f"  ✓ Successfully processed image {i+1}")

            except Exception as e:
                print(f"  ✗ Error processing image {i+1}: {e}")
                # Create error page
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.text(0.5, 0.5, f'Error processing image {i+1}\n{str(e)}',
                       ha='center', va='center', fontsize=12,
                       bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral", alpha=0.8))
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

        # Add summary page
        print("Adding summary page...")
        fig, ax = plt.subplots(figsize=(10, 8))
        # Create summary text based on whether random text was added
        if has_random_text:
            summary_text = f"""
DICOM Redaction Report Summary

Total Images Processed: {len(pairs_to_process)}
Report Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

This report shows:
- Original DICOM images (left) with red boxes showing detected areas
- Redacted DICOM images (center) showing the redacted result
- Difference maps (3rd column, red=changes)
- Final images (right) with random text added to redacted areas

Red boxes on the original image show the areas that were detected for redaction.
            """.strip()
        else:
            summary_text = f"""
DICOM Redaction Report Summary

Total Images Processed: {len(pairs_to_process)}
Report Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

This report shows:
- Original DICOM images (left) with red boxes showing detected areas
- Redacted DICOM images (center) showing the final result
- Difference maps (right, red=changes)

Red boxes on the original image show the areas that were detected for redaction.
            """.strip()

        ax.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=14,
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('DICOM Redaction Report', fontsize=18, fontweight='bold', pad=20)

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    print(f"✓ PDF generation completed successfully!")
    print(f"✓ Comparison PDF saved to: {output_pdf_path}")


# -------------------------
# CLI
# -------------------------

def main():
    parser = argparse.ArgumentParser(description="Leaf+transducer union redaction for DICOM clips.")
    parser.add_argument("--input", required=True, help="Path to input DICOM or directory (supports gs://bucket/prefix)")
    parser.add_argument("--output", required=True, help="Output directory (supports gs://bucket/prefix)")
    parser.add_argument("--headers", required=True, help="Directory for bbox JSONs (supports gs://bucket/prefix)")
    parser.add_argument("--recursive", action="store_true", default=True, help="Recurse into subdirectories")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Overwrite outputs if exist")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--top-ratio", type=float, default=0.25)
    parser.add_argument("--bottom-ratio", type=float, default=0.25)
    parser.add_argument("--left-ratio", type=float, default=0.25)
    parser.add_argument("--right-ratio", type=float, default=0.25)
    parser.add_argument("--padding", type=int, default=8)
    parser.add_argument("--merge-iou", type=float, default=0.2)
    parser.add_argument("--expand", type=int, default=8)
    parser.add_argument("--patient-prefix", default="anon")
    parser.add_argument("--redact-metadata", action="store_true", default=False, help="Redact metadata and rename output files")
    parser.add_argument("--random-text", action="store_true", default=False, help="Insert random text into boxes")
    parser.add_argument("--text-bold", action="store_true", default=False, help="Make text bold (may cause slight blur)")
    parser.add_argument("--pdf-report", type=str, help="Generate PDF comparison report (specify output path)")
    args = parser.parse_args()

    # Handle GCS vs local paths
    engine = get_engine()

    # Check if we're doing full GCS processing (input, output, and headers all from/to GCS)
    if (is_gcs_path(args.input) and is_gcs_path(args.output) and is_gcs_path(args.headers)):
        print("Using GCS streaming mode (processing files one at a time)")
        input_bucket, input_prefix = parse_gcs_path(args.input)
        output_bucket, output_prefix = parse_gcs_path(args.output)
        headers_bucket, headers_prefix = parse_gcs_path(args.headers)

        dicom_pairs = redact_from_gcs(
            input_bucket=input_bucket,
            input_prefix=input_prefix,
            output_bucket=output_bucket,
            output_prefix=output_prefix,
            headers_bucket=headers_bucket,
            headers_prefix=headers_prefix,
            engine=engine,
            overwrite=args.overwrite,
            samples=args.samples,
            top_ratio=args.top_ratio, bottom_ratio=args.bottom_ratio,
            left_ratio=args.left_ratio, right_ratio=args.right_ratio,
            padding_width=args.padding, expand_margin=args.expand,
            redact_metadata=args.redact_metadata,
            ocr_kwargs={
                "config": "--oem 1 --psm 6 -c tessedit_do_invert=0 preserve_interword_spaces=1 tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789:/.-"
            },
            dicom_tags=["PatientName", "PatientID", "PatientBirthDate", "OtherPatientIDs", "ReferringPhysicianName", "InstitutionName", "AccessionNumber", "StudyDate", "SeriesDate", "ContentDate", "StudyTime", "SeriesTime", "ContentTime"],
            entities=["PERSON", "DATE_TIME"],
            language="en",
        )
    else:
        # Fall back to the original approach for mixed local/GCS or all local
        print("Using traditional mode (downloading/uploading directories)")
        temp_dirs = []
        # Initialize GCS variables in case we're not using GCS
        output_bucket = output_prefix = headers_bucket = headers_prefix = None

        try:
            # Process input path
            if is_gcs_path(args.input):
                print(f"Downloading input from GCS: {args.input}")
                bucket_name, prefix = parse_gcs_path(args.input)
                temp_input_dir = tempfile.mkdtemp(prefix="gcs_input_")
                temp_dirs.append(temp_input_dir)
                # Use the old download function for full directory download
                from google.cloud import storage
                client = storage.Client()
                bucket = client.bucket(bucket_name)
                Path(temp_input_dir).mkdir(parents=True, exist_ok=True)
                blobs = bucket.list_blobs(prefix=prefix)
                for blob in blobs:
                    if blob.name.endswith('/'):
                        continue
                    relative_path = blob.name[len(prefix):].lstrip('/')
                    local_file_path = Path(temp_input_dir) / relative_path
                    local_file_path.parent.mkdir(parents=True, exist_ok=True)
                    blob.download_to_filename(str(local_file_path))
                input_path = temp_input_dir
            else:
                input_path = args.input

            # Process output path
            if is_gcs_path(args.output):
                temp_output_dir = tempfile.mkdtemp(prefix="gcs_output_")
                temp_dirs.append(temp_output_dir)
                output_path = temp_output_dir
                output_is_gcs = True
                output_bucket, output_prefix = parse_gcs_path(args.output)
            else:
                output_path = args.output
                output_is_gcs = False

            # Process headers path
            if is_gcs_path(args.headers):
                temp_headers_dir = tempfile.mkdtemp(prefix="gcs_headers_")
                temp_dirs.append(temp_headers_dir)
                headers_path = temp_headers_dir
                headers_is_gcs = True
                headers_bucket, headers_prefix = parse_gcs_path(args.headers)
            else:
                headers_path = args.headers
                headers_is_gcs = False

            dicom_pairs = redact_from_directory(
                input_dicom_path=input_path,
                output_dir=output_path,
                headers_dir=headers_path,
                engine=engine,
                recursive=args.recursive,
                overwrite=args.overwrite,
                samples=args.samples,
                top_ratio=args.top_ratio, bottom_ratio=args.bottom_ratio,
                left_ratio=args.left_ratio, right_ratio=args.right_ratio,
                padding_width=args.padding, expand_margin=args.expand,
                redact_metadata=args.redact_metadata,
                ocr_kwargs={
                    "config": "--oem 1 --psm 6 -c tessedit_do_invert=0 preserve_interword_spaces=1 tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789:/.-"
                },
                dicom_tags=["PatientName", "PatientID", "PatientBirthDate", "OtherPatientIDs", "ReferringPhysicianName", "InstitutionName", "AccessionNumber", "StudyDate", "SeriesDate", "ContentDate", "StudyTime", "SeriesTime", "ContentTime"],
                entities=["PERSON", "DATE_TIME"],
                language="en",
            )

            # Upload results to GCS if needed
            if output_is_gcs:
                print(f"Uploading output to GCS: {args.output}")
                from google.cloud import storage
                client = storage.Client()
                bucket = client.bucket(output_bucket)
                local_path = Path(output_path)
                for file_path in local_path.rglob('*'):
                    if file_path.is_file():
                        relative_path = file_path.relative_to(local_path)
                        blob_name = f"{output_prefix.rstrip('/')}/{relative_path.as_posix()}"
                        blob = bucket.blob(blob_name)
                        blob.upload_from_filename(str(file_path))

            if headers_is_gcs:
                print(f"Uploading headers to GCS: {args.headers}")
                from google.cloud import storage
                client = storage.Client()
                bucket = client.bucket(headers_bucket)
                local_path = Path(headers_path)
                for file_path in local_path.rglob('*'):
                    if file_path.is_file():
                        relative_path = file_path.relative_to(local_path)
                        blob_name = f"{headers_prefix.rstrip('/')}/{relative_path.as_posix()}"
                        blob = bucket.blob(blob_name)
                        blob.upload_from_filename(str(file_path))

        finally:
            # Clean up temporary directories
            for temp_dir in temp_dirs:
                if Path(temp_dir).exists():
                    shutil.rmtree(temp_dir)
                    print(f"Cleaned up temporary directory: {temp_dir}")

    print(f"Redacted {len(dicom_pairs)} items.")

    # Add random text first (if requested)
    if args.random_text:
        print("Adding random text to redacted areas...")
        for item in dicom_pairs:
            # Load the redacted image (without text)
            ds = pydicom.dcmread(item["output"])
            boxes = item.get("boxes", [])
            if isinstance(boxes, list):
                ds = add_random_text_to_boxes(
                    ds,
                    boxes,
                    seed=1234,
                    per_frame_boxes=False,
                    prefer_dark_ink=False,
                    strong_contrast=False,  # Use softer contrast for less bright text
                    bold=args.text_bold,   # Use command line parameter
                    thickness_denominator=8  # Smaller denominator for thicker text
                )
            # Save the final image with text
            ds.save_as(item["output"])
        print("Random text added to all redacted areas.")

    # Generate PDF comparison report if requested (after random text if applicable)
    if args.pdf_report:
        pdf_path = Path(args.pdf_report)

        # Validate PDF path
        if pdf_path.is_dir():
            print(f"Error: PDF report path is a directory: {pdf_path}")
            print("Please specify a file path (e.g., /path/to/report.pdf)")
            return

        # Ensure parent directory exists
        pdf_path.parent.mkdir(parents=True, exist_ok=True)

        # Ensure .pdf extension
        if pdf_path.suffix.lower() != '.pdf':
            pdf_path = pdf_path.with_suffix('.pdf')

        print(f"Generating PDF comparison report with {len(dicom_pairs)} images...")
        print(f"PDF will be saved to: {pdf_path}")
        create_comparison_pdf(dicom_pairs, str(pdf_path), has_random_text=args.random_text)

if __name__ == "__main__":
    # Helpful defaults for OCR determinism (optional)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("TESSERACT_NUM_THREADS", "1")
    main()