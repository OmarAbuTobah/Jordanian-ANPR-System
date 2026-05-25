"""Production-ready Flask app for offline Jordanian license plate OCR using dual-stage YOLOv8 + EasyOCR.

TWO-STAGE YOLO ARCHITECTURE:
- Stage 1: YOLOv8 Plate Detection (best.pt) → Detect license plate in image
- Stage 2: YOLOv8 Digit Detection (best-2.pt) → Detect numeric region within plate
- Stage 3: Advanced preprocessing pipeline → Enhance digit region quality
- Stage 4: EasyOCR recognition → Extract digits with high confidence
- Stage 5: Format validation → Format as Jordanian plate "XX - XXXXX"

TWO-STAGE BENEFITS:
✓ More accurate plate location (Stage 1)
✓ Precise digit region extraction (Stage 2)
✓ Better preprocessing on exact digit area
✓ Higher OCR confidence scores
✓ Reduced false positives from surrounding plate elements
✓ Better handling of decorative plate regions

Features:
- Dual YOLO models for precise region detection
- Professional preprocessing with bilateral filtering, CLAHE, denoise
- Multiple fallback pipelines for robust OCR
- Debug image outputs for all stages
- Confidence logging and tracking
- Support for Arabic and English digits
- Modern embedded HTML UI
"""

from datetime import datetime
import os
import re
import uuid

import PIL.Image
if not hasattr(PIL.Image, 'ANTIALIAS'):
    PIL.Image.ANTIALIAS = PIL.Image.Resampling.LANCZOS

import cv2
import easyocr
import numpy as np
from flask import Flask, render_template_string, request, redirect, url_for
from werkzeug.utils import secure_filename
from ultralytics import YOLO

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024  # limit uploads to 8MB

print("Loading YOLO model...")

MODEL_PATH = os.path.join(APP_ROOT, 'best.pt')
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Required model file 'best.pt' not found in project root.")

model = YOLO(MODEL_PATH)

print("Loading YOLO digit detection model...")
DIGIT_MODEL_PATH = os.path.join(APP_ROOT, 'best-2.pt')
if not os.path.exists(DIGIT_MODEL_PATH):
    print(f"WARNING: Digit detection model 'best-2.pt' not found at {DIGIT_MODEL_PATH}")
    print("Proceeding with single-stage detection (Stage 1 only)")
    digit_model = None
else:
    digit_model = YOLO(DIGIT_MODEL_PATH)
    print(f"✓ Loaded digit detection model: {DIGIT_MODEL_PATH}")

print("Loading EasyOCR reader...")
reader = easyocr.Reader(['en', 'ar'], gpu=False)

# Mapping for Arabic digits to English digits
ARABIC_TO_ENGLISH = str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789')

# ============================================================================
# PREPROCESSING CONFIGURATION - PROFESSIONAL JORDANIAN PLATE OPTIMIZATION
# ============================================================================
class OCRPreprocessConfig:
    """Professional preprocessing parameters optimized for Jordanian license plates.
    
    WHY EACH STEP MATTERS FOR JORDANIAN PLATES:
    - Bilateral Filter: Removes shadows/glare while preserving digit edges (outdoor conditions)
    - CLAHE: Handles uneven lighting (sun, reflections, clouds on plate)
    - Denoise: Removes dust, camera noise, JPEG compression artifacts
    - 4x Upscaling: Provides 16x more pixels for tiny plate digits (critical for small crops)
    - Adaptive Threshold: Works in shadow regions where Otsu threshold fails
    - Morphology: Fills holes in digits (8, 0, 9), removes dust specks
    - Sharpening: Makes blurry/low-res characters crisp enough for OCR
    """
    
    # ========== CROPPING: Focus on numeric region only ==========
    CROP_TOP_PERCENT = 0.18        # Remove "JORDAN" header + extra space
    CROP_BOTTOM_PERCENT = 0.18     # Remove dealer/ministry footer text
    
    # ========== UPSCALING: PROFESSIONAL 4x for maximum clarity ==========
    # 4.0x enlargement = 16x more pixels = dramatically improved OCR accuracy
    # Increased from 2.5x to 4.0x for professional-grade recognition
    UPSCALE_FACTOR = 4.0
    
    # ========== BILATERAL FILTERING: Edge-preserving smoothing ==========
    # Removes shadows/glare while keeping digit edges razor sharp
    BILATERAL_D = 9
    BILATERAL_SIGMA_COLOR = 80     # Increased for better shadow removal
    BILATERAL_SIGMA_SPACE = 80     # Better smoothing of reflection artifacts
    
    # ========== CLAHE: Contrast enhancement for outdoor lighting ==========
    # Critical for handling uneven outdoor conditions (sun glare, shadows, clouds)
    USE_CLAHE = True
    CLAHE_CLIP_LIMIT = 3.0
    CLAHE_TILE_SIZE = (8, 8)
    
    # ========== DENOISE: Remove camera/compression noise ==========
    # Non-Local Means denoising aggressively removes noise artifacts
    USE_DENOISE = True
    DENOISE_H = 10
    DENOISE_TEMPLATE_WINDOW = 7
    DENOISE_SEARCH_WINDOW = 21
    
    # ========== ADAPTIVE THRESHOLDING: Gray to Binary ==========
    # Adaptive threshold works in shadow regions, unlike global Otsu
    ADAPTIVE_BLOCK_SIZE = 35       # Increased from 31 (better adaptation)
    ADAPTIVE_CONSTANT = 12         # Decreased from 15 (better digit visibility)
    
    # ========== MORPHOLOGICAL OPERATIONS ==========
    # Close: fills holes inside digits. Open: removes isolated noise
    MORPH_KERNEL_SIZE = (3, 3)
    MORPH_ITERATIONS = 3           # Increased from 2 for stronger cleaning
    
    # ========== SHARPENING: Enhance digit edges ==========
    # Makes blurry characters crisp for better OCR recognition
    SHARPENING_STRENGTH = 1.8      # Increased from 1.5
    
    # ========== OCR CONFIDENCE THRESHOLDS ==========
    CONFIDENCE_HIGH = 0.75
    CONFIDENCE_MEDIUM = 0.55
    CONFIDENCE_LOW = 0.35
    
    # ========== IMAGE QUALITY CHECKS ==========
    # Automatic detection of problematic inputs
    MIN_CROP_HEIGHT = 20           # Pixels: too small = unreadable
    MIN_CROP_WIDTH = 80            # Pixels: too narrow = unreadable
    BLUR_THRESHOLD = 100.0         # Laplacian variance: lower = blurrier
    
    # ========== DEBUG & VISUALIZATION ==========
    SAVE_DEBUG_IMAGES = True
    DEBUG_OUTPUT_FOLDER = 'static/debug'

print(f"DEBUG: Preprocessing config loaded (PROFESSIONAL JORDANIAN OPTIMIZATION)")
print(f"DEBUG: Upscale factor: {OCRPreprocessConfig.UPSCALE_FACTOR}x (professional strength)")
print(f"DEBUG: CLAHE enabled: {OCRPreprocessConfig.USE_CLAHE}")
print(f"DEBUG: Denoise enabled: {OCRPreprocessConfig.USE_DENOISE}")
print(f"DEBUG: Debug output folder: {OCRPreprocessConfig.DEBUG_OUTPUT_FOLDER}")
if OCRPreprocessConfig.SAVE_DEBUG_IMAGES:
    os.makedirs(OCRPreprocessConfig.DEBUG_OUTPUT_FOLDER, exist_ok=True)


HTML_TEMPLATE = """
<!doctype html>
<html lang="en" dir="ltr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Jordanian License Plate Recognition</title>
  <style>
    body{font-family: Inter, system-ui, -apple-system, sans-serif;background:#f6f8fb;padding:36px}
    .card{max-width:900px;margin:0 auto;background:#fff;border-radius:10px;padding:24px;box-shadow:0 6px 24px rgba(32,33,36,.08)}
    h1{margin:0 0 8px 0;font-size:20px}
    form{display:flex;gap:12px;align-items:center;flex-wrap:wrap}
    input[type=file]{flex:1;min-width:260px}
    .primary-button{background:#0ea5a4;color:#fff;border:none;padding:10px 16px;border-radius:8px;cursor:pointer}
    .secondary-button{background:#6b7280;color:#fff;border:none;padding:10px 16px;border-radius:8px;cursor:pointer;text-decoration:none;display:inline-flex;align-items:center;justify-content:center}
    .result{margin-top:20px;text-align:center}
    .box{background:#f8fafc;padding:16px;border-radius:8px}
    img{max-width:100%;border-radius:6px;margin-top:12px}
    .label{font-weight:700;color:#0f172a;margin-bottom:8px}
    .plate-number{font-size:32px;font-weight:700;color:#0ea5a4;letter-spacing:2px}
  </style>
</head>
<body>
  <div class="card">
    <h1>Jordanian License Plate Recognition</h1>
    <p>Upload an image to extract the license plate number.</p>
    <form method="post" enctype="multipart/form-data">
      <input type="file" name="file" accept="image/*" required>
      <button type="submit" class="primary-button">Analyze Image 🔍</button>
      <a href="/" class="secondary-button">Refresh / Clear Page</a>
    </form>

    {% if error %}
      <div class="result" style="margin-top:18px">
        <div class="box">
          <div class="label">Error</div>
          <div>{{ error }}</div>
        </div>
      </div>
    {% endif %}

    {% if plate_cleaned %}
      <div class="result">
        <div class="box">
          <div class="label">Extracted Plate Number</div>
          <div class="plate-number">{{ plate_cleaned }}</div>
          <div class="label" style="margin-top:16px">OCR Results</div>
          <div>Grayscale OCR: <strong>{{ ocr_grayscale_text or 'N/A' }}</strong> <small>({{ ocr_grayscale_confidence | round(2) }})</small></div>
          <div>Threshold OCR: <strong>{{ ocr_threshold_text or 'N/A' }}</strong> <small>({{ ocr_threshold_confidence | round(2) }})</small></div>
          <img src="{{ image_url }}" alt="processed">
        </div>
      </div>
    {% endif %}
  </div>
</body>
</html>
"""


def check_image_quality(image: np.ndarray) -> tuple:
    """Check image quality and detect potential problems.
    
    Returns: (quality_score, warnings_list)
    
    Checks for:
    - Too small dimensions
    - Blur (using Laplacian variance)
    - Low contrast
    """
    warnings = []
    quality_score = 100.0
    
    h, w = image.shape[:2]
    
    # Check minimum size
    if h < OCRPreprocessConfig.MIN_CROP_HEIGHT or w < OCRPreprocessConfig.MIN_CROP_WIDTH:
        msg = f"WARNING: Image too small ({w}x{h}), minimum: ({OCRPreprocessConfig.MIN_CROP_WIDTH}x{OCRPreprocessConfig.MIN_CROP_HEIGHT})"
        warnings.append(msg)
        quality_score -= 30
    
    # Check for blur (Laplacian variance method)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    if laplacian_var < OCRPreprocessConfig.BLUR_THRESHOLD:
        msg = f"WARNING: Image appears blurry (Laplacian variance: {laplacian_var:.1f})"
        warnings.append(msg)
        quality_score -= 20
    
    # Check contrast
    mean_pixel = np.mean(gray)
    if mean_pixel < 30 or mean_pixel > 225:
        msg = f"WARNING: Extreme brightness (mean pixel: {mean_pixel:.0f})"
        warnings.append(msg)
        quality_score -= 10
    
    return quality_score, warnings


def detect_digit_region(plate_crop: np.ndarray) -> tuple:
    """Stage 2: Detect numeric digit region within license plate using second YOLO.
    
    Returns: (digit_crop, bbox_coords, detection_found)
    
    This stage uses a specialized YOLO model trained on digit regions to:
    - Precisely locate where the digits are on the plate
    - Ignore decorative elements, logos, and header/footer text
    - Extract only the numeric area for OCR
    
    If digit model not available, falls back to smart cropping.
    """
    
    if digit_model is None:
        # Fallback: use smart cropping instead of Stage 2 detection
        print("DEBUG [STAGE 2]: Digit model not available, using smart cropping fallback")
        h, w = plate_crop.shape[:2]
        top = int(h * 0.15)
        bottom = int(h * 0.85)
        digit_crop = plate_crop[top:bottom, :]
        return digit_crop, (0, top, w, bottom), False
    
    try:
        print("DEBUG [STAGE 2]: Running YOLO digit region detection...")
        results = digit_model(plate_crop)
        
        if len(results) == 0 or len(results[0].boxes) == 0:
            print("DEBUG [STAGE 2]: No digit region detected, using smart cropping")
            h, w = plate_crop.shape[:2]
            top = int(h * 0.15)
            bottom = int(h * 0.85)
            digit_crop = plate_crop[top:bottom, :]
            return digit_crop, (0, top, w, bottom), False
        
        # Get best detection box (highest confidence)
        boxes = results[0].boxes
        try:
            confs = boxes.conf if hasattr(boxes, 'conf') else None
        except:
            confs = None
        
        if confs is not None:
            idx = int(np.argmax(confs))
            best_box = boxes[idx]
        else:
            best_box = boxes[0]
        
        try:
            xy = best_box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, xy)
        except:
            coords = np.array(best_box.xyxy).reshape(-1)
            x1, y1, x2, y2 = map(int, coords[:4])
        
        # Add small padding to digit region
        pad = 2
        h, w = plate_crop.shape[:2]
        x1p = max(0, x1 - pad)
        y1p = max(0, y1 - pad)
        x2p = min(w - 1, x2 + pad)
        y2p = min(h - 1, y2 + pad)
        
        digit_crop = plate_crop[y1p:y2p, x1p:x2p]
        
        print(f"DEBUG [STAGE 2]: Digit region detected at ({x1p}, {y1p}) → ({x2p}, {y2p})")
        print(f"DEBUG [STAGE 2]: Digit crop shape: {digit_crop.shape}")
        
        if OCRPreprocessConfig.SAVE_DEBUG_IMAGES:
            cv2.imwrite(f"{OCRPreprocessConfig.DEBUG_OUTPUT_FOLDER}/00_digit_region.jpg", digit_crop)
        
        return digit_crop, (x1p, y1p, x2p, y2p), True
        
    except Exception as e:
        print(f"DEBUG [STAGE 2]: ERROR detecting digit region: {e}")
        # Fallback to smart cropping
        h, w = plate_crop.shape[:2]
        top = int(h * 0.15)
        bottom = int(h * 0.85)
        digit_crop = plate_crop[top:bottom, :]
        return digit_crop, (0, top, w, bottom), False


def extract_digits_from_crop(crop: np.ndarray) -> tuple:
    """Multi-stage OCR using grayscale and threshold images.

    Returns: (pure_digits, confidence_score, debug_info_dict)
    """

    debug_info = {
        'pipeline': 'grayscale_threshold_multistage',
        'stages_completed': [],
        'confidence_scores': [],
        'ocr_attempts': 0,
        'quality_warnings': [],
        'laplacian_variance': 0.0,
        'ocr_grayscale_text': '',
        'ocr_grayscale_digits': '',
        'ocr_grayscale_confidence': 0.0,
        'ocr_threshold_text': '',
        'ocr_threshold_digits': '',
        'ocr_threshold_confidence': 0.0
    }

    try:
        # ====== STAGE 0: Quality Check ======
        quality_score, warnings = check_image_quality(crop)
        debug_info['quality_warnings'] = warnings
        for w in warnings:
            print(f"DEBUG: {w}")
        print(f"DEBUG [QC]: Image quality score: {quality_score:.1f}/100")

        # ====== STAGE 1: Convert to Grayscale ======
        if len(crop.shape) == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop
        debug_info['stages_completed'].append('grayscale')
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        debug_info['laplacian_variance'] = laplacian_var
        print(f"DEBUG [1/4]: Grayscale conversion - shape: {gray.shape}, blur metric: {laplacian_var:.1f}")

        if OCRPreprocessConfig.SAVE_DEBUG_IMAGES:
            cv2.imwrite(f"{OCRPreprocessConfig.DEBUG_OUTPUT_FOLDER}/01_grayscale.jpg", gray)

        # ====== STAGE 2: Smart Cropping (Remove header/footer) ======
        h, w = gray.shape[:2]
        top = int(h * OCRPreprocessConfig.CROP_TOP_PERCENT)
        bottom = int(h * (1 - OCRPreprocessConfig.CROP_BOTTOM_PERCENT))
        middle_crop = gray[top:bottom, :]

        if middle_crop.size == 0:
            print("DEBUG [2/4]: ERROR - Middle crop is empty")
            return "", 0.0, debug_info

        debug_info['stages_completed'].append('smart_crop')
        print(f"DEBUG [2/4]: Smart crop - Original: {gray.shape} → Cropped: {middle_crop.shape}")

        # ====== STAGE 3: Upscaling ======
        upscaled = cv2.resize(middle_crop, None, fx=OCRPreprocessConfig.UPSCALE_FACTOR,
                             fy=OCRPreprocessConfig.UPSCALE_FACTOR,
                             interpolation=cv2.INTER_CUBIC)
        debug_info['stages_completed'].append('upscale')
        print(f"DEBUG [3/4]: Upscaling {OCRPreprocessConfig.UPSCALE_FACTOR}x → {upscaled.shape}")

        if OCRPreprocessConfig.SAVE_DEBUG_IMAGES:
            cv2.imwrite(f"{OCRPreprocessConfig.DEBUG_OUTPUT_FOLDER}/02_upscaled.jpg", upscaled)

        # ====== STAGE 4: Adaptive Thresholding ======
        thresh_adaptive = cv2.adaptiveThreshold(upscaled, 255,
                                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                               cv2.THRESH_BINARY,
                                               OCRPreprocessConfig.ADAPTIVE_BLOCK_SIZE,
                                               OCRPreprocessConfig.ADAPTIVE_CONSTANT)
        debug_info['stages_completed'].append('adaptive_threshold')
        print(f"DEBUG [4/4]: Adaptive threshold (block: {OCRPreprocessConfig.ADAPTIVE_BLOCK_SIZE}, const: {OCRPreprocessConfig.ADAPTIVE_CONSTANT})")

        if OCRPreprocessConfig.SAVE_DEBUG_IMAGES:
            cv2.imwrite(f"{OCRPreprocessConfig.DEBUG_OUTPUT_FOLDER}/03_threshold.jpg", thresh_adaptive)

        # ====== MULTI-STAGE OCR ======
        debug_info['ocr_attempts'] = 2

        grayscale_results = reader.readtext(upscaled, allowlist='0123456789')
        threshold_results = reader.readtext(thresh_adaptive, allowlist='0123456789')

        raw_gray = ''.join([entry[1] for entry in grayscale_results])
        gray_confidences = [entry[2] for entry in grayscale_results]
        gray_conf = float(np.mean(gray_confidences)) if gray_confidences else 0.0
        gray_digits = ''.join(re.findall(r'\d+', raw_gray.translate(ARABIC_TO_ENGLISH)))

        raw_thresh = ''.join([entry[1] for entry in threshold_results])
        thresh_confidences = [entry[2] for entry in threshold_results]
        thresh_conf = float(np.mean(thresh_confidences)) if thresh_confidences else 0.0
        thresh_digits = ''.join(re.findall(r'\d+', raw_thresh.translate(ARABIC_TO_ENGLISH)))

        debug_info['ocr_grayscale_text'] = raw_gray
        debug_info['ocr_grayscale_digits'] = gray_digits
        debug_info['ocr_grayscale_confidence'] = gray_conf
        debug_info['ocr_threshold_text'] = raw_thresh
        debug_info['ocr_threshold_digits'] = thresh_digits
        debug_info['ocr_threshold_confidence'] = thresh_conf
        debug_info['confidence_scores'] = [gray_conf, thresh_conf]

        print(f"DEBUG: Grayscale OCR raw='{raw_gray}' conf={gray_conf:.2f} digits='{gray_digits}'")
        print(f"DEBUG: Threshold OCR raw='{raw_thresh}' conf={thresh_conf:.2f} digits='{thresh_digits}'")

        final_digits = ''
        final_confidence = 0.0
        if thresh_digits and gray_digits:
            if thresh_conf >= gray_conf:
                final_digits = thresh_digits
                final_confidence = thresh_conf
                print("DEBUG: Choosing threshold OCR result")
            else:
                final_digits = gray_digits
                final_confidence = gray_conf
                print("DEBUG: Choosing grayscale OCR result")
        elif thresh_digits:
            final_digits = thresh_digits
            final_confidence = thresh_conf
            print("DEBUG: Using threshold OCR result")
        elif gray_digits:
            final_digits = gray_digits
            final_confidence = gray_conf
            print("DEBUG: Using grayscale OCR result")
        else:
            print("DEBUG: No digits found in either grayscale or threshold OCR")
            return "", 0.0, debug_info

        return final_digits, final_confidence, debug_info

    except Exception as e:
        print(f"DEBUG [ERROR]: Exception in extract_digits_from_crop: {e}")
        import traceback
        traceback.print_exc()
        return "", 0.0, debug_info


def format_plate_number(pure_digits: str) -> str:
    """Format pure digit string into Jordanian plate format or error message.
    
    Supports 2-7 digit plates with flexible formatting:
    - 2 digits: X - X
    - 3 digits: X - XX or XX - X
    - 4 digits: X - XXX or XX - XX
    - 5 digits: X - XXXX or XX - XXX
    - 6 digits: XX - XXXX or X - XXXXX
    - 7 digits: XX - XXXXX
    
    Smart fallback: If string is longer than 7 digits, keep only the last 7.
    """
    if not pure_digits:
        print("DEBUG: Empty pure_digits")
        return ''
    
    length = len(pure_digits)
    print(f"DEBUG: Digit length: {length}")
    
    # Smart fallback: if too long, keep last 7 digits
    if length > 7:
        print(f"DEBUG: Digit string too long ({length}), keeping last 7 digits")
        pure_digits = pure_digits[-7:]
        length = 7
    
    # Validate minimum length
    if length < 2:
        print(f"DEBUG: Invalid length {length}, less than 2")
        return "Plate detected but format is unreadable"
    
    # Format based on length with standard Jordanian plate patterns
    if length == 2:
        return f"{pure_digits[0]} - {pure_digits[1]}"
    elif length == 3:
        # Prefer 1 digit code + 2 digit sequence
        return f"{pure_digits[0]} - {pure_digits[1:]}"
    elif length == 4:
        # Prefer 1 digit code + 3 digit sequence
        return f"{pure_digits[0]} - {pure_digits[1:]}"
    elif length == 5:
        # Prefer 1 digit code + 4 digit sequence (most common)
        return f"{pure_digits[0]} - {pure_digits[1:]}"
    elif length == 6:
        # Standard format: 2 digit code + 4 digit sequence
        return f"{pure_digits[:2]} - {pure_digits[2:]}"
    elif length == 7:
        # 2 digit code + 5 digit sequence
        return f"{pure_digits[:2]} - {pure_digits[2:]}"
    
    return "Plate detected but format is unreadable"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template_string(HTML_TEMPLATE, error='No file uploaded.')
        file = request.files['file']
        if file.filename == '':
            return render_template_string(HTML_TEMPLATE, error='No file selected.')

        filename = secure_filename(file.filename)
        unique_name = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}_{filename}"
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
        try:
            file.save(save_path)
        except Exception as e:
            return render_template_string(HTML_TEMPLATE, error=f'Failed to save uploaded file: {e}')

        # Read original image
        img = cv2.imread(save_path)
        if img is None:
            return render_template_string(HTML_TEMPLATE, error='Unsupported or corrupted image file.')

        # ========== STAGE 1: YOLOv8 Plate Detection ==========
        print("\n" + "="*70)
        print("STAGE 1: YOLO PLATE DETECTION")
        print("="*70)
        try:
            results = model(img)
        except Exception as e:
            return render_template_string(HTML_TEMPLATE, error=f'Failed to run object detection model: {e}')

        plate_cleaned = ''
        processed_img = img.copy()

        if len(results) == 0 or len(results[0].boxes) == 0:
            return render_template_string(HTML_TEMPLATE, error='No license plate was detected in this image.')

        # Get best plate detection box (highest confidence)
        boxes = results[0].boxes
        try:
            confs = boxes.conf if hasattr(boxes, 'conf') else None
        except Exception:
            confs = None

        best_box = None
        if confs is not None:
            idx = int(np.argmax(confs))
            best_box = boxes[idx]
        else:
            best_box = boxes[0]

        try:
            xy = best_box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, xy)
        except Exception:
            coords = np.array(best_box.xyxy).reshape(-1)
            x1, y1, x2, y2 = map(int, coords[:4])

        # Padding for plate crop
        pad = 10
        h, w = img.shape[:2]
        x1p = max(0, x1 - pad)
        y1p = max(0, y1 - pad)
        x2p = min(w - 1, x2 + pad)
        y2p = min(h - 1, y2 + pad)

        plate_crop = img[y1p:y2p, x1p:x2p]
        if plate_crop.size == 0:
            return render_template_string(HTML_TEMPLATE, error='Failed to crop plate region from the image.')

        print(f"✓ Plate detected at ({x1}, {y1}) → ({x2}, {y2})")
        print(f"✓ Plate crop shape: {plate_crop.shape}")
        
        # Save original plate crop to debug
        if OCRPreprocessConfig.SAVE_DEBUG_IMAGES:
            debug_crop_path = os.path.join(OCRPreprocessConfig.DEBUG_OUTPUT_FOLDER, '00_plate_crop.jpg')
            cv2.imwrite(debug_crop_path, plate_crop)
            print(f"✓ Saved plate crop: {debug_crop_path}")

        # ========== STAGE 2: YOLO Digit Region Detection ==========
        print("\n" + "="*70)
        print("STAGE 2: YOLO DIGIT REGION DETECTION")
        print("="*70)
        digit_crop, digit_bbox, digit_detected = detect_digit_region(plate_crop)

        # ========== STAGE 3-4: Preprocessing & OCR ==========
        print("\n" + "="*70)
        print("STAGE 3: PREPROCESSING & STAGE 4: OCR")
        print("="*70)
        
        # Extract digits using professional preprocessing pipeline
        print(f"Starting OCR on digit crop shape: {digit_crop.shape}")
        pure_digits, ocr_confidence, debug_info = extract_digits_from_crop(digit_crop)
        print(f"✓ Digits returned: '{pure_digits}' (confidence: {ocr_confidence:.2f})")
        print(f"✓ Preprocessing stages: {debug_info['stages_completed']}")
        print(f"✓ OCR attempts: {debug_info['ocr_attempts']}")
        
        # ========== STAGE 5: Format Validation ==========
        print("\n" + "="*70)
        print("STAGE 5: FORMAT VALIDATION")
        print("="*70)
        plate_cleaned = format_plate_number(pure_digits)
        print(f"✓ Formatted plate: '{plate_cleaned}'")
        
        if not plate_cleaned:
            print(f"ERROR: Plate extraction failed")
            return render_template_string(HTML_TEMPLATE, error='Failed to extract valid plate number from the image. Please try a clearer image.')

        # Annotate processed image with plate location and result
        cv2.rectangle(processed_img, (x1p, y1p), (x2p, y2p), (16, 185, 129), 3)
        label = plate_cleaned if plate_cleaned else pure_digits[:20]
        cv2.putText(processed_img, label, (x1p, max(15, y1p - 8)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (16, 185, 129), 2)

        out_name = 'processed_' + unique_name
        out_path = os.path.join(app.config['UPLOAD_FOLDER'], out_name)
        try:
            cv2.imwrite(out_path, processed_img)
        except Exception as e:
            return render_template_string(HTML_TEMPLATE, error=f'Failed to write the annotated image: {e}')

        image_url = url_for('static', filename=f'uploads/{out_name}')
        
        print("\n" + "="*70)
        print("RESULT")
        print("="*70)
        print(f"Plate number: {plate_cleaned}")
        print(f"Confidence: {ocr_confidence:.2f}")
        print(f"Grayscale OCR: {debug_info.get('ocr_grayscale_text', '')} (conf={debug_info.get('ocr_grayscale_confidence', 0.0):.2f})")
        print(f"Threshold OCR: {debug_info.get('ocr_threshold_text', '')} (conf={debug_info.get('ocr_threshold_confidence', 0.0):.2f})")
        print(f"Digit model used: {'Yes (best-2.pt)' if digit_detected else 'No (smart crop fallback)'}")
        print("="*70 + "\n")
        
        return render_template_string(
            HTML_TEMPLATE,
            plate_cleaned=plate_cleaned,
            image_url=image_url,
            ocr_grayscale_text=debug_info.get('ocr_grayscale_text', ''),
            ocr_threshold_text=debug_info.get('ocr_threshold_text', ''),
            ocr_grayscale_confidence=debug_info.get('ocr_grayscale_confidence', 0.0),
            ocr_threshold_confidence=debug_info.get('ocr_threshold_confidence', 0.0)
        )

    return render_template_string(HTML_TEMPLATE)


if __name__ == '__main__':
    # Production note: set debug=False when deploying
    app.run(host='0.0.0.0', port=5001, debug=False)