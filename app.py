import os
import glob
import cv2
import numpy as np
from typing import Optional
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ==========================================================
# 1. CORE ROTATION DETECTION ENGINE
# ==========================================================

class UniversalRotationDetector:
    def __init__(self, search_range=180.0, step=1.0, osd_margin=0.10):
        self.search_range = search_range
        self.step = step
        self.osd_margin = osd_margin
        self.ref_histogram = None

    def _compute_orientation_distribution(self, frame: np.ndarray) -> Optional[np.ndarray]:
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)

        # Blank outer 10% margins to strip timestamps and logos
        my, mx = int(h * self.osd_margin), int(w * self.osd_margin)
        gx[:my, :] = 0; gx[h - my:, :] = 0; gx[:, :mx] = 0; gx[:, w - mx:] = 0
        gy[:my, :] = 0; gy[h - my:, :] = 0; gy[:, :mx] = 0; gy[:, w - mx:] = 0

        magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

        inner_mag = magnitude[my:h-my, mx:w-mx]
        if inner_mag.size == 0:
            return None

        thresh = np.percentile(inner_mag, 75)
        valid_mask = magnitude > thresh

        valid_angles = angle[valid_mask]
        valid_weights = magnitude[valid_mask]

        if len(valid_angles) < 500:
            return None

        hist, _ = np.histogram(valid_angles, bins=360, range=(0, 360), weights=valid_weights)

        kernel = cv2.getGaussianKernel(9, 2.0).ravel()
        padded = np.pad(hist, 4, mode='wrap')
        smoothed_hist = np.convolve(padded, kernel, mode='valid')

        smoothed_hist -= np.mean(smoothed_hist)
        norm = np.linalg.norm(smoothed_hist)
        return smoothed_hist / (norm + 1e-7)

    def set_reference(self, reference_frame: np.ndarray):
        hist = self._compute_orientation_distribution(reference_frame)
        if hist is None:
            raise ValueError("Reference image has insufficient edge contrast.")
        self.ref_histogram = hist

    def detect_rotation(self, live_frame: np.ndarray) -> dict:
        if self.ref_histogram is None:
            raise ValueError("Reference must be set first.")

        live_hist = self._compute_orientation_distribution(live_frame)
        if live_hist is None:
            return {
                "status": "INSUFFICIENT_EVIDENCE",
                "rotation_angle": 0.0,
                "is_rotated": False,
                "confidence": 0.0
            }

        test_angles = np.arange(-self.search_range, self.search_range + self.step, self.step)
        scores = []

        for alpha in test_angles:
            shift_bins = int(np.round(alpha))
            shifted = np.roll(live_hist, shift_bins)
            scores.append(np.dot(self.ref_histogram, shifted))

        scores = np.array(scores)
        best_idx = np.argmax(scores)
        raw_angle = test_angles[best_idx]
        best_score = float(scores[best_idx])

        # Parabolic refinement
        if 0 < best_idx < len(scores) - 1:
            y0, y1, y2 = scores[best_idx - 1], scores[best_idx], scores[best_idx + 1]
            denom = 2 * (2 * y1 - y0 - y2)
            if denom != 0:
                delta = (y2 - y0) / denom
                raw_angle += delta * self.step

        camera_tilt = -raw_angle
        camera_tilt = (camera_tilt + 180.0) % 360.0 - 180.0

        abs_tilt = abs(camera_tilt)
        is_rotated = abs_tilt >= 2.0  # Threshold of 2.0 degrees

        return {
            "rotation_angle": round(float(camera_tilt), 1),
            "is_rotated": is_rotated,
            "confidence": round(best_score, 3)
        }


# ==========================================================
# 2. RTSP UTILITIES & REFERENCE REPO
# ==========================================================

STORAGE_DIR = Path("./reference_images")
STORAGE_DIR.mkdir(parents=True, exist_ok=True)


def grab_rtsp_frame(stream_url: str) -> np.ndarray:
    """Connects to live RTSP feed over TCP, flushes stale buffer, returns frame."""
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        raise HTTPException(status_code=502, detail=f"Cannot connect to RTSP stream: {stream_url}")

    # Read a few frames to clear the RTSP buffer and grab the latest live image
    frame = None
    for _ in range(5):
        ret, temp = cap.read()
        if ret:
            frame = temp

    cap.release()

    if frame is None:
        raise HTTPException(status_code=504, detail="Connected to stream but failed to retrieve video frames.")
    return frame


def resolve_reference_path(nvr: str, channel: str, seq: Optional[str] = None) -> Path:
    """
    Finds file matching pattern:
      - Specific: reference_images/{nvr}_{channel}_{seq}.jpg
      - Wildcard: reference_images/{nvr}_{channel}_*.jpg (takes latest)
    """
    if seq:
        expected = STORAGE_DIR / f"{nvr}_{channel}_{seq}.jpg"
        if not expected.exists():
            # Try png as fallback
            expected = STORAGE_DIR / f"{nvr}_{channel}_{seq}.png"
            if not expected.exists():
                raise HTTPException(
                    status_code=404,
                    detail=f"Reference file '{nvr}_{channel}_{seq}.jpg' not found in {STORAGE_DIR}"
                )
        return expected

    # Wildcard search for the latest reference image for this NVR and channel
    matches = sorted(glob.glob(str(STORAGE_DIR / f"{nvr}_{channel}_*.*")))
    if not matches:
        raise HTTPException(
            status_code=404,
            detail=f"No reference image found matching pattern '{nvr}_{channel}_*.*' in {STORAGE_DIR}"
        )
    return Path(matches[-1])


# ==========================================================
# 3. FASTAPI SCHEMAS & ENDPOINTS
# ==========================================================

app = FastAPI(
    title="CCTV Rotation & Tamper Guard API",
    description="Microservice to verify CCTV alignment from live streams against stored reference images.",
    version="1.0.0"
)


class CheckRotationRequest(BaseModel):
    nvr: str = Field(..., example="NVR01", description="NVR Identifier")
    channel: str = Field(..., example="CH02", description="Camera Channel")
    seq: Optional[str] = Field(None, example="001", description="Reference version/number (optional)")
    stream_url: str = Field(..., example="rtsp://admin:pass@192.168.1.50:554/stream1", description="Live RTSP stream URL")


class SaveReferenceRequest(BaseModel):
    nvr: str = Field(..., example="NVR01", description="NVR Identifier")
    channel: str = Field(..., example="CH02", description="Camera Channel")
    seq: str = Field(..., example="001", description="Number/version (e.g., 001)")
    stream_url: str = Field(..., example="rtsp://admin:pass@192.168.1.50:554/stream1", description="Live RTSP stream URL")


@app.post("/check-rotation")
def check_camera_rotation(payload: CheckRotationRequest):
    """
    1. Loads reference image matching {nvr}_{channel}_{seq}.jpg
    2. Pulls a fresh frame from the live CCTV RTSP stream
    3. Returns rotation status and actionable human instructions
    """
    # 1. Locate reference image
    ref_path = resolve_reference_path(payload.nvr, payload.channel, payload.seq)
    ref_img = cv2.imread(str(ref_path))
    if ref_img is None:
        raise HTTPException(status_code=500, detail=f"Failed to decode reference image from disk: {ref_path.name}")

    # 2. Grab live frame from CCTV stream
    live_frame = grab_rtsp_frame(payload.stream_url)

    # 3. Run rotation detection
    detector = UniversalRotationDetector(search_range=180.0, step=1.0, osd_margin=0.10)
    try:
        detector.set_reference(ref_img)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    result = detector.detect_rotation(live_frame)

    # 4. Formulate human-friendly response
    tilt = abs(result["rotation_angle"])
    is_rotated = result["is_rotated"]

    if not is_rotated:
        status_label = "NORMAL"
        action_required = "None. Camera is properly aligned."
    else:
        status_label = "TILTED"
        if 175.0 <= tilt <= 180.0:
            action_required = "Camera is mounted/flipped upside down. Flip camera upright."
        elif result["rotation_angle"] > 0:
            action_required = f"Turn camera {tilt:.0f}° to the LEFT (Counter-Clockwise) to align."
        else:
            action_required = f"Turn camera {tilt:.0f}° to the RIGHT (Clockwise) to align."

    return {
        "nvr": payload.nvr,
        "channel": payload.channel,
        "reference_file_used": ref_path.name,
        "is_rotated": is_rotated,
        "status": status_label,
        "tilt_angle_deg": result["rotation_angle"],
        "action_required": action_required,
        "match_confidence": result["confidence"]
    }


@app.post("/save-reference")
def save_new_reference(payload: SaveReferenceRequest):
    """
    Captures a frame from the live stream and stores it as the new baseline
    under ./reference_images/{nvr}_{channel}_{seq}.jpg
    """
    frame = grab_rtsp_frame(payload.stream_url)
    target_filename = f"{payload.nvr}_{payload.channel}_{payload.seq}.jpg"
    target_path = STORAGE_DIR / target_filename

    success = cv2.imwrite(str(target_path), frame)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to save image to disk.")

    return {
        "message": "Reference baseline captured successfully.",
        "saved_path": str(target_path),
        "filename": target_filename
    }


@app.get("/references")
def list_reference_images():
    """Lists all stored camera references."""
    files = [f.name for f in STORAGE_DIR.glob("*.*") if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    return {"total_references": len(files), "files": sorted(files)}