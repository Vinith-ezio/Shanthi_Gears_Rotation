import os
import glob
import cv2
import numpy as np
from typing import Optional
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

# ==========================================================
# 1. CORE ROTATION & INDUSTRIAL BLUR ENGINE
# ==========================================================

class UniversalRotationDetector:
    def __init__(self, search_range=180.0, step=1.0, osd_margin=0.10, blur_drop_ratio=0.45, min_sharpness_floor=50.0):
        """
        :param search_range: Search range in degrees (+/- 180°).
        :param step: Search step in degrees.
        :param osd_margin: Outer margin to mask (0.10 = 10% strips timestamps).
        :param blur_drop_ratio: If sharpness drops below 45% of baseline, trigger alert.
        :param min_sharpness_floor: Absolute minimum sharpness threshold.
        """
        self.search_range = search_range
        self.step = step
        self.osd_margin = osd_margin
        self.blur_drop_ratio = blur_drop_ratio
        self.min_sharpness_floor = min_sharpness_floor

        self.ref_histogram = None
        self.ref_sharpness = None

    def _get_inner_gray(self, frame: np.ndarray) -> np.ndarray:
        """Crops out outer 10% OSD border so timestamps don't fake sharpness."""
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        my, mx = int(h * self.osd_margin), int(w * self.osd_margin)
        return gray[my:h - my, mx:w - mx]

    def compute_sharpness(self, frame: np.ndarray) -> float:
        """
        Industrial-Grade Contrast-Normalized Patch Laplacian:
        1. Divides image into a 4x4 grid (16 patches).
        2. Rejects glare/specular reflections from metal surfaces.
        3. Normalizes contrast per patch (invariant to industrial lighting changes).
        4. Computes Laplacian variance per patch.
        5. Takes the MEDIAN patch variance (moving workers/forklifts are rejected).
        """
        inner = self._get_inner_gray(frame)
        h, w = inner.shape

        grid_y, grid_x = 4, 4
        patch_h, patch_w = h // grid_y, w // grid_x
        patch_scores = []

        for i in range(grid_y):
            for j in range(grid_x):
                patch = inner[i * patch_h:(i + 1) * patch_h, j * patch_w:(j + 1) * patch_w]

                # 1. Skip patches with extreme glare (>245) or deep shadows (<10)
                if np.mean(patch >= 245) > 0.35 or np.mean(patch <= 10) > 0.70:
                    continue

                std_dev = np.std(patch)
                if std_dev < 8.0:
                    continue

                # 2. Denoise: 3x3 Gaussian filter removes digital ISO grain
                blurred = cv2.GaussianBlur(patch, (3, 3), 0)

                # 3. Contrast Normalization: Z-score scaling removes factory light shift effects
                norm_patch = ((blurred - np.mean(blurred)) / (std_dev + 1e-5)) * 50.0

                # 4. Laplacian Operator (2nd spatial derivative)
                lap = cv2.Laplacian(norm_patch, cv2.CV_64F, ksize=3)

                # 5. Trim extreme 1% outlier spikes (welding arcs, glints)
                lap_abs = np.abs(lap)
                cap_val = np.percentile(lap_abs, 99)
                lap_trimmed = np.clip(lap, -cap_val, cap_val)

                patch_scores.append(float(np.var(lap_trimmed)))

        if not patch_scores:
            std_dev = max(1.0, np.std(inner))
            norm_inner = ((inner - np.mean(inner)) / std_dev) * 50.0
            return round(float(cv2.Laplacian(norm_inner, cv2.CV_64F).var()), 2)

        return round(float(np.median(patch_scores)), 2)

    def _compute_orientation_distribution(self, frame: np.ndarray) -> Optional[np.ndarray]:
        h, w = frame.shape[:2]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)

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
        self.ref_sharpness = self.compute_sharpness(reference_frame)

    def detect_status(self, live_frame: np.ndarray) -> dict:
        if self.ref_histogram is None or self.ref_sharpness is None:
            raise ValueError("Reference must be set first.")

        # 1. Industrial Blur Evaluation
        live_sharpness = self.compute_sharpness(live_frame)
        sharpness_ratio = live_sharpness / (self.ref_sharpness + 1e-5)
        sharpness_drop_pct = max(0.0, round((1.0 - sharpness_ratio) * 100, 1))

        is_blurry = (live_sharpness < self.min_sharpness_floor) or (sharpness_ratio < self.blur_drop_ratio)

        # 2. Tilt Evaluation
        live_hist = self._compute_orientation_distribution(live_frame)
        if live_hist is None:
            return {
                "rotation_angle": 0.0,
                "is_rotated": False,
                "is_blurry": is_blurry,
                "sharpness_drop_pct": sharpness_drop_pct,
                "live_sharpness": live_sharpness,
                "ref_sharpness": self.ref_sharpness,
                "confidence": 0.0,
                "insufficient_evidence": True
            }

        test_angles = np.arange(-self.search_range, self.search_range + self.step, self.step)
        scores = []

        for alpha in test_angles:
            shift_bins = int(np.round(alpha))
            shifted = np.roll(live_hist, shift_bins)
            scores.append(np.dot(self.ref_histogram, shifted))

        scores = np.array(scores)
        best_idx = np.argmax(scores)
        best_score = float(scores[best_idx])
        raw_angle = test_angles[best_idx]

        if 0 < best_idx < len(scores) - 1:
            y0, y1, y2 = scores[best_idx - 1], scores[best_idx], scores[best_idx + 1]
            denom = 2 * (2 * y1 - y0 - y2)
            if denom != 0:
                delta = (y2 - y0) / denom
                raw_angle += delta * self.step

        camera_tilt = -raw_angle
        camera_tilt = (camera_tilt + 180.0) % 360.0 - 180.0
        abs_tilt = abs(camera_tilt)
        is_rotated = abs_tilt >= 2.0

        return {
            "rotation_angle": round(float(camera_tilt), 1),
            "is_rotated": is_rotated,
            "is_blurry": is_blurry,
            "sharpness_drop_pct": sharpness_drop_pct,
            "live_sharpness": live_sharpness,
            "ref_sharpness": self.ref_sharpness,
            "confidence": round(best_score, 3),
            "insufficient_evidence": False
        }


# ==========================================================
# 2. STORAGE & RTSP UTILITIES
# ==========================================================

STORAGE_DIR = Path("./reference_images")
STORAGE_DIR.mkdir(parents=True, exist_ok=True)


def grab_rtsp_frame(stream_url: str) -> np.ndarray:
    """Pulls frame from live camera over TCP and clears buffer."""
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    
    # Handle webcam index ("0") or RTSP string
    src = int(stream_url) if stream_url.isdigit() else stream_url
    cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        raise HTTPException(status_code=502, detail=f"Cannot connect to stream: {stream_url}")

    frame = None
    for _ in range(5):
        ret, temp = cap.read()
        if ret:
            frame = temp
    cap.release()

    if frame is None:
        raise HTTPException(status_code=504, detail="Connected to stream but failed to retrieve frame.")
    return frame


def resolve_reference_path(nvr: str, channel: str, seq: Optional[str] = None) -> Path:
    """Matches files in format: {nvr}_{channel}_{seq}.jpg"""
    if seq:
        for ext in [".jpg", ".jpeg", ".png"]:
            expected = STORAGE_DIR / f"{nvr}_{channel}_{seq}{ext}"
            if expected.exists():
                return expected
        raise HTTPException(status_code=404, detail=f"Reference '{nvr}_{channel}_{seq}.jpg' not found in {STORAGE_DIR}")

    matches = sorted(glob.glob(str(STORAGE_DIR / f"{nvr}_{channel}_*.*")))
    if not matches:
        raise HTTPException(status_code=404, detail=f"No reference image found for '{nvr}_{channel}_*' in {STORAGE_DIR}")
    return Path(matches[-1])


# ==========================================================
# 3. FASTAPI ENDPOINTS
# ==========================================================

app = FastAPI(
    title="CCTV Health & Tamper Guard API",
    description="Microservice to monitor CCTV camera rotation/tilt and industrial lens blur/defocus.",
    version="2.0.0"
)

@app.get("/", include_in_schema=False)
def redirect_to_docs():
    return RedirectResponse(url="/docs")


class CheckRequest(BaseModel):
    nvr: str = Field(..., example="NVR01", description="NVR Identifier")
    channel: str = Field(..., example="CH02", description="Camera Channel")
    seq: Optional[str] = Field(None, example="001", description="Reference version (optional)")
    stream_url: str = Field(..., example="rtsp://admin:password@192.168.1.50:554/stream1", description="Live RTSP stream URL or '0' for webcam")


class SaveRequest(BaseModel):
    nvr: str = Field(..., example="NVR01", description="NVR Identifier")
    channel: str = Field(..., example="CH02", description="Camera Channel")
    seq: str = Field(..., example="001", description="Reference sequence/number")
    stream_url: str = Field(..., example="rtsp://admin:password@192.168.1.50:554/stream1", description="Live RTSP stream URL or '0' for webcam")


@app.post("/check-camera")
def check_camera_health(payload: CheckRequest):
    """
    1. Loads reference image matching {nvr}_{channel}_{seq}.jpg
    2. Pulls real-time frame from live camera stream
    3. Runs tilt cross-correlation + industrial contrast-normalized patch blur detection
    """
    ref_path = resolve_reference_path(payload.nvr, payload.channel, payload.seq)
    ref_img = cv2.imread(str(ref_path))
    if ref_img is None:
        raise HTTPException(status_code=500, detail=f"Failed to decode reference: {ref_path.name}")

    live_frame = grab_rtsp_frame(payload.stream_url)

    detector = UniversalRotationDetector(search_range=180.0, step=1.0, osd_margin=0.10)
    try:
        detector.set_reference(ref_img)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    res = detector.detect_status(live_frame)

    # Human-friendly actions
    issues = []
    actions = []

    if res["is_rotated"]:
        tilt = abs(res["rotation_angle"])
        if 175.0 <= tilt <= 180.0:
            issues.append("CAMERA_FLIPPED_UPSIDE_DOWN")
            actions.append("Flip camera upright.")
        elif res["rotation_angle"] > 0:
            issues.append(f"CAMERA_TILTED_RIGHT (+{tilt}°)")
            actions.append(f"Turn camera {tilt:.0f}° to the LEFT.")
        else:
            issues.append(f"CAMERA_TILTED_LEFT (-{tilt}°)")
            actions.append(f"Turn camera {tilt:.0f}° to the RIGHT.")

    if res["is_blurry"]:
        issues.append("LENS_BLURRY_OR_DIRTY")
        actions.append(f"Clean lens or adjust focus ring (Lost {res['sharpness_drop_pct']}% sharpness).")

    is_healthy = (not res["is_rotated"]) and (not res["is_blurry"])
    overall_status = "HEALTHY" if is_healthy else " | ".join(issues)
    action_required = "None. Camera is operating normally." if is_healthy else " | ".join(actions)

    return {
        "nvr": payload.nvr,
        "channel": payload.channel,
        "reference_file_used": ref_path.name,
        "is_healthy": is_healthy,
        "overall_status": overall_status,
        "rotation": {
            "is_rotated": res["is_rotated"],
            "tilt_angle_deg": res["rotation_angle"],
            "confidence": res.get("confidence", 0.0)
        },
        "focus_clarity": {
            "is_blurry": res["is_blurry"],
            "live_sharpness_score": res["live_sharpness"],
            "baseline_sharpness_score": res["ref_sharpness"],
            "sharpness_drop_pct": res["sharpness_drop_pct"]
        },
        "action_required": action_required
    }


@app.post("/save-reference")
def save_reference_baseline(payload: SaveRequest):
    """
    Captures baseline frame directly from live camera and saves it as:
    ./reference_images/{nvr}_{channel}_{seq}.jpg
    """
    frame = grab_rtsp_frame(payload.stream_url)
    target_filename = f"{payload.nvr}_{payload.channel}_{payload.seq}.jpg"
    target_path = STORAGE_DIR / target_filename

    success = cv2.imwrite(str(target_path), frame)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to save image to disk.")

    detector = UniversalRotationDetector()
    initial_sharpness = detector.compute_sharpness(frame)

    return {
        "message": "Reference baseline captured and stored successfully.",
        "filename": target_filename,
        "saved_path": str(target_path),
        "baseline_sharpness_score": initial_sharpness
    }


@app.get("/references")
def list_references():
    """Lists all stored reference images."""
    files = [f.name for f in STORAGE_DIR.glob("*.*") if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    return {"total_references": len(files), "files": sorted(files)}