import cv2
import numpy as np
import argparse
import sys
import os


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

        # 4x4 grid (16 patches)
        grid_y, grid_x = 4, 4
        patch_h, patch_w = h // grid_y, w // grid_x
        patch_scores = []

        for i in range(grid_y):
            for j in range(grid_x):
                patch = inner[i * patch_h:(i + 1) * patch_h, j * patch_w:(j + 1) * patch_w]

                # 1. Skip patches dominated by extreme glare (>245) or deep shadows (<10)
                if np.mean(patch >= 245) > 0.35 or np.mean(patch <= 10) > 0.70:
                    continue

                std_dev = np.std(patch)
                if std_dev < 8.0:
                    # Skip flat surfaces (e.g. smooth floor patches)
                    continue

                # 2. Denoise: 3x3 Gaussian blur removes digital sensor/ISO noise
                blurred = cv2.GaussianBlur(patch, (3, 3), 0)

                # 3. Contrast Normalization: Makes the patch lighting-invariant
                # Scaling by standard deviation ensures dim lighting does NOT drop variance
                norm_patch = ((blurred - np.mean(blurred)) / (std_dev + 1e-5)) * 50.0

                # 4. Laplacian Operator (2nd spatial derivative)
                lap = cv2.Laplacian(norm_patch, cv2.CV_64F, ksize=3)

                # 5. Trim extreme 1% outlier spikes (e.g. welding arcs, glints)
                lap_abs = np.abs(lap)
                cap_val = np.percentile(lap_abs, 99)
                lap_trimmed = np.clip(lap, -cap_val, cap_val)

                patch_scores.append(float(np.var(lap_trimmed)))

        if not patch_scores:
            # Fallback to global normalized Laplacian if all patches were skipped
            std_dev = max(1.0, np.std(inner))
            norm_inner = ((inner - np.mean(inner)) / std_dev) * 50.0
            return round(float(cv2.Laplacian(norm_inner, cv2.CV_64F).var()), 2)

        # Median patch score rejects localized motion from passing workers/forklifts
        return round(float(np.median(patch_scores)), 2)

    def _compute_orientation_distribution(self, frame: np.ndarray) -> np.ndarray:
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

        # --- 1. INDUSTRIAL BLUR EVALUATION ---
        live_sharpness = self.compute_sharpness(live_frame)
        sharpness_ratio = live_sharpness / (self.ref_sharpness + 1e-5)
        sharpness_drop_pct = max(0.0, round((1.0 - sharpness_ratio) * 100, 1))

        # Flag blur if sharpness dropped > 45% compared to baseline OR hit absolute floor
        is_blurry = (live_sharpness < self.min_sharpness_floor) or (sharpness_ratio < self.blur_drop_ratio)

        # --- 2. ROTATION EVALUATION ---
        live_hist = self._compute_orientation_distribution(live_frame)
        if live_hist is None:
            return {
                "rotation_angle": 0.0,
                "is_rotated": False,
                "is_blurry": is_blurry,
                "sharpness_drop_pct": sharpness_drop_pct,
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
            "insufficient_evidence": False
        }


# ==========================================================
# USER-FRIENDLY VISUAL SCREEN
# ==========================================================

def create_simple_dashboard(ref_img: np.ndarray, curr_img: np.ndarray, result: dict) -> np.ndarray:
    target_height = 460

    def resize_to_height(img):
        h, w = img.shape[:2]
        new_w = int(w * (target_height / h))
        return cv2.resize(img, (new_w, target_height))

    p1 = resize_to_height(ref_img)
    p2 = resize_to_height(curr_img)

    is_rotated = result["is_rotated"]
    is_blurry = result["is_blurry"]
    tilt = abs(result["rotation_angle"])

    if result["rotation_angle"] > 0:
        tilt_side = "RIGHT (Clockwise)"
        tilt_fix = f"Turn camera {tilt:.0f} deg to the LEFT"
    else:
        tilt_side = "LEFT (Counter-Clockwise)"
        tilt_fix = f"Turn camera {tilt:.0f} deg to the RIGHT"

    if 175.0 <= tilt <= 180.0:
        tilt_side = "UPSIDE DOWN (180 deg)"
        tilt_fix = "Flip camera upright"

    has_issue = is_rotated or is_blurry
    curr_border_color = (0, 0, 255) if has_issue else (0, 200, 0)

    def label_panel(img, title, color):
        cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), color, 4)
        cv2.rectangle(img, (0, img.shape[0] - 40), (img.shape[1], img.shape[0]), (0, 0, 0), -1)
        cv2.putText(img, title, (20, img.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return img

    p1 = label_panel(p1, "[ORIGINAL VIEW] Baseline", (0, 200, 0))

    if has_issue:
        status_sub = []
        if is_rotated: status_sub.append(f"Tilted {tilt:.0f} deg")
        if is_blurry: status_sub.append("Blurry")
        p2 = label_panel(p2, f"[CURRENT VIEW] {', '.join(status_sub)}", curr_border_color)
    else:
        p2 = label_panel(p2, "[CURRENT VIEW] Normal Alignment & Sharp", (0, 200, 0))

    body = np.hstack([p1, p2])
    total_w = body.shape[1]

    header_h = 120
    header = np.full((header_h, total_w, 3), 30, dtype=np.uint8)

    if has_issue:
        if is_rotated and is_blurry:
            badge_text = "! TILTED & BLURRY"
            prob_text = f"Problems: Tilted {tilt:.0f} deg to the {tilt_side} AND lens is blurry."
            act_text = f"Actions : {tilt_fix}. Clean lens or adjust focus."
        elif is_rotated:
            badge_text = "! CAMERA TILTED"
            prob_text = f"Problem : Camera is tilted {tilt:.0f} deg to the {tilt_side}."
            act_text = f"Action  : {tilt_fix}."
        else:
            badge_text = "! LENS BLURRY"
            prob_text = f"Problem : Image lost {result['sharpness_drop_pct']}% sharpness (out of focus or dirty)."
            act_text = "Action  : Clean camera lens or adjust focus ring."

        cv2.rectangle(header, (15, 15), (290, 105), (0, 0, 220), -1)
        cv2.putText(header, badge_text, (25, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2)
        cv2.putText(header, prob_text, (310, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 120, 255), 2)
        cv2.putText(header, act_text, (310, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
    else:
        cv2.rectangle(header, (15, 15), (260, 105), (0, 180, 0), -1)
        cv2.putText(header, "CAMERA OK", (35, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
        cv2.putText(header, "Status: Camera alignment is correct and lens is sharp.",
                    (280, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.putText(header, "Action: No action needed.",
                    (280, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 1)

    return np.vstack([header, body])


def main():
    parser = argparse.ArgumentParser(description="CCTV Rotation & Lens Blur Detector")
    parser.add_argument("--ref", type=str, required=True, help="Path to reference image")
    parser.add_argument("--current", type=str, default=None, help="Current image to verify")
    parser.add_argument("--source", type=str, default=None, help="Camera index ('0') or RTSP stream")
    parser.add_argument("--range", type=float, default=180.0, help="Search range (default: 180.0)")
    args = parser.parse_args()

    ref_img = cv2.imread(args.ref)
    if ref_img is None:
        print(f"[Error] Failed to open reference image: {args.ref}")
        sys.exit(1)

    detector = UniversalRotationDetector(search_range=args.range, step=1.0, osd_margin=0.10)
    detector.set_reference(ref_img)

    if args.current:
        curr_img = cv2.imread(args.current)
        if curr_img is None:
            print(f"[Error] Failed to open current image: {args.current}")
            sys.exit(1)

        result = detector.detect_status(curr_img)
        tilt = abs(result["rotation_angle"])

        print("\n" + "=" * 45)
        print(" INDUSTRIAL CAMERA HEALTH REPORT")
        print("=" * 45)
        print(f" Tilted?        : {'YES (' + str(tilt) + ' deg)' if result['is_rotated'] else 'NO'}")
        print(f" Blurry/Dirty?  : {'YES (Sharpness dropped by ' + str(result['sharpness_drop_pct']) + '%)' if result['is_blurry'] else 'NO'}")
        print(f" Live Score     : {result.get('live_sharpness', 'N/A')}")
        print(f" Baseline Score : {result.get('ref_sharpness', 'N/A')}")
        print("=" * 45 + "\n")

        dashboard = create_simple_dashboard(ref_img, curr_img, result)
        cv2.namedWindow("CCTV Camera Monitor", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("CCTV Camera Monitor", dashboard)
        print("Press any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif args.source:
        src = int(args.source) if args.source.isdigit() else args.source
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            print(f"[Error] Failed to open stream: {args.source}")
            sys.exit(1)

        print("[Streaming] Monitoring camera. Press 'q' to quit.")
        cv2.namedWindow("CCTV Camera Monitor", cv2.WINDOW_AUTOSIZE)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = detector.detect_status(frame)
            dashboard = create_simple_dashboard(ref_img, frame, result)
            cv2.imshow("CCTV Camera Monitor", dashboard)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()