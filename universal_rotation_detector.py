import cv2
import numpy as np
import argparse
import sys
import os


class UniversalRotationDetector:
    def __init__(self, search_range=180.0, step=1.0, osd_margin=0.10):
        self.search_range = search_range
        self.step = step
        self.osd_margin = osd_margin
        self.ref_histogram = None

    def _compute_orientation_distribution(self, frame: np.ndarray) -> np.ndarray:
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
            return {"status": "INSUFFICIENT EVIDENCE", "rotation_angle": 0.0, "is_rotated": False}

        test_angles = np.arange(-self.search_range, self.search_range + self.step, self.step)
        scores = []

        for alpha in test_angles:
            shift_bins = int(np.round(alpha))
            shifted = np.roll(live_hist, shift_bins)
            score = np.dot(self.ref_histogram, shifted)
            scores.append(score)

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
        is_rotated = abs_tilt >= 2.0  # 2 degree threshold

        return {
            "rotation_angle": round(float(camera_tilt), 1),
            "is_rotated": is_rotated
        }


# ==========================================================
# SIMPLE, USER-FRIENDLY VISUAL SCREEN
# ==========================================================

def create_simple_dashboard(ref_img: np.ndarray, curr_img: np.ndarray, result: dict) -> np.ndarray:
    """
    Creates an ultra-simple, intuitive display that any non-technical operator can understand:
    Left: Reference (How it should look)
    Right: Current (What it looks like now)
    """
    target_height = 460

    def resize_to_height(img):
        h, w = img.shape[:2]
        new_w = int(w * (target_height / h))
        return cv2.resize(img, (new_w, target_height))

    p1 = resize_to_height(ref_img)
    p2 = resize_to_height(curr_img)

    is_rotated = result["is_rotated"]
    tilt = abs(result["rotation_angle"])

    # Human-friendly directions
    if result["rotation_angle"] > 0:
        tilt_side = "RIGHT (Clockwise)"
        fix_action = f"Turn camera {tilt:.0f} deg to the LEFT"
    else:
        tilt_side = "LEFT (Counter-Clockwise)"
        fix_action = f"Turn camera {tilt:.0f} deg to the RIGHT"

    if 175.0 <= tilt <= 180.0:
        tilt_side = "UPSIDE DOWN (180 deg)"
        fix_action = "Flip camera upright"

    # Add friendly bottom labels to panels
    def label_panel(img, title, color):
        # Outline border
        cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), color, 4)
        # Bottom label bar
        cv2.rectangle(img, (0, img.shape[0] - 40), (img.shape[1], img.shape[0]), (0, 0, 0), -1)
        cv2.putText(img, title, (20, img.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return img

    p1 = label_panel(p1, "[ORIGINAL VIEW] Normal Alignment", (0, 200, 0))

    if is_rotated:
        p2 = label_panel(p2, f"[CURRENT VIEW] Tilted {tilt:.0f} deg", (0, 0, 255))
    else:
        p2 = label_panel(p2, "[CURRENT VIEW] Normal Alignment", (0, 200, 0))

    # Combine side-by-side
    body = np.hstack([p1, p2])
    total_w = body.shape[1]

    # Create top notification banner
    header_h = 110
    header = np.full((header_h, total_w, 3), 30, dtype=np.uint8)

    if is_rotated:
        # Red Alert Banner
        cv2.rectangle(header, (15, 15), (260, 95), (0, 0, 220), -1)
        cv2.putText(header, "! CAMERA TILTED", (25, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        # Message
        cv2.putText(header, f"Problem: Camera is tilted {tilt:.0f} deg to the {tilt_side}",
                    (280, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 100, 255), 2)
        cv2.putText(header, f"Action : {fix_action}",
                    (280, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    else:
        # Green All-OK Banner
        cv2.rectangle(header, (15, 15), (260, 95), (0, 180, 0), -1)
        cv2.putText(header, "CAMERA OK", (35, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

        # Message
        cv2.putText(header, "Status: Camera alignment is correct.",
                    (280, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.putText(header, "Action: No action needed.",
                    (280, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 1)

    # Combine Header + Body
    dashboard = np.vstack([header, body])
    return dashboard


def main():
    parser = argparse.ArgumentParser(description="Simple CCTV Rotation Detector")
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

    # 1. Single Image Mode
    if args.current:
        curr_img = cv2.imread(args.current)
        if curr_img is None:
            print(f"[Error] Failed to open current image: {args.current}")
            sys.exit(1)

        result = detector.detect_rotation(curr_img)
        tilt = abs(result["rotation_angle"])

        # Simple terminal message
        print("\n" + "="*45)
        if result["is_rotated"]:
            print(f" [ALERT] CAMERA IS TILTED by {tilt:.0f} degrees!")
        else:
            print(" [OK] Camera is aligned properly.")
        print("="*45 + "\n")

        # Show Simple Screen
        dashboard = create_simple_dashboard(ref_img, curr_img, result)
        cv2.namedWindow("CCTV Camera Monitor", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("CCTV Camera Monitor", dashboard)
        print("Press any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # 2. Live Stream Mode
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

            result = detector.detect_rotation(frame)
            dashboard = create_simple_dashboard(ref_img, frame, result)
            cv2.imshow("CCTV Camera Monitor", dashboard)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()