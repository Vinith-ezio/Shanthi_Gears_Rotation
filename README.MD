# CCTV Health & Tamper Guard

A Python-based CCTV monitoring utility for detecting camera rotation/tilt and image blur using OpenCV. The project supports two execution modes: a **FastAPI service** for API-based integration and a **local CLI application** for image comparison and live visual monitoring.

## 1. Project Overview

The system evaluates a live camera frame against a reference image to identify potential camera health issues.

### Detection engines

| Engine | Method | Purpose |
|---|---|---|
| Camera Tilt / Rotation | Sobel gradients, orientation histogram, circular correlation | Estimates rotation relative to the reference frame |
| Blur / Focus Clarity | Patch-based Laplacian variance | Detects a reduction in image sharpness |
| Combined Health Evaluation | Rule-based decision logic | Combines tilt and blur results into a health status |

**Important:** These are image-based indicators. A rotation alert does not prove physical tampering, and a blur alert does not prove the lens is dirty. Both require validation against the target camera and environment.

## 2. Project Structure

```text
Shanthi Gears/
│
├── .venv/                         # Python virtual environment
├── __pycache__/                   # Generated Python bytecode
│
├── reference_images/               # Reference images used by the API
│
├── app.py                          # FastAPI application and endpoints
├── universal_rotation_detector.py  # Local CLI and visual monitoring
│
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── .gitignore                      # Git exclusion rules
│
├── image.jpg                       # Local test image
├── images_1.jpg                    # Local test image
├── images_2.jpg                    # Local test image
├── images.jpeg                     # Local test image
├── new_1.jpeg                      # Local test image
├── new_2.jpeg                      # Local test image
├── new.jpeg                        # Local test image
└── ref.jpeg                        # Local reference image
```

The image filenames above reflect the current development folder. They are not automatically used by the API unless copied into `reference_images/` with the expected naming convention or supplied directly to the local CLI.

### File responsibilities

#### `app.py` — API execution

This file provides:

- FastAPI application initialization.
- HTTP request validation through Pydantic.
- Reference image lookup and storage.
- Live webcam or RTSP frame capture.
- Camera tilt and blur evaluation.
- JSON responses containing health status and recommended actions.

Available endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | Redirects to API documentation |
| `POST` | `/save-reference` | Captures and saves a reference frame |
| `POST` | `/check-camera` | Checks a live camera against a reference |
| `GET` | `/references` | Lists saved reference images |

#### `universal_rotation_detector.py` — Local execution

This file provides:

- The `UniversalRotationDetector` class.
- Reference image initialization.
- Rotation and blur detection.
- Command-line argument parsing.
- Image-to-image comparison.
- Live webcam or RTSP monitoring.
- An OpenCV dashboard showing the reference and current frames.

Use this file when developing or testing the algorithm without making HTTP requests.

#### `reference_images/` — Reference storage

The API looks for reference images using the naming pattern:

```text
{nvr}_{channel}_{seq}.jpg
```

Example:

```text
reference_images/
├── NVR01_CH02_001.jpg
├── NVR01_CH02_002.jpg
└── NVR02_CH01_001.jpg
```

The implementation also supports `.jpeg` and `.png` files for explicitly requested reference sequences.

A reference should be captured when the camera is correctly positioned and the image is acceptably sharp. Store a separate reference for each relevant camera and baseline version.

## 3. System Requirements

- Python 3.10 or later recommended.
- OpenCV.
- NumPy.
- FastAPI.
- Uvicorn.
- Pydantic.
- A webcam, accessible image files, or an RTSP camera stream.
- A graphical desktop session for the local OpenCV dashboard.

The exact Python version and package versions should be confirmed against the team's environment and `requirements.txt`.

## 4. Installation

Run these commands from the project root.

### Windows PowerShell

```powershell
# Create a virtual environment if one does not already exist
python -m venv .venv

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip

# Install project dependencies
pip install -r requirements.txt
```

If PowerShell blocks virtual environment activation, use an approved activation method for your environment or run the environment's Python executable directly.

If the dependency file has not yet been prepared, the main direct dependencies inferred from the source imports are:

```text
numpy
opencv-python
fastapi
uvicorn[standard]
pydantic
```

These are inferred from the code imports, not a verified copy of the current `requirements.txt`. Use pinned and tested versions for reproducible deployment.

## 5. Execution Mode A — Run the FastAPI Service

Start the API from the project root:

```powershell
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

Open the interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

Use `--reload` for development only. For deployment, configure a production server process, access controls, logging, and appropriate network exposure.

### 5.1 Save a reference image

Send a `POST` request to `/save-reference`.

Example JSON request:

```json
{
  "nvr": "NVR01",
  "channel": "CH02",
  "seq": "001",
  "stream_url": "rtsp://CAMERA_HOST:554/STREAM_PATH"
}
```

Replace the example stream with your actual camera stream.

Expected behavior:

1. The API attempts to connect to the stream.
2. It captures a frame.
3. It saves the image under `reference_images/`.
4. It returns the filename and baseline sharpness score.

Example reference filename:

```text
reference_images/NVR01_CH02_001.jpg
```

Capture the reference only after confirming the camera's intended orientation, field of view, and focus.

### 5.2 Check camera health

Send a `POST` request to `/check-camera`.

Example JSON request:

```json
{
  "nvr": "NVR01",
  "channel": "CH02",
  "seq": "001",
  "stream_url": "rtsp://CAMERA_HOST:554/STREAM_PATH"
}
```

The API loads the selected reference, captures a current frame, and evaluates both tilt and blur.

The response includes fields such as:

- `is_healthy`
- `overall_status`
- `rotation.is_rotated`
- `rotation.tilt_angle_deg`
- `rotation.confidence`
- `focus_clarity.is_blurry`
- `focus_clarity.live_sharpness_score`
- `focus_clarity.baseline_sharpness_score`
- `focus_clarity.sharpness_drop_pct`
- `action_required`

The exact returned values depend on the images and configured thresholds.

### 5.3 List saved references

Send a `GET` request to:

```text
http://127.0.0.1:8000/references
```

This returns the number and filenames of the supported reference images found in the storage directory.

## 6. Execution Mode B — Run Locally

Use `universal_rotation_detector.py` when testing image pairs or monitoring a camera through an OpenCV window.

### 6.1 Compare two images

```powershell
python universal_rotation_detector.py --ref ref.jpeg --current new.jpeg
```

The application:

1. Loads the reference image.
2. Initializes the detector.
3. Evaluates the current image.
4. Prints a camera health report.
5. Opens a dashboard with the reference and current images.

Press any key to close the dashboard.

### 6.2 Monitor a webcam

```powershell
python universal_rotation_detector.py --ref ref.jpeg --source 0
```

Use `0` for the default webcam. Other numeric indices may select different local cameras.

### 6.3 Monitor an RTSP camera

```powershell
python universal_rotation_detector.py --ref ref.jpeg --source "rtsp://CAMERA_HOST:554/STREAM_PATH"
```

Press `q` to exit the live monitoring window.

### 6.4 Configure the rotation search range

The default search range is ±180 degrees.

For a smaller expected rotation range:

```powershell
python universal_rotation_detector.py --ref ref.jpeg --current new.jpeg --range 45
```

This searches within ±45 degrees. A smaller range can reduce computation, but it will not detect rotations outside that range.

## 7. How the Detection Logic Works

### 7.1 Camera rotation

The rotation engine:

1. Converts the frame to grayscale.
2. Applies Gaussian blur.
3. Calculates horizontal and vertical Sobel gradients.
4. Derives gradient magnitude and orientation.
5. Builds a 360-bin orientation histogram, weighted by gradient magnitude.
6. Smooths and normalizes the histogram.
7. Compares the reference histogram with circularly shifted versions of the current histogram.
8. Selects the best-scoring shift and estimates the rotation angle.

The current implementation flags rotation when the absolute estimated angle is at least approximately 2 degrees.

The returned angle is an estimate relative to the reference image. It is not a direct measurement from a physical angle sensor.

### 7.2 Blur detection

The blur engine:

1. Removes the outer image margins.
2. Divides the remaining image into a 4 × 4 grid.
3. Skips patches dominated by extreme brightness, deep shadows, or low contrast.
4. Applies a small Gaussian filter.
5. Normalizes patch contrast.
6. Calculates the variance of the Laplacian.
7. Uses the median of usable patch scores as the frame sharpness score.
8. Compares the current score against the baseline.

The current implementation flags blur if either:

- The live sharpness score is below `min_sharpness_floor`, configured by default as `50.0`; or
- The live-to-reference sharpness ratio is below `blur_drop_ratio`, configured by default as `0.45`.

These values are initial configuration choices, not universal industrial thresholds. They must be calibrated with representative recordings.

## 8. Configuration Parameters

| Parameter | Default | Meaning |
|---|---:|---|
| `search_range` | `180.0` | Maximum rotation search range in degrees, in each direction |
| `step` | `1.0` | Rotation search interval in degrees |
| `osd_margin` | `0.10` | Fraction of each outer border excluded from processing |
| `blur_drop_ratio` | `0.45` | Relative sharpness threshold |
| `min_sharpness_floor` | `50.0` | Absolute sharpness threshold |

The local script and API currently instantiate the detector with their own defaults or explicit arguments. If you change a parameter, confirm the behavior in both execution modes.

## 9. Important Implementation Notes

### Reference handling

The reference image is required for relative tilt estimation and baseline-relative blur evaluation. A missing, unreadable, or unsuitable reference should be treated as an error.

### Insufficient evidence

If the live image does not contain enough usable gradient information, the rotation estimator can return an insufficient-evidence result.

The API currently needs an additional health-status guard for this case. An unknown rotation result must not automatically be interpreted as a healthy camera.

### Moving scenes

The orientation histogram summarizes gradient directions across the image but does not preserve feature locations. Moving vehicles, changing traffic, scene changes, or camera movement can alter the histogram independently of camera tampering.

For road-facing cameras mounted on moving vehicles, validate the method carefully. Consider temporal confirmation and visual motion estimation; an IMU should be rigidly attached to the camera assembly if it is used to measure camera-specific movement.

### Blur interpretation

A low sharpness score can indicate defocus, motion blur, obstruction, compression artifacts, or a scene with insufficient texture. The algorithm does not independently identify the exact cause.

### Dashboard limitations

The local dashboard visualizes the detector's results. It is a development interface, not a production web dashboard.

## 10. Troubleshooting

| Problem | Checks |
|---|---|
| `ModuleNotFoundError` | Activate the correct virtual environment and install dependencies |
| Reference image not found | Verify the file path or the NVR/channel/sequence naming convention |
| Failed to decode reference | Confirm the image is valid and readable by OpenCV |
| RTSP connection failure | Check the camera address, credentials, network access, stream path, and transport settings |
| No live frames | Check camera availability and stream permissions |
| OpenCV window does not appear | Use a desktop environment with GUI support |
| Incorrect tilt alerts | Validate reference quality, moving-scene behavior, search range, and angle sign convention |
| Frequent blur alerts | Calibrate thresholds against representative sharp and blurred recordings |
| API reports healthy with insufficient evidence | Update the health decision logic so insufficient evidence becomes an explicit unknown/error state |

## 11. Recommended Development Workflow

1. Install the dependencies in the project's virtual environment.
2. Select a known-good reference image.
3. Test known rotations and blur conditions using the local CLI.
4. Verify angle signs and detection behavior against labeled examples.
5. Run the FastAPI service and test all endpoints through `/docs`.
6. Test with actual RTSP streams and representative operating conditions.
7. Add logging, temporal alert confirmation, and explicit insufficient-evidence handling.
8. Validate performance before using alerts operationally.

## 12. Current Scope and Future Improvements

**Currently implemented:**

- Reference image handling.
- Gradient-histogram-based rotation estimation.
- Patch-based sharpness evaluation.
- Local image comparison and live monitoring.
- FastAPI endpoints for reference capture, camera checking, and reference listing.

**Recommended next steps:**

- Extract the shared detector into a dedicated module to avoid duplicated implementations.
- Add a distinct `UNKNOWN` or `INSUFFICIENT_EVIDENCE` health status.
- Add temporal validation and alert recovery hysteresis.
- Calibrate thresholds for each camera type and environment.
- Add camera offline, obstruction, and scene-change detection as separate features if required.
- Add authentication, secret management, access restrictions, and structured logging before production deployment.

## 13. Security Notes

- Do not commit real camera passwords or sensitive RTSP URLs.
- Use environment variables or a secret manager for stream credentials.
- Restrict API access to authorized clients.
- Avoid exposing the development server directly to the public internet.
- Apply authentication, network controls, and appropriate operational logging before production use.

## Summary

`app.py` is the HTTP API entry point, while `universal_rotation_detector.py` is the standalone local execution entry point. Both currently contain the camera tilt and blur detection logic.

This separation supports API integration and local testing, but the duplicated detector code should eventually be consolidated to maintain consistent behavior across both modes.
