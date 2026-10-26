# CCTV Rotation & Tamper Guard API

A Python-based computer vision system that detects potential CCTV camera roll rotation by comparing a stored reference image with a live CCTV frame.

The system uses image-gradient orientation analysis to estimate the relative rotation angle, identify whether the camera may be tilted, and provide a corrective instruction through a REST API.

## 1. Problem Statement

CCTV cameras installed in office and industrial environments can accidentally rotate due to physical disturbances or improper handling.

When a camera rotates, its viewing image becomes tilted, potentially affecting surveillance coverage and monitoring reliability.

Manually checking every camera is time-consuming, especially when multiple cameras are connected to an NVR.

The objective of this project is to develop an automated system that:

- Stores a reference image captured when the camera is correctly aligned.
- Retrieves a current frame from the CCTV's RTSP stream.
- Compares the reference and live images without feature detection or feature matching.
- Estimates the relative image rotation angle.
- Identifies whether the estimated rotation exceeds a configurable threshold.
- Provides an actionable correction instruction.
- Supports multiple NVRs, camera channels, and reference versions.

## 2. Proposed Solution

The system uses **image-gradient orientation distribution analysis** to estimate the relative rotation between a reference image and a live CCTV frame.

Instead of matching individual keypoints, the algorithm analyzes the distribution of edge-gradient orientations across each image.

### Processing workflow

1. **Reference capture:** Save an image from the correctly aligned CCTV camera.
2. **Image preprocessing:** Convert the image to grayscale and apply Gaussian blur.
3. **Gradient calculation:** Use Sobel operators to calculate horizontal and vertical image gradients.
4. **Orientation distribution:** Calculate gradient magnitudes and orientations, exclude the outer image margins, and construct a weighted orientation histogram.
5. **Rotation estimation:** Compare the reference and live orientation histograms using circular shifts and select the best-scoring alignment.
6. **Rotation classification:** Compare the estimated angle against the configured threshold.
7. **API response:** Return the estimated angle, rotation status, confidence score, and recommended corrective action.

### Workflow diagram

```text
        Correctly Aligned CCTV
                 |
                 v
        Capture Reference Image
                 |
                 v
        Calculate Gradient Histogram
                 |
                 |
Live CCTV --> Capture Current Frame
                 |
                 v
        Calculate Gradient Histogram
                 |
                 v
        Compare Circular Histogram Shifts
                 |
                 v
        Estimate Relative Rotation
                 |
                 v
        Apply Rotation Threshold
                 |
          +------+------+
          |             |
          v             v
       NORMAL        ROTATED
          |             |
          v             v
     No Action     Correction Advice
```

## 3. Key Features

- **Reference-based comparison:** Uses a stored baseline for each NVR and camera channel.
- **No feature matching:** Does not use SIFT, ORB, keypoint descriptors, or descriptor matching.
- **Rotation estimation:** Searches a configurable angular range to estimate relative image rotation.
- **Confidence score:** Returns a histogram-correlation score for the selected rotation.
- **RTSP integration:** Retrieves live frames from compatible CCTV streams.
- **Reference management:** Stores multiple reference versions for different camera channels.
- **REST API:** Exposes endpoints for rotation checks, reference capture, and reference listing.
- **Corrective guidance:** Generates instructions to help the operator realign the camera.

## 4. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core implementation |
| OpenCV | Image processing, Sobel gradients, RTSP frame capture |
| NumPy | Histogram calculations and numerical operations |
| FastAPI | REST API framework |
| Pydantic | Request validation |
| Uvicorn | ASGI server |

## 5. Project Structure

```text
cctv-rotation-guard/
├── app.py
├── requirements.txt
├── README.md
└── reference_images/
    ├── NVR01_CH01_001.jpg
    ├── NVR01_CH02_001.jpg
    └── NVR02_CH01_001.jpg
```

`app.py` contains the rotation detector, RTSP utilities, reference-image management, request schemas, and API endpoints.

`requirements.txt` lists the Python dependencies.

`reference_images/` stores reference images captured from correctly aligned cameras. The directory is created automatically when the application starts.

## 6. Installation

### Prerequisites

- Python 3.10 or newer.
- Access to the CCTV camera or NVR RTSP stream.
- Network connectivity to the camera.
- A reference image captured when the camera is correctly aligned.

### Step 1: Clone the repository

```bash
git clone https://github.com/Vinith-ezio/Shanthi_Gears_Rotation.git
cd shanthi_gears_rotation
```

### Step 2: Create a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Create `requirements.txt`

```text
opencv-python
numpy
fastapi
uvicorn[standard]
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## 7. Run the Application

Start the FastAPI server:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

If the Python file has a different name, replace `main` with the corresponding module name.

Open the interactive API documentation:

- Swagger UI: `http://127.0.0.1:8000/docs`
- OpenAPI schema: `http://127.0.0.1:8000/openapi.json`

For remote access, configure appropriate network access and authentication before exposing the service.

## 8. API Endpoints

### 8.1 Save a reference image

**Endpoint:** `POST /save-reference`

Captures a frame from the live RTSP stream and saves it as the reference image for a camera channel.

Example request:

```json
{
  "nvr": "NVR01",
  "channel": "CH02",
  "seq": "001",
  "stream_url": "rtsp://<CAMERA_HOST>:554/<STREAM_PATH>"
}
```

The RTSP URL must be replaced with the actual stream URL supported by your camera or NVR.

Example response:

```json
{
  "message": "Reference baseline captured successfully.",
  "saved_path": "reference_images/NVR01_CH02_001.jpg",
  "filename": "NVR01_CH02_001.jpg"
}
```

**Important:** Capture the reference only when the camera is correctly aligned.

### 8.2 Check camera rotation

**Endpoint:** `POST /check-rotation`

Loads a stored reference image, retrieves a current CCTV frame, and estimates the relative rotation.

Example request:

```json
{
  "nvr": "NVR01",
  "channel": "CH02",
  "seq": "001",
  "stream_url": "rtsp://<CAMERA_HOST>:554/<STREAM_PATH>"
}
```

The `seq` field is optional. When omitted, the application selects the latest matching reference file according to its current filename-sorting logic.

Example response when a rotation is detected:

```json
{
  "nvr": "NVR01",
  "channel": "CH02",
  "reference_file_used": "NVR01_CH02_001.jpg",
  "is_rotated": true,
  "status": "TILTED",
  "tilt_angle_deg": -5.0,
  "action_required": "Turn camera 5° to the RIGHT (Clockwise) to align.",
  "match_confidence": 0.85
}
```

The numerical values above are illustrative, not measured results.

### 8.3 List reference images

**Endpoint:** `GET /references`

Returns the stored reference-image filenames.

Example response:

```json
{
  "total_references": 2,
  "files": [
    "NVR01_CH01_001.jpg",
    "NVR01_CH02_001.jpg"
  ]
}
```

## 9. Configuration

The current detector is initialized with the following parameters:

| Parameter | Current value | Purpose |
|---|---:|---|
| `search_range` | 180 degrees | Angular search range in either direction |
| `step` | 1 degree | Rotation-search interval |
| `osd_margin` | 0.10 | Excludes the outer 10% margins during orientation analysis |
| Rotation threshold | 2 degrees | Classifies an estimate of at least 2 degrees in magnitude as rotated |
| Histogram bins | 360 | Represents gradient orientations |
| Gaussian blur | 5 × 5 | Reduces image noise before gradient calculation |

These are the defaults in the current implementation. They should be tuned and validated against the target CCTV cameras and recorded footage.

## 10. Important Technical Limitations

This project is a rotation-detection prototype and requires experimental validation before production deployment.

### Moving people and vehicles

Moving objects introduce additional image gradients and may influence the orientation histogram. The current implementation does not explicitly detect or mask people, vehicles, or moving objects.

### Wall decorations and environmental changes

Balloons, banners, furniture, lighting changes, and other scene modifications can change the gradient distribution. The algorithm may consequently report an incorrect angle.

### Rotation angle accuracy

The estimated angle is derived from histogram correlation rather than direct geometric correspondence. Accuracy depends on scene structure, image content, and the uniqueness of the orientation distribution.

### Confidence interpretation

The returned `confidence` is a correlation score. It is not a calibrated probability that the camera has rotated correctly, and a production system should establish a confidence threshold using validation data.

### Temporal validation

The current implementation analyzes one live frame per request. It does not yet validate a suspected rotation across consecutive frames.

### Camera movement versus image rotation

The detector estimates relative image rotation. It does not reliably distinguish physical roll rotation from every possible scene change or camera movement.

### Error and direction validation

The angle sign, corrective direction, and upside-down-camera classification should be tested using images with known clockwise, counterclockwise, and 180-degree rotations.

## 11. Testing and Validation Plan

Before production deployment, evaluate the system with recorded CCTV footage and controlled image transformations.

Recommended test cases:

1. Reference and live images with no rotation.
2. Clockwise rotations of 2, 5, 10, and 15 degrees.
3. Counterclockwise rotations of 2, 5, 10, and 15 degrees.
4. A 180-degree rotation.
5. People walking through the scene.
6. Vehicles moving through the scene.
7. Balloons, banners, and other wall decorations.
8. Changes in brightness and illumination.
9. Partial obstruction of the scene.
10. Images with insufficient structural information.

Measure:

- Rotation angle error against the known angle.
- False-positive rate when the camera has not rotated.
- False-negative rate when the camera has rotated.
- Detection confidence and failure rate under environmental changes.
- Processing time per frame.

Do not assume a target accuracy until it has been measured using representative footage.

## 12. Security Considerations

- Do not commit RTSP credentials, passwords, tokens, or private camera addresses to GitHub.
- Use environment variables or a secure configuration system for sensitive connection details.
- Restrict access to the API using authentication and network controls.
- Use HTTPS or a trusted reverse proxy when accessing the service remotely.
- Validate NVR and channel identifiers before constructing filesystem paths.
- Avoid exposing internal network details or credentials in API error responses.

## 13. Future Improvements

Potential improvements include:

- Temporal validation across multiple frames.
- Stable architectural-region selection.
- Robust handling of temporary decorations and moving objects.
- Confidence calibration and uncertainty reporting.
- Adjustable rotation thresholds.
- Camera-specific configuration.
- Logging and historical rotation alerts.
- Automated evaluation on a labeled CCTV test dataset.
- Dashboard integration for camera status and corrective instructions.

## 14. Conclusion

CCTV Rotation & Tamper Guard is designed to investigate whether a camera's image orientation has changed relative to a correctly aligned reference image.

The current implementation uses image-gradient orientation distributions and histogram correlation instead of feature detection and feature matching.

The next development priority is to validate the angle estimates and false-alarm behavior on realistic office and industrial CCTV footage before relying on the system for operational alerts.

**Project status:** Prototype — accuracy, environmental robustness, and production readiness remain to be validated.
