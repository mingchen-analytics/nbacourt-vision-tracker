# Coordinate systems

Source of truth: `src/nbatrack/court/geometry.py` and `src/nbatrack/calib/camera.py`.

## Court (world) frame — meters

| Axis | Direction |
|---|---|
| Origin | Center of the court, on the floor |
| +X | Along the sidelines, toward the basket on the **right of the main broadcast camera** |
| +Y | Along the baselines, toward the **far sideline** (away from the main camera) |
| +Z | Up; floor is Z = 0 (right-handed) |

"Left/right basket" is relative to the main camera, not to a team (teams switch ends at halftime).

Top-down plots draw +X to the right and +Y up, so the camera-side sideline is at the bottom, as in the broadcast.

### NBA dimensions used (1 ft = 0.3048 m exactly)

| Element | Value |
|---|---|
| Playing area | X ∈ [−14.3256, 14.3256], Y ∈ [−7.62, 7.62] (94 × 50 ft) |
| Rim centers | (±12.7254, 0, 3.048): 5.25 ft from the baseline, 10 ft high |
| Backboard face | \|X\| = 13.1064 (4 ft from the baseline) |
| Lane | 16 ft wide (\|Y\| ≤ 2.4384), free-throw line at \|X\| = 8.5344 (19 ft from the baseline) |
| Free-throw circle / center circle radius | 6 ft = 1.8288 m |
| Three-point arc | 23.75 ft = 7.239 m from the rim center. Corner lines at \|Y\| = 22 ft = 6.7056 m. The corner line meets the arc at 14.2 ft from the baseline (computed geometrically; the rulebook rounds this to 14 ft). |
| Restricted area | 4 ft = 1.2192 m radius |

Painted lines are 2 in (≈5 cm) wide. Landmark positions use the nominal rulebook reference, so expect a systematic floor of about ±5 cm.

### Landmarks
32 named point landmarks (IDs 0–31, frozen because they are stored in output files) are defined in `court_landmarks()`:
- court corners;
- midcourt line ends;
- center-circle points;
- lane corners;
- free-throw circle apex;
- three-point corner and break points;
- three-point arc apex;
- the two **3D rim centers**, which are not on the floor.

## Camera frame
OpenCV convention: x right, y down, z forward. A world point X maps to Xc = R (X − C), where R maps world → camera and C is the camera center in world meters.

Pixels are in the **source video resolution**, with the origin at the top-left. Integer coordinates are pixel centers (OpenCV convention).

### PTZ angles (degrees)
- pan > 0: the camera turns toward +X.
- tilt > 0: the camera looks down.
- roll: rotation about the optical axis.

With pan = tilt = roll = 0, the camera looks horizontally along +Y. The rotation is R = Rz(roll) · Rx(tilt) · R_base · Rz_world(pan).

## Calibration models (`calib/models.py`)
| Model | Valid for | Notes |
|---|---|---|
| `HomographyCalibration` | floor points only (Z = 0) | Refuses off-floor points: returns NULL/NaN. |
| `CameraCalibration` (free) | 3D | Per-frame pinhole camera; the center may differ per frame. |
| `CameraCalibration` (ptz) | 3D | Fixed center shared across a segment. **This is a hypothesis that is tested, not assumed.** |
