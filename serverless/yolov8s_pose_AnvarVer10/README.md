# Integrate a custom YOLOv8 pose model into CVAT automatic annotation

This function is a YOLOv8-pose variant of the custom YOLOv8 detector blueprint. It returns CVAT `skeleton` shapes instead of `rectangle` shapes.

## Model file

Put your YOLOv8 pose model in this directory as:

```bash
custom-yolov8-pose.pt
```

If you want a different file name, update `MODEL_PATH` in `function.yaml`.

## Thresholds

CVAT's auto annotation threshold is sent to the Nuclio function as `threshold`.

- Detection threshold: uses the CVAT UI value and filters whole person detections by `result.boxes.conf`.
- Keypoint threshold: defaults to `0.3` via `KEYPOINT_THRESHOLD`. Points below this value are returned with `outside: 1`.

You can tune `KEYPOINT_THRESHOLD` in `function.yaml` without changing the CVAT UI.

## Labels

The included `function.yaml` assumes the standard COCO 17-keypoint YOLOv8-pose order:

```text
nose, left_eye, right_eye, left_ear, right_ear,
left_shoulder, right_shoulder, left_elbow, right_elbow,
left_wrist, right_wrist, left_hip, right_hip,
left_knee, right_knee, left_ankle, right_ankle
```

If your custom model was trained with a different keypoint count or order, update both `sublabels` and the skeleton `svg` in `function.yaml`.

## Deployment

From the CVAT repository root, after adding the model file:

```bash
./serverless/deploy_cpu.sh serverless/custom-yolov8-pose-auto-annotation-cvat-blueprint
```

The deployed function name is `custom-model-yolov8-pose`.
