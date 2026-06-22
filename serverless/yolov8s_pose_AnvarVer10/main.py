import base64
import io
import json
import os

import cv2
import numpy as np
import yaml
from ultralytics import YOLO


MODEL_PATH = os.getenv("MODEL_PATH", "yolov8s_pose_AnvarVer10.pt")
DEFAULT_DETECTION_THRESHOLD = float(os.getenv("DETECTION_THRESHOLD", "0.5"))
DEFAULT_KEYPOINT_THRESHOLD = float(os.getenv("KEYPOINT_THRESHOLD", "0.3"))


def init_context(context):
    context.logger.info("Init YOLOv8 pose context... 0%")

    context.user_data.model_handler = YOLO(MODEL_PATH)

    with open("/opt/nuclio/function.yaml", "rb") as function_file:
        function_config = yaml.safe_load(function_file)
        labels_spec = function_config["metadata"]["annotations"]["spec"]
        context.user_data.labels = json.loads(labels_spec)

    context.logger.info("Init YOLOv8 pose context... 100%")


def _event_body(event):
    body = event.body
    if isinstance(body, bytes):
        return json.loads(body.decode("utf-8"))
    if isinstance(body, str):
        return json.loads(body)
    return body


def _decode_image(image_payload):
    image_buffer = io.BytesIO(base64.b64decode(image_payload))
    image = cv2.imdecode(np.frombuffer(image_buffer.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode image")
    return image


def _to_numpy(tensor):
    if tensor is None:
        return np.empty((0,))
    return tensor.cpu().numpy() if hasattr(tensor, "cpu") else np.asarray(tensor)


def handler(context, event):
    context.logger.info("Run custom YOLOv8 pose model")

    data = _event_body(event)
    detection_threshold = float(data.get("threshold", DEFAULT_DETECTION_THRESHOLD))
    keypoint_threshold = float(data.get("keypoint_threshold", DEFAULT_KEYPOINT_THRESHOLD))
    image = _decode_image(data["image"])

    inference_results = context.user_data.model_handler(image, conf=detection_threshold)
    result = inference_results[0]

    if result.keypoints is None or result.boxes is None:
        return context.Response(
            body=json.dumps([]), headers={}, content_type="application/json", status_code=200
        )

    box_confidences = _to_numpy(result.boxes.conf)
    boxes_xyxy = _to_numpy(result.boxes.xyxy)
    class_ids = _to_numpy(result.boxes.cls).astype(int)
    keypoints_xy = _to_numpy(result.keypoints.xy)
    keypoints_conf = _to_numpy(result.keypoints.conf)

    if keypoints_conf.size == 0:
        keypoints_conf = np.ones(keypoints_xy.shape[:2], dtype=np.float32)

    detections = []
    skeleton_labels = [label for label in context.user_data.labels if label.get("type") == "skeleton"]
    skeleton_labels_by_name = {label["name"]: label for label in skeleton_labels}

    for box_confidence, bbox, class_id, points, scores in zip(
        box_confidences, boxes_xyxy, class_ids, keypoints_xy, keypoints_conf
    ):
        if float(box_confidence) < detection_threshold:
            continue

        model_label = result.names.get(int(class_id), str(int(class_id)))
        if model_label in skeleton_labels_by_name:
            labels = [skeleton_labels_by_name[model_label]]
        elif len(skeleton_labels) == 1:
            labels = skeleton_labels
        else:
            labels = []

        for label in labels:
            elements = []
            for element in label["sublabels"]:
                keypoint_index = element["id"]
                if keypoint_index >= len(points):
                    continue

                score = float(scores[keypoint_index])
                elements.append(
                    {
                        "label": element["name"],
                        "type": "points",
                        "outside": 0 if score >= keypoint_threshold else 1,
                        "points": [
                            float(points[keypoint_index][0]),
                            float(points[keypoint_index][1]),
                        ],
                        "confidence": str(score),
                    }
                )

            if elements and not all(element["outside"] for element in elements):
                detections.append(
                    {
                        "confidence": str(float(box_confidence)),
                        "label": label["name"],
                        "type": "skeleton",
                        "bbox": [float(coord) for coord in bbox[:4]],
                        "elements": elements,
                    }
                )

    return context.Response(
        body=json.dumps(detections), headers={}, content_type="application/json", status_code=200
    )
