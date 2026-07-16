import base64
import io
import json
import os

import cv2
import numpy as np
from ultralytics import YOLO


MODEL_PATH = os.getenv("MODEL_PATH", "best.pt")
DEFAULT_THRESHOLD = float(os.getenv("DETECTION_THRESHOLD", "0.25"))
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "960"))


def init_context(context):
    context.logger.info("Initializing tinySmoke_260710... 0%")
    context.user_data.model = YOLO(MODEL_PATH)
    context.logger.info("Initializing tinySmoke_260710... 100%")


def _event_body(event):
    body = event.body
    if isinstance(body, bytes):
        return json.loads(body.decode("utf-8"))
    if isinstance(body, str):
        return json.loads(body)
    return body


def _decode_image(encoded_image):
    image_bytes = base64.b64decode(encoded_image, validate=True)
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode the input image")
    return image


def handler(context, event):
    context.logger.info("Running tinySmoke_260710 inference")

    data = _event_body(event)
    threshold = float(data.get("threshold", DEFAULT_THRESHOLD))
    threshold = min(max(threshold, 0.0), 1.0)
    image = _decode_image(data["image"])

    result = context.user_data.model.predict(
        source=image,
        conf=threshold,
        imgsz=IMAGE_SIZE,
        device="cpu",
        verbose=False,
    )[0]

    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls.item())
            detections.append(
                {
                    "confidence": str(float(box.conf.item())),
                    "label": result.names[class_id],
                    "points": [float(value) for value in box.xyxy[0].tolist()],
                    "type": "rectangle",
                }
            )

    return context.Response(
        body=json.dumps(detections),
        headers={},
        content_type="application/json",
        status_code=200,
    )
