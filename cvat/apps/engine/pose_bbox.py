# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

from collections.abc import Sequence

POSE_BBOX_LABEL_SUFFIX = "__bbox"


def pose_bbox_label_name(label_name: str) -> str:
    return f"{label_name}{POSE_BBOX_LABEL_SUFFIX}"


def normalize_bbox_points(points: Sequence[float]) -> list[float]:
    if len(points) != 4:
        raise ValueError("Pose bbox must have exactly 4 coordinates")

    xtl, ytl, xbr, ybr = map(float, points)
    return [min(xtl, xbr), min(ytl, ybr), max(xtl, xbr), max(ytl, ybr)]
