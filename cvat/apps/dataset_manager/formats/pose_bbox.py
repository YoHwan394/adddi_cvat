# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import datumaro as dm

from cvat.apps.engine.pose_bbox import normalize_bbox_points, pose_bbox_label_name


class _SkeletonWithCompanionBbox(dm.Skeleton):
    __slots__ = ("_companion_bbox",)

    def __init__(self, skeleton: dm.Skeleton, companion_bbox: Sequence[float]) -> None:
        super().__init__(
            skeleton.elements,
            id=skeleton.id,
            attributes=skeleton.attributes,
            group=skeleton.group,
            object_id=skeleton.object_id,
            label=skeleton.label,
            z_order=skeleton.z_order,
        )
        object.__setattr__(self, "_companion_bbox", normalize_bbox_points(companion_bbox))

    def get_bbox(self):
        xtl, ytl, xbr, ybr = self._companion_bbox
        return [xtl, ytl, xbr - xtl, ybr - ytl]


def _remap_skeleton(ann: dm.Skeleton, *, label: int, elements: Sequence[dm.Points]) -> dm.Skeleton:
    skeleton = dm.Skeleton(
        list(elements),
        id=ann.id,
        attributes=ann.attributes,
        group=ann.group,
        object_id=ann.object_id,
        label=label,
        z_order=ann.z_order,
    )
    if isinstance(ann, _SkeletonWithCompanionBbox):
        return _SkeletonWithCompanionBbox(skeleton, ann._companion_bbox)

    return skeleton


def use_pose_bbox_companions(
    annotations: Sequence[dm.Annotation], *, bbox_label_to_skeleton_label: dict[int, int]
) -> list[dm.Annotation]:
    bbox_by_group_and_label = {}
    for ann in annotations:
        if ann.type is not dm.AnnotationType.bbox:
            continue

        skeleton_label_id = bbox_label_to_skeleton_label.get(ann.label)
        if skeleton_label_id is not None and ann.group:
            bbox_by_group_and_label[(ann.group, skeleton_label_id)] = ann.points

    converted = []
    for ann in annotations:
        if ann.type is dm.AnnotationType.bbox and ann.label in bbox_label_to_skeleton_label:
            continue

        if ann.type is dm.AnnotationType.skeleton:
            bbox = bbox_by_group_and_label.get((ann.group, ann.label))
            if bbox is not None:
                converted.append(_SkeletonWithCompanionBbox(ann, bbox))
                continue

        converted.append(ann)

    return converted


class UsePoseBboxCompanions(dm.ItemTransform):
    def __init__(self, extractor):
        super().__init__(extractor)
        label_cat = self._extractor.categories().get(dm.AnnotationType.label)
        self._bbox_label_to_skeleton_label = {}
        self._skeleton_label_to_bbox_label = {}

        if label_cat is None:
            return

        label_ids_by_name = {label.name: label_id for label_id, label in enumerate(label_cat.items)}
        for skeleton_label_id, skeleton_label in enumerate(label_cat.items):
            bbox_label_id = label_ids_by_name.get(pose_bbox_label_name(skeleton_label.name))
            if bbox_label_id is not None:
                self._skeleton_label_to_bbox_label[skeleton_label_id] = bbox_label_id
                self._bbox_label_to_skeleton_label[bbox_label_id] = skeleton_label_id

    def transform_item(self, item):
        item_annotations = list(item.annotations)

        return item.wrap(
            annotations=use_pose_bbox_companions(
                item_annotations,
                bbox_label_to_skeleton_label=self._bbox_label_to_skeleton_label,
            )
        )


class RemapPoseBboxCompanionsForYolo(dm.ItemTransform):
    def __init__(self, extractor):
        super().__init__(extractor)
        self._source_categories = self._extractor.categories()
        self._skeleton_label_id_map = {}
        self._label_id_map = {}
        self._categories = self._make_categories()

    @property
    def is_applicable(self) -> bool:
        return bool(self._skeleton_label_id_map)

    def _make_categories(self):
        label_cat = self._source_categories.get(dm.AnnotationType.label)
        point_cat = self._source_categories.get(dm.AnnotationType.points)
        if label_cat is None or point_cat is None:
            return self._source_categories

        new_label_cat = dm.LabelCategories(attributes=set(label_cat.attributes))
        new_point_cat = dm.PointsCategories()

        for skeleton_label_id, skeleton_points in point_cat.items.items():
            skeleton_label = label_cat[skeleton_label_id]
            if label_cat.find(pose_bbox_label_name(skeleton_label.name))[0] is None:
                continue

            new_skeleton_label_id = new_label_cat.add(
                skeleton_label.name,
                attributes=set(skeleton_label.attributes),
            )
            self._skeleton_label_id_map[skeleton_label_id] = new_skeleton_label_id
            self._label_id_map[skeleton_label_id] = new_skeleton_label_id

            for point_label_name in skeleton_points.labels:
                point_label_id, point_label = label_cat.find(point_label_name, skeleton_label.name)
                if point_label_id is None:
                    continue

                new_point_label_id = new_label_cat.add(
                    point_label.name,
                    parent=skeleton_label.name,
                    attributes=set(point_label.attributes),
                )
                self._label_id_map[point_label_id] = new_point_label_id

            new_point_cat.add(
                new_skeleton_label_id,
                labels=skeleton_points.labels,
                joints=skeleton_points.joints,
                positions=skeleton_points.positions,
            )

        if not self.is_applicable:
            return self._source_categories

        return {
            **self._source_categories,
            dm.AnnotationType.label: new_label_cat,
            dm.AnnotationType.points: new_point_cat,
        }

    def categories(self):
        return self._categories

    def transform_item(self, item):
        if not self.is_applicable:
            return item

        annotations = []
        for ann in item.annotations:
            if ann.type is not dm.AnnotationType.skeleton:
                continue

            new_label_id = self._skeleton_label_id_map.get(ann.label)
            if new_label_id is None:
                continue

            elements = [
                element.wrap(label=self._label_id_map[element.label])
                for element in ann.elements
                if element.label in self._label_id_map
            ]
            annotations.append(_remap_skeleton(ann, label=new_label_id, elements=elements))

        return item.wrap(annotations=annotations)


class AddPoseBboxCompanions(dm.ItemTransform):
    def __init__(self, extractor, *, bboxes_by_item: dict[str, list], bbox_label_names: set[str]):
        super().__init__(extractor)
        self._bboxes_by_item = bboxes_by_item
        self._skeleton_label_to_bbox_label = {}

        label_cat = self._extractor.categories().get(dm.AnnotationType.label)
        if label_cat is None:
            return

        label_ids_by_name = {label.name: label_id for label_id, label in enumerate(label_cat.items)}
        for skeleton_label_id, skeleton_label in enumerate(label_cat.items):
            bbox_label_name = pose_bbox_label_name(skeleton_label.name)
            if bbox_label_name not in bbox_label_names:
                continue

            bbox_label_id = label_ids_by_name.get(bbox_label_name)
            if bbox_label_id is None:
                bbox_label_id = label_cat.add(bbox_label_name)
                label_ids_by_name[bbox_label_name] = bbox_label_id

            self._skeleton_label_to_bbox_label[skeleton_label_id] = bbox_label_id

    def transform_item(self, item):
        bboxes_by_label = defaultdict(list)
        for bbox in self._bboxes_by_item.get(item.id, []):
            bboxes_by_label[bbox["label_name"]].append(bbox["points"])

        label_cat = self._extractor.categories().get(dm.AnnotationType.label)
        next_group = max((ann.group for ann in item.annotations), default=0) + 1

        def convert_annotations():
            nonlocal next_group
            annotations = []
            for ann in item.annotations:
                if ann.type is not dm.AnnotationType.skeleton:
                    annotations.append(ann)
                    continue

                bbox_label_id = self._skeleton_label_to_bbox_label.get(ann.label)
                label_name = label_cat[ann.label].name if label_cat is not None else None
                if bbox_label_id is None or not label_name or not bboxes_by_label[label_name]:
                    annotations.append(ann)
                    continue

                group = ann.group or next_group
                if not ann.group:
                    next_group += 1

                xtl, ytl, xbr, ybr = normalize_bbox_points(bboxes_by_label[label_name].pop(0))
                annotations.append(ann.wrap(group=group))
                annotations.append(
                    dm.Bbox(
                        xtl,
                        ytl,
                        xbr - xtl,
                        ybr - ytl,
                        label=bbox_label_id,
                        group=group,
                        z_order=ann.z_order,
                    )
                )

            return annotations

        return item.wrap(annotations=convert_annotations)
