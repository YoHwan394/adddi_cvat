# Copyright (C) 2018-2022 Intel Corporation
# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

import zipfile
from pathlib import Path
from typing import BinaryIO

from datumaro.components.annotation import AnnotationType
from datumaro.components.dataset import StreamDataset
from datumaro.components.transformer import ItemTransform
from datumaro.plugins.data_formats.coco.importer import CocoImporter

from cvat.apps.dataset_manager.bindings import (
    GetCVATDataExtractor,
    NoMediaInAnnotationFileError,
    detect_dataset,
    import_dm_annotations,
)
from cvat.apps.dataset_manager.util import make_zip_archive
from cvat.apps.engine.models import ShapeType
from cvat.apps.engine.pose_bbox import pose_bbox_label_name

from .pose_bbox import UsePoseBboxCompanions
from .registry import dm_env, exporter, importer
from .transformations import EllipsesToMasks


@exporter(name="COCO", ext="ZIP", version="1.0")
def _export_instances(dst_file, temp_dir, instance_data, save_images=False):
    with GetCVATDataExtractor(instance_data, include_images=save_images) as extractor:
        dataset = StreamDataset.from_extractors(extractor, env=dm_env)
        dataset.transform(EllipsesToMasks)
        dataset.export(temp_dir, "coco_instances", save_media=save_images, merge_images=False)

    make_zip_archive(temp_dir, dst_file)


@importer(name="COCO", ext="JSON, ZIP", version="1.0")
def _import_instances(
    src_file: BinaryIO, temp_dir, instance_data, load_data_callback=None, **kwargs
):
    if zipfile.is_zipfile(src_file):
        zipfile.ZipFile(src_file).extractall(temp_dir)
        # We use coco importer because it gives better error message
        detect_dataset(temp_dir, format_name="coco", importer=CocoImporter)
        dataset = StreamDataset.import_from(temp_dir, "coco_instances", env=dm_env)
        if load_data_callback is not None:
            load_data_callback(dataset, instance_data)
        import_dm_annotations(dataset, instance_data)
    else:
        if load_data_callback:
            raise NoMediaInAnnotationFileError()

        tmp_src_file_link = Path(temp_dir) / "annotations" / "default.json"
        tmp_src_file_link.parent.mkdir()
        tmp_src_file_link.symlink_to(src_file.name)
        dataset = StreamDataset.import_from(
            str(tmp_src_file_link.absolute()), "coco_instances", env=dm_env
        )
        import_dm_annotations(dataset, instance_data)


@exporter(name="COCO Keypoints", ext="ZIP", version="1.0")
def _export_keypoints(dst_file, temp_dir, instance_data, save_images=False):
    with GetCVATDataExtractor(instance_data, include_images=save_images) as extractor:
        dataset = StreamDataset.from_extractors(extractor, env=dm_env)
        dataset.transform(EllipsesToMasks)
        dataset = dataset.transform(UsePoseBboxCompanions)
        dataset.export(
            temp_dir, "coco_person_keypoints", save_media=save_images, merge_images=False
        )

    make_zip_archive(temp_dir, dst_file)


def _make_pose_bbox_label_mapping(dataset, instance_data) -> dict[int, int]:
    task_labels = {
        label["name"]: label.get("type", "any")
        for _, label in instance_data.meta[instance_data.META_FIELD]["labels"]
        if "parent" not in label
    }
    label_cat = dataset.categories()[AnnotationType.label]
    label_ids_by_name = {label.name: label_id for label_id, label in enumerate(label_cat.items)}
    bbox_label_mapping = {}

    for label_id, label in enumerate(label_cat.items):
        bbox_label_name = pose_bbox_label_name(label.name)
        if task_labels.get(bbox_label_name) not in {str(ShapeType.RECTANGLE), "any"}:
            continue

        bbox_label_id = label_ids_by_name.get(bbox_label_name)
        if bbox_label_id is None:
            bbox_label_id = label_cat.add(bbox_label_name)
            label_ids_by_name[bbox_label_name] = bbox_label_id

        bbox_label_mapping[label_id] = bbox_label_id

    return bbox_label_mapping


class ConvertPoseBboxAnnotations(ItemTransform):
    # COCO keypoints annotations include bbox, but CVAT stores it as a companion rectangle.
    # If the task has no companion label, keep the old behavior and drop the bbox.
    def __init__(self, extractor, bbox_label_mapping: dict[int, int]):
        super().__init__(extractor)
        self._bbox_label_mapping = bbox_label_mapping

    def transform_item(self, item):
        def convert_annotations():
            annotations = []
            for ann in item.annotations:
                if ann.type == AnnotationType.bbox:
                    bbox_label_id = self._bbox_label_mapping.get(ann.label)
                    if bbox_label_id is not None:
                        annotations.append(ann.wrap(label=bbox_label_id))
                else:
                    annotations.append(ann)

            return annotations

        return item.wrap(annotations=convert_annotations)


@importer(name="COCO Keypoints", ext="JSON, ZIP", version="1.0")
def _import_keypoints(src_file, temp_dir, instance_data, load_data_callback=None, **kwargs):
    if zipfile.is_zipfile(src_file):
        zipfile.ZipFile(src_file).extractall(temp_dir)
        # We use coco importer because it gives better error message
        detect_dataset(temp_dir, format_name="coco", importer=CocoImporter)
        dataset = StreamDataset.import_from(temp_dir, "coco_person_keypoints", env=dm_env)
        dataset = dataset.transform(
            ConvertPoseBboxAnnotations,
            bbox_label_mapping=_make_pose_bbox_label_mapping(dataset, instance_data),
        )
        if load_data_callback is not None:
            load_data_callback(dataset, instance_data)
        import_dm_annotations(dataset, instance_data)
    else:
        if load_data_callback:
            raise NoMediaInAnnotationFileError()

        tmp_src_file_link = Path(temp_dir) / "annotations" / "default.json"
        tmp_src_file_link.parent.mkdir()
        tmp_src_file_link.symlink_to(src_file.name)
        dataset = StreamDataset.import_from(
            str(tmp_src_file_link.absolute()), "coco_person_keypoints", env=dm_env
        )
        dataset = dataset.transform(
            ConvertPoseBboxAnnotations,
            bbox_label_mapping=_make_pose_bbox_label_mapping(dataset, instance_data),
        )
        import_dm_annotations(dataset, instance_data)
