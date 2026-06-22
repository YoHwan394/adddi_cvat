# Copyright (C) 2019-2022 Intel Corporation
# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT
import os.path as osp
from collections.abc import Callable
from glob import glob
from pathlib import Path

from datumaro.components.annotation import AnnotationType
from datumaro.components.dataset import StreamDataset
from datumaro.components.dataset_base import DatasetItem
from pyunpack import Archive
import yaml

from cvat.apps.dataset_manager.bindings import (
    CommonData,
    CVATDataExtractorMixin,
    GetCVATDataExtractor,
    ProjectData,
    convert_cvat_anno_to_dm,
    detect_dataset,
    find_dataset_root,
    import_dm_annotations,
    match_dm_item,
)
from cvat.apps.dataset_manager.util import make_zip_archive

from .registry import dm_env, exporter, importer
from .pose_bbox import (
    AddPoseBboxCompanions,
    RemapPoseBboxCompanionsForYolo,
    use_pose_bbox_companions,
)
from .transformations import EllipsesToMasks, SetKeyframeForEveryTrackShape


def _export_common(
    dst_file: str,
    temp_dir: str,
    instance_data: ProjectData | CommonData,
    format_name: str,
    *,
    save_images: bool = False,
    **kwargs,
):
    with GetCVATDataExtractor(instance_data, include_images=save_images) as extractor:
        dataset = StreamDataset.from_extractors(extractor, env=dm_env)
        dataset.export(temp_dir, format_name, save_media=save_images, **kwargs)

    make_zip_archive(temp_dir, dst_file)


@exporter(name="YOLO", ext="ZIP", version="1.1")
def _export_yolo(*args, **kwargs):
    _export_common(*args, format_name="yolo", **kwargs)


def _import_common(
    src_file,
    temp_dir: str,
    instance_data: ProjectData | CommonData,
    format_name: str,
    *,
    load_data_callback: Callable | None = None,
    import_kwargs: dict | None = None,
    after_import_callback: Callable | None = None,
    **kwargs,
):
    Archive(src_file.name).extractall(temp_dir)

    detect_dataset(temp_dir, format_name=format_name, importer=dm_env.importers.get(format_name))

    detected_sources = dm_env.make_importer(format_name)(temp_dir)

    image_info = {}
    extractor = dm_env.extractors.get(format_name)
    frames = [
        extractor.name_from_path(osp.relpath(p, folder))
        for folder in set(osp.dirname(source["url"]) for source in detected_sources)
        for p in glob(osp.join(folder, "**", "*.txt"), recursive=True)
    ]
    root_hint = find_dataset_root([DatasetItem(id=frame) for frame in frames], instance_data)
    for frame in frames:
        frame_info = None
        try:
            frame_id = match_dm_item(DatasetItem(id=frame), instance_data, root_hint=root_hint)
            frame_info = instance_data.frame_info[frame_id]
        except Exception:  # nosec
            pass
        if frame_info is not None:
            image_info[frame] = (frame_info["height"], frame_info["width"])

    dataset = StreamDataset.import_from(
        temp_dir, format_name, env=dm_env, image_info=image_info, **(import_kwargs or {})
    )
    dataset = dataset.transform(SetKeyframeForEveryTrackShape)
    if after_import_callback is not None:
        dataset = after_import_callback(dataset, temp_dir, image_info)
    if load_data_callback is not None:
        load_data_callback(dataset, instance_data)
    import_dm_annotations(dataset, instance_data)


def _load_yolo_ultralytics_label_names(temp_dir: str) -> dict[int, str]:
    config_path = next(Path(temp_dir).rglob("data.yaml"), None)
    if config_path is None:
        return {}

    with config_path.open(encoding="utf-8") as config_file:
        names = (yaml.safe_load(config_file) or {}).get("names", {})

    if isinstance(names, list):
        return {index: str(name) for index, name in enumerate(names)}
    if isinstance(names, dict):
        return {int(index): str(name) for index, name in names.items()}

    return {}


def _possible_yolo_item_ids(path: Path, temp_dir: str, extractor) -> set[str]:
    aliases = set()
    rel_path = path.relative_to(temp_dir).as_posix()
    aliases.add(extractor.name_from_path(rel_path))

    parts = path.parts
    if "labels" in parts:
        labels_index = parts.index("labels")
        labels_rel = Path(*parts[labels_index + 1 :]).as_posix()
        aliases.add(extractor.name_from_path(labels_rel))
        if len(parts) > labels_index + 2:
            subset_rel = Path(*parts[labels_index + 2 :]).as_posix()
            aliases.add(extractor.name_from_path(subset_rel))

    aliases.add(path.stem)
    return aliases


def _collect_yolo_pose_bboxes(temp_dir: str, image_info: dict[str, tuple[int, int]]) -> dict[str, list]:
    label_names = _load_yolo_ultralytics_label_names(temp_dir)
    if not label_names:
        return {}

    extractor = dm_env.extractors.get("yolo_ultralytics_pose")
    bboxes_by_item = {}

    for path in Path(temp_dir).rglob("*.txt"):
        item_bboxes = []
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue

        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                label_name = label_names[int(parts[0])]
                cx, cy, rw, rh = map(float, parts[1:5])
            except (KeyError, ValueError):
                continue

            item_bboxes.append((label_name, cx, cy, rw, rh))

        if not item_bboxes:
            continue

        for item_id in _possible_yolo_item_ids(path, temp_dir, extractor):
            if item_id not in image_info:
                continue

            height, width = image_info[item_id]
            bboxes_by_item.setdefault(item_id, [])
            for label_name, cx, cy, rw, rh in item_bboxes:
                bboxes_by_item[item_id].append(
                    {
                        "label_name": label_name,
                        "points": [
                            (cx - rw / 2) * width,
                            (cy - rh / 2) * height,
                            (cx + rw / 2) * width,
                            (cy + rh / 2) * height,
                        ],
                    }
                )

    return bboxes_by_item


@importer(name="YOLO", ext="ZIP", version="1.1")
def _import_yolo(*args, **kwargs):
    _import_common(*args, format_name="yolo", **kwargs)


@exporter(name="Ultralytics YOLO Detection", ext="ZIP", version="1.0")
def _export_yolo_ultralytics_detection(*args, **kwargs):
    _export_common(*args, format_name="yolo_ultralytics_detection", **kwargs)


@exporter(name="Ultralytics YOLO Detection Track", ext="ZIP", version="1.0")
def _export_yolo_ultralytics_detection_track(*args, **kwargs):
    _export_common(*args, format_name="yolo_ultralytics_detection", write_track_id=True, **kwargs)


@exporter(name="Ultralytics YOLO Oriented Bounding Boxes", ext="ZIP", version="1.0")
def _export_yolo_ultralytics_oriented_boxes(*args, **kwargs):
    _export_common(*args, format_name="yolo_ultralytics_oriented_boxes", **kwargs)


@exporter(name="Ultralytics YOLO Segmentation", ext="ZIP", version="1.0")
def _export_yolo_ultralytics_segmentation(dst_file, temp_dir, instance_data, *, save_images=False):
    with GetCVATDataExtractor(instance_data, include_images=save_images) as extractor:
        dataset = StreamDataset.from_extractors(extractor, env=dm_env)
        dataset.transform(EllipsesToMasks)
        dataset = dataset.transform("masks_to_polygons")
        dataset.export(temp_dir, "yolo_ultralytics_segmentation", save_media=save_images)

    make_zip_archive(temp_dir, dst_file)


def _make_pose_bbox_converter(instance_data):
    instance_meta = instance_data.meta[instance_data.META_FIELD]
    label_cat = CVATDataExtractorMixin.load_categories(instance_meta["labels"])[AnnotationType.label]
    top_level_labels = {
        label["name"]: label
        for _, label in instance_meta["labels"]
        if "parent" not in label
    }

    bbox_label_to_skeleton_label = {}
    for label_name, label in top_level_labels.items():
        if label.get("type", "any") != "skeleton":
            continue

        bbox_label = top_level_labels.get(f"{label_name}__bbox")
        if bbox_label is None or bbox_label.get("type", "any") not in {"rectangle", "any"}:
            continue

        skeleton_label_id = label_cat.find(label_name)[0]
        bbox_label_id = label_cat.find(bbox_label["name"])[0]
        bbox_label_to_skeleton_label[bbox_label_id] = skeleton_label_id

    def convert_annotations(cvat_frame_anno, label_attrs, map_label, format_name=None, dimension=None):
        annotations = convert_cvat_anno_to_dm(
            cvat_frame_anno,
            label_attrs,
            map_label,
            format_name,
            dimension,
        )
        return use_pose_bbox_companions(
            annotations,
            bbox_label_to_skeleton_label=bbox_label_to_skeleton_label,
        )

    return convert_annotations


@exporter(name="Ultralytics YOLO Pose", ext="ZIP", version="1.0")
def _export_yolo_ultralytics_pose(dst_file, temp_dir, instance_data, *, save_images=False):
    with GetCVATDataExtractor(
        instance_data,
        include_images=save_images,
        convert_annotations=_make_pose_bbox_converter(instance_data),
    ) as extractor:
        yolo_extractor = RemapPoseBboxCompanionsForYolo(extractor)
        if not yolo_extractor.is_applicable:
            yolo_extractor = extractor

        dataset = StreamDataset.from_extractors(yolo_extractor, env=dm_env)
        dataset.export(temp_dir, "yolo_ultralytics_pose", save_media=save_images)

    make_zip_archive(temp_dir, dst_file)


@exporter(name="Ultralytics YOLO Classification", ext="ZIP", version="1.0")
def _export_yolo_ultralytics_classification(*args, **kwargs):
    _export_common(*args, format_name="yolo_ultralytics_classification", **kwargs)


@importer(name="Ultralytics YOLO Detection", ext="ZIP", version="1.0")
def _import_yolo_ultralytics_detection(*args, **kwargs):
    _import_common(*args, format_name="yolo_ultralytics_detection", **kwargs)


@importer(name="Ultralytics YOLO Segmentation", ext="ZIP", version="1.0")
def _import_yolo_ultralytics_segmentation(*args, **kwargs):
    _import_common(*args, format_name="yolo_ultralytics_segmentation", **kwargs)


@importer(name="Ultralytics YOLO Oriented Bounding Boxes", ext="ZIP", version="1.0")
def _import_yolo_ultralytics_oriented_boxes(*args, **kwargs):
    _import_common(*args, format_name="yolo_ultralytics_oriented_boxes", **kwargs)


@importer(name="Ultralytics YOLO Pose", ext="ZIP", version="1.0")
def _import_yolo_ultralytics_pose(src_file, temp_dir, instance_data, **kwargs):
    instance_meta = instance_data.meta[instance_data.META_FIELD]
    categories = CVATDataExtractorMixin.load_categories(instance_meta["labels"])
    point_categories = categories.get(AnnotationType.points)
    label_categories = categories.get(AnnotationType.label)
    true_skeleton_point_labels = {
        label_categories[label_id].name: category.labels
        for label_id, category in point_categories.items.items()
    }
    bbox_label_names = {
        label["name"]
        for _, label in instance_meta["labels"]
        if "parent" not in label and label.get("type", "any") in {"rectangle", "any"}
    }

    _import_common(
        src_file,
        temp_dir,
        instance_data,
        format_name="yolo_ultralytics_pose",
        import_kwargs=dict(skeleton_sub_labels=true_skeleton_point_labels),
        after_import_callback=lambda dataset, _temp_dir, image_info: dataset.transform(
            AddPoseBboxCompanions,
            bboxes_by_item=_collect_yolo_pose_bboxes(_temp_dir, image_info),
            bbox_label_names=bbox_label_names,
        ),
        **kwargs,
    )


@importer(name="Ultralytics YOLO Classification", ext="ZIP", version="1.0")
def _import_yolo_ultralytics_classification(*args, **kwargs):
    _import_common(*args, format_name="yolo_ultralytics_classification", **kwargs)
