# from pathlib import Path

# images = list(Path("C:/workspace/srcipt/raw/img").glob("*.*"))

# print("Total images:", len(images))

"""Load/export a YOLO-format directory into FiftyOne and launch the app.

Why this exists:
- `fo.load_dataset(...)` expects a *dataset name* that already exists in FiftyOne's DB.
- A filesystem path like `C:/workspace/srcipt/auto` must be imported with `Dataset.from_dir(...)`.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import fiftyone as fo
import fiftyone.utils.ultralytics as fou
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import YOLO dataset dir into FiftyOne")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("C:/workspace/srcipt/clean_dataset"),
        help="Directory containing images/, labels/, and dataset.yaml",
    )
    parser.add_argument(
        "--dataset-name",
        default="auto_yolo_dataset",
        help="FiftyOne dataset name saved in local DB",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Ultralytics model path/name",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing FiftyOne dataset with the same --dataset-name first",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for manual Ultralytics fallback prediction",
    )
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="Skip model inference and only open/edit/export dataset",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=None,
        help="Optional: export dataset to YOLOv5 format at this path",
    )
    parser.add_argument(
        "--export-label-field",
        default="ground_truth",
        help="Label field to export (for example: ground_truth or predictions)",
    )
    parser.add_argument(
        "--export-overwrite",
        action="store_true",
        help="Delete --export-dir before export to avoid stale dataset.yaml conflicts",
    )
    return parser.parse_args()


def _collect_detection_classes(dataset: fo.Dataset, label_field: str) -> list[str]:
    """Collect a stable global class list for YOLO export across all splits."""
    labels = dataset.distinct(f"{label_field}.detections.label")
    labels = [str(x) for x in labels if x is not None and str(x).strip()]
    return sorted(set(labels))


def _manual_apply_ultralytics_model(dataset: fo.Dataset, model: YOLO, conf: float) -> None:
    """Version-safe fallback that avoids FiftyOne<->Ultralytics integration mismatches.

    This directly runs YOLO per sample and writes `fo.Detections` to `predictions`.
    """
    for sample in dataset.iter_samples(progress=True, autosave=True):
        result = model.predict(sample.filepath, conf=conf, verbose=False)[0]

        detections = []
        names = result.names or {}
        img_h, img_w = result.orig_shape

        if result.boxes is not None and len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.cpu().numpy()
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()

            for (x1, y1, x2, y2), cls_id, score in zip(xyxy, cls_ids, confs):
                w = max(0.0, x2 - x1)
                h = max(0.0, y2 - y1)
                if w <= 0 or h <= 0:
                    continue

                rel_box = [
                    float(x1 / img_w),
                    float(y1 / img_h),
                    float(w / img_w),
                    float(h / img_h),
                ]
                detections.append(
                    fo.Detection(
                        label=str(names.get(int(cls_id), cls_id)),
                        confidence=float(score),
                        bounding_box=rel_box,
                    )
                )

        sample["predictions"] = fo.Detections(detections=detections)


def _apply_predictions_with_compat(dataset: fo.Dataset, model: YOLO, conf: float) -> None:
    """Apply predictions with best-effort compatibility across package versions."""
    apply_helper = getattr(fou, "apply_ultralytics_model", None)
    if callable(apply_helper):
        try:
            apply_helper(dataset, model, label_field="predictions")
            print("Applied model via fiftyone.utils.ultralytics.apply_ultralytics_model")
            return
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: helper path failed, falling back. Reason: {exc}")

    try:
        dataset.apply_model(model, label_field="predictions")
        print("Applied model via dataset.apply_model")
        return
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: dataset.apply_model failed, using manual fallback. Reason: {exc}")

    _manual_apply_ultralytics_model(dataset, model, conf=conf)
    print("Applied model via manual YOLO->FiftyOne conversion fallback")


def main() -> None:
    args = parse_args()

    dataset_dir = args.dataset_dir
    yaml_path = dataset_dir / "dataset.yaml"

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    if not yaml_path.exists():
        raise FileNotFoundError(f"dataset.yaml not found: {yaml_path}")

    if args.overwrite and fo.dataset_exists(args.dataset_name):
        fo.delete_dataset(args.dataset_name)

    if fo.dataset_exists(args.dataset_name):
        dataset = fo.load_dataset(args.dataset_name)
        print(f"Loaded existing FiftyOne dataset: {args.dataset_name}")
    else:
        dataset = fo.Dataset.from_dir(
            dataset_type=fo.types.YOLOv5Dataset,
            dataset_dir=str(dataset_dir),
            yaml_path=str(yaml_path),
            name=args.dataset_name,
        )
        print(f"Imported YOLO dataset from: {dataset_dir}")

    if args.skip_model:
        print("Skip model inference (--skip-model).")
    else:
        model = YOLO(args.model)
        _apply_predictions_with_compat(dataset, model, conf=args.conf)

    print("Launching FiftyOne app. Edit your dataset, then close the app to continue...")
    session = fo.launch_app(dataset)
    session.wait()

    # Refresh dataset handle so app-side edits are reflected before export.
    dataset.reload()
    print("Dataset reloaded after app session.")

    if args.export_dir is not None:
        if args.export_overwrite and args.export_dir.exists():
            shutil.rmtree(args.export_dir)

        classes = _collect_detection_classes(dataset, args.export_label_field)
        if not classes:
            raise ValueError(
                f"No classes found in label field '{args.export_label_field}'. "
                "Choose another --export-label-field (e.g. ground_truth/predictions)."
            )

        dataset.export(
            export_dir=str(args.export_dir),
            dataset_type=fo.types.YOLOv5Dataset,
            label_field=args.export_label_field,
            classes=classes,
        )
        print(
            f"Export finished: {args.export_dir} "
            f"(label_field={args.export_label_field}, classes={len(classes)})"
        )


if __name__ == "__main__":
    main()

