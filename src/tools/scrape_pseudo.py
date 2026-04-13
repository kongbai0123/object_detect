import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import uuid
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from tqdm import tqdm
from pipeline_notice import print_pipeline_notice


# =========================================================================
# Utility functions
# =========================================================================

def _md5(path: str) -> Optional[str]:
    """Compute file MD5 for deduplication."""
    try:
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return None


def _safe_link(src: str, dest: str) -> None:
    """Prefer hardlink, fallback to copy across filesystems."""
    try:
        os.link(src, dest)
    except (OSError, NotImplementedError):
        shutil.copy2(src, dest)


def _bbox_area_ratio(box_xyxy: Sequence[float], img_w: int, img_h: int) -> float:
    """bbox area ratio in [0, 1]."""
    x1, y1, x2, y2 = box_xyxy
    box_area = max(0.0, (x2 - x1) * (y2 - y1))
    img_area = img_w * img_h
    return float(box_area / img_area) if img_area > 0 else 0.0


def _iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    a_area = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    b_area = max(0.0, (bx2 - bx1) * (by2 - by1))
    denom = a_area + b_area - inter
    return float(inter / denom) if denom > 0 else 0.0


def _sequence_key_and_frame_idx(path: str) -> Tuple[str, Optional[int]]:
    """Infer sequence key and frame index from filename.

    Examples:
      cam1_000123.jpg -> (cam1, 123)
      sceneA-frame-45.png -> (sceneA-frame, 45)
      image.jpg -> (image, None)
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    m = re.match(r"^(.*?)[_\-]?(\d+)$", stem)
    if not m:
        return stem, None
    return m.group(1) or stem, int(m.group(2))


@dataclass
class CandidateBox:
    cls_id: int
    conf: float
    xyxy: Tuple[float, float, float, float]
    area_ratio: float


@dataclass
class ScoredImage:
    path: str
    score: float
    max_conf: float
    boxes: List[CandidateBox]
    h_img: int
    w_img: int


class Level2DataScraper:
    def __init__(self, download_dir: Optional[str] = None, hard_neg_dir: Optional[str] = None):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if download_dir is None:
            download_dir = os.path.normpath(os.path.join(script_dir, "../data/1_raw"))
        if hard_neg_dir is None:
            hard_neg_dir = os.path.normpath(os.path.join(script_dir, "../data/1_rawless"))

        self.download_dir = download_dir
        self.hard_neg_dir = hard_neg_dir
        os.makedirs(self.download_dir, exist_ok=True)
        os.makedirs(self.hard_neg_dir, exist_ok=True)

        self._hash_db_path = os.path.join(self.download_dir, ".hash_db.json")
        self._hash_db = self._load_hash_db()

    def _load_hash_db(self) -> set:
        if os.path.exists(self._hash_db_path):
            try:
                with open(self._hash_db_path, "r", encoding="utf-8") as f:
                    return set(json.load(f))
            except Exception:
                pass
        return set()

    def _save_hash_db(self) -> None:
        with open(self._hash_db_path, "w", encoding="utf-8") as f:
            json.dump(list(self._hash_db), f)

    def _add_image(self, src_path: str, prefix: str = "img") -> bool:
        h = _md5(src_path)
        if h is None or h in self._hash_db:
            return False
        dest = os.path.join(self.download_dir, f"{prefix}_{os.path.basename(src_path)}")
        if os.path.exists(dest):
            return False
        _safe_link(src_path, dest)
        self._hash_db.add(h)
        return True

    def fetch_openimages(self, classes: Sequence[str], max_samples: int = 100, splits: Sequence[str] = ("validation", "train")) -> None:
        import fiftyone.zoo as foz

        count = 0
        for split in splits:
            print(f" [OpenImages/{split}] classes={classes}, max={max_samples}")
            try:
                dataset = foz.load_zoo_dataset(
                    "open-images-v7",
                    split=split,
                    label_types=["detections"],
                    classes=list(classes),
                    max_samples=max_samples,
                    dataset_name=f"oi_{split}_{uuid.uuid4().hex[:8]}",
                )
                for sample in tqdm(dataset, desc=f"[OpenImages/{split}]"):
                    if self._add_image(sample.filepath, prefix="oi"):
                        count += 1
            except Exception as e:
                print(f" OpenImages/{split} failed: {e}")
        self._save_hash_db()
        print(f" OpenImages complete, added {count} images")

    def fetch_coco(self, classes: Sequence[str], max_samples: int = 100, splits: Sequence[str] = ("val", "train")) -> None:
        import fiftyone.zoo as foz

        count = 0
        for split in splits:
            print(f" [COCO/{split}] classes={classes}, max={max_samples}")
            try:
                dataset = foz.load_zoo_dataset(
                    "coco-2017",
                    split=split,
                    label_types=["detections"],
                    classes=list(classes),
                    max_samples=max_samples,
                    dataset_name=f"coco_{split}_{uuid.uuid4().hex[:8]}",
                )
                for sample in tqdm(dataset, desc=f"[COCO/{split}]"):
                    if self._add_image(sample.filepath, prefix="coco"):
                        count += 1
            except Exception as e:
                print(f" COCO/{split} failed: {e}")
        self._save_hash_db()
        print(f" COCO complete, added {count} images")

    def _compute_image_score(
        self,
        valid_boxes: List[CandidateBox],
        alpha: float,
        beta: float,
        gamma: float,
    ) -> float:
        if not valid_boxes:
            return 0.0
        max_conf = max(b.conf for b in valid_boxes)
        mean_area = sum(b.area_ratio for b in valid_boxes) / len(valid_boxes)
        obj_count = len(valid_boxes)
        return alpha * max_conf + beta * mean_area + gamma * obj_count

    def _temporal_consistency_filter(
        self,
        scored: List[ScoredImage],
        min_consecutive_hits: int,
    ) -> Tuple[List[ScoredImage], List[str]]:
        if min_consecutive_hits <= 1:
            return scored, []

        groups: Dict[str, List[Tuple[Optional[int], ScoredImage]]] = defaultdict(list)
        for item in scored:
            key, frame_idx = _sequence_key_and_frame_idx(item.path)
            groups[key].append((frame_idx, item))

        keep: List[ScoredImage] = []
        drop_paths: List[str] = []

        for _, entries in groups.items():
            with_idx = [x for x in entries if x[0] is not None]
            no_idx = [x for x in entries if x[0] is None]

            # If frame indices not inferable, keep as-is.
            for _, item in no_idx:
                keep.append(item)

            if not with_idx:
                continue

            with_idx.sort(key=lambda x: x[0])
            run: List[ScoredImage] = []
            prev_idx: Optional[int] = None
            for idx, item in with_idx:
                if prev_idx is None or idx == prev_idx + 1:
                    run.append(item)
                else:
                    if len(run) >= min_consecutive_hits:
                        keep.extend(run)
                    else:
                        drop_paths.extend(x.path for x in run)
                    run = [item]
                prev_idx = idx

            if len(run) >= min_consecutive_hits:
                keep.extend(run)
            else:
                drop_paths.extend(x.path for x in run)

        return keep, drop_paths

    def filter_with_model(
        self,
        model_path: Optional[str] = None,
        conf: float = 0.25,
        batch: int = 16,
        target_classes: Optional[Iterable[int]] = None,
        min_bbox_ratio: float = 0.005,
        top_k: float = 0.7,
        save_pseudo_labels: bool = True,
        alpha: float = 0.7,
        beta: float = 0.2,
        gamma: float = 0.1,
        min_consecutive_hits: int = 1,
        hard_low_conf_min: float = 0.1,
        hard_low_conf_max: float = 0.4,
        conflict_iou: float = 0.6,
        pseudo_conf: float = 0.35,
    ) -> None:
        from ultralytics import YOLO
        import cv2

        if model_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            root = os.path.normpath(os.path.join(script_dir, ".."))
            candidates = glob.glob(os.path.join(root, "weight", "**", "best.pt"), recursive=True)
            candidates = [p for p in candidates if "cls_" not in p.replace("\\", "/")]
            if not candidates:
                print(" No detection weight found: expected */weight/**/best.pt")
                return
            model_path = sorted(candidates, key=os.path.getmtime)[-1]

        print(f"\n [Model Filter] model={model_path}")
        print(f"   batch={batch} conf={conf} top_k={top_k:.0%}")
        print(
            "   scoring weights: "
            f"alpha={alpha}, beta={beta}, gamma={gamma}; temporal={min_consecutive_hits}"
        )

        model = YOLO(model_path)

        img_paths = (
            glob.glob(os.path.join(self.download_dir, "*.jpg"))
            + glob.glob(os.path.join(self.download_dir, "*.jpeg"))
            + glob.glob(os.path.join(self.download_dir, "*.png"))
        )
        if not img_paths:
            print(" No images in download_dir for filtering")
            return

        target_set = set(target_classes) if target_classes else None
        scored: List[ScoredImage] = []
        hard_neg_buckets: Dict[str, List[str]] = {
            "no_detection": [],
            "low_confidence": [],
            "conflicting_detection": [],
            "temporal_inconsistent": [],
            "below_top_k": [],
        }

        for i in tqdm(range(0, len(img_paths), batch), desc="[Batch Inference]"):
            batch_paths = img_paths[i : i + batch]
            results = model.predict(batch_paths, conf=min(conf, hard_low_conf_min), verbose=False)

            for img_path, r in zip(batch_paths, results):
                if len(r.boxes) == 0:
                    hard_neg_buckets["no_detection"].append(img_path)
                    continue

                img = cv2.imread(img_path)
                if img is None:
                    hard_neg_buckets["no_detection"].append(img_path)
                    continue
                h_img, w_img = img.shape[:2]

                raw_max_conf = max(float(box.conf) for box in r.boxes)
                if hard_low_conf_min < raw_max_conf < hard_low_conf_max:
                    hard_neg_buckets["low_confidence"].append(img_path)

                valid_boxes: List[CandidateBox] = []
                conflicting = False

                parsed_boxes: List[CandidateBox] = []
                for box in r.boxes:
                    cls_id = int(box.cls)
                    xyxy = tuple(float(v) for v in box.xyxy[0].tolist())
                    area_ratio = _bbox_area_ratio(xyxy, w_img, h_img)
                    parsed_boxes.append(
                        CandidateBox(cls_id=cls_id, conf=float(box.conf), xyxy=xyxy, area_ratio=area_ratio)
                    )

                for idx_a in range(len(parsed_boxes)):
                    for idx_b in range(idx_a + 1, len(parsed_boxes)):
                        a = parsed_boxes[idx_a]
                        b = parsed_boxes[idx_b]
                        if a.cls_id != b.cls_id and _iou_xyxy(a.xyxy, b.xyxy) > conflict_iou:
                            conflicting = True
                            break
                    if conflicting:
                        break

                if conflicting:
                    hard_neg_buckets["conflicting_detection"].append(img_path)

                for pb in parsed_boxes:
                    if target_set is not None and pb.cls_id not in target_set:
                        continue
                    if pb.conf < conf:
                        continue
                    if pb.area_ratio < min_bbox_ratio:
                        continue
                    valid_boxes.append(pb)

                if not valid_boxes:
                    hard_neg_buckets["no_detection"].append(img_path)
                    continue

                score = self._compute_image_score(valid_boxes, alpha=alpha, beta=beta, gamma=gamma)
                scored.append(
                    ScoredImage(
                        path=img_path,
                        score=score,
                        max_conf=max(b.conf for b in valid_boxes),
                        boxes=valid_boxes,
                        h_img=h_img,
                        w_img=w_img,
                    )
                )

        # Temporal consistency
        scored, temporal_drop = self._temporal_consistency_filter(scored, min_consecutive_hits=min_consecutive_hits)
        hard_neg_buckets["temporal_inconsistent"].extend(temporal_drop)

        scored.sort(key=lambda x: x.score, reverse=True)
        keep_n = max(1, int(len(scored) * top_k)) if scored else 0
        keep_set = scored[:keep_n]
        discard_set = scored[keep_n:]
        hard_neg_buckets["below_top_k"].extend(x.path for x in discard_set)

        moved_count = 0
        all_hard_paths = []
        for _, paths in hard_neg_buckets.items():
            all_hard_paths.extend(paths)
        for img_path in sorted(set(all_hard_paths)):
            dest = os.path.join(self.hard_neg_dir, os.path.basename(img_path))
            if os.path.exists(img_path) and not os.path.exists(dest):
                try:
                    os.rename(img_path, dest)
                    moved_count += 1
                except Exception:
                    pass

        summary_path = os.path.join(self.hard_neg_dir, "hard_negative_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({k: sorted(set(v)) for k, v in hard_neg_buckets.items()}, f, ensure_ascii=False, indent=2)

        label_count = 0
        if save_pseudo_labels:
            labels_dir = os.path.join(self.download_dir, "labels")
            os.makedirs(labels_dir, exist_ok=True)
            for item in keep_set:
                name = os.path.splitext(os.path.basename(item.path))[0]
                label_path = os.path.join(labels_dir, f"{name}.txt")
                with open(label_path, "w", encoding="utf-8") as f:
                    for box in item.boxes:
                        if box.conf < pseudo_conf:
                            continue
                        x1, y1, x2, y2 = box.xyxy
                        cx = ((x1 + x2) / 2) / item.w_img
                        cy = ((y1 + y2) / 2) / item.h_img
                        bw = (x2 - x1) / item.w_img
                        bh = (y2 - y1) / item.h_img
                        f.write(f"{box.cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                label_count += 1

        print("\n Filtering completed")
        print(f"    Kept images: {len(keep_set)}")
        print(f"    Hard negatives moved: {moved_count} -> {self.hard_neg_dir}")
        for k, v in hard_neg_buckets.items():
            print(f"      - {k}: {len(set(v))}")
        if save_pseudo_labels:
            print(f"    Pseudo labels: {label_count}")
        print(f"    Hard negative summary: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Level-2 quality image scraping pipeline")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_dir = os.path.normpath(os.path.join(script_dir, "../data/1_raw"))

    parser.add_argument("--source", type=str, default="openimages", choices=["openimages", "coco", "all"])
    parser.add_argument("--dir", type=str, default=default_dir)
    parser.add_argument("--max", type=int, default=100)
    parser.add_argument("--classes", type=str, nargs="+", default=["Car door open"])
    parser.add_argument("--splits", type=str, nargs="+", default=["validation"])

    parser.add_argument("--filter_model", action="store_true")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--top_k", type=float, default=0.7)
    parser.add_argument("--min_bbox", type=float, default=0.005)
    parser.add_argument("--target_classes", type=int, nargs="*", default=None)

    parser.add_argument("--score_alpha", type=float, default=0.7)
    parser.add_argument("--score_beta", type=float, default=0.2)
    parser.add_argument("--score_gamma", type=float, default=0.1)
    parser.add_argument("--min_consecutive_hits", type=int, default=1)

    parser.add_argument("--hard_low_conf_min", type=float, default=0.1)
    parser.add_argument("--hard_low_conf_max", type=float, default=0.4)
    parser.add_argument("--conflict_iou", type=float, default=0.6)

    parser.add_argument("--no_pseudo_labels", action="store_true")
    parser.add_argument("--pseudo_conf", type=float, default=0.35)

    args = parser.parse_args()

    scraper = Level2DataScraper(download_dir=args.dir)
    if args.source in ("openimages", "all"):
        scraper.fetch_openimages(classes=args.classes, max_samples=args.max, splits=args.splits)
    if args.source in ("coco", "all"):
        scraper.fetch_coco(classes=args.classes, max_samples=args.max, splits=args.splits)
    if args.filter_model:
        scraper.filter_with_model(
            model_path=args.model,
            conf=args.conf,
            batch=args.batch,
            target_classes=args.target_classes,
            min_bbox_ratio=args.min_bbox,
            top_k=args.top_k,
            save_pseudo_labels=not args.no_pseudo_labels,
            alpha=args.score_alpha,
            beta=args.score_beta,
            gamma=args.score_gamma,
            min_consecutive_hits=args.min_consecutive_hits,
            hard_low_conf_min=args.hard_low_conf_min,
            hard_low_conf_max=args.hard_low_conf_max,
            conflict_iou=args.conflict_iou,
            pseudo_conf=args.pseudo_conf,
        )
    print_pipeline_notice(
        output_paths=args.dir,
        next_script="src/clip_filter.py",
        notes=[
            "此步驟會把下載素材與可選的 pseudo label 結果存回 1_raw 或其子資料夾。",
            "若啟用模型過濾，建議先抽查下載樣本品質，再接續做 CLIP 過濾。",
        ],
    )
