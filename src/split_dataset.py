import argparse
import shutil
import random
from pathlib import Path
from typing import Optional, List, Tuple, Dict
from anti_gravity.settings import settings
from anti_gravity.logger import logger
from anti_gravity.storage import DatasetStorage
from anti_gravity.splitter import Splitter
from anti_gravity.entities import ImageMetadata

class SplitService:
    """
    [Industrial Grade] 協調 Split 任務的服務層。
    具備：
    1. Per-version Dynamic Seeding (防止 Shuffle 偏誤)
    2. Bidirectional Open Safeguard (下限 15% 監督能力，上限 30% 防止過度修正，保護 Train 60% 覆蓋率)
    3. Dynamic Domain Cap (防止單一版本過載)
    """
    def __init__(self, storage: Optional[DatasetStorage] = None):
        self.storage = storage or DatasetStorage()
        self.base_seed = settings.train.patience

    def get_stats(self, metadata_list) -> Tuple[int, int, int, int]:
        """[Audit] 統計 Open, Close, Background 數量與含有 Open 的圖片數"""
        o_box, c_box, bg_img = 0, 0, 0
        o_img = 0
        for m in metadata_list:
            if m.is_background:
                bg_img += 1
            else:
                o_box += m.open_cnt
                c_box += m.close_cnt
                if m.open_cnt > 0:
                    o_img += 1
        return o_box, c_box, bg_img, o_img

    def apply_open_safeguard(self, train_set: List[ImageMetadata], val_set: List[ImageMetadata]) -> Tuple[List[ImageMetadata], List[ImageMetadata], str]:
        """
        [Safeguard] 雙向場景級 Open 樣本保護。
        平衡 [驗證集監督能力] 與 [訓練集吸收能力]。
        """
        total_open_imgs = len([m for m in (train_set + val_set) if m.open_cnt > 0])
        # 若總數太少(例如 < 3)，直接跳過，不具備切分保護意義
        if total_open_imgs < 3:
            return train_set, val_set, ""

        # 定義門檻 (基於專家建議)
        min_val_target = max(3, round(total_open_imgs * 0.15))
        max_val_cap = max(3, round(total_open_imgs * 0.30))
        min_train_floor = max(3, round(total_open_imgs * 0.60))
        
        notes = []
        while True:
            cur_val_open = len([m for m in val_set if m.open_cnt > 0])
            cur_train_open = len([m for m in train_set if m.open_cnt > 0])
            
            # 停止條件 1: Val 已達標 (15%) 或已達上限 (30%)
            if cur_val_open >= min_val_target or cur_val_open >= max_val_cap:
                break
            
            # 尋找 Train 中最小的 Open 場景
            train_scenes = {}
            for m in train_set:
                if m.scene not in train_scenes: train_scenes[m.scene] = []
                train_scenes[m.scene].append(m)

            candidates = []
            for s_name, s_imgs in train_scenes.items():
                s_open_cnt = len([m for m in s_imgs if m.open_cnt > 0])
                if s_open_cnt > 0:
                    candidates.append((s_name, s_imgs, s_open_cnt))
            
            if not candidates: break
            candidates.sort(key=lambda x: len(x[1])) # 最小影響優先

            # 選中目標
            best_s_name, best_s_imgs, best_s_open = candidates[0]
            
            # 停止條件 2: 檢查搬遷後是否會破壞 Train 的 60% 保底
            if (cur_train_open - best_s_open) < min_train_floor:
                # print(f"  [Safeguard Stop] Cannot move {best_s_name}, would drop train open below {min_train_floor}")
                break
            
            # 執行遷移
            train_set = [m for m in train_set if m.scene != best_s_name]
            val_set.extend(best_s_imgs)
            notes.append(f"Moved {best_s_name}(+{best_s_open}O)")

        if not notes: return train_set, val_set, ""
        return train_set, val_set, f"SG: {', '.join(notes)}"

    def execute(self, input_arg: str, train_ratio: float = 0.8, balance_domain: bool = True):
        logger.info(f"[SplitService] Starting split task, mode: {input_arg}, balance_domain: {balance_domain}")
        
        train_out = settings.paths.split / "current/train_src"
        val_out = settings.paths.split / "current/val_candidate"
        
        all_train_set = []
        all_val_set = []

        target_dirs = []
        if input_arg.lower() == "all":
            versions_root = settings.paths.assets / "goldenset/versions"
            target_dirs = sorted([d for d in versions_root.iterdir() if d.is_dir()])
            print(f"\n[Full-Merge Mode] Industrial Split-then-Merge (Total {len(target_dirs)} versions)...")
        else:
            target_dirs = [Path(input_arg)]

        # 預計算動態 Domain Cap (40%)
        total_raw = 0
        for d in target_dirs:
            total_raw += len(list(d.glob("images/*.jpg"))) + len(list(d.glob("images/*.png")))
        max_allowed = int(total_raw * train_ratio * 0.40)
        print(f"[*] Global Capacity: {total_raw} imgs | Domain Cap: {max_allowed} per version\n")

        # 1. 逐版本執行 [Split-then-Merge]
        print("-" * 125)
        print(f"{'Version':<12} | {'Train (Total/Scene/O-Img/C-Box)':<32} | {'Val (Total/Scene/O-Img/C-Box)':<32} | {'Note'}")
        print("-" * 125)

        for i, v_dir in enumerate(target_dirs):
            metadata_list = self.storage.scan_directories([v_dir])
            if not metadata_list: continue
            
            # --- [動態 Seed] ---
            v_splitter = Splitter(seed=self.base_seed + i)
            v_splitter.split_ratio = train_ratio
            train_set, val_set = v_splitter.perform_split(metadata_list)
            
            # --- [修正: 雙向 Open Safeguard] ---
            train_set, val_set, sg_note = self.apply_open_safeguard(train_set, val_set)

            # --- [動態 Domain Balancing] ---
            db_note = ""
            if balance_domain and len(train_set) > max_allowed:
                original_len = len(train_set)
                random.seed(self.base_seed + 99)
                train_set = random.sample(train_set, max_allowed)
                db_note = f"Capped {original_len}->{max_allowed}"

            all_train_set.extend(train_set)
            all_val_set.extend(val_set)

            # --- [Audit] ---
            to_box, tc_box, tb_img, to_img = self.get_stats(train_set)
            vo_box, vc_box, vb_img, vo_img = self.get_stats(val_set)
            t_scenes = len(set(m.scene for m in train_set))
            v_scenes = len(set(m.scene for m in val_set))
            
            print(f"{v_dir.name:<12} | {len(train_set):>4} ({t_scenes:>2}/{to_img:>3}/{tc_box:>3}) {'':<4} | "
                  f"{len(val_set):>4} ({v_scenes:>2}/{vo_img:>2}/{vc_box:>2}) {'':<4} | {sg_note} {db_note}")

        if not all_train_set: return

        # 全域統計
        go_box, gc_box, gb_img, go_img = self.get_stats(all_train_set)
        vo_box, vc_box, vb_img, vo_img = self.get_stats(all_val_set)
        print("-" * 125)
        print(f"{'GLOBAL TOTAL':<12} | {len(all_train_set):>4} (O-Img:{go_img}/C-Box:{gc_box}/BG:{gb_img}) | "
              f"{len(all_val_set):>4} (O-Img:{vo_img}/C-Box:{vc_box}/BG:{vb_img})")
        print("-" * 125)

        # 2. 實體部署
        for d in [train_out, val_out]:
            if d.exists(): shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)

        self.storage.deploy_dataset(all_train_set, train_out)
        self.storage.deploy_dataset(all_val_set, val_out)
        self._inject_replay_core(train_out)

    def _inject_replay_core(self, train_dir: Path):
        replay_dir = settings.paths.replay_core
        if not replay_dir.exists(): return
        metadata_list = self.storage.scan_directories([replay_dir])
        if metadata_list:
            self.storage.deploy_dataset(metadata_list, train_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--ratio", type=float, default=0.8)
    parser.add_argument("--balance_domain", action="store_true", default=True)
    args = parser.parse_args()
    service = SplitService()
    service.execute(args.input, args.ratio, args.balance_domain)
