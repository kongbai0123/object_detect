from pathlib import Path
from collections import Counter
from anti_gravity.utils import extract_scene_key
from anti_gravity.storage import DatasetStorage

def audit_5_img():
    path = Path(r"C:\antigravity\storage\assets\goldenset\versions\5_img")
    storage = DatasetStorage()
    metadata_list = storage.scan_directories([path])
    
    scenes = [m.scene for m in metadata_list]
    scene_counts = Counter(scenes)
    
    print(f"\n--- [Audit] 5_img Scene Distribution ---")
    print(f"Total Images: {len(metadata_list)}")
    print(f"Total Unique Scenes: {len(scene_counts)}")
    print("-" * 40)
    for scene, count in scene_counts.most_common():
        # 額外檢查這組場景內的標籤分佈
        group_meta = [m for m in metadata_list if m.scene == scene]
        o = sum(m.open_cnt for m in group_meta)
        c = sum(m.close_cnt for m in group_meta)
        print(f"Scene: {scene:<25} | Images: {count:>3} | O: {o:>3} | C: {c:>3}")

def audit_val_open_coverage():
    # 這裡我們模擬目前的 split 邏輯來查看分佈
    from anti_gravity.splitter import Splitter
    from anti_gravity.settings import settings
    
    root = Path(r"C:\antigravity\storage\assets\goldenset\versions")
    versions = sorted([d for d in root.iterdir() if d.is_dir()])
    storage = DatasetStorage()
    
    print(f"\n--- [Audit] Validation Set Open Image Coverage ---")
    print(f"{'Version':<12} | {'Val Images':<10} | {'Images with Open':<15}")
    print("-" * 45)
    
    for i, v in enumerate(versions):
        meta_list = storage.scan_directories([v])
        splitter = Splitter(seed=settings.train.patience + i)
        _, val_set = splitter.perform_split(meta_list)
        
        open_imgs = [m for m in val_set if m.open_cnt > 0]
        print(f"{v.name:<12} | {len(val_set):>10} | {len(open_imgs):>15}")

if __name__ == "__main__":
    audit_5_img()
    audit_val_open_coverage()
