import os
from pathlib import Path

def dedup_hierarchical(base_dir):
    # 定義優先順序，前面的會保留，後面的若重複則剔除
    versions = ["1_img", "2_img", "3_img", "temp", "temp2"]
    
    # 1. 建立完整路徑對照表並檢查
    paths = {v: Path(base_dir) / v for v in versions}
    for v, p in paths.items():
        if not p.exists():
            print(f"⚠️ 警告: 找不到資料夾 {p}，將跳過該目錄。")
            continue
        # 確保子目錄存在
        (p / "images").mkdir(parents=True, exist_ok=True)
        (p / "labels").mkdir(parents=True, exist_ok=True)

    # 2. 執行階層式去重
    # 邏輯：從第一個開始遍歷，收集它有的檔案，並從後續所有目錄中移除重複項
    seen_images = set()
    seen_labels = set()

    for i, current_v in enumerate(versions):
        current_path = paths[current_v]
        if not current_path.exists(): continue

        print(f"\n🔍 正在掃描優先目錄: {current_v} ...")
        
        # --- A. 處理當前目錄 ---
        # 獲取當前目錄的檔案列表
        current_images = set(f.name for f in (current_path / "images").iterdir() if f.is_file())
        current_labels = set(f.name for f in (current_path / "labels").iterdir() if f.is_file())

        # --- B. 檢查與「前面目錄」的重複 (實際上這步是防禦性檢查) ---
        # 我們遍歷後續所有目錄 (i+1 之後)，並從中刪除與 seen_images/labels 重複的檔案
        # 將當前檔案加入全局已見集合
        seen_images.update(current_images)
        seen_labels.update(current_labels)

        # --- C. 遍歷後續所有目錄進行剔除 ---
        for later_v in versions[i+1:]:
            later_path = paths[later_v]
            if not later_path.exists(): continue
            
            # 剔除影像
            for img_file in (later_path / "images").iterdir():
                if img_file.name in seen_images:
                    print(f"  [-] 移除 {later_v}/images/{img_file.name} (與優先目錄重複)")
                    try:
                        os.remove(img_file)
                    except Exception as e:
                        print(f"  ❌ 刪除失敗: {e}")

            # 剔除標籤
            for lbl_file in (later_path / "labels").iterdir():
                if lbl_file.name in seen_labels:
                    print(f"  [-] 移除 {later_v}/labels/{lbl_file.name} (與優先目錄重複)")
                    try:
                        os.remove(lbl_file)
                    except Exception as e:
                        print(f"  ❌ 刪除失敗: {e}")

    print("\n" + "="*40)
    print("✅ 階層式去重任務完成！")
    print(f"清理鏈條: {' > '.join(versions)}")
    print("="*40)

    # 最終檢查
    print("\n--- 最終分佈檢查 ---")
    global_manifest = {}
    for v in versions:
        p = paths[v]
        if not p.exists(): continue
        imgs = [f.name for f in (p / "images").iterdir() if f.is_file()]
        print(f" {v}: {len(imgs)} 張影像")
        for img in imgs:
            if img in global_manifest:
                print(f" ❌ [錯誤] 檔案 {img} 竟然同時存在於 {global_manifest[img]} 與 {v}！")
            global_manifest[img] = v

if __name__ == "__main__":
    # 黃金集版本的根目錄
    base = r"C:\antigravity\storage\assets\goldenset\versions"
    dedup_hierarchical(base)
