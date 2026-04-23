import cv2
import shutil
from pathlib import Path
from tqdm import tqdm
from anti_gravity.settings import settings
from anti_gravity.logger import logger
from anti_gravity.storage import DatasetStorage
from anti_gravity.augmenter import Augmenter

class AugmentService:
    """
    協調增強任務的服務層。
    """
    def __init__(self, storage: DatasetStorage = None, augmenter: Augmenter = None):
        self.storage = storage or DatasetStorage()
        self.augmenter = augmenter or Augmenter(settings.augment.model_dump())

    def execute(self, input_raw: str):
        logger.info(f"[AugmentService] Starting augmentation task, input: {input_raw}")
        
        input_path = Path(input_raw)
        # 修正：指向新的 workspace/augment 結構
        output_dir = settings.paths.augment / "current"
        
        out_img = output_dir / "images"
        out_lbl = output_dir / "labels"
        
        # 修正：每次執行前清理工作區，避免舊資料干擾
        if output_dir.exists():
            shutil.rmtree(output_dir)
            
        for d in [out_img, out_lbl]:
            d.mkdir(parents=True, exist_ok=True)

        # 1. 取得資料清單
        metadata_list = self.storage.scan_directories([input_path])
        if not metadata_list:
            logger.error(f"[AugmentService] No valid data found at: {input_path}")
            return

        # 2. 逐一處理並增強
        for meta in tqdm(metadata_list, desc="Augmenting"):
            # 複製原始一份
            shutil.copy2(meta.path, out_img / meta.path.name)
            if meta.label_path and meta.label_path.exists():
                # 修正：不使用 shutil.copy2，改用讀寫方式自動移除 BOM (\ufeff)
                with open(meta.label_path, 'r', encoding='utf-8-sig') as f_in:
                    content = f_in.read()
                with open(out_lbl / meta.label_path.name, 'w', encoding='utf-8') as f_out:
                    f_out.write(content)
            else:
                (out_lbl / f"{meta.stem}.txt").touch()

            # 判斷增強需求
            profile, multiplier = self.augmenter.select_profile(meta)
            if multiplier <= 1:
                continue

            # 執行 OpenCV 載入
            image = cv2.imread(str(meta.path))
            if image is None: continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            for i in range(multiplier - 1):
                transformed = self.augmenter.apply(image, meta, profile)
                if not transformed: continue
                
                aug_stem = f"{meta.stem}_{profile}_aug_{i}"
                
                # 儲存影像
                cv2.imwrite(
                    str(out_img / f"{aug_stem}.jpg"), 
                    cv2.cvtColor(transformed['image'], cv2.COLOR_RGB2BGR)
                )
                
                # 儲存標籤 (顯式指定 utf-8)
                with open(out_lbl / f"{aug_stem}.txt", 'w', encoding='utf-8') as f:
                    for lbl, bbox in zip(transformed['class_labels'], transformed['bboxes']):
                        f.write(f"{lbl} {' '.join([f'{x:.6f}' for x in bbox])}\n")

        # 3. 產出 dataset.yaml (MLOps 自動化關鍵)
        import yaml
        dataset_cfg = {
            'path': str(output_dir.absolute()).replace('\\', '/'),
            'train': 'images',
            'val': str((settings.paths.val_frozen / 'images').absolute()).replace('\\', '/'),
            'nc': 2,
            'names': ['open', 'close']
        }
        with open(output_dir / "dataset.yaml", 'w', encoding='utf-8') as f:
            yaml.dump(dataset_cfg, f, default_flow_style=False)

        logger.info(f"[AugmentService] Augmentation finished & dataset.yaml created. Result at: {output_dir}")
        print("")
        print(f"[完成] 增強結果已輸出至:")
        print(f"       {output_dir}")
        print("")
        print("[下一步] 執行模型訓練:")
        print("         python train.py")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=str(settings.paths.balance / "current"), help="增強輸入來源 (預設為 Balance 輸出)")
    args = parser.parse_args()
    
    AugmentService().execute(input_raw=args.input)
