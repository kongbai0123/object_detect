import os
import zipfile
import shutil
import tempfile
import socket
import urllib.request
import webbrowser
import time
import subprocess
import argparse
from pathlib import Path
import sys
from tqdm import tqdm
from anti_gravity.settings import settings

# 引入 MLOps 通知模組
from anti_gravity.pipeline_notice import print_pipeline_notice

class CVATBridge:
    def __init__(self, dataset_dir=None):
        """
        跨平台通用 GT 導入橋接器 (支援 ZIP 與 資料夾模式)
        """
        if dataset_dir is None:
            dataset_dir = settings.paths.goldenset
            
        self.dataset_dir = Path(dataset_dir)
        self.images_dir = self.dataset_dir / 'images'
        self.labels_dir = self.dataset_dir / 'labels'
        
        # 原始影像池 (用於缺失補位)
        self.raw_source_dir = settings.paths.raw
        
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)
        
    def _is_safe_path(self, basedir, path, symlinks=True):
        matchpath = os.path.abspath(path)
        if symlinks and os.path.islink(matchpath):
            matchpath = os.path.realpath(matchpath)
        return basedir == os.path.commonpath((basedir, matchpath))

    def _resolve_input_path(self, input_path):
        """解析輸入來源，支援 .zip 檔案或資料夾目錄"""
        candidate = Path(input_path).expanduser()
        if not candidate.exists():
            # 嘗試讀取 Windows 下常用的 Downloads 路徑
            if not candidate.is_absolute():
                downloads = Path(os.environ.get('USERPROFILE', '')) / 'Downloads'
                if (downloads / input_path).exists():
                    candidate = downloads / input_path
            
        if not candidate.exists():
            raise FileNotFoundError(f"找不到輸入來源: {candidate}")
        return candidate

    def _collect_txt_files(self, root_dir):
        """智慧搜集標籤檔：自動過濾 YOLO 的 meta 檔案"""
        root_dir = Path(root_dir)
        labels_dir = root_dir / "labels"
        search_dir = labels_dir if labels_dir.is_dir() else root_dir

        txt_files = []
        for p in search_dir.rglob("*.txt"):
            if p.name in ['train.txt', 'val.txt', 'test.txt', 'obj.names', 'obj.data', 'classes.txt']:
                continue
            txt_files.append(p)
        return txt_files

    def _ensure_image_in_gold_pool(self, base_name):
        """
        智慧對位機制：
        1. 檢查 3_processed/images 是否已有影像。
        2. 若無，則自動去原始池 (2_filtered) 尋找並晉升 (Promotion) 到黃金池。
        """
        # A. 檢查是否已在黃金池
        for ext in ['.jpg', '.jpeg', '.png', '.webp', '.JPG', '.PNG']:
            gold_path = self.images_dir / f"{base_name}{ext}"
            if gold_path.exists():
                return True
        
        # B. 嘗試自動補位
        if self.raw_source_dir.exists():
            for ext in ['.jpg', '.jpeg', '.png', '.webp', '.JPG', '.PNG']:
                raw_paths = list(self.raw_source_dir.rglob(f"{base_name}{ext}"))
                if raw_paths:
                    raw_path = raw_paths[0]
                    dest_path = self.images_dir / f"{base_name}{ext}"
                    try:
                        shutil.copy2(str(raw_path), str(dest_path))
                        return True
                    except Exception as e:
                        print(f" [錯誤] 影像補位失敗: {base_name} ({e})")
        
        return False

    def merge_labels_from_source(self, input_path):
        """
        [MLOps 閉環引擎] 標籤同步與合併
        支援由 CVAT 匯出的 ZIP 或由 FiftyOne 編輯過的資料夾。
        """
        try:
            source = self._resolve_input_path(input_path)
        except Exception as exc:
            print(f" [錯誤] {exc}")
            return
            
        print(f" [同步啟動] 準備同步來源: {source}")
        
        temp_extract_dir = None
        working_dir = None
        source_type = "folder"
        
        try:
            if source.is_file() and source.suffix.lower() == ".zip":
                source_type = "zip"
                temp_extract_dir = tempfile.mkdtemp(prefix="cvat_mlops_")
                with zipfile.ZipFile(source, 'r') as zip_ref:
                    for member in zip_ref.namelist():
                        member_path = os.path.join(temp_extract_dir, member)
                        if not self._is_safe_path(temp_extract_dir, member_path):
                            continue
                        zip_ref.extract(member, temp_extract_dir)
                working_dir = Path(temp_extract_dir)
            elif source.is_dir():
                working_dir = source
            else:
                print(f" [錯誤] 不支援的輸入格式: {source.suffix}")
                return

            txt_files = self._collect_txt_files(working_dir)
            
            merge_count = 0
            skip_count = 0
            updated_files = []

            for txt_path in tqdm(txt_files, desc=f"[{source_type.upper()} 同步中]", unit="file"):
                base_name = txt_path.stem

                # 智慧對位
                if self._ensure_image_in_gold_pool(base_name):
                    dest_path = self.labels_dir / txt_path.name
                    shutil.copy2(str(txt_path), str(dest_path))
                    updated_files.append(txt_path.name)
                    merge_count += 1
                else:
                    skip_count += 1
            
            print("\n" + "="*45)
            print(" [同步報告] 標籤審核完成")
            print("="*45)
            print(f" 來源格式: {source_type}")
            print(f" 成功合併: {merge_count} 份標籤")
            if skip_count > 0:
                print(f" 略過幽靈標籤: {skip_count} 份 (查無對應影像)")
            
            print(f" 註記: 系統已自動從原始池補位缺失影像至 {self.images_dir}")
            print("="*45)
            print("\n 同步完成！現在您可以執行 split_dataset.py 重新切分訓練集。")
        finally:
            if temp_extract_dir and os.path.exists(temp_extract_dir):
                shutil.rmtree(temp_extract_dir)

if __name__ == '__main__':
    # 針對 Windows CMD/PowerShell 的 UTF-8 優化
    if sys.stdout.encoding.lower() != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except AttributeError:
            pass
            
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("="*65)
    print(" MLOps 人工審查閉環主控台 (CVAT Orchestrator)")
    print("="*65)
    print("[1] 啟動本地 CVAT 標註伺服器 (Mode 1: Start)")
    print("[2] 匯入標籤與同步影像 (Mode 2: Merge)")
    print("[3] 關閉本地 CVAT 標註伺服器 (Mode 3: Stop)")
    print("="*65)
    
    choice = input(" 請輸入執行模式代號 (1, 2 或 3): ").strip()
    
    # 自動偵測可用埠號
    target_port = 8080
    if os.environ.get('CVAT_HTTP_PORT'):
        target_port = int(os.environ.get('CVAT_HTTP_PORT'))
    else:
        def is_port_in_use(port):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                return s.connect_ex(('localhost', port)) == 0
        if is_port_in_use(target_port):
            target_port = 8088
    
    dataset_dir_path = settings.paths.goldenset
    bridge = CVATBridge(dataset_dir=dataset_dir_path)
    
    if choice == '1':
        print(f"\n [Mode 1] 正在啟動 CVAT 伺服器 (PORT: {target_port})...")
        try:
            env = os.environ.copy()
            env['CVAT_HTTP_PORT'] = str(target_port)
            subprocess.Popen(["docker", "compose", "-f", "cvat/docker-compose.yml", "up", "-d"], env=env)
            
            print(f" 等待伺服器就緒 (http://localhost:{target_port})...")
            # 簡單輪詢
            for i in range(30):
                time.sleep(3)
                try:
                    with urllib.request.urlopen(f"http://localhost:{target_port}/api/server/about", timeout=5) as r:
                        if r.getcode() == 200:
                            print("\n [成功] CVAT 已就緒！")
                            break
                except:
                    print(f"  [偵測中] ... ({i+1}/30)", end='\r')
            
            webbrowser.open(f"http://localhost:{target_port}")
            print_pipeline_notice(
                output_paths=None,
                next_script="src/cvat_import.py",
                notes=["請在標註售完成後，將 ZIP 下載至下載資料夾，再回來選擇 Mode 2。"]
            )
        except Exception as e:
            print(f" 啟動失敗: {e}")
            
    elif choice == '2':
        print("\n [Mode 2] 標籤同步程序啟動...")
        default_input = os.path.join(os.environ.get('USERPROFILE', ''), 'Downloads', 'cvat.zip')
        user_input = input(f" 請輸入路徑 (直接 Enter 使用預設下載路徑):\n > ").strip()
        target_input_path = user_input if user_input else default_input
        
        bridge.merge_labels_from_source(target_input_path)
        print_pipeline_notice(
            output_paths=[os.path.abspath(dataset_dir_path)],
            next_script="src/split_dataset.py",
            notes=["同步完成，建議檢查 labels/ 目錄內容是否更新。"]
        )
        
    elif choice == '3':
        print("\n [Mode 3] 正在關閉 CVAT 伺服器...")
        subprocess.run(["docker", "compose", "-f", "cvat/docker-compose.yml", "down"])
        print(" 關閉完成。")
    
    else:
        print(" 無效的選項，請重新執行並輸入 1, 2 或 3。")
