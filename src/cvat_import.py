import os
import zipfile
import shutil
from pathlib import Path
import sys
# ⚠️ 解決 'def' 保留字問題
sys.path.append(str(Path(__file__).resolve().parent / "def"))
from pipeline_notice import print_pipeline_notice

#還未產出labels之前，先自行開啟CVAT，匯入圖片，並手動標註   
#command:
#  docker compose up -d 


class CVATBridge:
    def __init__(self, dataset_dir=None):
        """
        初始化通用 GT 匯入閘門 (支持 ZIP 與 資料夾合併)
        """
        if dataset_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            dataset_dir = os.path.normpath(os.path.join(script_dir, '../data/3_processed'))
            
        self.dataset_dir = Path(dataset_dir)
        self.images_dir = self.dataset_dir / 'images'
        self.labels_dir = self.dataset_dir / 'labels'
        
        # 預設的生肉影像池 (用於自動補位)
        self.raw_source_dir = self.dataset_dir.parent / '2_filtered' / 'open'
        
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)
        
    def _is_safe_path(self, basedir, path, symlinks=True):
        matchpath = os.path.abspath(path)
        if symlinks and os.path.islink(matchpath):
            matchpath = os.path.realpath(matchpath)
        return basedir == os.path.commonpath((basedir, matchpath))

    def _resolve_input_path(self, input_path):
        """解析輸入來源，支援 .zip 檔或資料夾目錄"""
        candidate = Path(input_path).expanduser()
        if not candidate.exists():
            # 嘗試自動補回 Windows 可能漏掉的 Downloads 前綴
            if not candidate.is_absolute():
                downloads = Path(os.environ.get('USERPROFILE', '')) / 'Downloads'
                if (downloads / input_path).exists():
                    candidate = downloads / input_path
            
        if not candidate.exists():
            raise FileNotFoundError(f"找不到輸入來源: {candidate}")
        return candidate

    def _collect_txt_files(self, root_dir):
        """智慧蒐集標籤檔：優先看 labels/ 子目錄，否則掃描整個根目錄"""
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
        2. 若無，則嘗試從 2_filtered/open (生肉池) 進行「影像晉升 (Promotion)」。
        """
        # A. 檢查現有黃金池
        for ext in ['.jpg', '.jpeg', '.png', '.webp', '.JPG', '.PNG']:
            gold_path = self.images_dir / f"{base_name}{ext}"
            if gold_path.exists():
                return True
        
        # B. 嘗試從整個 2_filtered 母目錄及其子目錄中自動補位
        if self.raw_source_dir.exists():
            # 這裡將 raw_source_dir 上提到 2_filtered 以便搜索所有子目錄 (open, open_plus, negative_samples)
            parent_source = self.raw_source_dir.parent
            for ext in ['.jpg', '.jpeg', '.png', '.webp', '.JPG', '.PNG']:
                # 利用 rglob 進行遞迴搜索，應付多樣化的來源池
                raw_paths = list(parent_source.rglob(f"{base_name}{ext}"))
                if raw_paths:
                    raw_path = raw_paths[0] # 抓取第一個匹配項
                    dest_path = self.images_dir / f"{base_name}{ext}"
                    try:
                        shutil.copy2(str(raw_path), str(dest_path))
                        return True
                    except Exception as e:
                        print(f" ERROR: 影像補位失敗 {base_name} ({e})")
        
        return False

    def merge_labels_from_source(self, input_path):
        """
        [Level 5 閉環引擎] 標籤回流總閘門：
        支援從 CVAT zip 或 FiftyOne edited folder 合併最新 GT 至黃金池
        """
        try:
            source = self._resolve_input_path(input_path)
        except Exception as exc:
            print(f" {exc}")
            return
            
        print(f" [GT Bridge] 開始準備同步來源: {source}")
        
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
                            print(f" 跳過可疑壓縮路徑: {member}")
                            continue
                        zip_ref.extract(member, temp_extract_dir)
                working_dir = Path(temp_extract_dir)
            elif source.is_dir():
                working_dir = source
            else:
                print(f" ERROR: 不支援的輸入格式 {source.suffix}")
                return

            txt_files = self._collect_txt_files(working_dir)
            
            merge_count = 0
            capture_count = 0 
            skip_count = 0
            updated_files = []

            from tqdm import tqdm

            for txt_path in tqdm(txt_files, desc=f"[{source_type.upper()} 合併中]", unit="file"):
                base_name = txt_path.stem

                # 改用智慧對位機制 (含自動補位)
                is_ready = False
                # 先檢查是否已經在黃金池
                if self._ensure_image_in_gold_pool(base_name):
                    # 如果原本不在，但後來補進去了 (偵測邏輯)
                    # 這裡簡化邏輯：只要能通過 ensure 就算成功
                    is_ready = True
                
                if not is_ready:
                    skip_count += 1
                    continue
                    
                dest_path = self.labels_dir / txt_path.name
                shutil.copy2(str(txt_path), str(dest_path))
                updated_files.append(txt_path.name)
                merge_count += 1
            
            print("\n" + "="*45)
            print(" [Merge Report] 標籤回流審核清單 ")
            print("="*45)
            print(f" 來源型態: {source_type}")
            print(f" 來源路徑: {source}")
            print(f" 成功合併: {merge_count} 份 Ground Truth")
            if skip_count > 0:
                print(f" 過濾跳過: {skip_count} 份 (幽靈標籤: 找不到對應圖片)")
            
            print(f" 註記: 系統已自動從生肉池補位缺失影像至 {self.images_dir}")

            if updated_files:
                print("-" * 45)
                print(" 最終更新清單 (前 20 筆):")
                for name in updated_files[:20]:
                    print(f"  - {name}")
                if len(updated_files) > 20:
                    print(f"  ... 另外還有 {len(updated_files)-20} 份檔案已寫入")
            print("="*45)
            print("\n 閉環同步完成！現在可以跑 split_dataset.py 重新切分訓練集了。")
        finally:
            if temp_extract_dir and os.path.exists(temp_extract_dir):
                shutil.rmtree(temp_extract_dir)

    def upload_to_cvat(self):
        pass

if __name__ == '__main__':
    import sys
    # 防禦 Windows cmd/powershell 預設 cp950 編碼無法印出 Emoji 的錯誤
    if sys.stdout.encoding.lower() != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
        
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("="*65)
    print(" MLOps 人工審查閉環主控台 (CVAT Orchestrator)")
    print("="*65)
    print("[1] 啟動本地端 CVAT 標註伺服器 (Mode 1: Start)")
    print("[2] 執行核心標籤防呆同步回流 (Mode 2: Merge)")
    print("[3] 關閉本地端 CVAT 標註伺服器 (Mode 3: Stop)")
    print("="*65)
    
    choice = input(" 請輸入要執行的模式代碼 (1, 2 或 3): ").strip()
    
    # 自動偵測可用埠號
    default_port = 8080
    if os.environ.get('CVAT_HTTP_PORT'):
        target_port = int(os.environ.get('CVAT_HTTP_PORT'))
    else:
        # 如果 8080 被佔用，自動切換到 8088
        import socket
        def check_port(port):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                return s.connect_ex(('localhost', port)) == 0
        
        target_port = default_port
        if check_port(target_port):
            print(f" [提示] 偵測到埠號 {target_port} 已被佔用，自動嘗試切換至 8088...")
            target_port = 8088
            if check_port(target_port):
                print(f" [警告] 埠號 {target_port} 也被佔用，將嘗試隨機可用埠號...")
    
    # 這裡綁定的是專案預設資料庫根目錄
    dataset_dir_path = os.path.normpath(os.path.join(script_dir, '../data/3_processed'))
    bridge = CVATBridge(dataset_dir=dataset_dir_path)
    
    if choice == '1':
        print("\n [Mode 1] 正在為您啟動本地端 CVAT 標註伺服器...")
        import subprocess
        import webbrowser
        import time
        
        try:
            print(f" 喚醒 Docker 容器服務 (PORT: {target_port})...")
            # 設定環境變數傳遞給 docker compose
            env = os.environ.copy()
            env['CVAT_HTTP_PORT'] = str(target_port)
            
            subprocess.Popen(
                ["docker", "compose", "-f", "cvat/docker-compose.yml", "up", "-d"],
                env=env
            )
            
            print(f" 等待 CVAT 後端微服務完全啟動 (目標: http://localhost:{target_port})...")
            
            # 使用更聰明的輪詢偵測 (不需要額外安裝 requests，用內建 urllib)
            import urllib.request
            max_retries = 30
            success = False
            for i in range(max_retries):
                try:
                    time.sleep(3)
                    with urllib.request.urlopen(f"http://localhost:{target_port}/api/server/about", timeout=5) as response:
                        if response.getcode() == 200:
                            success = True
                            break
                except Exception:
                    print(f"  [偵測中] 服務載入中... ({i+1}/{max_retries})", end='\r')
            
            if success:
                print(f"\n [成功] CVAT 已就緒！")
            else:
                print(f"\n [提醒] 啟動超時，但容器可能仍在後端啟動中...")

            print(f" 開啟瀏覽器進入標註畫面: http://localhost:{target_port}")
            webbrowser.open(f"http://localhost:{target_port}")
            print_pipeline_notice(
                output_paths=None,
                next_script="src/cvat_import.py",
                notes=[
                    "Mode 1 只會啟動 CVAT，不會直接把資料寫回正式資料集。",
                    "完成人工標註或 FiftyOne 校閱後，請再執行同一支腳本的 Mode 2 進行合併。",
                ],
            )
            print(" 小提示：若網頁出現拒絕連線，代表後端還在載入，請稍等片刻後重新整理網頁")
        except FileNotFoundError:
            print(" 找不到 Docker 環境！請確認 Docker Desktop 是否已經開啟並安裝就緒")
            
    elif choice == '2':
        print("\n [Mode 2] 進入標籤閉環同步 (Merge) 程序...")
        print(" 下一步：請提供『CVAT 匯出的 zip』或『FiftyOne 匯出的 edited 資料夾』路徑")
        
        default_input = "C:/Users/user/Downloads/cvat.zip"
        user_input = input(f"請輸入路徑 (Enter 使用預設值: {default_input}):\n> ").strip()
        
        target_input_path = user_input.strip('\"\'') if user_input else default_input
        
        print("-" * 65)
        bridge.merge_labels_from_source(target_input_path)
        print_pipeline_notice(
            output_paths=dataset_dir_path,
            next_script="src/split_dataset.py",
            notes=[
                "合併完成後建議先抽查 3_processed/labels 的內容是否更新。",
                "此程序只同步標籤，不會自動把 edited dataset 的圖片寫回 3_processed/images。",
            ],
        )
        
    elif choice == '3':
        print("\n [Mode 3] 正在為您關閉本地端 CVAT 標註伺服器...")
        import subprocess
        try:
            print(" 停止 Docker 容器服務 (docker compose -f cvat/docker-compose.yml down)...")
            subprocess.run(["docker", "compose", "-f", "cvat/docker-compose.yml", "down"])
            print(" CVAT 服務已成功關閉。")
        except FileNotFoundError:
            print(" 找不到 Docker 環境！")
            
    else:
        print(" 無效的選擇，請在終端機輸入 1, 2 或是 3！程式結束")
