import argparse
import shutil
import subprocess
import zipfile
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# 引入 MLOps 通知模組
from anti_gravity.pipeline_notice import print_pipeline_notice
VIDEO_EXTS = ("*.mp4", "*.mkv", "*.avi", "*.mov", "*.MP4", "*.MKV", "*.AVI", "*.MOV")

def find_ffmpeg_from_path():
    """從系統環境變數中尋找 ffmpeg"""
    return shutil.which("ffmpeg")

def find_local_ffmpeg_exe(search_root):
    """從本地專案目錄中遞迴尋找 ffmpeg 執行檔"""
    search_root = Path(search_root)

    exe_candidates = sorted(search_root.rglob("ffmpeg.exe"))
    if exe_candidates:
        return exe_candidates[0]

    unix_candidates = sorted(search_root.rglob("ffmpeg"))
    if unix_candidates:
        return unix_candidates[0]

    return None

def extract_ffmpeg_zip(zip_path, extract_root):
    """解壓縮預先準備好的 FFmpeg ZIP 包"""
    zip_path = Path(zip_path)
    extract_root = Path(extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)
    marker = extract_root / ".ffmpeg_extracted"

    if not marker.exists():
        print(f" [系統] 偵測到 FFmpeg ZIP，正在進行自動解壓: {zip_path.name}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_root)
        marker.write_text(zip_path.name, encoding="utf-8")

    return find_local_ffmpeg_exe(extract_root)

def resolve_ffmpeg_executable(explicit_path=None):
    """
    智慧解析 FFmpeg 路徑：
    1. 優先使用手動指定路徑。
    2. 其次使用系統路徑 (PATH)。
    3. 最後嘗試從專案根目錄自動解壓帶有的 ZIP 包。
    """
    if explicit_path:
        ffmpeg_path = Path(explicit_path)
        if ffmpeg_path.exists():
            return str(ffmpeg_path)
        raise FileNotFoundError(f"找不到指定的 FFmpeg 路徑: {ffmpeg_path}")

    from_path = find_ffmpeg_from_path()
    if from_path:
        return from_path

    local_extracted = find_local_ffmpeg_exe(ROOT)
    if local_extracted:
        return str(local_extracted)

    zip_candidates = sorted(ROOT.glob("ffmpeg*.zip"))
    if not zip_candidates:
        return None

    extract_root = ROOT / "tools" / "ffmpeg"
    extracted = extract_ffmpeg_zip(zip_candidates[0], extract_root)
    return str(extracted) if extracted else None

def collect_videos(input_dir):
    """標註尋集所有支援格式的影片"""
    input_dir = Path(input_dir)
    videos = []
    for pattern in VIDEO_EXTS:
        videos.extend(input_dir.rglob(pattern))
    return sorted(videos)

def extract_frames(input_dir, output_dir, fps=5, ffmpeg_path=None):
    """呼叫 FFmpeg 執行抽幀與降冗餘 (Redundancy Removal)"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        print(f" [錯誤] 找不到影像來源資料夾: {input_dir}")
        return

    ffmpeg_exe = resolve_ffmpeg_executable(ffmpeg_path)
    if ffmpeg_exe is None:
        print(" [錯誤] 系統未安裝 FFmpeg。請安裝至 PATH 或確保專案目錄下有 ffmpeg zip。")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    vid_files = collect_videos(input_dir)

    if not vid_files:
        print(f" [提示] {input_dir} 內找不到任何可處理的影片檔案。")
        return

    print(f" [啟動] 使用 FFmpeg: {ffmpeg_exe}")
    print(f" [執行] 發現 {len(vid_files)} 部影片，準備以 {fps} FPS 進行影格提取...")

    for vid in vid_files:
        vid_name = vid.stem
        out_pattern = output_dir / f"{vid_name}_%06d.jpg"
        cmd = [
            ffmpeg_exe,
            "-y",
            "-i",
            str(vid),
            "-vf",
            f"fps={fps},scale=1280:-1",
            "-q:v",
            "2",
            str(out_pattern),
        ]

        print(f"  -> 正在處理: {vid.name}")
        try:
            subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
        except Exception as e:
            print(f" [失敗] {vid.name} 處理出錯: {e}")
            continue

    print("\n [完成] 影片抽幀作業結束，產出影格可供後續流程 (如 CLIP 篩選) 使用。")

if __name__ == "__main__":
    from anti_gravity.settings import settings
    parser = argparse.ArgumentParser(description="影片轉影格提取工具 (降冗餘版)")
    parser.add_argument("--input", type=str, default=str(settings.paths.assets / "videos"), help="影片源資料夾")
    parser.add_argument("--output", type=str, default=str(settings.paths.raw / "door_opening_frames"), help="影格輸出資料夾")
    parser.add_argument("--fps", type=int, default=4, help="提取頻率 (預設為 5 FPS)")
    parser.add_argument("--ffmpeg", type=str, default="", help="可選：手動指定 ffmpeg 執行檔路徑")
    args = parser.parse_args()

    extract_frames(
        args.input,
        args.output,
        args.fps,
        ffmpeg_path=args.ffmpeg or None,
    )
    
    print_pipeline_notice(
        output_paths=[os.path.abspath(args.output)],
        next_script="src/archive/clip_filter.py",
        notes=[
            "抽幀後的 JPEG 影格可以直接交由 CLIP 進行自動篩選。",
            "如果您已經手動安裝過 FFmpeg，系統會優先自動偵測並使用之。",
        ],
    )
