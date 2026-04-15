import argparse
import shutil
import subprocess
import zipfile
from pathlib import Path
import sys
# ⚠️ 解決 'def' 保留字問題
sys.path.append(str(Path(__file__).resolve().parent / "def"))
from pipeline_notice import print_pipeline_notice

ROOT = Path(__file__).resolve().parent.parent
VIDEO_EXTS = ("*.mp4", "*.mkv", "*.avi", "*.mov", "*.MP4", "*.MKV", "*.AVI", "*.MOV")


def find_ffmpeg_from_path():
    return shutil.which("ffmpeg")


def find_local_ffmpeg_exe(search_root):
    search_root = Path(search_root)

    exe_candidates = sorted(search_root.rglob("ffmpeg.exe"))
    if exe_candidates:
        return exe_candidates[0]

    unix_candidates = sorted(search_root.rglob("ffmpeg"))
    if unix_candidates:
        return unix_candidates[0]

    return None


def extract_ffmpeg_zip(zip_path, extract_root):
    zip_path = Path(zip_path)
    extract_root = Path(extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)
    marker = extract_root / ".ffmpeg_extracted"

    if not marker.exists():
        print(f"偵測到 FFmpeg zip，正在解壓: {zip_path.name}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_root)
        marker.write_text(zip_path.name, encoding="utf-8")

    return find_local_ffmpeg_exe(extract_root)


def resolve_ffmpeg_executable(explicit_path=None):
    if explicit_path:
        ffmpeg_path = Path(explicit_path)
        if ffmpeg_path.exists():
            return str(ffmpeg_path)
        raise FileNotFoundError(f"指定的 FFmpeg 路徑不存在: {ffmpeg_path}")

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
    input_dir = Path(input_dir)
    videos = []
    for pattern in VIDEO_EXTS:
        videos.extend(input_dir.rglob(pattern))
    return sorted(videos)


def extract_frames(input_dir, output_dir, fps=5, ffmpeg_path=None):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        print(f"找不到影片資料夾: {input_dir}")
        return

    ffmpeg_exe = resolve_ffmpeg_executable(ffmpeg_path)
    if ffmpeg_exe is None:
        print("找不到 FFmpeg。請安裝到 PATH，或把 ffmpeg zip 放在專案根目錄。")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    vid_files = collect_videos(input_dir)

    if not vid_files:
        print(f"{input_dir} 內沒有可處理的影片檔。")
        return

    print(f"使用 FFmpeg: {ffmpeg_exe}")
    print(f"找到 {len(vid_files)} 支影片，準備以 {fps} FPS 進行解壓縮 (去冗餘策略)...")

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

        print(f"正在抽幀: {vid.name} -> {output_dir}")
        try:
            subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
        except FileNotFoundError:
            print(f"FFmpeg 執行檔不存在: {ffmpeg_exe}")
            return
        except subprocess.CalledProcessError:
            print(f"抽幀失敗: {vid.name}")
            return

    print("所有影片抽幀完成，輸出結果可進一步交給後續流程。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FFmpeg Video to Frame Extractor (Low FPS for Redundancy Removal)"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(ROOT / "data/1_raw/videos"),
        help="影片資料夾路徑",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(ROOT / "data/1_raw/door_opening_frames"),
        help="抽幀輸出資料夾",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=5,
        help="抽幀 FPS，預設 5。",
    )
    parser.add_argument(
        "--ffmpeg",
        type=str,
        default="",
        help="可選，直接指定 ffmpeg 或 ffmpeg.exe 路徑。",
    )
    args = parser.parse_args()

    extract_frames(
        args.input,
        args.output,
        args.fps,
        ffmpeg_path=args.ffmpeg or None,
    )
    print_pipeline_notice(
        output_paths=args.output,
        next_script="src/clip_filter.py",
        notes=[
            "此步驟會輸出抽幀 JPEG，後續可直接做 CLIP 過濾。",
            "若已自動解壓 FFmpeg，執行檔位於 tools/ffmpeg/ 下，可重複使用。",
        ],
    )
