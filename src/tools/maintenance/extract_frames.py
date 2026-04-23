import cv2
from pathlib import Path

def extract_frames(video_path, output_dir, interval_sec=5):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30
    
    interval_frames = int(fps * interval_sec)
    count = 0
    frame_id = 0
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        if frame_id % interval_frames == 0:
            out_name = Path(output_dir) / f"frame_{count:03d}.jpg"
            cv2.imwrite(str(out_name), frame)
            print(f"Saved: {out_name}")
            count += 1
        frame_id += 1
    
    cap.release()

if __name__ == "__main__":
    v_path = Path(r"storage/artifacts/evaluations/videos/4_Video Project_rigorous_eval.mp4")
    out_dir = Path(r"storage/artifacts/evaluations/frames")
    extract_frames(v_path, out_dir)
