import os
from pathlib import Path

def consolidate():
    log_dir = Path("storage/artifacts/logs")
    master_log = log_dir / "training_master.log"
    
    # 獲取所有 training_*.log 並排序
    log_files = sorted(list(log_dir.glob("training_*.log")))
    
    if not log_files:
        print("No training logs found.")
        return

    print(f"Consolidating {len(log_files)} files into {master_log}...")
    
    with open(master_log, "a", encoding="utf-8") as master:
        for f in log_files:
            if f.name == "training_master.log": continue
            
            master.write(f"\n--- [LOG START: {f.name}] ---\n")
            try:
                content = f.read_text(encoding="utf-8")
                master.write(content)
                master.write(f"\n--- [LOG END: {f.name}] ---\n")
                
                # 刪除原檔案
                f.unlink()
            except Exception as e:
                print(f"Error reading {f.name}: {e}")

    print("Consolidation complete.")

if __name__ == "__main__":
    consolidate()
