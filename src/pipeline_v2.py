import argparse
import sys
import os
from pathlib import Path
from anti_gravity.settings import settings
from anti_gravity.logger import logger

def main():
    """
    🔵 展示層 (Presentation)：統一 CLI 入口
    整合所有 Sub-commands 並負責頂層的環境校驗。
    """
    parser = argparse.ArgumentParser(
        description="Antigravity Industrial MLOps Pipeline v2.0",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="subcommand", help="支援的任務命令")

    # --- 1. Balance 命令 ---
    balance_parser = subparsers.add_parser("balance", help="執行資料類別平衡與場景治理")
    balance_parser.add_argument("--input", type=str, default="5", help="輸入來源：可為 '5', '6' 或路徑")
    balance_parser.add_argument("--output", type=str, help="輸出路徑")
    balance_parser.add_argument("--ratio", type=float, help="覆寫目標平衡比例")

    # --- 2. Split 命令 ---
    split_parser = subparsers.add_parser("split", help="執行場景感知切分")
    split_parser.add_argument("--input", type=str, default=str(settings.paths.processed), help="輸入來源資料夾")
    split_parser.add_argument("--ratio", type=float, default=0.8, help="訓練集比例")

    # --- 3. Augment 命令 ---
    augment_parser = subparsers.add_parser("augment", help="執行離線資料增強")
    augment_parser.add_argument("--input", type=str, help="欲增強的輸入源 (例如 data/6_augmented)")

    args = parser.parse_args()

    # 🟣 穩定性檢查
    try:
        settings.paths.validate_paths()
    except FileNotFoundError as e:
        logger.error(f"[CLI] Init Failed: {e}")
        sys.exit(1)

    # 路由邏輯與路徑自動解析
    if args.subcommand == "balance":
        from balance_dataset import BalanceService
        if args.ratio: settings.balance.target_ratio = args.ratio
        
        # 針對 balance 的特別簡寫處理在 service 內部執行
        BalanceService().execute(input_raw=args.input, output_path=Path(args.output).resolve() if args.output else None)
        
    elif args.subcommand == "split":
        from split_dataset import SplitService
        input_abs = str(Path(args.input).resolve())
        SplitService().execute(input_raw=input_abs, train_ratio=args.ratio)
        
    elif args.subcommand == "augment":
        from augment_dataset import AugmentService
        # 如果沒傳入，嘗試自動尋找最新的平衡產出
        input_raw = args.input or str(settings.paths.augmented)
        input_abs = str(Path(input_raw).resolve())
        AugmentService().execute(input_raw=input_abs)
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
