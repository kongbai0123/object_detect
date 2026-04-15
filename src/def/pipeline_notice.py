from pathlib import Path


def _normalize_paths(output_paths):
    if output_paths is None:
        return []
    if isinstance(output_paths, (str, Path)):
        return [str(output_paths)]
    return [str(path) for path in output_paths]


def print_pipeline_notice(output_paths=None, next_script="", notes=None):
    normalized_paths = _normalize_paths(output_paths)
    normalized_notes = notes or []

    print("\n" + "=" * 72)
    print("Pipeline Summary")
    print("=" * 72)

    if normalized_paths:
        print("1. 輸出到哪裡:")
        for path in normalized_paths:
            print(f"   - {path}")
    else:
        print("1. 輸出到哪裡:")
        print("   - 無固定輸出路徑，請依互動流程或參數指定位置確認。")

    print("2. 下一個可執行檔案:")
    print(f"   - {next_script or '請依 readme.md 流程選擇下一步'}")

    print("3. 其他說明:")
    if normalized_notes:
        for note in normalized_notes:
            print(f"   - {note}")
    else:
        print("   - 請依 readme.md 檢查輸出品質與資料夾結構。")

    print("=" * 72)
