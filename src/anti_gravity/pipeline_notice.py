def print_pipeline_notice(output_paths: list, next_script: str, notes: list = None):
    """
    訓練或任意階段完成後，印出輸出路徑、下一步指令與注意事項。
    """
    print("")
    print("=" * 60)
    print("[完成] 輸出路徑:")
    for p in output_paths:
        print(f"       {p}")

    print("")
    print(f"[下一步] 執行:")
    print(f"         python {next_script}")

    if notes:
        print("")
        print("[注意事項]")
        for note in notes:
            print(f"  - {note}")

    print("=" * 60)
    print("")
