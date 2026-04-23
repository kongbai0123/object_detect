import cv2
from time import perf_counter

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("無法開啟攝影機")

    window_name = "Camera (press q to quit)"

    while True:
        t0 = perf_counter()

        ok, frame = cap.read()
        if not ok:
            print("讀取影像失敗")
            break

        # 計算 FPS
        fps = 1.0 / max(perf_counter() - t0, 1e-6)

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (12, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            2
        )

        cv2.imshow(window_name, frame)

        # q 離開
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()