import cv2
import numpy as np
from yolov8_utils import colors

def letterbox(img, new_size=640, color=(114, 114, 114)):
    h, w = img.shape[:2]
    scale = min(new_size / w, new_size / h)

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((new_size, new_size, 3), color, dtype=np.uint8)
    dw = (new_size - new_w) // 2
    dh = (new_size - new_h) // 2

    canvas[dh:dh + new_h, dw:dw + new_w] = resized
    return canvas, scale, dw, dh


def plot_image(img, det, names):
    h, w = img.shape[:2]

    font_scale = max((h + w) / 4000, 0.6)
    font_thickness = max(int(font_scale * 2), 2)


    if det is not None and len(det):
        det = det.cpu().numpy()

        for x1, y1, x2, y2, conf, cls in det:
            c = int(cls)
            label = f"{names[c]} {conf:.2f}"

            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), colors(c), 2)

            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )

            cv2.rectangle(
                img,
                (int(x1), int(y1) - th - 6),
                (int(x1) + tw, int(y1)),
                colors(c),
                -1,
            )

            cv2.putText(
                img,
                label,
                (int(x1), int(y1) - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                font_thickness,
                cv2.LINE_AA,
            )
    cv2.namedWindow("YOLOv8 Output", cv2.WINDOW_NORMAL)
    cv2.imshow("YOLOv8 Output", img)
    return cv2.waitKey(1)
