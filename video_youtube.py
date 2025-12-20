import argparse
from pathlib import Path
import cv2
import torch
import numpy as np
import onnxruntime as ort
from vidgear.gears import CamGear

from yolov8_utils import (
    post_process,
    non_max_suppression,
    output_to_target,
    plot_images,
)

# -------------------------------------------------
# NPU / ONNX Runtime setup
# -------------------------------------------------
print(ort.get_available_providers())

model = "yolov8m.onnx"
providers = ["VitisAIExecutionProvider"]
xclbin_file = Path(__file__).resolve().parent / "4x4.xclbin"
provider_options = [{
    "target": "X1",
    "xclbin": xclbin_file
}]

print("Creating ONNX Runtime session...")
npu_session = ort.InferenceSession(
    model,
    providers=providers,
    provider_options=provider_options
)
print("InferenceSession created")

# -------------------------------------------------
# Load COCO labels
# -------------------------------------------------
with open("coco.names", "r") as f:
    names = f.read().splitlines()

# -------------------------------------------------
# YouTube stream
# -------------------------------------------------
parser = argparse.ArgumentParser(description="YOLOv8 NPU YouTube inference")
parser.add_argument(
    "--url",
    type = str,
    default ="https://www.youtube.com/watch?v=SeRUThVhlc4",
    help ="YouTube video URL"
)
args = parser.parse_args()
youtube_url = args.url

stream = CamGear(
    source=youtube_url,
    stream_mode=True,
    logging=False
).start()

# -------------------------------------------------
# Main loop
# -------------------------------------------------
print("Running YOLOv8 on YouTube stream")
frame = stream.read()
print("Frame shape:", frame.shape)  # (H, W, C)

while True:
    if frame is None:
       break

    # BGR → RGB + resize
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb = cv2.resize(frame_rgb, (640, 640))

    # NHWC float32 [0–1] for NPU
    im_npu = frame_rgb.astype(np.float32) / 255.0
    im_npu = im_npu[None, ...]  # [1,H,W,C]

    # NPU inference
    outputs = npu_session.run(
        None,
        {npu_session.get_inputs()[0].name: im_npu}
    )

    # Post-processing (NHWC → NCHW only here)
    outputs = [torch.from_numpy(o).permute(0, 3, 1, 2) for o in outputs]
    preds = post_process(outputs)
    preds = non_max_suppression(preds, 0.25, 0.7, max_det=300)
    targets = output_to_target(preds, max_det=300)

    # Visualization (plot_images expects NCHW)
    im_plot = np.transpose(im_npu, (0, 3, 1, 2))
    key = plot_images(im_plot, *targets, names=names)

    # Exit on 'q'
    if key & 0xFF == ord("q"):
        break

    frame = stream.read()
    frame = stream.read()
# -------------------------------------------------
# Cleanup
# -------------------------------------------------
stream.stop()
cv2.destroyAllWindows()
