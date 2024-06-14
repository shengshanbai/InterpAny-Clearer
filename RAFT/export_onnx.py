import argparse
from data.dis_index import FlowEstimator
import torch
import onnx
import onnxruntime as ort
from PIL import Image
import numpy as np
import albumentations as A
import cv2
from RAFT.core.utils import flow_viz


def pad_to_8(img):
    height, width = img.shape[:2]
    if width % 8 != 0 or height % 8 != 0:
        padding_width = 8 - (width % 8)
        padding_height = 8 - (height % 8)
        pad_left = padding_width // 2
        pad_right = padding_width - pad_left
        pad_top = padding_height // 2
        pad_bottom = padding_height - pad_top
        img = A.pad_if_needed(
            img, padding=[pad_left, pad_top, pad_right, pad_bottom])
    return img


class FlowEstimatorOnnx:
    def __init__(self, onnx_weight):
        # 加载onnx模型并运行
        self.ort_session = ort.InferenceSession(
            onnx_weight, providers=['CPUExecutionProvider'])

    def estimate_flow(self, image_0_path, image_1_path, visualize=False):
        # 加载图像
        image_0 = np.array(Image.open(image_0_path)).astype(np.uint8)
        image_1 = np.array(Image.open(image_1_path)).astype(np.uint8)
        image_0_pad = pad_to_8(image_0)
        image_1_pad = pad_to_8(image_1)
        image_0_in = image_0_pad.transpose(
            2, 0, 1)[None, :, :, :].astype(np.float32)
        image_1_in = image_1_pad.transpose(
            2, 0, 1)[None, :, :, :].astype(np.float32)
        flow_low, flow_up = self.ort_session.run(["flow_low", "flow_up"], {
            "image_1": image_0_in,
            "image_2": image_1_in
        })
        flow_rgb = self.viz(image_0_in, flow_up)
        if visualize:
            cv2.imwrite("./temp_onnx.jpg", flow_rgb[:, :, ::-1])
        return flow_up, flow_rgb

    def viz(self, img, flo):
        img = img[0].transpose(1, 2, 0)
        flo = flo[0].transpose(1, 2, 0)

        # map flow to rgb image
        flo = flow_viz.flow_to_image(flo)
        img_flo = np.concatenate([img, flo], axis=0)

        return img_flo


def main(args):
    checkpoint = './RAFT/models/raft-things.pth'
    device = "cpu"
    flow_estimator = FlowEstimator(checkpoint=checkpoint, device=device)
    image_1 = torch.randn(args.image_shape, device=device)
    image_2 = torch.randn(args.image_shape, device=device)
    flow_estimator.model(image_1, image_2, 20, None, True, True)
    # torch.onnx.export(
    #     flow_estimator.model,
    #     args=(image_1, image_2, 20, None, True, True),
    #     f=args.onnx_file,
    #     export_params=True,
    #     input_names=["image_1", "image_2"],
    #     output_names=["flow_low", "flow_up"],
    #     opset_version=16,
    #     do_constant_folding=True,
    # )


def test_onnx(args):
    flow_estimator = FlowEstimatorOnnx(args.onnx_file)
    flow_estimator.estimate_flow(args.image_0, args.image_1, visualize=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_shape", nargs="+", default=[64, 3, 224, 224])
    parser.add_argument("--onnx_file", default="./RAFT/models/raft.onnx")
    parser.add_argument("--test", default=False)
    parser.add_argument("--image_0", default="/home/ssbai/datas/vfi/1_s.jpg")
    parser.add_argument("--image_1", default="/home/ssbai/datas/vfi/3_s.jpg")
    args = parser.parse_args()
    if args.test:
        test_onnx(args)
    else:
        main(args)
