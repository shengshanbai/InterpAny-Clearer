import argparse
from data.dis_index import FlowEstimator
from pathlib import Path
import cv2


def main(args):
    checkpoint = './RAFT/models/raft-things.pth'
    flow_estimator = FlowEstimator(checkpoint=checkpoint)
    flow_estimator.estimate_flow(args.image_0, args.image_1, visualize=True)


def resize_images(root_dir="/home/ssbai/datas/vfi"):
    STD_LONG = 640
    root_dir = Path(root_dir)
    for image_file in root_dir.iterdir():
        image_np = cv2.imread(str(image_file), cv2.IMREAD_COLOR)
        h, w, _ = image_np.shape
        max_side = max(h, w)
        if max_side > STD_LONG:
            ratio = STD_LONG/max_side
            image_np = cv2.resize(image_np, (int(w*ratio), int(h*ratio)))
            cv2.imwrite(str(root_dir.joinpath(
                image_file.stem+"_s.jpg")), image_np)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("--image_0", default="/home/ssbai/datas/vfi/1_s.jpg")
    parser.add_argument("--image_1", default="/home/ssbai/datas/vfi/3_s.jpg")
    parser.add_argument("--gt", default="/home/ssbai/datas/vfi/2_s.jpg")
    args = parser.parse_args()
    main(args)
