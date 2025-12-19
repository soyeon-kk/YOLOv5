import argparse
import os
import sys
from pathlib import Path
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from changedetection import ChangeDetection
from person_tracker import PersonTracker

from ultralytics.utils.plotting import Annotator, colors
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (
    check_file, check_img_size, check_imshow, cv2,
    increment_path, non_max_suppression, print_args, scale_boxes
)
from utils.torch_utils import select_device, smart_inference_mode


def classify_person_box(x1, y1, x2, y2, img_h, img_w):
    """
    단순 규칙 기반으로 사람 박스를 'seated' / 'standing' 분류.
    - 너무 작은 박스는 노이즈로 무시
    - 세로로 많이 길고, 화면 아래쪽까지 내려오면 standing
    - 그 외는 seated 로 취급
    """
    w = max(x2 - x1, 1.0)
    h = max(y2 - y1, 1.0)

    aspect = h / w  # 세로/가로 비율
    # center_y = (y1 + y2) / 2  # 필요하면 튜닝용으로 사용

    # 너무 작은 건 무시 (멀리 있는 사람/노이즈)
    if h < img_h * 0.05 or w < img_w * 0.02:
        return "ignore"

    # 세로로 길고, 바닥 쪽(이미지 아래 60% 이후)까지 내려오면 서 있는 사람
    if aspect > 2.0 and y2 > img_h * 0.6:
        return "standing"

    # 나머지는 일단 앉은 사람으로 본다
    return "seated"


@smart_inference_mode()
def run(
    weights=ROOT / "yolov5s.pt",
    source="0",
    data=ROOT / "data/coco128.yaml",
    imgsz=(640, 640),
    conf_thres=0.25,
    iou_thres=0.45,
    device="cpu",
    view_img=False,
    nosave=False,
    agnostic_nms=False,
    augment=False,
    project=ROOT / "runs/detect",
    name="exp",
    exist_ok=False,
    line_thickness=2,
    hide_conf=True,
    vid_stride=1,
):
    source = str(source)

    save_img = not nosave and not source.endswith(".txt")
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.startswith(("rtsp", "rtmp", "http"))

    webcam = source.isnumeric() or (is_url and not is_file)

    if is_url and is_file:
        source = check_file(source)

    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, data=data)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    # 혼잡도 / 서버 업로드 담당
    cd = ChangeDetection()

    # ID 추적 담당 (노이즈 줄이기 용)
    tracker = PersonTracker(max_dist=250.0, max_lost=15)

    # 입력 소스 로딩
    if webcam:
        view_img = check_imshow(warn=False)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)

    model.warmup(imgsz=(1 if pt else 1, 3, *imgsz))

    for path, im, im0s, vid_cap, s in dataset:
        im = torch.from_numpy(im).to(device).float() / 255.0
        if im.ndimension() == 3:
            im = im.unsqueeze(0)

        pred = model(im)
        pred = non_max_suppression(pred, conf_thres, iou_thres, None, agnostic_nms)

        for i, det in enumerate(pred):
            if webcam:
                im0 = im0s[i].copy()
            else:
                im0 = im0s.copy()

            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            person_boxes = []
            seated_count = 0
            waiting_count = 0

            if len(det):
                # YOLO 좌표를 원본 이미지 스케일로 변환
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                img_h, img_w = im0.shape[:2]

                for *xyxy, conf, cls in det:
                    cls_id = int(cls)
                    if names[cls_id] != "person":
                        continue

                    x1, y1, x2, y2 = [float(x) for x in xyxy]
                    person_boxes.append((x1, y1, x2, y2))

                    state = classify_person_box(x1, y1, x2, y2, img_h, img_w)
                    if state == "seated":
                        seated_count += 1
                    elif state == "standing":
                        waiting_count += 1
                    # "ignore" 는 카운트 안 함

            # 추적기로 ID 부여 (필요하면 나중에 add에서 current_ids 사용 가능)
            current_ids, track_boxes = tracker.update(person_boxes)

            # 바운딩 박스 + ID 시각화
            for x1, y1, x2, y2, tid in track_boxes:
                annotator.box_label([x1, y1, x2, y2], f"ID {tid}", color=colors(0, True))

            im0 = annotator.result()

            # 혼잡도 계산 + 상태 변경 시 서버로 업로드
            cd.add(
                current_ids=current_ids,
                save_dir=save_dir,
                image=im0,
                seated_count=seated_count,
                waiting_count=waiting_count,
            )

            # 디버그용 출력 (원하면 지워도 됨)
            print(
                f"[Occupancy] seated={seated_count}, waiting={waiting_count}, "
                f"current_ids={len(current_ids)}"
            )

            if save_img:
                cv2.imwrite(str(save_dir / "last.jpg"), im0)

            if view_img:
                cv2.imshow("YOLOv5", im0)
                if cv2.waitKey(1) == ord("q"):
                    return

    print("Done.")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=ROOT / "yolov5s.pt")
    parser.add_argument("--source", type=str, default="0")
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml")
    parser.add_argument("--imgsz", nargs="+", type=int, default=[640])
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--iou-thres", type=float, default=0.45)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--view-img", action="store_true")
    parser.add_argument("--nosave", action="store_true")
    parser.add_argument("--agnostic-nms", action="store_true")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--project", default=ROOT / "runs/detect")
    parser.add_argument("--name", default="exp")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--line-thickness", default=3, type=int)
    parser.add_argument("--hide-conf", action="store_true")
    parser.add_argument("--vid-stride", type=int, default=1)

    opt = parser.parse_args()

    if len(opt.imgsz) == 1:
        opt.imgsz = opt.imgsz * 2

    print_args(vars(opt))
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
