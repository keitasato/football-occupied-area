import os
import random
import sys

sys.path.append("../yolov5")
sys.path.append("../SST")

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


from tracker import SSTTracker, TrackerConfig, Track
# from sst_tracker import TrackSet as SSTTracker
from data.mot_data_reader import MOTDataReader
import numpy as np
from config.config import config
# from utils.timer import Timer
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from footballerModel import FootballerModel, Net

import pickle

import scipy
from PIL import Image

import csv

def get_cor_points(file_path):
    #print("input cor points")
    #print(file_path)
    csv_name = './' + file_path.split('/')[-1].split('.')[0] + '.csv'
    with open(csv_name) as f:
        reader = csv.reader(f)
        points = [row for row in reader]
    points = np.float32(points)

    return points

def get_homography(x1, x2):
    A = np.zeros((x1.shape[0]*2, 9))
    for i, (a, b) in enumerate(zip(x1, x2)):
        A[i * 2] = np.array([a[0], a[1], 1, 0, 0, 0, -b[0] * a[0], -b[0] * a[1], -b[0]])
        A[i * 2 + 1] = np.array([0, 0, 0, a[0], a[1], 1, -b[1] * a[0], -b[1] * a[1], -b[1]])
    u, s, vh = np.linalg.svd(A)
    min = 8
    Hest = vh[min].reshape((3, 3))
    Hest = Hest / Hest[2, 2]
    return Hest

field_h, field_w = int(105*10/2), int(68*10)
margin = 10

save_id = 0

velocity_min = 5.0
velocity_max = 44.0

def get_foot_coord(det, h, w):
    #各物体のバウンディングボックス下辺の中点を返す
    foot_coord = det[:, 0:2] + det[:, 2:4]
    foot_coord[:, 0] -= det[:, 2]/2
    foot_coord[:, 0] *= w #x
    foot_coord[:, 1] *= h #y
    return foot_coord

def transform_coord(det, h, w, M):
    foot = get_foot_coord(det, h, w)#まず検出した人の足元の座標を計算
    foot_np = foot.to('cpu').detach().numpy().copy()#detはPyTorchによるテンソルだが射影変換はnumpyのテンソルで行う必要あり
    foot_np = np.array([foot_np], dtype='float32')
    #射影変換
    foot_trans = cv2.perspectiveTransform(foot_np, M)
    return foot_trans

def draw_foot(img, foot_coord, h, w):
    a, x, y = foot_coord.shape
    foot_coord = foot_coord.reshape([x, y])
    for coord in foot_coord:
        x, y = int(coord[0]), int(coord[1])
        if x > 0+margin and x < field_w-margin and y > 0+margin and y < field_h-margin:
            center = (x, y)
            img = cv2.circle(img, center, 10, (0, 0, 0), thickness=1)
    return img

def exclude_out_of_feild_object(det, trans_coord):
    if trans_coord is None:
        return det
    #射影変換した座標からサッカーコート外のオブジェクトを除外する trans_coordは既に射影変換済みの座標
    a, x, y = trans_coord.shape
    trans_coord = trans_coord.reshape([x, y])
    #フィールド内の選手のindexを調べる
    idx = np.where((trans_coord[:, 0] >= 0+margin) & (trans_coord[:, 0] <= field_w-margin) & (trans_coord[:, 1] >= 0+margin) & (trans_coord[:, 1] <= field_h-margin))
    _, numObj, element = det[idx, :].shape
    return det[idx, :].reshape([numObj, element])

def save_objects(image, det, h, w, save_id):
    save_dir = "./data/test/"
    det[:, 2:4] += det[:, 0:2]
    det[:, 0] *= w
    det[:, 2] *= w
    det[:, 1] *= h
    det[:, 3] *= h
    det = det.to(torch.int)
    for d in det:
        img = image[d[1]:d[3], d[0]:d[2]]
        save_path = save_dir + format(save_id, '04') + '.jpg'
        print(save_path)
        cv2.imwrite(save_path, img)
        save_id += 1
    return save_id

def test_triplet(model):
    root = "../triplet/data/train/"
    dirs = ["blue/", "red/", "black/"]

    imgs = []
    for d in dirs:
        filenames = os.listdir(root + d)
        count = 0
        for n in filenames:
            if count > 220:
                continue
            #print(n)
            img = cv2.imread(root + d + n)
            img = cv2.resize(img, dsize=(64, 32))
            img = img.transpose(2, 0, 1)
            #print(img.shape)
            imgs.append(img)
            count += 1
    imgs = np.array(imgs)
    print(imgs.shape)
    imgs = torch.from_numpy(imgs.astype(np.float32)).clone()
    metric = model(imgs).detach().cpu().numpy()
    metric = metric.reshape(metric.shape[0], metric.shape[1])
    tSNE_metric = TSNE(n_components=2, random_state=0).fit_transform(metric)
    plt.scatter(tSNE_metric[:, 0], tSNE_metric[:, 1])
    plt.colorbar()
    plt.savefig("output.png")
    plt.show()

def greenback(img):
    print("at greenback, input shape : ", img.shape)
    img = img.transpose(1, 2, 0)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lower = (20, 0, 0)
    upper = (45, 255, 255)

    bin_img = cv2.inRange(hsv, lower, upper)
    return ~bin_img

def exclude_judge(det, img, chroma_img, w, h):
    chroma_img_hsv = cv2.cvtColor(chroma_img, cv2.COLOR_RGB2HSV)
    detection = [[int(d[0]*w), int(d[1]*h), int((d[0]+d[2])*w), int((d[1]+d[3])*h)] for d in det]
    idx = []
    for i, d in enumerate(detection):
        person_hsv = chroma_img_hsv[d[1]:d[3], d[0]:d[2]]
        person_img = img[d[1]:d[3], d[0]:d[2], :]
        person_img = cv2.resize(person_img, dsize=(32, 64))
        person_chroma_img_hsv = chroma_img_hsv[d[1]:d[3], d[0]:d[2], :]
        person_chroma_img_hsv = cv2.resize(person_chroma_img_hsv, dsize=(32, 64))
        person_chroma_img = chroma_img[d[1]:d[3], d[0]:d[2], :]
        person_chroma_img = cv2.resize(person_chroma_img, dsize=(32, 64))
        hue_hist = np.histogram(person_chroma_img_hsv[:, :, 0], bins=6)[0]
        saturation_hist = np.histogram(person_chroma_img_hsv[:, :, 1], bins=6)[0]
        value_hist = np.histogram(person_chroma_img_hsv[:, :, 2], bins=6)[0]
        if value_hist[0] > 32*64*0.90:
            continue
        hue_hist = hue_hist[1:]
        #saturation_hist = saturation_hist[1:]
        #value_hist = value_hist[1:]
        #print("hue : ", hue_hist, np.argmax(hue_hist))
        #print("saturation : ", saturation_hist, np.argmax(saturation_hist))
        #print("value : ", value_hist, np.argmax(value_hist))
        #print("")
        idx.append(i)
        #cv2.imshow("img", person_img)
        #cv2.imshow("chroma", person_chroma_img)
        #cv2.waitKey(500)

    return det[idx, :]

def draw_ball(field_image, trans_ball, ball_pic):
    if trans_ball is not None and trans_ball.shape[0] != 0:
        for ball in trans_ball[0]:
            center = (int(ball[0]), int(ball[1]))
            #field_image = cv2.circle(field_image, center, 10, (0, 0, 0), thickness=20)

            field_image = Image.fromarray(field_image).convert('RGBA')
            ball_pic = Image.fromarray(ball_pic).convert('RGBA')  # PIL

            ball_pic_clear = Image.new("RGBA", field_image.size, (255,255,255,0))
            ball_pic_clear.paste(ball_pic, (center[0] - int(40 / 2), center[1] - int(40 / 2)))

            field_image = Image.alpha_composite(field_image, ball_pic_clear)
            field_image = np.array(field_image, dtype='uint8')
    return field_image

def draw_ball2(field_image, trans_ball, ball_pic, ball_mask):
    if trans_ball is not None and trans_ball.shape[0] != 0:
        for ball in trans_ball[0]:
            center = (int(ball[0]), int(ball[1]))

            field_image = np.array(field_image, dtype='float64')

            x_start = center[0]-int(40/2) if center[0]-int(40/2) > 0 else 0
            x_end = center[0] + int(40/2) if center[0]+int(40/2) < field_image.shape[1] else field_image.shape[1]
            y_start = center[1]-int(40/2) if center[1]-int(40/2) > 0 else 0
            y_end = center[1] + int(40/2) if center[1]+int(40/2) < field_image.shape[0] else field_image.shape[0]

            #print("field_size:", field_image.shape)
            #print(x_start , " < ", x_end, "   ", y_start, " < ", y_end)

            #field_image[center[1]-int(40/2):center[1]+int(40/2):,center[0]-int(40/2):center[0]+int(40/2)] *= 1 - ball_mask  # 透過率に応じて元の画像を暗くする。
            #field_image[center[1]-int(40/2):center[1]+int(40/2):,center[0]-int(40/2):center[0]+int(40/2)] += ball_pic * ball_mask  # 貼り付ける方の画像に透過率をかけて加算。
            field_image[y_start:y_end:,x_start:x_end] *= 1 - ball_mask  # 透過率に応じて元の画像を暗くする。
            field_image[y_start:y_end:,x_start:x_end] += ball_pic * ball_mask  # 貼り付ける方の画像に透過率をかけて加算。
            field_image = np.array(field_image, dtype='uint8')

    return field_image

def draw_ball_on_video(img, ball):
    if ball.shape[0] != 0:
        for b in ball:
            #ball = ball[0]
            x, y, w, h = int(b[0]), int(b[1]), int(b[2]), int(b[3])
            #print("here!!! ", x ,y, w, h)
            img = cv2.rectangle(img, (x, y, w, h), (0,0,0), 3)
    return img


@torch.no_grad()
def detect(opt):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    #names[0] = person, names[32] = sports ball
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Load Tracker
    tracker = SSTTracker()
    print("saving frame length = ", tracker.recorder.max_record_frame)
    frame_index = 0

    # 射影変換行列の計算
    #p_original_dlt = np.float32([[0, 700], [229, 699], [446, 693], [756, 685], [892, 684], [1053, 677], [729, 703], [374, 729], [682, 713],[931, 705], [1102, 699], [524, 936], [1406, 808], [1562, 781], [1686, 767], [1889, 734]])
    #print(p_original_dlt, p_original_dlt.shape, type(p_original_dlt))
    p_original_dlt = get_cor_points(opt.source)
    print(p_original_dlt, p_original_dlt.shape, type(p_original_dlt))
    p_trans_dlt = np.float32([[0, 0], [132.81, 0], [239.06, 0], [440.94, 0], [547.19, 0], [680, 0], [255, 109.38], [132.81, 167.34],[239.06, 167.34], [440.94, 167.34], [547.19, 167.34], [0, 525], [212.5, 525], [255, 525], [467.5, 525],[680, 525]])
    M = get_homography(p_original_dlt, p_trans_dlt)

    #俯瞰図のための画像読み込み
    field_image = cv2.imread('field.png')
    field_image = cv2.resize(field_image, dsize=(field_w, field_h))
    field_image_0 = field_image.copy()

    #俯瞰図の録画
    vid_writer2 = cv2.VideoWriter("field_video.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 60, (field_w, field_h))

    ball_pic = cv2.imread("ball.png", -1)
    ball_pic = cv2.resize(ball_pic, dsize=(40, 40))

    ball_mask = ball_pic[:, :, 3]
    ball_mask = cv2.cvtColor(ball_mask, cv2.COLOR_GRAY2BGR)  # 3色分に増やす。
    ball_mask = ball_mask / 255  # 0-255だと使い勝手が悪いので、0.0-1.0に変更。
    ball_pic = ball_pic[:, :, :3]

    save_id = 1

    # Run inference(main loop)
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        fps = vid_cap.get(cv2.CAP_PROP_FPS)
        if frame_index < 0:
            frame_index += 1
            continue

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        field_image = field_image_0.copy()

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms,
                                   max_det=opt.max_det)
        t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            print()

            #det[x, y, w, h, ?, ?]
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if opt.save_crop else im0  # for opt.save_crop
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # mask person and ball
                det_int = det.int()
                mask_person = det_int[:, -1] == 0
                mask_ball = det_int[:, -1] == 32
                #mask = torch.logical_or(mask_person, mask_ball)
                ball = det[mask_ball]
                det = det[mask_person]


                det[:, 0:4] = xyxy2xywh(det[:, 0:4]) # to xywh
                det[:, 0:2] -= det[:, 2:4]/2 # fix coord
                det[:, 0:4] /= gn # normalize

                ball[:, 0:4] = xyxy2xywh(ball[:, 0:4])  # to xywh
                ball[:, 0:2] -= ball[:, 2:4] / 2  # fix coord
                draw_ball_on_video(im0s, ball)
                ball[:, 0:4] /= gn  # normalize

                h, w, _ = im0s.shape
                trans_coord = transform_coord(det[:, 0:4], h, w, M)
                trans_ball = transform_coord(ball[:, 0:4], h, w, M)

                det = exclude_out_of_feild_object(det, trans_coord)
                ball = exclude_out_of_feild_object(ball, trans_ball)
                trans_ball = transform_coord(ball[:, 0:4], h, w, M)

                track_img = tracker.update(im0s, det[:, 0:4], True, frame_index)#追跡の肝
                tracker.tracks.set_verocity2(frame_index, tracker.recorder, M, h, w, fps)
                if frame_index > 0:
                    occupied_area, occupied_area_prob = tracker.tracks.get_occupied_area3(field_h, field_w, velocity_min, velocity_max)

                cv2.imshow("tracking", track_img)
                cv2.waitKey(1)
                if frame_index > 0:
                    blended = tracker.tracks.draw_occupied2(field_image.copy(), occupied_area, occupied_area_prob, field_w, field_h)
                    blended = tracker.tracks.draw_on_field2(blended, im0s, frame_index, M, tracker.recorder, h, w)  # 追跡中の選手を俯瞰図的に描画(切り抜き)
                    blended = draw_ball2(blended, trans_ball, ball_pic, ball_mask)  # ボールを描画
                    cv2.imshow("blended", blended)
                    cv2.waitKey(1)
                    vid_writer2.write(blended)

                frame_index += 1

                # Write results
                for *xyxy, conf, cls_id in reversed(det):
                    c = int(cls_id)
                    if (c == 0 or c == 32):
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        xywh = [round(xywh[n], 3) for n in range(len(xywh))]
                        line = (cls_id, *xywh, conf) if opt.save_conf else (cls_id, *xywh)  # label format

                        if save_img or opt.save_crop or view_img:  # Add bbox to image
                            c = int(cls_id)  # integer class
                            label = None if opt.hide_labels else (names[c] if opt.hide_conf else f'{names[c]} {conf:.2f}')
                            plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=opt.line_thickness)
                            if opt.save_crop:
                                save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    #vid_writer.write(im0)
                    vid_writer.write(track_img)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=1920, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.30, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.10, help='IOU threshold for NMS')
    parser.add_argument('--max-det', type=int, default=100, help='maximum number of detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('tensorboard', 'pycocotools', 'thop'))

    if opt.update:  # update all models (to fix SourceChangeWarning)
        for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
            detect(opt=opt)
            strip_optimizer(opt.weights)
    else:
        detect(opt=opt)
