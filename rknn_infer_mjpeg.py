#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import sys
import numpy as np
from http.server import BaseHTTPRequestHandler, HTTPServer

from rknnlite.api import RKNNLite

# ==================== 配置 ====================
MODEL_PATH  = "/userdata/rknn_yolov8_demo_new/model/best_int8_9t.rknn"
DEVICE      = "/dev/video11"
WIDTH       = 1280
HEIGHT      = 720
FPS         = 30
PORT        = 8080

IMG_SIZE    = (640, 640)
OBJ_THRESH  = 0.5
NMS_THRESH  = 0.45
CLASSES     = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")

PIPELINE = (
    "v4l2src device=" + DEVICE +
    " ! video/x-raw,format=NV12,width=" + str(WIDTH) + ",height=" + str(HEIGHT) + ",framerate=" + str(FPS) + "/1"
    " ! videoconvert"
    " ! video/x-raw,format=BGR"
    " ! appsink drop=true max-buffers=1"
)

# ==================== RKNN ====================
print("[INFO] Loading model: " + MODEL_PATH, flush=True)
rknn = RKNNLite()
ret = rknn.load_rknn(MODEL_PATH)
if ret != 0:
    print("[ERROR] load_rknn failed: " + str(ret), flush=True)
    sys.exit(1)
ret = rknn.init_runtime()
if ret != 0:
    print("[ERROR] init_runtime failed: " + str(ret), flush=True)
    sys.exit(1)
print("[INFO] RKNN model loaded OK", flush=True)

# ==================== 后处理 ====================
def dfl_numpy(position):
    n, c, h, w = position.shape
    p_num = 4
    mc = c // p_num
    y = position.reshape(n, p_num, mc, h, w)
    y = np.exp(y - y.max(axis=2, keepdims=True))
    y = y / y.sum(axis=2, keepdims=True)
    acc = np.arange(mc).reshape(1, 1, mc, 1, 1).astype(np.float32)
    y = (y * acc).sum(axis=2)
    return y

def box_process(position):
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([IMG_SIZE[1] // grid_h, IMG_SIZE[0] // grid_w]).reshape(1, 2, 1, 1)

    position = dfl_numpy(position)
    box_xy  = grid + 0.5 - position[:, 0:2, :, :]
    box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
    xyxy = np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)
    return xyxy

def filter_boxes(boxes, box_confidences, box_class_probs):
    box_confidences = box_confidences.reshape(-1)
    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)
    _class_pos = np.where(class_max_score * box_confidences >= OBJ_THRESH)
    scores = (class_max_score * box_confidences)[_class_pos]
    boxes = boxes[_class_pos]
    classes = classes[_class_pos]
    return boxes, classes, scores

def nms_boxes(boxes, scores):
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    areas = w * h
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])
        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    return np.array(keep)

def post_process(input_data):
    boxes, scores, classes_conf = [], [], []
    default_branch = 3
    pair_per_branch = len(input_data) // default_branch
    for i in range(default_branch):
        boxes.append(box_process(input_data[pair_per_branch * i]))
        classes_conf.append(input_data[pair_per_branch * i + 1])
        scores.append(np.ones_like(input_data[pair_per_branch * i + 1][:, :1, :, :], dtype=np.float32))

    def sp_flatten(_in):
        ch = _in.shape[1]
        _in = _in.transpose(0, 2, 3, 1)
        return _in.reshape(-1, ch)

    boxes = np.concatenate([sp_flatten(_v) for _v in boxes])
    classes_conf = np.concatenate([sp_flatten(_v) for _v in classes_conf])
    scores = np.concatenate([sp_flatten(_v) for _v in scores])

    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf)

    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c_ = classes[inds]
        s = scores[inds]
        keep = nms_boxes(b, s)
        if len(keep) != 0:
            nboxes.append(b[keep])
            nclasses.append(c_[keep])
            nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    return np.concatenate(nboxes), np.concatenate(nclasses), np.concatenate(nscores)

# ==================== letterbox ====================
def letterbox(im, new_shape=(640, 640), color=(0, 0, 0)):
    shape = im.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (left, top)

def my_process(frame_bgr):
    img, ratio, (dw, dh) = letterbox(frame_bgr, IMG_SIZE, color=(0, 0, 0))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0)
    outputs = rknn.inference(inputs=[img])
    boxes, classes, scores = post_process(outputs)
    return boxes, classes, scores, ratio, dw, dh
# ==================== 摄像头 ====================
print("[INFO] Opening camera: " + PIPELINE, flush=True)
cap = cv2.VideoCapture(PIPELINE, cv2.CAP_GSTREAMER)
if not cap.isOpened():
    print("[ERROR] Cannot open camera", flush=True)
    sys.exit(1)
print("[INFO] Camera opened OK", flush=True)

# ==================== MJPEG HTTP ====================
class MJPEGHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/":
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            try:
                boxes, classes, scores, ratio, dw, dh = my_process(frame)
            except Exception as e:
                print("[ERROR] my_process: " + str(e), flush=True)
                boxes = None

            if boxes is not None:
                for box, score, cl in zip(boxes, scores, classes):
                    x1 = int((box[0] - dw) / ratio)
                    y1 = int((box[1] - dh) / ratio)
                    x2 = int((box[2] - dw) / ratio)
                    y2 = int((box[3] - dh) / ratio)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, CLASSES[int(cl)] + " " + ("%.2f" % score),
                                (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (0, 0, 255), 2)

            ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ok:
                continue
            data = jpg.tobytes()
            try:
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(("Content-Length: " + str(len(data)) + "\r\n\r\n").encode())
                self.wfile.write(data)
                self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                break
            except Exception as e:
                print("[ERROR] send: " + str(e), flush=True)
                break

    def log_message(self, fmt, *args):
        pass


if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", PORT), MJPEGHandler)
    print("[INFO] MJPEG server on http://0.0.0.0:" + str(PORT), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        rknn.release()