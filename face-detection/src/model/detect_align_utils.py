import numpy as np
import cv2
import os.path as osp
import base64
from io import BytesIO
import torchvision.transforms as transforms
from typing import List

from src.model.pipeline_utils.inference import  predict_sparseVert
from src.model.pipeline_utils.ddfa import ToTensor, Normalize
from src.model.pipeline_utils.process import distance2bbox, distance2kps
from src.model.pipeline_utils.process import crop_img, FaceAligner

def get_img_string(pil_img):
    buffered = BytesIO()
    pil_img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue())

def render_bbox_landmark(img: np.ndarray, bbox: List, landmark: List)-> np.ndarray:
    """Plot bbox and landmark on image

    """
    temp_img = img.copy()
    x1,y1,x2,y2 = bbox
    cv2.rectangle(temp_img, (x1,y1)  , (x2,y2) , (255,0,0) , 2)          
    landmark = np.array(landmark)    
    for p in landmark:
        cv2.circle(temp_img, (p[0], p[1]), 3, (0, 255, 0), -1)
    return temp_img

def align_crop_face(img: np.ndarray, bbox: List, landmark: List)-> np.ndarray:
    """Get crop face using face landmark

    """    
    temp_img = img.copy()
    landmark = np.array(landmark) 
    bbox = np.array(bbox)
    alginer = FaceAligner(shape=landmark, desiredFaceWidth=120)
    img_align, status = alginer.align(temp_img)  # can't align will return False
    if status == False: # can't align will crop face using original bbox
        result_img_array = crop_img(temp_img, bbox)
    else:
        roi_box_final = alginer.get_align_box(img_align)
        result_img_array = crop_img(img_align, roi_box_final)   
    return result_img_array 


def pre_detect(img: np.ndarray, input_size):
    """Get resized image and scale

    Args:
        img (np.ndarray): cv2 image array
        input_size (tuple): (nxn)
    Returns:
        det_img(np.ndarray)     
        det_scale(float)
    """
    im_ratio = float(img.shape[0]) / img.shape[1]
    model_ratio = float(input_size[1]) / input_size[0]
    if im_ratio > model_ratio:
        new_height = input_size[1]
        new_width = int(new_height / im_ratio)
    else:
        new_width = input_size[0]
        new_height = int(new_width * im_ratio)
    det_scale = float(new_height) / img.shape[0]
    resized_img = cv2.resize(img, (new_width, new_height))
    det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
    det_img[:new_height, :new_width, :] = resized_img
    return det_img, det_scale

def post_detection_process(img: np.ndarray, bbox, filter_confidence=0.6, filter_size=20):
    """Get cropped image and corresponding roi_box after filtering

    Args:
        img (np.ndarray): original cv2 image array
        bbox (list): original bbox from detection model
        filter_confidence (float): confidence threshold ratio, between 0 and 1
        filter_size (int): filter size
    """
    roi_box = bbox

    h_len = roi_box[3] - roi_box[1]
    w_len = roi_box[2] - roi_box[0]

    HCenter = (bbox[1] + bbox[3]) / 2
    WCenter = (bbox[0] + bbox[2]) / 2
    side_len = roi_box[3] - roi_box[1]
    margin = side_len * 1.2 // 2
    roi_box[0], roi_box[1], roi_box[2], roi_box[3] = (
        WCenter - margin,
        HCenter - margin,
        WCenter + margin,
        HCenter + margin,
    )

    roi_box[0] = max(roi_box[0], 0)
    roi_box[1] = max(roi_box[1], 0)
    roi_box[2] = min(roi_box[2], img.shape[1])
    roi_box[3] = min(roi_box[3], img.shape[0])

    if bbox[-1] < filter_confidence:
        return None
    if h_len < filter_size or w_len < filter_size:
        return None
    else:
        return roi_box[:-1]

def predict_face_landmark(align_model, img_detection_crop, roi_box):
    """Get face landmark preidction

    Args:
        img_detection_crop (np.ndarray): cv2 image cropped array processed by detection model
        roi_box (np.ndarray): roi_box array predicted by detection model
    Returns:
        shape (np.ndarray): (2, 68)        
    """          
    transform = transforms.Compose([ToTensor(), Normalize(mean=127.5, std=128)])
    input = transform(img_detection_crop).unsqueeze(0).cpu()
    param = align_model.run(["output"], {"input": input.numpy()})[0].flatten().astype(np.float32)

    lmks = predict_sparseVert(param, roi_box, transform=True)
    shape = lmks[:2, :].T  # facial landmark
    return shape


class SCRFD: # Detection Model
    def __init__(self, model_file=None, session=None):
        import onnxruntime

        self.model_file = model_file
        self.session = session
        self.taskname = "detection"
        self.batched = False
        if self.session is None:
            assert self.model_file is not None
            assert osp.exists(self.model_file)

            self.session = onnxruntime.InferenceSession(
                self.model_file, providers=["CUDAExecutionProvider"])
        self.center_cache = {}
        self.nms_thresh = 0.4
        self._init_vars()

    def _init_vars(self):
        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape
        if isinstance(input_shape[2], str):
            self.input_size = None
        else:
            self.input_size = tuple(input_shape[2:4][::-1])
        input_name = input_cfg.name
        outputs = self.session.get_outputs()
        if len(outputs[0].shape) == 3:
            self.batched = True
        output_names = []
        for o in outputs:
            output_names.append(o.name)
        self.input_name = input_name
        self.output_names = output_names
        self.use_kps = False
        self._num_anchors = 1
        if len(outputs) == 6:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
        elif len(outputs) == 9:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
            self.use_kps = True
        elif len(outputs) == 10:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
        elif len(outputs) == 15:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
            self.use_kps = True

    def prepare(self, ctx_id, **kwargs):
        if ctx_id < 0:
            self.session.set_providers(["CUDAExecutionProvider"])
        nms_thresh = kwargs.get("nms_thresh", None)
        if nms_thresh is not None:
            self.nms_thresh = nms_thresh
        input_size = kwargs.get("input_size", None)
        if input_size is not None:
            if self.input_size is not None:
                print("warning: det_size is already set in scrfd model, ignore")
            else:
                self.input_size = input_size

    def single_detect(self, img, thresh=0.5, input_size=None, max_num=0, metric="default"):
        assert input_size is not None or self.input_size is not None
        input_size = self.input_size if input_size is None else input_size

        det_img, det_scale = pre_detect(img, input_size)

        scores_list = []
        bboxes_list = []
        kpss_list = []
        input_size = tuple(det_img.shape[0:2][::-1])
        blob = cv2.dnn.blobFromImage(det_img, 1.0 / 128, input_size, (127.5, 127.5, 127.5), swapRB=True)
        net_outs = self.session.run(self.output_names, {self.input_name: blob})

        input_height = blob.shape[2]
        input_width = blob.shape[3]
        fmc = self.fmc
        for idx, stride in enumerate(self._feat_stride_fpn):
            if self.batched:
                scores = net_outs[idx][0]
                bbox_preds = net_outs[idx + fmc][0]
                bbox_preds = bbox_preds * stride
                if self.use_kps:
                    kps_preds = net_outs[idx + fmc * 2][0] * stride
            else:
                scores = net_outs[idx]
                bbox_preds = net_outs[idx + fmc]
                bbox_preds = bbox_preds * stride
                if self.use_kps:
                    kps_preds = net_outs[idx + fmc * 2] * stride

            height = input_height // stride
            width = input_width // stride
            K = height * width
            key = (height, width, stride)
            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)

                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if self._num_anchors > 1:
                    anchor_centers = np.stack([anchor_centers] * self._num_anchors, axis=1).reshape((-1, 2))
                if len(self.center_cache) < 100:
                    self.center_cache[key] = anchor_centers

            pos_inds = np.where(scores >= thresh)[0]
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
            if self.use_kps:
                kpss = distance2kps(anchor_centers, kps_preds)
                kpss = kpss.reshape((kpss.shape[0], -1, 2))
                pos_kpss = kpss[pos_inds]
                kpss_list.append(pos_kpss)

        # return (det, kpss)
        return self.get_final_result(
            scores_list, bboxes_list, kpss_list, det_scale, img.shape[0], img.shape[1], max_num, metric,
        )

    def get_final_result(
        self, scores_list, bboxes_list, kpss_list, det_scale, img_shape_0, img_shape_1, max_num=0, metric="default",
    ):
        """
        Returns:
            det(np.ndarray): bbox
        """        
        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        bboxes = np.vstack(bboxes_list) / det_scale
        if self.use_kps:
            kpss = np.vstack(kpss_list) / det_scale
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = self.nms(pre_det)
        det = pre_det[keep, :]
        if self.use_kps:
            kpss = kpss[order, :, :]
            kpss = kpss[keep, :, :]
        else:
            kpss = None
        if max_num > 0 and det.shape[0] > max_num:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            img_center = img_shape_0 // 2, img_shape_1 // 2
            offsets = np.vstack([(det[:, 0] + det[:, 2]) / 2 - img_center[1], (det[:, 1] + det[:, 3]) / 2 - img_center[0], ])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            if metric == "max":
                values = area
            else:
                values = area - offset_dist_squared * 2.0  # some extra weight on the centering
            bindex = np.argsort(values)[::-1]  # some extra weight on the centering
            bindex = bindex[0:max_num]
            det = det[bindex, :]
            if kpss is not None:
                kpss = kpss[bindex, :]
        return (det, kpss)

    def nms(self, dets):
        thresh = self.nms_thresh
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep

