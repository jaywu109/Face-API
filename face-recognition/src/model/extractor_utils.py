from PIL.Image import AFFINE
from numpy.linalg import norm
import numpy as np
import cv2
import os.path as osp
import torchvision.transforms as transforms
from PIL import Image, UnidentifiedImageError
from io import BytesIO
from src.base.logger import Logger
import validators
import urllib
import binascii
import base64

from src.model.pipeline_utils.inference import crop_img, predict_sparseVert
from src.model.pipeline_utils.ddfa import ToTensor, Normalize
from src.model.pipeline_utils.process import crop_img, FaceAligner, FACIAL_LANDMARKS_68_IDXS, distance2bbox, distance2kps
from src.base.exceptions import InternalProgramException
from src.utils import  CVInvalidInputsException, get_detailed_error

def response_handler(response, exception_retInfo_list, exception_retInfo):
    if 'retInfo' in response.keys() and response['retInfo'] == 'OK':  
        return response['retData']
    else:
        if response['retInfo'] in exception_retInfo_list:            
            raise InternalProgramException(ret_info=response['retInfo'])
        else:    
            raise InternalProgramException(ret_info=exception_retInfo)

def get_image(image_url_or_bytestring)-> Image:

    if validators.url(image_url_or_bytestring):
        try:
            image = Image.open(urllib.request.urlopen(image_url_or_bytestring)).convert("RGB")
        except urllib.error.HTTPError as e:
            Logger.error(f"Loading Error: {get_detailed_error(e)}")  
            raise CVInvalidInputsException(ret_info="Invalid image url.")
    else:
        try:
            img_bytes = base64.b64decode(image_url_or_bytestring)
            image = Image.open(BytesIO(img_bytes)).convert("RGB")
        except (UnidentifiedImageError, binascii.Error) as e:
            Logger.error(f"Loading Error: {get_detailed_error(e)}")  
            raise CVInvalidInputsException(ret_info="Invalid image content.")     

    return image      

def get_image_url(image_url):

    try:
        image = Image.open(urllib.request.urlopen(image_url)).convert("RGB")
    except urllib.error.HTTPError as e:
        Logger.error(f"Loading Error: {get_detailed_error(e)}")  
        raise CVInvalidInputsException(ret_info="Invalid image url.")

    return image  

def get_image_bytestring(image_bytestring):

    try:
        img_bytes = base64.b64decode(image_bytestring)
        image = Image.open(BytesIO(img_bytes)).convert("RGB")
    except (UnidentifiedImageError, binascii.Error) as e:
        Logger.error(f"Loading Error: {get_detailed_error(e)}")  
        raise CVInvalidInputsException(ret_info="Invalid image content.")     

    return image      

def align_process(img, np_bbox, landmark, target_size):
    alginer = FaceAligner(shape=landmark, desiredFaceWidth=target_size)
    img_align, status = alginer.align(img)  # can't align will return None
    if status == False:
        align_result =  crop_img(img, np_bbox)
    else:
        M = alginer.M  # affine matrix
        shape_trans = cv2.transform(np.array([landmark]), M)[
            0
        ]  # transform facial landmark to new direction using info provided by align model
        shape_min_x = max(shape_trans[:, 0].min(), 0)
        shape_max_x = max(shape_trans[:, 0].max(), 0)
        shape_min_y = min(shape_trans[:, 1].min(), img_align.shape[0])
        shape_max_y = min(shape_trans[:, 1].max(), img_align.shape[0])

        (nStart, nEnd) = FACIAL_LANDMARKS_68_IDXS["nose"]
        nosePts = shape_trans[nStart:nEnd]
        noseCenter = nosePts.mean(axis=0).astype(int).astype(float)
        offset_H = 1.1 * max(shape_max_y - noseCenter[1], noseCenter[1] - shape_min_y)
        offset_W = 1.1 * max(shape_max_x - noseCenter[0], noseCenter[0] - shape_min_x)

        roi_box_final = np.array(
            [noseCenter[0] - offset_W, noseCenter[1] - offset_H, noseCenter[0] + offset_W, noseCenter[1] + offset_H, -1, ]
        )
        roi_box_final[0] = max(roi_box_final[0], 0)
        roi_box_final[1] = max(roi_box_final[1], 0)
        roi_box_final[2] = min(roi_box_final[2], img_align.shape[1])
        roi_box_final[3] = min(roi_box_final[3], img_align.shape[0])
        roi_box_final = roi_box_final.astype(int)  # use face landmark to crop image
        align_result = crop_img(img_align, roi_box_final)

    if align_result.shape[0] > 0 and align_result.shape[1] > 0:
        factor_0 = target_size / align_result.shape[0]
        factor_1 = target_size / align_result.shape[1]
        factor = min(factor_0, factor_1)

        dsize = (
            int(align_result.shape[1] * factor),
            int(align_result.shape[0] * factor),
        )
        align_result = cv2.resize(align_result, dsize)

        diff_0 = target_size - align_result.shape[0]
        diff_1 = target_size - align_result.shape[1]

        align_result = np.pad(
            align_result, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2), (0, 0),), "constant",
        )

    if align_result.shape[0] != target_size or align_result.shape[1] != target_size:
        align_result = cv2.resize(align_result, (target_size, target_size))        

    return align_result


class SCRFD:
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
            self.session.set_providers(["CPUExecutionProvider"])
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


def pre_detect(img, input_size):
    """Get resized image and scale

    Args:
        img (array): cv2 image array
        input_size (tuple): (nxn)
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
