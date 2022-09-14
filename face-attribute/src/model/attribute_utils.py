import numpy as np
import cv2
import torch
import base64
from PIL import Image, UnidentifiedImageError
import validators
import binascii
import urllib
from io import BytesIO
from src.base.logger import Logger
import torchvision.transforms as transforms

import src.model.parsing.reference_impl as ref
from src.utils import CVInvalidInputsException, get_detailed_error
from src.base.exceptions import InternalProgramException
from src.model.parsing.utils import flip_image
from src.model.pipeline_utils.process import crop_img, FaceAligner, FACIAL_LANDMARKS_68_IDXS

def response_handler(response, exception_retInfo_list, exception_retInfo):
    if 'retInfo' in response.keys() and response['retInfo'] == 'OK':  
        return response['retData']
    else:
        if response['retInfo'] in exception_retInfo_list:            
            raise InternalProgramException(ret_info=response['retInfo'])
        else:    
            raise InternalProgramException(ret_info=exception_retInfo)

def get_image(image_url_or_bytestring):

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

def predict_face_parsing(parse_model, img, bbox, parse_target_size=513):
    """Get face parsing preidction

    Args:
        img (array): cv2 original image array
        bbox (array): bbox array predicted by detection model
    Returns:
        low(np.ndarray, dtype=float32): (N, 512, 65, 65)        
        high(np.ndarray, dtype=float32): (N, 512, 65, 65)        
        logits(np.ndarray, dtype=float32): (N, 14, 65, 65)          
    """          
    bbox = np.array([bbox]) # convert to numpy array and add a dimension    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])    
    h, w = img.shape[:2]
    num_faces = len(bbox)
    imgs = [
        ref.roi_tanh_polar_warp(img, b, *(parse_target_size, parse_target_size), keep_aspect_ratio=True) # roi_tanh_polar_warp: https://github.com/ibug-group/roi_tanh_warping
        for b in bbox
    ]
    img_flip, bboxes_flip = flip_image(img, bbox) # add horizontal flip image as another input image
    imgs += [
        ref.roi_tanh_polar_warp(img_flip, b, *(parse_target_size, parse_target_size), keep_aspect_ratio=True)
        for b in bboxes_flip
    ]
    bbox = np.concatenate([bbox, bboxes_flip], axis=0)
    num_faces *= 2
    imgs = [transform(img) for img in imgs]
    img = torch.stack(imgs) 
    low, high, logits, _ = parse_model.run(None, {'img': img.numpy()})  
    return low, high, logits

def align_process(img, np_bbox, landmark, target_size):
    """Get align crop face using face landmark recieved from Face Detection API
        
    """          
    alginer = FaceAligner(shape=landmark, desiredFaceWidth=target_size)
    img_align, status = alginer.align(img)  # can't align will return False
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



