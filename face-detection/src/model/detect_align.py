import cv2
import numpy as np
from PIL import Image
import onnxruntime

from src.base.logger import Logger
from src.model.detect_align_utils import *
from src.model.pipeline_utils.process import crop_img
from src.base.logger import Logger
from src.utils import CVInvalidInputsException

class Detect_Align(object):
    def __init__(self, model):
        self.detect_model = model["detection_model"]
        self.align_model = model["alignment_model"]
        self.align_target_size = 120
        self.parse_target_size = 513     

    def get_bbox(self, img: Image , filter_confidence=0.6, filter_size=20, face_limit_num=1):
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

        bboxes, _ = self.detect_model.single_detect(img, thresh=0.5, input_size=(640, 640)) # bboxes:(np.ndarray), should not be changed for thresh and input_size

        if face_limit_num < -1 or face_limit_num == 0:
            raise CVInvalidInputsException(ret_info="Invalid face_limit_num input.")
        elif len(bboxes) == 0: # if no bounding box detected
            return [{"bbox": 'None'}]
        elif face_limit_num == -1 or face_limit_num > len(bboxes):
            bboxes = bboxes
        else:
            bboxes = bboxes[np.argsort(bboxes[:, 3] - bboxes[:, 1])[::-1][:face_limit_num]]

        bbox_result_list = []
        for bbox in bboxes:
            roi_box = post_detection_process(
                img, bbox, filter_confidence, filter_size
            ) # filter out bbox with low confidence and too small bbox size
            if roi_box is None:  
                continue

            bbox_result_list.append({'bbox': roi_box.astype(int).tolist()})

        if len(bbox_result_list) == 0:
            return [{"bbox": "None"}]
        else:
            return bbox_result_list

    def get_face_landmark(self, img: Image, filter_confidence=0.6, filter_size=20, face_limit_num=1):
        detect_result_list = self.get_bbox(img, filter_confidence, filter_size, face_limit_num)
        if detect_result_list[0]['bbox'] == "None":
            return [{"landmark": "None", "bbox": "None"}]

        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

        align_result_list = []
        for detect_result in detect_result_list:
            bbox = detect_result['bbox']
            np_bbox = np.array(bbox)
            img_crop = crop_img(img, np_bbox)
            img_crop = cv2.resize(img_crop, dsize=( self.align_target_size,  self.align_target_size), interpolation=cv2.INTER_LINEAR)              
            landmark = predict_face_landmark(self.align_model, img_crop, np_bbox).astype(int).tolist()
            align_result_list.append({"landmark": landmark, "bbox": bbox})
            
        return align_result_list

    def plot_bbox_landmark(self, img: Image, filter_confidence=0.6, filter_size=20, face_limit_num=1):
        align_result_list = self.get_face_landmark(img, filter_confidence, filter_size, face_limit_num)
        if align_result_list[0]['bbox'] == "None":
            return [{'img_bytestring': "None", "landmark": "None", "bbox": "None"}]

        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

        return_result_list = []
        for align_result in align_result_list:
            bbox = align_result['bbox']
            landmark = align_result['landmark']
    
            plot_result = render_bbox_landmark(img, bbox, landmark)

            pil_image=Image.fromarray(cv2.cvtColor(plot_result, cv2.COLOR_BGR2RGB))
            img_bytestring = get_img_string(pil_image)
            return_result_list.append({'img_bytestring': img_bytestring, "landmark": landmark, "bbox": bbox})
        
        return return_result_list  

    def get_align_crop(self, img: Image, filter_confidence=0.6, filter_size=20, face_limit_num=1):
        align_result_list = self.get_face_landmark(img, filter_confidence, filter_size, face_limit_num)
        if align_result_list[0]['bbox'] == "None":
            return [{'img_bytestring': "None", "landmark": "None", "bbox": "None"}]

        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

        return_result_list = []
        for align_result in align_result_list:
            bbox = align_result['bbox']
            landmark = align_result['landmark']
            
            crop_result = align_crop_face(img, bbox, landmark)

            pil_image=Image.fromarray(cv2.cvtColor(crop_result, cv2.COLOR_BGR2RGB))
            img_bytestring = get_img_string(pil_image)    
            return_result_list.append({'img_bytestring': img_bytestring, "landmark": landmark, "bbox": bbox})

        return return_result_list   

def get_model(detect_path, align_path):
    """Load and prepare models in pipeline"""

    Logger.info("Loading detection model")  
    detect_model = SCRFD(model_file=detect_path)
    detect_model.prepare(1)
    Logger.info("Loading detection model done")
    Logger.info("Loading alignmet model") 
    align_model = onnxruntime.InferenceSession(align_path, providers=["CUDAExecutionProvider"])
    Logger.info("Loading alignmet model done") 
    return {"detection_model": detect_model, "alignment_model": align_model}


