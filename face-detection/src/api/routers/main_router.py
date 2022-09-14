import os

from numpy import single
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
from fastapi import Body
from fastapi import Depends, Query

import base64
import binascii
import urllib
from io import BytesIO
from typing import List
from PIL import Image, UnidentifiedImageError

from src.base.exceptions import InternalProgramException
from src.base.logger import Logger
from src.base.router import APIRouter
from src.base.configs import get_conf
from src.base.schema import APIResponse
from src.utils import CVInvalidInputsException, get_detailed_error
from src.api.model import image_io
from src.model.detect_align import get_model, Detect_Align

model_conf = get_conf("MODEL")

detection_model_path = os.environ.get("DETECTION_MODEL_PATH", model_conf["DETECTION_MODEL_PATH"])
alignment_model_path = os.environ.get("ALIGNMENT_MODEL_PATH", model_conf["ALIGNMENT_MODEL_PATH"])
model = get_model(detection_model_path, alignment_model_path)
Logger.info("Loading model finish")  
pipeline = Detect_Align(model)
Logger.info("Import pipeline finish")  
detection_router = APIRouter(prefix="/detect", tags=["Detection Process"])
alignment_router = APIRouter(prefix="/align", tags=["Alignment Process"])
image_router = APIRouter(prefix="/image", tags=["Return Image Process"])
Logger.info("Import router and response finish")  

class InputQueryParams:
    def __init__(
        self,
        confidence: float = Query(0.6, description="Confidence Threshold (0.0 ~ 1.0). Should be higher than \"0.6\""),
        size: int = Query(20, description="Face Area Pixel Threshold. Should be higher than \"20\""),
        face_limit_num: int = Query(1, description="Number of faces to be returned in the image. If larger than number of faces in the image, return all faces. Return all faces if \"-1\""),
    ):
        self.confidence = confidence
        self.size = size
        self.face_limit_num = face_limit_num

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

def inference_process(function, image, confindence, size, face_limit_num):

    try:
        result = function(image, confindence, size, face_limit_num)
    except (RuntimeError, OSError) as e:
        Logger.error(f"Inference Error: {get_detailed_error(e)}")  
        raise InternalProgramException(ret_info="Inference is crashed.")
    except ValueError as e:
        Logger.error(f"Inference Error: {get_detailed_error(e)}")  
        raise InternalProgramException(ret_info=str(e))

    return result

Logger.info("import helper function finish")  

@detection_router.post("/get_bbox_url", response_model=List[image_io.UrlBboxResult])
async def get_bbox_url(request: image_io.ImageUrlInputSchema = Body(..., examples=image_io.image_input_examples_url), params: InputQueryParams = Depends()):
    
    data = request.dict()
    image_url = data["url"]  

    image = get_image_url(image_url)
    result = inference_process(pipeline.get_bbox, image, params.confidence, params.size, params.face_limit_num)
    response = APIResponse(retData=result)
    return response

@detection_router.post("/get_bbox_bytestring", response_model=List[image_io.StrBboxResult])
async def get_bbox_bytestring(request: image_io.ImageBytestringInputSchema = Body(..., examples=image_io.image_input_examples_str), params: InputQueryParams = Depends()):
    
    data = request.dict()
    image_bytestring = data["bytestring"]  

    image = get_image_bytestring(image_bytestring)
    result = inference_process(pipeline.get_bbox, image, params.confidence, params.size, params.face_limit_num)
    response = APIResponse(retData=result)
    return response

@alignment_router.post("/get_landmark_url", response_model=List[image_io.UrlLandmarkResult])
async def get_landmark_url(request: image_io.ImageUrlInputSchema = Body(..., examples=image_io.image_input_examples_url), params: InputQueryParams = Depends()):
    data = request.dict()
    image_url = data["url"]  

    image = get_image_url(image_url)
    result = inference_process(pipeline.get_face_landmark, image, params.confidence, params.size, params.face_limit_num)
    response = APIResponse(retData=result)
    return response

@alignment_router.post("/get_landmark_bytestring", response_model=List[image_io.StrLandmarkResult])
async def get_landmark_bytestring(request: image_io.ImageBytestringInputSchema = Body(..., examples=image_io.image_input_examples_str), params: InputQueryParams = Depends()):
    data = request.dict()
    image_bytestring = data["bytestring"]  

    image = get_image_bytestring(image_bytestring)
    result = inference_process(pipeline.get_face_landmark, image, params.confidence, params.size, params.face_limit_num)
    response = APIResponse(retData=result)
    return response

@image_router.post("/get_align_crop_url", response_model=List[image_io.UrlAlignCropResult])
async def get_align_crop(request: image_io.ImageUrlInputSchema = Body(..., examples=image_io.image_input_examples_url), params: InputQueryParams = Depends()):
    data = request.dict()
    image_url = data["url"]  

    image = get_image_url(image_url)
    result = inference_process(pipeline.get_align_crop, image, params.confidence, params.size, params.face_limit_num)
    response = APIResponse(retData=result)
    return response

@image_router.post("/get_align_crop_bytestring", response_model=List[image_io.StrAlignCropResult])
async def get_align_crop(request: image_io.ImageBytestringInputSchema = Body(..., examples=image_io.image_input_examples_str), params: InputQueryParams = Depends()):
    data = request.dict()
    image_bytestring = data["bytestring"]  

    image = get_image_bytestring(image_bytestring)
    result = inference_process(pipeline.get_align_crop, image, params.confidence, params.size, params.face_limit_num)
    response = APIResponse(retData=result)
    return response  

@image_router.post("/plot_bbox_landmark_url", response_model=List[image_io.UrlPlotBboxLandmarkResult])
async def plot_bbox_landmark(request: image_io.ImageUrlInputSchema = Body(..., examples=image_io.image_input_examples_url), params: InputQueryParams = Depends()):
    data = request.dict()
    image_url = data["url"]  

    image = get_image_url(image_url)
    result = inference_process(pipeline.plot_bbox_landmark, image, params.confidence, params.size, params.face_limit_num)
    response = APIResponse(retData=result)
    return response    

@image_router.post("/plot_bbox_landmark_bytestring", response_model=List[image_io.StrPlotBboxLandmarkResult])
async def plot_bbox_landmark(request: image_io.ImageBytestringInputSchema = Body(..., examples=image_io.image_input_examples_str), params: InputQueryParams = Depends()):
    data = request.dict()
    image_bytestring = data["bytestring"]  

    image = get_image_bytestring(image_bytestring)
    result = inference_process(pipeline.plot_bbox_landmark, image, params.confidence, params.size, params.face_limit_num)
    response = APIResponse(retData=result)
    return response    

