import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
from typing import List

from fastapi import Body
from fastapi import Depends, Query

from src.base.exceptions import  InternalProgramException
from src.base.logger import Logger
from src.base.router import APIRouter
from src.base.configs import get_conf
from src.base.schema import APIResponse
from src.utils import get_detailed_error
from src.api.model import image_io
from src.model.attribute import get_model, Attribute_Model

model_conf = get_conf("MODEL")
parsing_model_path = os.environ.get("PARSING_MODEL_PATH", model_conf["PARSING_MODEL_PATH"])
age_model_path = os.environ.get("AGE_MODEL_PATH", model_conf["AGE_MODEL_PATH"])
emotion_model_path = os.environ.get("EMOTION_MODEL_PATH", model_conf["EMOTION_MODEL_PATH"])
gender_model_path = os.environ.get("GENDER_MODEL_PATH", model_conf["GENDER_MODEL_PATH"])
model = get_model(parsing_model_path, age_model_path, emotion_model_path, gender_model_path)
Logger.info("Loading model finish")  

api_links_conf = get_conf("API_LINKS")

pipeline = Attribute_Model(model, api_links_conf)
Logger.info("Import pipeline finish")  
single_router = APIRouter(prefix="/single", tags=["Single Attribute"])
all_router = APIRouter(prefix="/all", tags=["All Attribute"])
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

def inference_process(function, image_url_or_bytestring, data_type, confindence, size, face_limit_num):

    try:
        result = function(image_url_or_bytestring, data_type, confindence, size, face_limit_num)
    except (RuntimeError, OSError) as e:
        Logger.error(f"Inference Error: {get_detailed_error(e)}")  
        raise InternalProgramException(ret_info="Inference is crashed.")
    except ValueError as e:
        Logger.error(f"Inference Error: {get_detailed_error(e)}")  
        raise InternalProgramException(ret_info=str(e))

    return result

Logger.info("import helper function finish")  

@single_router.post("/get_age_url", response_model=List[image_io.UrlAgeResult])
async def get_age_url(request: image_io.ImageUrlInputSchema = Body(..., examples=image_io.image_input_examples_url), params: InputQueryParams = Depends()):
    
    data = request.dict()
    image_url = data["url"]  

    result = await inference_process(pipeline.get_age, image_url, 'url', params.confidence, params.size, params.face_limit_num)   
    response = APIResponse(retData=result)
    return response

@single_router.post("/get_age_bytestring", response_model=List[image_io.StrAgeResult])
async def get_age_bytestring(request: image_io.ImageBytestringInputSchema = Body(..., examples=image_io.image_input_examples_str), params: InputQueryParams = Depends()):
    
    data = request.dict()
    image_bytestring = data["bytestring"]  

    result = await inference_process(pipeline.get_age, image_bytestring, 'bytestring', params.confidence, params.size, params.face_limit_num)   
    response = APIResponse(retData=result)
    return response

@single_router.post("/get_emotion_url", response_model=List[image_io.UrlEmotionResult])
async def get_emotion_url(request: image_io.ImageUrlInputSchema = Body(..., examples=image_io.image_input_examples_url), params: InputQueryParams = Depends()):
    
    data = request.dict()
    image_url = data["url"]   

    result = await inference_process(pipeline.get_emotion, image_url, 'url', params.confidence, params.size, params.face_limit_num)
    response = APIResponse(retData=result)
    return response

@single_router.post("/get_emotion_bytestring", response_model=List[image_io.StrEmotionResult])
async def get_emotion_bytestring(request: image_io.ImageBytestringInputSchema = Body(..., examples=image_io.image_input_examples_str), params: InputQueryParams = Depends()):
    
    data = request.dict()
    image_bytestring = data["bytestring"]  

    result = await inference_process(pipeline.get_emotion, image_bytestring, 'bytestring', params.confidence, params.size, params.face_limit_num)
    response = APIResponse(retData=result)
    return response

@single_router.post("/get_gender_url", response_model=List[image_io.UrlGenderResult])
async def get_gender_url(request: image_io.ImageUrlInputSchema = Body(..., examples=image_io.image_input_examples_url), params: InputQueryParams = Depends()):
    
    data = request.dict()
    image_url = data["url"]  

    result = await inference_process(pipeline.get_gender, image_url, 'url', params.confidence, params.size, params.face_limit_num)
    response = APIResponse(retData=result)
    return response

@single_router.post("/get_gender_bytestring", response_model=List[image_io.StrGenderResult])
async def get_gender_bytestring(request: image_io.ImageBytestringInputSchema = Body(..., examples=image_io.image_input_examples_str), params: InputQueryParams = Depends()):
    
    data = request.dict()
    image_bytestring = data["bytestring"]  

    result = await inference_process(pipeline.get_gender, image_bytestring, 'bytestring', params.confidence, params.size, params.face_limit_num)
    response = APIResponse(retData=result)
    return response

@all_router.post("/get_all_url", response_model=List[image_io.UrlAllResult])
async def get_all_url(request: image_io.ImageUrlInputSchema = Body(..., examples=image_io.image_input_examples_url), params: InputQueryParams = Depends()):
    
    data = request.dict()
    image_url = data["url"]  

    result = await inference_process(pipeline.get_all, image_url, 'url', params.confidence, params.size, params.face_limit_num)
    response = APIResponse(retData=result)
    return response

@all_router.post("/get_all_bytestring", response_model=List[image_io.StrAllResult])
async def get_all_bytestring(request: image_io.ImageBytestringInputSchema = Body(..., examples=image_io.image_input_examples_str), params: InputQueryParams = Depends()):
    
    data = request.dict()
    image_bytestring = data["bytestring"]  

    result = await inference_process(pipeline.get_all, image_bytestring, 'bytestring', params.confidence, params.size, params.face_limit_num)
    response = APIResponse(retData=result)
    return response
