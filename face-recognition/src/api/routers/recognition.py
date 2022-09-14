import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
from typing import List
from fastapi import Body, Depends, Query
from fastapi.responses import FileResponse
from collections import Counter

from src.base.exceptions import ExternalDependencyException, InternalProgramException
from src.base.logger import Logger
from src.base.router import APIRouter
from src.base.schema import APIResponse
from src.base.configs import get_conf
# from src.base.schema import ResponseSchema
from src.utils import CVInvalidInputsException, get_db_info, get_detailed_error
from src.api.model import image_io
from src.model.database import HDF5DATABASE
from src.model.extractor import Predictor_, Enroll, Predict, get_model, Embedding_Model

Logger.info("Import finish")  

model_conf = get_conf("MODEL")

recognition_model_path = os.environ.get("RECOGNITION_MODEL_PATH", model_conf["RECOGNITION_MODEL_PATH"])
image_root = os.environ.get("IMAGEROOT", model_conf["IMAGEROOT"])
reference_embedding_path = os.environ.get("EMBEDDING_ROOT", model_conf["EMBEDDING_ROOT"])

model = get_model(recognition_model_path)
Logger.info("Load model finish")  
database = HDF5DATABASE(reference_embedding_path, image_root)
embedding_model = Embedding_Model(model, model_conf)
predictor = Predictor_(reference_embedding_path)
enroll_process = Enroll(embedding_model, database)
prediction_process = Predict(embedding_model, database, predictor)
Logger.info("Load pipeline finish")  

class ErollQueryParams:
    def __init__(
        self,
        confidence: float = Query(0.6, description="Confidence Threshold (0.0 ~ 1.0). Should be higher than \"0.6\""),
        size: int = Query(20, description="Face Area Pixel Threshold. Should be higher than \"20\""),
    ):
        self.confidence = confidence
        self.size = size

class PredictQueryParams:
    def __init__(
        self,
        confidence: float = Query(0.6, description="Confidence Threshold (0.0 ~ 1.0). Should be higher than \"0.6\""),
        size: int = Query(20, description="Face Area Pixel Threshold. Should be higher than \"20\""),
        face_limit_num: int = Query(1, description="Number of faces to be returned in the image. If larger than number of faces in the image, return all faces. Return all faces if \"-1\""),
    ):
        self.confidence = confidence
        self.size = size    
        self.face_limit_num = face_limit_num

enroll_router = APIRouter(prefix="/enroll", tags=["Enrollment Process"])
predict_router = APIRouter(prefix="/predict", tags=["Prediction Process"])
database_router = APIRouter(prefix="/database", tags=["Database Process"])
# EnrollSingleResponse = ResponseSchema("EnrollSingleResponse", ret_data=image_io.ImageEnrollSingleResultSchema)
# PredictSingleResponse = ResponseSchema("PredictSingleResponse", ret_data=image_io.ImagePredictSingleResultSchema, is_list=True)
# DatabaseDeleteResponse = ResponseSchema("DatabaseDeleteResponse", ret_data=image_io.DatabaseDeleteSchema)
# DatabaseResetResponse = ResponseSchema("DatabaseResetResponse", ret_data=image_io.DatabaseResetSchema)
# DatabaseValidStatResponse = ResponseSchema("DatabasenumberResponse", ret_data=image_io.DatabaseValidStatSchema)
# DatabaseGetDataResponse = ResponseSchema("DatabaseGetDataResponse", ret_data=image_io.DatabaseGetDataSchema)

# @enroll_router.post("/single", response_model=image_io.ImageEnrollSingleResultSchema)
# async def enroll_single_image(request: image_io.ImageEnrollSingleSchema = Body(..., examples=image_io.enroll_single_examples), params: ErollQueryParams = Depends()):
#     data = request.dict()
#     image_url_or_bytestring = data["url_or_bytestring"]
#     image_name = data["name"]
#     image_label = data["label"]      

#     try:
#         result = enroll_process.single_enroll(image_url_or_bytestring, image_name, image_label, params.confidence, params.size)
#     except (RuntimeError, OSError) as e:
#         Logger.error(f"Enroll single image with following extracting error: {get_detailed_error(e)}")  
#         raise InternalProgramException(ret_info="Inference is crashed.")
#     except ValueError as e:
#         Logger.error(f"Enroll single image with following extracting error: {get_detailed_error(e)}")  
#         raise InternalProgramException(ret_info=str(e))

#     Logger.info(f"Insert {image_name} into database")

#     prediction_process.predictor.load_faiss_index()

#     response = APIResponse(retData=result)
#     return response

# @predict_router.post("/single", response_model=List[image_io.ImagePredictSingleResultSchema])
# async def predict_single_image_by_url(request: image_io.ImagePredictSingleSchema = Body(..., examples=image_io.predict_single_examples), params: PredictQueryParams = Depends()):
#     if database.num_valid_records == 0:
#         raise CVInvalidInputsException(ret_info="No reference image in database")   

#     index_labels, _ = get_db_info(reference_embedding_path)
#     label_num = len(list({**Counter(index_labels)}.keys()))
#     if label_num < 2:
#         raise CVInvalidInputsException(ret_info="Should enroll at least two face label in database") 

#     data = request.dict()
#     image_url_or_bytestring = data["url_or_bytestring"]

#     try:
#         result = prediction_process.single_predict(image_url_or_bytestring, params.confidence, params.size, params.face_limit_num)
#     except (RuntimeError, OSError) as e:
#         Logger.error(f"Preidct single image with following pipeline error: {get_detailed_error(e)}")  
#         raise InternalProgramException(ret_info="Inference is crashed.")
#     except ValueError as e:
#         Logger.error(f"Predict single image with following pipeline error: {get_detailed_error(e)}")  
#         raise InternalProgramException(ret_info=str(e))

#     response = APIResponse(retData=result)
#     return response

@enroll_router.post("/single_url", response_model=image_io.UrlImageEnrollResult)
async def enroll_single_image_url(request: image_io.ImageUrlEnrollInput = Body(..., examples=image_io.enroll_single_examples_url), params: ErollQueryParams = Depends()):
    data = request.dict()
    image_url = data["url"]  
    image_name = data["name"]
    image_label = data["label"]      

    try:
        result = enroll_process.single_enroll(image_url, 'url', image_name, image_label, params.confidence, params.size)
    except (RuntimeError, OSError) as e:
        Logger.error(f"Enroll single image with following extracting error: {get_detailed_error(e)}")  
        raise InternalProgramException(ret_info="Inference is crashed.")
    except ValueError as e:
        Logger.error(f"Enroll single image with following extracting error: {get_detailed_error(e)}")  
        raise InternalProgramException(ret_info=str(e))

    Logger.info(f"Insert {image_name} into database")

    prediction_process.predictor.load_faiss_index()

    response = APIResponse(retData=result)
    return response

@enroll_router.post("/single_bytestring", response_model=image_io.StrImageEnrollResult)
async def enroll_single_image_bytestring(request: image_io.ImageBytestringEnrollInput = Body(..., examples=image_io.enroll_single_examples_str), params: ErollQueryParams = Depends()):
    data = request.dict()
    image_bytestring = data["bytestring"]  
    image_name = data["name"]
    image_label = data["label"]      

    try:
        result = enroll_process.single_enroll(image_bytestring, 'bytestring', image_name, image_label, params.confidence, params.size)
    except (RuntimeError, OSError) as e:
        Logger.error(f"Enroll single image with following extracting error: {get_detailed_error(e)}")  
        raise InternalProgramException(ret_info="Inference is crashed.")
    except ValueError as e:
        Logger.error(f"Enroll single image with following extracting error: {get_detailed_error(e)}")  
        raise InternalProgramException(ret_info=str(e))

    Logger.info(f"Insert {image_name} into database")

    prediction_process.predictor.load_faiss_index()

    response = APIResponse(retData=result)
    return response

@predict_router.post("/single_url", response_model=List[image_io.UrlImagePredictResult])
async def predict_single_image_url(request: image_io.ImageUrlPredictInput = Body(..., examples=image_io.predict_single_examples_url), params: PredictQueryParams = Depends()):
    if database.num_valid_records == 0:
        raise CVInvalidInputsException(ret_info="No reference image in database")   

    index_labels, _ = get_db_info(reference_embedding_path)
    label_num = len(list({**Counter(index_labels)}.keys()))
    if label_num < 2:
        raise CVInvalidInputsException(ret_info="Should enroll at least two face label in database") 

    data = request.dict()
    image_url = data["url"]  

    try:
        result = prediction_process.single_predict(image_url, 'url', params.confidence, params.size, params.face_limit_num)
    except (RuntimeError, OSError) as e:
        Logger.error(f"Preidct single image with following pipeline error: {get_detailed_error(e)}")  
        raise InternalProgramException(ret_info="Inference is crashed.")
    except ValueError as e:
        Logger.error(f"Predict single image with following pipeline error: {get_detailed_error(e)}")  
        raise InternalProgramException(ret_info=str(e))

    response = APIResponse(retData=result)
    return response

@predict_router.post("/single_bytestring", response_model=List[image_io.StrImagePredictResult])
async def predict_single_image_bytestring(request: image_io.ImageBytestringPredictInput = Body(..., examples=image_io.predict_single_examples_str), params: PredictQueryParams = Depends()):
    if database.num_valid_records == 0:
        raise CVInvalidInputsException(ret_info="No reference image in database")   

    index_labels, _ = get_db_info(reference_embedding_path)
    label_num = len(list({**Counter(index_labels)}.keys()))
    if label_num < 2:
        raise CVInvalidInputsException(ret_info="Should enroll at least two face label in database") 

    data = request.dict()
    image_bytestring = data["bytestring"]  

    try:
        result = prediction_process.single_predict(image_bytestring, 'bytestring', params.confidence, params.size, params.face_limit_num)
    except (RuntimeError, OSError) as e:
        Logger.error(f"Preidct single image with following pipeline error: {get_detailed_error(e)}")  
        raise InternalProgramException(ret_info="Inference is crashed.")
    except ValueError as e:
        Logger.error(f"Predict single image with following pipeline error: {get_detailed_error(e)}")  
        raise InternalProgramException(ret_info=str(e))

    response = APIResponse(retData=result)
    return response

@database_router.get("/valid_stat", response_model=image_io.DatabaseValidStatSchema)
async def get_valid_records():
    try:
        valid_num = database.num_valid_records
        index_labels, name_list = get_db_info(reference_embedding_path)
        label_count = {**Counter(index_labels)}  
    except Exception as e:
        Logger.error(f"Database valid_stat with following error: {get_detailed_error(e)}")  
        raise InternalProgramException(ret_info="Internal error.")
    result = {"valid_records": valid_num, "valid_names": name_list, "valid_label_count":label_count}
    response = APIResponse(retData=result)
    return response

@database_router.get("/reset", response_model=image_io.DatabaseResetSchema)
async def reset_database():
    try:
        database._reset_db()
        status = "Database is reseted."
    except Exception as e:
        Logger.error(f"Database reset with following error: {get_detailed_error(e)}")  
        status = "Database reset failed."
    response = APIResponse(retData={"status": status})
    return response

@database_router.get("/get_data/{image_name}", response_model=image_io.DatabaseGetDataSchema)
async def get_reference_metadata_by_name(image_name: str):
    try:
        status, result = database.get_data(image_name)
    except Exception as e:
        Logger.error(f"Database get_data with following error: {get_detailed_error(e)}")
        raise InternalProgramException(ret_info="Internal error.")

    if status != 'success':
        raise CVInvalidInputsException(ret_info=status)
    else:
        response = APIResponse(retData=result)
    return response

@database_router.get("/get/{image_name}")
async def download_image_by_name(image_name: str):
    if image_name not in database.db:
        raise CVInvalidInputsException(ret_info="not in db")
    item = database.db[image_name]
    if not item["valid"]:
        raise CVInvalidInputsException(ret_info="not valid")
    try:
        image_path = database.db[image_name]["path"]
    except KeyError:
        raise ExternalDependencyException().DatabaseException(ret_info="Cannot find {} in database".format(image_name))
    if not os.path.isfile(image_path):
        raise ExternalDependencyException().DatabaseException(ret_info="Cannot find {} in database".format(image_name))

    return FileResponse(image_path, filename=image_name)

@database_router.delete("/delete/{image_name}", response_model=image_io.DatabaseDeleteSchema)
async def delete_image_by_name(image_name: str):
    try:
        status = database.delete(image_name)
    except Exception as e:
        Logger.error(f"Database delete with following error: {get_detailed_error(e)}")
        raise InternalProgramException(ret_info="Internal error.")

    if status != 'success':
        raise CVInvalidInputsException(ret_info=status)
    else:
        response = APIResponse(retData={"status": status, "valid_records": database.num_valid_records})
    prediction_process.predictor.load_faiss_index()
    return response
