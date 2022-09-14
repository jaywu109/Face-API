import os
import numpy as np
import cv2
from PIL import Image
import validators
import urllib
import base64
import h5py
import sys
import traceback
from io import BytesIO
from typing import Dict, Iterable

from src.base.exceptions import InvalidInputsException
from src.base.logger import Logger

def l2norm_numpy(x):
    return x / np.linalg.norm(x, ord=2, axis=0, keepdims=True)

def img_imread(url_path_base64, return_array=True):
    """ read image from url, path or base64

    Args:
        url_path_base64 (string): string of url, path or base64 
        return_array (bool): return array or PIL.Image

    Returns:
        np.ndarray or PIL.Image
    """

    isurl = validators.url(url_path_base64)
    isfile = os.path.isfile(url_path_base64)
    if isurl:
        image = Image.open(urllib.request.urlopen(url_path_base64)).convert("RGB")
    elif isfile:
        image = Image.open(url_path_base64).convert("RGB")
    else:
        string_to_bytes = bytes(url_path_base64, "ascii")
        img_bytes = base64.b64decode(string_to_bytes)
        image = Image.open(BytesIO(img_bytes)).convert("RGB")
    if return_array:
        cv_img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        return cv_img
    else:
        return image

def find_image(root: str) -> Iterable:
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            _, ext = os.path.splitext(name)
            ext = ext.lower()
            if ext == ".jpg" or ext == ".jpeg" or ext == ".png":
                yield os.path.join(dirpath, name)

def find_image_label(root: str) -> Iterable:
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            _, ext = os.path.splitext(name)
            ext = ext.lower()
            if ext == ".jpg" or ext == ".jpeg" or ext == ".png":
                yield [os.path.join(dirpath, name), dirpath.split("/")[-1]]

def create_image_db(image_root: str) -> Dict:
    Logger.info("Searching all images under {}".format(image_root))
    image_db = {}
    for image_path in find_image(image_root):
        image_db[os.path.basename(image_path)] = image_path
    Logger.info("Total number of index images: {}".format(len(image_db)))
    return image_db

def get_db_info(reference_embedding_path):
    with h5py.File(reference_embedding_path, "r+") as hdf5:
        index_valids = hdf5["valid"][:]
        index_labels = hdf5["label"][:][index_valids]  
        name_list = hdf5["name"][:][index_valids].tolist()   
    return index_labels, name_list

def get_detailed_error(error):
    try:
        error_class = error.__class__.__name__ #取得錯誤類型
        detail = error.args[0] #取得詳細內容
        cl, exc, tb = sys.exc_info() #取得Call Stack
        lastCallStack = traceback.extract_tb(tb)[-1] #取得Call Stack的最後一筆資料
        fileName = lastCallStack[0] #取得發生的檔案名稱
        lineNum = lastCallStack[1] #取得發生的行號
        funcName = lastCallStack[2] #取得發生的函數名稱
        errMsg = "File \"{}\", line {}, in {}: [{}] {}".format(fileName, lineNum, funcName, error_class, detail)
        return errMsg
    except:
        return error
        

class CVInvalidInputsException(InvalidInputsException):
    def __init__(self, ret_info) -> None:
        super().__init__()
        self.retInfo = ret_info
