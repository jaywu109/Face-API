import cv2
import numpy as np
import asyncio
from PIL import Image
import requests
import onnxruntime

from src.base.logger import Logger
from src.model.attribute_utils import *

exception_retInfo_list = ['Invalid face_limit_num input.', "Inference is crashed.", "Invalid image url.", "Inference is crashed."]

class Attribute_Model(object):
    def __init__(self, model, api_links):
        self.model = model
        self.api_model = api_links
        self.parse_target_size = 513     
        self.emotion_target_size = 260
        self.gender_target_size = 224
        self.emotion_idx_to_class = {0: 'Anger', 1: 'Contempt', 2: 'Disgust', 3: 'Fear', 4: 'Happiness', 5: 'Neutral', 6: 'Sadness', 7: 'Surprise'}
        self.emotion_transforms = transforms.Compose(
            [
                transforms.Resize((self.emotion_target_size, self.emotion_target_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
            ])

    async def get_age_process(self, img, bbox):
        low_feature, high_feature, logits = predict_face_parsing(self.model["parsing_model"], img, bbox)
        age_input = {'low': low_feature, 'high': high_feature, 'logits': logits}
        return self.model["age_model"].run(None, age_input)[0].mean()      

    async def get_gender_process(self, img, np_bbox, np_landmark):
        img_align = align_process(img, np_bbox, np_landmark, target_size=self.gender_target_size)
        img_tensor = (np.expand_dims(img_align, axis = 0).astype(float) / float(255)).astype(np.float32)
        gender_input = {'img': img_tensor}
        gender_prediction = self.model["gender_model"].run(None, gender_input)[0][0,:]
        return "Female" if np.argmax(gender_prediction) == 0 else "Male"

    async def get_emotion_process(self, img, np_bbox, np_landmark):
        img_align = align_process(img, np_bbox, np_landmark, target_size=self.emotion_target_size)
        img_tensor = self.emotion_transforms(Image.fromarray(img_align)).unsqueeze(0).numpy()
        emotion_input = {'img': img_tensor}
        scores = self.model["emotion_model"].run(None, emotion_input)[0][0]
        return self.emotion_idx_to_class[np.argmax(scores)]

    async def get_age(self, image_url_or_bytestring, data_type, filter_confidence=0.6, filter_size=20, face_limit_num=1):
        if data_type == 'url':
            request_link = self.api_model['DETECTION_URL'] + f'?confidence={filter_confidence}&size={filter_size}&face_limit_num={str(face_limit_num)}'
            response = requests.post(request_link, json={"url": image_url_or_bytestring}).json()    
        else:
            request_link = self.api_model['DETECTION_STR'] + f'?confidence={filter_confidence}&size={filter_size}&face_limit_num={str(face_limit_num)}'
            response = requests.post(request_link, json={"bytestring": image_url_or_bytestring}).json()    

        detect_result_list = response_handler(response, exception_retInfo_list, exception_retInfo='Face detection failed')
        if detect_result_list[0]['bbox'] == "None":
            return [{ "age": "None", "bbox": "None"}]

        if data_type == 'url':
            img = get_image_url(image_url_or_bytestring)
        else:
            img = get_image_bytestring(image_url_or_bytestring)
            
        img_org = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

        age_result_list = []
        for detect_result in detect_result_list:
            img = img_org.copy()
            bbox = detect_result['bbox'] # list

            age_pred = await self.get_age_process(img, bbox)
            age_result_list.append({ "age": int(age_pred), "bbox": bbox})

        return age_result_list

    async def get_emotion(self, image_url_or_bytestring, data_type, filter_confidence=0.6, filter_size=20, face_limit_num=1):
        if data_type == 'url':
            request_link = self.api_model['ALIGNMENT_URL'] + f'?confidence={filter_confidence}&size={filter_size}&face_limit_num={str(face_limit_num)}'
            response = requests.post(request_link, json={"url": image_url_or_bytestring}).json()    
        else:
            request_link = self.api_model['ALIGNMENT_STR'] + f'?confidence={filter_confidence}&size={filter_size}&face_limit_num={str(face_limit_num)}'
            response = requests.post(request_link, json={"bytestring": image_url_or_bytestring}).json()    

        align_result_list = response_handler(response, exception_retInfo_list, exception_retInfo="Face detection or alignment failed")      

        if align_result_list[0]['bbox'] == "None":
            return [{ "emotion": "None", "bbox": "None"}]

        if data_type == 'url':
            img = get_image_url(image_url_or_bytestring)
        else:
            img = get_image_bytestring(image_url_or_bytestring)

        img_org = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

        emotion_result_list = []
        for align_result in align_result_list:
            img = img_org.copy()
            bbox = align_result['bbox'] # list
            np_bbox = np.array(bbox) 
            np_landmark = np.array(align_result['landmark'])            

            emotion_pred = await self.get_emotion_process(img, np_bbox, np_landmark)
            emotion_result_list.append({ "emotion": emotion_pred, "bbox": bbox})

        return emotion_result_list

    async def get_gender(self, image_url_or_bytestring, data_type, filter_confidence=0.6, filter_size=20, face_limit_num=1):
        if data_type == 'url':
            request_link = self.api_model['ALIGNMENT_URL'] + f'?confidence={filter_confidence}&size={filter_size}&face_limit_num={str(face_limit_num)}'
            response = requests.post(request_link, json={"url": image_url_or_bytestring}).json()    
        else:
            request_link = self.api_model['ALIGNMENT_STR'] + f'?confidence={filter_confidence}&size={filter_size}&face_limit_num={str(face_limit_num)}'
            response = requests.post(request_link, json={"bytestring": image_url_or_bytestring}).json()    

        align_result_list = response_handler(response, exception_retInfo_list, exception_retInfo="Face detection or alignment failed")      

        if align_result_list[0]['bbox'] == "None":
            return [{ "gender": "None", "bbox": "None"}]

        if data_type == 'url':
            img = get_image_url(image_url_or_bytestring)
        else:
            img = get_image_bytestring(image_url_or_bytestring)

        img_org = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

        gender_result_list = []
        for align_result in align_result_list:
            img = img_org.copy()
            bbox = align_result['bbox'] # list
            np_bbox = np.array(bbox) 
            np_landmark = np.array(align_result['landmark'])            

            gender_pred = await self.get_gender_process(img, np_bbox, np_landmark)
            gender_result_list.append({ "gender": gender_pred, "bbox": bbox})

        return gender_result_list

    async def get_all(self, image_url_or_bytestring, data_type, filter_confidence=0.6, filter_size=20, face_limit_num=1):
        if data_type == 'url':
            request_link = self.api_model['ALIGNMENT_URL'] + f'?confidence={filter_confidence}&size={filter_size}&face_limit_num={str(face_limit_num)}'
            response = requests.post(request_link, json={"url": image_url_or_bytestring}).json()    
        else:
            request_link = self.api_model['ALIGNMENT_STR'] + f'?confidence={filter_confidence}&size={filter_size}&face_limit_num={str(face_limit_num)}'
            response = requests.post(request_link, json={"bytestring": image_url_or_bytestring}).json()    

        align_result_list = response_handler(response, exception_retInfo_list, exception_retInfo="Face detection or alignment failed")      

        if align_result_list[0]['bbox'] == "None":
            return [{ "age": "None", "emotion": "None", "gender": "None", "bbox": "None"}]

        if data_type == 'url':
            img = get_image_url(image_url_or_bytestring)
        else:
            img = get_image_bytestring(image_url_or_bytestring)
        img_org = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

        all_attribute_result_list = []
        for align_result in align_result_list:
            bbox = align_result['bbox'] # list
            np_bbox = np.array(bbox) 
            np_landmark = np.array(align_result['landmark'])            
            tasks = []

            tasks.append(asyncio.create_task(self.get_age_process(img_org.copy(), bbox)))     
            tasks.append(asyncio.create_task(self.get_emotion_process(img_org.copy(), np_bbox, np_landmark)))     
            tasks.append(asyncio.create_task(self.get_gender_process(img_org.copy(), np_bbox, np_landmark)))     
            all_result = await asyncio.gather(*tasks)

            all_attribute_result_list.append({ "age": int(all_result[0]), "emotion": all_result[1], "gender": all_result[2], "bbox": bbox})

        return all_attribute_result_list
         

def get_model(parsing_path, age_path, emotion_path, gender_path):
    """Load and prepare models in pipeline"""

    Logger.info("Loading parsing model")     
    parsing_model = onnxruntime.InferenceSession(parsing_path, providers=["CUDAExecutionProvider"])
    Logger.info("Loading parsing model done")
    Logger.info("Loading age model")     
    age_model = onnxruntime.InferenceSession(age_path, providers=["CUDAExecutionProvider"])
    Logger.info("Loading age model done")
    Logger.info("Loading emotion model")     
    emotion_model = onnxruntime.InferenceSession(emotion_path, providers=["CUDAExecutionProvider"])
    Logger.info("Loading emotion model done")
    Logger.info("Loading gender model")     
    gender_model = onnxruntime.InferenceSession(gender_path, providers=["CUDAExecutionProvider"])
    Logger.info("Loading gender model done")

    return {"parsing_model": parsing_model, "age_model": age_model, "emotion_model": emotion_model, "gender_model": gender_model}


