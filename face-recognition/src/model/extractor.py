from typing import Dict, Iterable, List
import faiss
import cv2
import numpy as np
import requests
from collections import Counter
import h5py

from .extractor_utils import response_handler, get_image, get_image_url, get_image_bytestring, align_process
from src.utils import l2norm_numpy
from src.model.insightface.model_zoo import model_zoo

exception_retInfo_list = ['Invalid face_limit_num input.', "Inference is crashed.", "Invalid image url.", "Inference is crashed."]

def get_model(recog_path):
    """Load and prepare models in pipeline"""

    rec_model = model_zoo.get_model(recog_path)
    rec_model.prepare(1)
    return {"recognition_model": rec_model}

class Embedding_Model(object):
    def __init__(self, model, api_links):
        self.model = model
        self.api_model = api_links
        self.embedding_target_size = 120

    def get_embedding(self, image_url_or_bytestring, data_type, filter_confidence=0.6, filter_size=20, face_limit_num=1):
        # """Get single image embedding

        # Args:
        #     model (dict): {'detection_model', 'alignment_model', 'recognition_model'}
        #     img(PIL.Image): single image
        #     single_face (bool): get the biggest bbox from image if there're multiple faces detected for reference database (allows only for single face)

        # """
        if data_type == 'url':
            request_link = self.api_model['ALIGNMENT_URL'] + f'?confidence={filter_confidence}&size={filter_size}&face_limit_num={str(face_limit_num)}'
            response = requests.post(request_link, json={"url": image_url_or_bytestring}).json()    
        else:
            request_link = self.api_model['ALIGNMENT_STR'] + f'?confidence={filter_confidence}&size={filter_size}&face_limit_num={str(face_limit_num)}'
            response = requests.post(request_link, json={"bytestring": image_url_or_bytestring}).json()    

        align_result_list = response_handler(response, exception_retInfo_list, exception_retInfo="Face detection or alignment failed")      

        if align_result_list[0]['bbox'] == "None":
            return [{"embedding": "None", "bbox": "None"}], None      
    
        if data_type == 'url':
            pil_img = get_image_url(image_url_or_bytestring)
        else:
            pil_img = get_image_bytestring(image_url_or_bytestring)

        img_org = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)

        embedding_result_list = []
        for align_result in align_result_list:
            img = img_org.copy()
            bbox = align_result['bbox'] # list
            np_bbox = np.array(bbox) 
            np_landmark = np.array(align_result['landmark'])            

            img_align = align_process(img, np_bbox, np_landmark, target_size=self.embedding_target_size)
            embedding = self.model["recognition_model"].get_feat(img_align).flatten()

            embedding_result_list.append({"embedding": embedding, "bbox": bbox})

        return embedding_result_list, pil_img


class Predictor_:
    def __init__(self, hdf5_path) -> None:
        """Get recognition result from reference database

        Args:
            hdf5_path (str): path to reference database

        """
        self.hdf5_path = hdf5_path
        self.load_faiss_index()

    def load_faiss_index(self):
        """Load reference embeding to faiss and get ready to search"""

        with h5py.File(self.hdf5_path, "r+") as hdf5:
            index_valids = hdf5["valid"][:]
            index_embeddings = hdf5["embedding"][:][index_valids]
            self.index_names = hdf5["name"][:][index_valids]
            self.index_labels = hdf5["label"][:][index_valids]
        embedding_dim = index_embeddings.shape[1]
        cpu_index = faiss.IndexFlatIP(embedding_dim)
        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        gpu_index.add(index_embeddings)
        self.gpu_index = gpu_index

    def get_prediction_result(self, embedding):
        topk = int(2 * min(list(Counter(self.index_labels).values()))) + 1  # get smallest sample amount of each label
        query_embedding = l2norm_numpy(embedding)
        topk_score, topk_idx = self.gpu_index.search(x=query_embedding[None, ...], k=min(topk, len(self.index_names)))
        label_result = []
        score_result = []

        for _, (i, score) in enumerate(zip(topk_idx[0], topk_score[0])):
            label_result.append(self.index_labels[i])
            score_result.append(score ** 4)  # L2 distance ** 4 (avoiding sensitive value)
        label_result = np.array(label_result)
        score_result = np.array(score_result)

        candidate_label = np.unique(label_result)  # get possible label from topk result
        label_score = []
        for label in candidate_label:
            label_score.append(score_result[label_result == label].sum())  # get final score of each label by summing
        label_score = np.array(label_score)
        label_score = np.around(label_score, decimals=6)

        top_result_index = (-1 * label_score).argsort()  # sorting candidate label by final score
        prediction_result = {
            "top1_label": candidate_label[top_result_index[0]],
            "top1_score": label_score[top_result_index[0]],
            "top2_label": candidate_label[top_result_index[1]],
            "top2_score": label_score[top_result_index[1]],
        }
        return prediction_result


class Enroll:
    def __init__(self, embedding_model, database):
        """Erollment process

        Args:
            model (dict): {'recognition_model'} 
            database (HDF5DATABASE): reference database
        """
        self.model = embedding_model
        self.database = database

    def single_enroll(self, image_url_or_bytestring, data_type, img_name, img_label, filter_confidence, filter_size):
        """
        Args:
            model (dict): {'recognition_model'} 
            database (HDF5DATABASE): reference database
            img (PIL.Image): single image
            img_name (str): image name
            img_label (str): image label

        """
        embedding_result_list, img = self.model.get_embedding(image_url_or_bytestring, data_type, filter_confidence=filter_confidence, filter_size=filter_size, face_limit_num=1)

        if embedding_result_list[0]['bbox'] == "None":
            return {"bbox": "None"}
        else:
            embedding = embedding_result_list[0]["embedding"]  # single face embedding for enrollment
            bbox = embedding_result_list[0]["bbox"]
            self.database.insert(img, img_name, embedding, bbox, img_label)
  
            return {"bbox": bbox}

class Predict:
    def __init__(self, model, database, predictor):
        """Prediction process

        Args:
            model (dict): {'detection_model', 'alignment_model', 'recognition_model'} 
            database (HDF5DATABASE): reference database
            predictor(Predictor_) 
        """
        self.model = model
        self.database = database
        self.predictor = predictor

    def single_predict(self, image_url_or_bytestring, data_type, filter_confidence, filter_size, face_limit_num=-1):
        """
        Args:
            model (dict): {'recognition_model'} 
            predictor(Predictor_) 
            img (PIL.Image): single image
        """


        embedding_result_list, _ = self.model.get_embedding(image_url_or_bytestring, data_type, filter_confidence=filter_confidence, filter_size=filter_size, face_limit_num=face_limit_num)
        output_result = []
        if embedding_result_list[0]['bbox'] == "None":
            return [{
                    "bbox": "None",
                    "top1_label": "None",
                    "top1_score": "None",
                    "top2_label": "None",
                    "top2_score": "None",
                    }]
        else:
            for embedding_result in embedding_result_list:
                prediction_result = self.predictor.get_prediction_result(embedding_result['embedding'])
                output_result.append(
                    {"bbox": embedding_result['bbox'], **prediction_result}
                )
            return output_result                