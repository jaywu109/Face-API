from typing import Dict, List, Union, Iterable
import os
import h5py
import numpy as np
from PIL import Image
import shutil
import aiohttp
import base64
from io import BytesIO
import validators
import asyncio
from src.utils import create_image_db, l2norm_numpy

async def string_save_img(byte_string, path):
    # string_to_bytes = bytes(byte_string, "ascii")
    # img_bytes = base64.b64decode(string_to_bytes)
    img_bytes = base64.b64decode(byte_string)
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    img.save(path)

async def get_image_url(session, url, path):
    async with session.get(url) as resp:
        try:
            img_raw = await resp.read()
            img = Image.open(BytesIO(img_raw))
            img.save(path)
            return path
        except Exception as e:
            return None

async def get_image_byte(byte_string, path):
    try:
        await string_save_img(byte_string, path)
        return path
    except Exception as e:
        return None

class HDF5DATABASE:
    def __init__(self, hdf5_path, root_path) -> None:
        """A tiny HDF5 database storing image name, label and embedding

        Args:
            hdf5_path (str): path of the hdf5 file
            root_path (str): root path storing the images
        """
        self.hdf5_path = hdf5_path
        os.makedirs(os.path.dirname(self.hdf5_path), exist_ok=True)
        self.root_path = root_path
        self.temp_path = os.path.join(os.path.dirname(root_path), "temp_image_folder")
        self._clear_temp_folder()

        ### initialize database
        self.db = None
        self.num_records = 0
        self.num_valid_records = 0
        self._create_or_load_db()

    def _create_or_load_db(self):
        name_to_path_mapping = create_image_db(self.root_path)

        self.db = {}
        if os.path.isfile(self.hdf5_path):
            with h5py.File(self.hdf5_path, "r+") as hdf5:
                self.num_records = hdf5.attrs["num_records"]
                self.num_valid_records = hdf5.attrs["num_valid_records"]
                valids = hdf5["valid"][: self.num_records]
                names = hdf5["name"][: self.num_records].astype(str)
            inds = np.arange(len(names))
            for name, valid, ind in zip(names, valids, inds):
                path = name_to_path_mapping[name]
                self.db[name] = {"valid": valid, "ind": ind, "path": path}
        else:
            with h5py.File(self.hdf5_path, "w") as hdf5:
                hdf5.create_dataset(
                    "name", data=["pre_defined.jpg"], maxshape=(None,), dtype="S100"
                )
                hdf5.create_dataset("valid", data=[False], maxshape=(None,), dtype=bool)
                pre_defined_embedding = np.zeros((1, 512))
                pre_defined_bbox = np.zeros((1, 4))
                hdf5.create_dataset(
                    "embedding",
                    data=pre_defined_embedding,
                    dtype=np.float32,
                    maxshape=(None, 512),
                    compression="gzip",
                    chunks=(1, 512),
                )
                hdf5.create_dataset(
                    "bbox",
                    data=pre_defined_bbox,
                    dtype=np.float32,
                    maxshape=(None, 4),
                    compression="gzip",
                    chunks=(1, 4),
                )
                hdf5.create_dataset("label", data=[0], maxshape=(None,), dtype=np.int32)
                hdf5.attrs["num_records"] = 0
                hdf5.attrs["num_valid_records"] = 0

    def _clear_temp_folder(self):
        if os.path.isdir(self.temp_path):
            shutil.rmtree(self.temp_path)

    def _reset_db(self):
        os.remove(self.hdf5_path)
        if os.path.isdir(self.root_path):
            shutil.rmtree(self.root_path)
        self._clear_temp_folder()

        ### re-initialize database
        self.db = None
        self.num_records = 0
        self.num_valid_records = 0
        self._create_or_load_db()

    # def save_image(self, img_name: str, img: Image, enroll=True) -> str:
    #     """Save image by image name

    #     Args:
    #         img_name (str): image name
    #         img (Image.Image): image to be saved
    #     """

    #     # store the raw image
    #     img_path = self.get_save_path(img_name, enroll)
    #     img.save(img_path)
    #     return img_path

    async def save_image_async(
        self, img_list: list, img_name_list: list, enroll=True
    ) -> dict:
        """Save image asynchronously (for batch)

        Args:
            img_name (str): image name
            img_embedding (np.ndarray): 512-dim img embedding
            img_label (int): image label
        """
        async with aiohttp.ClientSession() as session:
            tasks = []
            for i in range(len(img_list)):
                img_name = img_name_list[i]
                img_path = self.get_save_path(img_name, enroll)
                url_or_bytestring = img_list[i]
                if validators.url(url_or_bytestring):
                    url = url_or_bytestring
                    tasks.append(
                        asyncio.create_task(get_image_url(session, url, img_path))
                    )
                else:
                    bytestring = url_or_bytestring
                    tasks.append(
                        asyncio.create_task(get_image_byte(bytestring, img_path))
                    )

            save_result = await asyncio.gather(*tasks)
            img_path_map = {}
            loading_error_index_list = []
            for i, result in enumerate(save_result):
                if result is None:
                    loading_error_index_list.append(i)
                else:
                    img_path = result
                    img_path_map[img_path] = i
        return {'img_path_map': img_path_map, 'loading_error_index_list': loading_error_index_list}

    def insert(
        self,
        img: Image,
        img_name: str,
        img_embedding: np.ndarray,
        bbox: np.ndarray,
        img_label: int,
    ) -> None:
        """Insert a record into the database

        Args:
            img (Image.Image): image to be saved
            img_name (str): image name
            img_embedding (np.ndarray): 512-dim img embedding
            img_label (int): image label
        """
        img_path = self.get_save_path(img_name, enroll=True)
        img.save(img_path)
        
        with h5py.File(self.hdf5_path, "r+") as hdf5:
            # overwrite if it is already exists
            if img_name in self.db:
                if not self.db[img_name]["valid"]:
                    self.num_valid_records += 1
                self.db[img_name]["valid"] = True
                self.db[img_name]["path"] = img_path
                ind = self.db[img_name]["ind"]
                hdf5["valid"][ind] = True
                hdf5["embedding"][ind] = l2norm_numpy(img_embedding)
                hdf5["bbox"][ind] = bbox
                hdf5["label"][ind] = img_label
            else:
                max_length = hdf5["name"].shape[0]
                if self.num_records == max_length:
                    hdf5["name"].resize(max_length*2, axis=0)
                    hdf5["valid"].resize(max_length*2, axis=0)
                    hdf5["embedding"].resize(max_length*2, axis=0)
                    hdf5["bbox"].resize(max_length*2, axis=0)
                    hdf5["label"].resize(max_length*2, axis=0)

                self.db[img_name] = {
                    "valid": True,
                    "ind": self.num_records,
                    "path": img_path,
                }
                hdf5["name"][self.num_records] = img_name
                hdf5["valid"][self.num_records] = True
                hdf5["embedding"][self.num_records] = l2norm_numpy(img_embedding)
                hdf5["bbox"][self.num_records] = bbox
                hdf5["label"][self.num_records] = img_label
                self.num_records += 1
                self.num_valid_records += 1
                hdf5.attrs["num_records"] = self.num_records

            hdf5.attrs["num_valid_records"] = self.num_valid_records

    def delete(self, img_name: str) -> str:
        """Delete from database and image folder by image name

        Args:
            img_name (str): image name

        Returns:
            whether the record is delete or not
        """
        if img_name not in self.db:
            return "not in db"
        item = self.db[img_name]
        if item["valid"]:
            self.num_valid_records -= 1
            item["valid"] = False
            ind = item["ind"]
            with h5py.File(self.hdf5_path, "r+") as hdf5:
                hdf5["valid"][ind] = False
                hdf5.attrs["num_valid_records"] = self.num_valid_records
            return "success"
        else:
            return "already deleted"

    def get_data(self, img_name: str) -> dict:
        """Search by image name

        Args:
            img_name (str): image name

        Returns:
            A dictionary storing metadata
        """
        if img_name not in self.db:
            return "not in db", {}
        item = self.db[img_name]
        if not item["valid"]:
            return "not valid", {}
        ind = item["ind"]
        with h5py.File(self.hdf5_path, "r+") as hdf5:
            embedding = hdf5["embedding"][ind] # tolist
            label = hdf5["label"][ind]
            bbox = hdf5["bbox"][ind]
        return "success", {"embedding": embedding.tolist(), "bbox": bbox.tolist(), "label": label}

    def get_save_path(self, img_name, enroll):
        """Get saving path for different scenario"""
        if enroll:
            img_dir = os.path.join(
                self.root_path, img_name[0], img_name[1], img_name[2]
            )
            os.makedirs(img_dir, exist_ok=True)
            return os.path.join(img_dir, img_name)
        else:
            os.makedirs(self.temp_path, exist_ok=True)
            return os.path.join(self.temp_path, img_name)

