# -*- coding: utf-8 -*-
import typing
from enum import Enum
from typing import Any

from pydantic import BaseConfig
from pydantic.fields import FieldInfo, ModelField, Undefined

from .validator import Validator


class _AttrEnum(Validator, Enum):
    def __init__(self, dict_):
        self.__dict__.update(**dict_)
        self.field_info = FieldInfo(default=self.default, example=self.example, title=self.title)
        self.field_info._validate()

    def set_description(self, description: str):
        self.description = description
        return self


class _GroupEnum(Enum):
    def __init__(self, value):
        self.__dict__.update(**{attr.name: attr for attr in value})
        self.field_info = FieldInfo(default=Undefined)
        self.field_info._validate()

    def __iter__(self):
        for member in self.value:
            yield member


class Fields(_GroupEnum):
    class Doc(_AttrEnum):
        docid = {
            "name_": "docid",
            "type_": str,
            "required": True,
            "default": None,
            "class_validators": {},
            "model_config": BaseConfig,
            "title": "Document ID.",
            "example": None,
        }
        headline = {
            "name_": "headline",
            "type_": str,
            "required": True,
            "default": None,
            "class_validators": {},
            "model_config": BaseConfig,
            "title": "Document headline.",
            "example": None,
        }
        content = {
            "name_": "content",
            "type_": str,
            "required": True,
            "default": None,
            "class_validators": {},
            "model_config": BaseConfig,
            "title": "Document content.",
            "example": None,
        }

    class Senti(_AttrEnum):
        score = {
            "name_": "score",
            "type_": float,
            "required": True,
            "default": None,
            "class_validators": {},
            "model_config": BaseConfig,
            "title": "Sentiment average score.",
            "example": None,
        }
        positive_score = {
            "name_": "positive",
            "type_": float,
            "required": True,
            "default": None,
            "class_validators": {},
            "model_config": BaseConfig,
            "title": "Sentiment positive score.",
            "example": None,
        }
        neutral_score = {
            "name_": "neutral",
            "type_": float,
            "required": True,
            "default": None,
            "class_validators": {},
            "model_config": BaseConfig,
            "title": "Sentiment neutral score.",
            "example": None,
        }
        negative_score = {
            "name_": "negative",
            "type_": float,
            "required": True,
            "default": None,
            "class_validators": {},
            "model_config": BaseConfig,
            "title": "Sentiment negative score.",
            "example": None,
        }
        label = {
            "name_": "label",
            "type_": str,
            "required": True,
            "default": None,
            "class_validators": {},
            "model_config": BaseConfig,
            "title": "Label name.",
            "example": None,
        }

    class NER(_AttrEnum):
        label_name = {
            "name_": "label_name",
            "type_": str,
            "required": True,
            "default": None,
            "class_validators": {},
            "model_config": BaseConfig,
            "title": "Label name.",
            "example": None,
        }
        text_segement = {
            "name_": "text_segment",
            "type_": str,
            "required": True,
            "default": None,
            "class_validators": {},
            "model_config": BaseConfig,
            "title": "Text wrapped between start and end index.",
            "example": None,
        }
        start_ind = {
            "name_": "start_ind",
            "type_": int,
            "required": True,
            "default": None,
            "class_validators": {},
            "model_config": BaseConfig,
            "title": "Start index of the label in given text.",
            "example": None,
        }
        end_ind = {
            "name_": "end_ind",
            "type_": int,
            "required": True,
            "default": None,
            "class_validators": {},
            "model_config": BaseConfig,
            "title": "End index of the label in given text.",
            "example": None,
        }

    class General(_AttrEnum):
        lang = {
            "name_": "lang",
            "type_": str,
            "required": True,
            "default": None,
            "class_validators": {},
            "model_config": BaseConfig,
            "title": "Language.",
            "example": None,
        }
        ltp = {
            "name_": "lang",
            "type_": dict,
            "required": True,
            "default": None,
            "class_validators": {},
            "model_config": BaseConfig,
            "title": "LTP object.",
            "example": None,
        }

    class _Response(_AttrEnum):
        retInfo = {
            "name_": "retInfo",
            "type_": str,
            "required": True,
            "default": "OK",
            "class_validators": {},
            "model_config": BaseConfig,
            "title": "Message of the response.",
            "example": "OK",
        }
        retCode = {
            "name_": "retCode",
            "type_": str,
            "required": True,
            "default": "S",
            "class_validators": {},
            "model_config": BaseConfig,
            "title": "Return code for mapping response status.",
            "example": "S",
        }


class CustomField(Validator):
    name_: str
    type_: Any
    required: bool
    example: Any = None
    title: str = ""
    class_validators = {}
    model_config = BaseConfig
    structure = {}
    default = None

    def __init__(self, name: str = "", type: typing = None, required: bool = True):
        self.name_ = name
        self.type_ = type
        self.required = required
        self.field_info = FieldInfo(default=self.default, example=self.example, title=self.title)
        self.field_info._validate()


class CustMF(ModelField, CustomField):
    def __repr__(self) -> str:
        return str({"name": self.name, "type": self.type_, "required": self.required})

    def set_description(self, description):
        self.field_info.description = description
        return self

    def __dict__(self):
        return dict(self)

    @property
    def name_(self):
        return self.name

    @name_.setter
    def name_(self, value):
        self.name = value
