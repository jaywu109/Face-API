# -*- coding: utf-8 -*-
import sys
from typing import Any, List, Union

from pydantic import BaseConfig, BaseModel
from pydantic.fields import FieldInfo, Undefined

from .field import CustMF, Fields


if sys.version_info < (
    3,
    8,
):
    from singledispatchmethod import singledispatchmethod
else:
    from functools import singledispatchmethod


class _DefaultSchema:

    name_: str
    type_: Any
    required: bool = True
    example: Any = None
    title: str = ""
    class_validators = {}
    model_config = BaseConfig
    structure = {}

    @staticmethod
    def _add(cls, attr: Any, example=None, required=None, name_=None, type_=None):
        example = attr.example if not example else example
        required = attr.required if not isinstance(required, bool) else required
        name_ = attr.name_ if not name_ else name_
        type_ = attr.type_ if not type_ else type_
        # Set title, validation, example in FieldInfo class
        f = attr.field_info
        # Set name, type_ ...
        cls.__fields__[name_] = CustMF(
            name=name_,
            type_=type_,
            required=required,
            class_validators=attr.class_validators,
            model_config=attr.model_config,
            field_info=f,
        )
        if attr.default:
            cls.__fields__[name_].default = attr.default
        # Set sturcture
        sub_struct = getattr(attr, "structure", None)
        struct = {
            **cls.structure,
            name_: cls.__fields__[name_] if not getattr(attr, "structure", None) else sub_struct,
        }
        cls.structure = struct
        cls = _DefaultSchema._update_attr(cls, struct)
        return cls

    @staticmethod
    def _update_attr(cls, struct):
        for field_name in struct:
            if isinstance(struct[field_name], CustMF):
                setattr(cls, field_name, struct[field_name])
            else:
                temp = CustMF(
                    name=field_name,
                    type_=dict,
                    class_validators={},
                    model_config=BaseConfig,
                )
                setattr(
                    cls,
                    field_name,
                    _DefaultSchema._update_attr(temp, struct[field_name]),
                )
        return cls

    # Append fields method
    @classmethod
    def append(cls, *input_obj, **kwargs) -> "_DefaultSchema":
        """Append a Fields or CustomField into Schema.

        Returns:
            Schema: Result Schema.
        """
        for member in input_obj:
            cls = cls._distribute(member, **kwargs)
        return cls

    @singledispatchmethod
    @classmethod
    def _distribute(cls, input_obj, **kwargs):
        """Add fields by pydantic.fields.ModelFields.

        Field's Attributes:
            name, str: Name of fields.
            type_, Type[Any]: Type of fields.
            required, bool: Is required or not.
            default, Type[Any]: Default value of fields.
            class_validators, dict: validators for Class.
            model_config, BaseConfig: Configuration
        """
        return _DefaultSchema._add(cls, input_obj, **kwargs)

    @_distribute.register(list)
    @_distribute.register(Fields)
    @classmethod
    def _(cls, input_obj, **kwargs):
        for member in input_obj:
            cls.append(member, **kwargs)
        return cls

    @classmethod
    def append_group(cls, input_obj, group_name="", is_list=False) -> "_DefaultSchema":
        name_ = group_name
        type_ = List[input_obj.type_] if is_list else input_obj.type_
        return _DefaultSchema._add(cls, input_obj, name_=name_, type_=type_)

    @classmethod
    def set_example(cls, dict_: dict = None, **kwargs):
        """Update example of dynamic schema.

        Return:
            dict: Dictionary structure of example.
        """
        if not dict_:
            example_dict = cls(**kwargs).dict()
        else:
            example_dict = cls(**dict_).dict()
        return example_dict

    @classmethod
    def _init_info(cls, name, required):
        cls.name_ = name
        cls.type_ = cls
        cls.default = None
        cls.required = required
        config_ = type(f"{name}_Config", (BaseConfig,), {})
        cls.__config__ = config_
        cls.field_info = FieldInfo(default=Undefined)
        cls.field_info._validate()
        cls.model_field = CustMF(
            name=name,
            type_=cls,
            required=required,
            class_validators={},
            model_config=cls.model_config,
            field_info=cls.field_info,
        )
        return cls

    @classmethod
    def new_example(cls, description: str = None, summary: str = None, *args, **kwargs):
        """Generate a FastAPI structure example.

        Args:
            description (str, optional): Description of such example. Defaults to None.
            summary (str, optional): Summary of such example. Defaults to None.

        Returns:
            Dict: FastAPI standard structure example
        """
        example = {}
        example["value"] = cls(*args, **kwargs)
        if summary:
            example["summary"] = summary
        if description:
            example["description"] = description
        return example

    @classmethod
    def to_list(cls, key_name: str = None, required: float = -1):
        """Schema to list method.

        Args:
            key_name (str, optional): Set a new key name for result. Defaults to None.
            required (float, optional): If input -1, then uesd the required attr of input schema. Defaults to -1.

        Returns:
            _DefaultSchema: List of input schema.
        """
        name = cls.name_ if not key_name else key_name
        required = cls.required if required == -1 else required
        ListSchema = Schema()._init_info(name, required)
        ListSchema.type_ = List[cls.type_]
        return ListSchema


def Schema(schema_name="CustomizeSchema", required=True) -> _DefaultSchema:
    cls_ = type(schema_name, (BaseModel, _DefaultSchema), {})
    cls_ = cls_._init_info(schema_name, required)
    return cls_


class _ResponseSchema(_DefaultSchema):
    @classmethod
    def append(cls, input_obj, **kwargs) -> _DefaultSchema:
        cls._data = cls._data.append(input_obj, **kwargs)
        sub_struct = getattr(cls._data, "structure", None)
        struct = {
            **cls.structure,
            "retData": cls.__fields__["retData"] if not getattr(cls._data, "structure", None) else sub_struct,
        }
        cls.structure = struct
        return cls

    @classmethod
    def append_group(cls, input_obj, **kwargs) -> _DefaultSchema:
        cls._data = cls._data.append_group(input_obj, **kwargs)
        cls.structure["retData"] = cls._data.structure
        return cls


class ResBaseModel(BaseModel):
    def __init__(__pydantic_self__, **data: Any) -> None:
        data.setdefault("retCode", "S")
        data.setdefault("retInfo", "OK")
        super().__init__(**data)


def ResponseSchema(schema_name="", ret_data: _DefaultSchema = None, required=True, is_list=False) -> _DefaultSchema:
    cls_ = type(schema_name, (ResBaseModel, _ResponseSchema), {})
    cls_ = cls_._init_info(schema_name, required)
    if not ret_data:
        cls_._data = Schema("{}_retData".format(schema_name))
    else:
        cls_._data = ret_data
    cls_._data.name_ = "retData"
    if is_list:
        cls_._data.type_ = List[cls_._data.type_]
    cls_ = cls_._add(cls_, Fields._Response.retCode)._add(cls_, Fields._Response.retInfo)._add(cls_, cls_._data)
    return cls_


def GeneralResponseSchema(schema_name="", required=True) -> _DefaultSchema:
    cls_ = type(schema_name, (BaseModel, _ResponseSchema), {})
    cls_ = cls_._init_info(schema_name, required)
    cls_._data = Schema("{}_retData".format(schema_name))
    cls_._data.name_ = "retData"
    cls_._data.type_ = Union[list, dict, None]
    cls_ = cls_._add(cls_, Fields._Response.retCode)._add(cls_, Fields._Response.retInfo)._add(cls_, cls_._data)
    return cls_


MetaGeneralResponse = GeneralResponseSchema("GeneralResponse")


class APIResponse(MetaGeneralResponse):
    def __init__(self, retCode: str = "S", retInfo: str = "OK", retData: Union[list, dict] = None) -> None:
        """Standard api response schema which defined by project template.  Including three fields below:
        - retCode: return status code.
        - retInfo: return information.
        - retData: return data.

        Args:
            retCode (str, optional): The status code of response. Defaults to "S".
            retInfo (str, optional): The detail information of response. Defaults to "OK".
            retData (Union[list, dict], optional): The actually data of response. Defaults to None.
        """
        super().__init__(retCode=retCode, retInfo=retInfo, retData=retData)
