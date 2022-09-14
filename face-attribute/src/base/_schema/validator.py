from enum import Enum
from typing import Any, Union

from pydantic.fields import FieldInfo
from pydantic.types import confloat, conint, constr


class AvailibleAttributesOfSetValue(Enum):
    required = ("The field is required or not.", "ModelField.generic", bool)
    # alias = ("",)
    title = ("Title of field.", "FieldInfo.Generic", str)
    description = ("Description of the field.", "FieldInfo.generic", str)
    # deprecated = ("", "FieldInfo.Generic")
    default = ("Default value of the field.", "FieldInfo.generic", Any)
    example = ("Example value of the field.", "FieldInfo.generic", Any)
    min_length = ("Minimun length of str type field.", "FieldInfo.validations.str", int)
    max_length = ("Maximum length of str type field.", "FieldInfo.validations.str", int)
    regex = ("Regex to validate the string against.", "FieldInfo.validations.str", int)
    gt = ("The numeric field should greater than value assigned.", "FieldInfo.validations.numeric", Union[int, float])
    ge = ("The numeric field should greater than or equal value assigned", "FieldInfo.validations.numeric", Union[int, float])
    lt = ("The numeric field should less than value assigned.", "FieldInfo.validations.numeric", Union[int, float])
    le = ("The numeric field should greater than or equal value assigned", "FieldInfo.validations.numeric", Union[int, float])

    def __init__(self, descriptions, category, type_):
        self.descriptions = descriptions
        self.category = category.split(".")
        self.type = type_


class Validator:
    def _update_type(self, **kwargs):
        try:
            # type standardisation
            type_name = self.type_.__name__.replace("Constrained", "").replace("Value", "").lower()
            # Update validators_
            validators_ = getattr(self, "validators_", {})
            validators_.update(**validators_, **kwargs)
            # type constrained
            self.type_ = eval(f"con{type_name}(**validators_)")
            # If successed, then update self.validators_
            setattr(self, "validators_", validators_)
        except NameError:
            raise AttributeError(f"Field name `{self.name_}`, type: `{type_name}`, field validations is not support.")
        except TypeError as e:
            msg = str(e).split(" ")[-1]
            raise AttributeError(f"Unexpected keyword argument for type <{type_name}>: {msg}")

    def _set_value(self, **kwargs):

        # print(self, self.type_, type(self))
        # Check if key are all allow
        try:
            for key in kwargs:
                getattr(AvailibleAttributesOfSetValue, key)
        except AttributeError:
            raise AttributeError(f"Keyword argument `{key}` is illegal!  Please check.")
        # Pop ModelField attribute: "required"
        if "required" in kwargs:
            requrired_ = kwargs.pop("required")
            setattr(self, "required", requrired_)
        # Update validator
        validations_keys = {}
        for key in kwargs:
            info_ = getattr(AvailibleAttributesOfSetValue, key)
            if "validations" in info_.category:
                validations_keys[key] = kwargs.get(key)
        self._update_type(**validations_keys)
        # Implement FieldInfo
        self.field_info = FieldInfo(**self.validators_)
        self.field_info._validate()
        return self

    def set_str_value(
        self,
        required: bool = None,
        title: str = None,
        description: str = None,
        default: Any = None,
        example: Any = None,
        regex: str = None,
        min_length: int = None,
        max_length: int = None,
    ):
        """Set value method for str type fields.

        Attrs:
                required (<class 'bool'>): The field is required or not.
                title (<class 'str'>): Title of field.
                description (<class 'str'>): Description of the field.
                default (typing.Any): Default value of the field.
                example (typing.Any): Example value of the field.
                regex (<class 'str'>): Regex to validate the string against.
                min_length (<class 'int'>): Minimun length of str type field.
                max_length (<class 'int'>): Maximum length of str type field.
        """
        kwargs = {
            "required": required,
            "title": title,
            "description": description,
            "default": default,
            "example": example,
            "regex": regex,
            "min_length": min_length,
            "max_length": max_length,
        }
        inputs = {kw: kwargs[kw] for kw in kwargs if kwargs[kw] is not None}
        return self._set_value(**inputs)

    def set_num_value(
        self,
        required: bool = None,
        title: str = None,
        description: str = None,
        default: Any = None,
        example: Any = None,
        gt: Union[int, float] = None,
        ge: Union[int, float] = None,
        lt: Union[int, float] = None,
        le: Union[int, float] = None,
    ):
        """Set value method for numeric fields.

        Attrs:
                required (<class 'bool'>): The field is required or not.
                title (<class 'str'>): Title of field.
                description (<class 'str'>): Description of the field.
                default (typing.Any): Default value of the field.
                example (typing.Any): Example value of the field.
                gt ([<class 'int'>, <class 'float'>]):The numeric field should greater than value assigned.
                ge ([<class 'int'>, <class 'float'>]): The numeric field should greater than or equal value assigned
                lt ([<class 'int'>, <class 'float'>]):The numeric field should less than value assigned.
                le ([<class 'int'>, <class 'float'>]): The numeric field should less than or equal value assigned
        """
        kwargs = {
            "required": required,
            "title": title,
            "description": description,
            "default": default,
            "example": example,
            "gt": gt,
            "ge": ge,
            "lt": lt,
            "le": le,
        }
        inputs = {kw: kwargs[kw] for kw in kwargs if kwargs[kw] is not None}
        return self._set_value(**inputs)
