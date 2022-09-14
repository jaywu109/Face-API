from typing import Dict, List, Optional, Union
from io import BytesIO
import base64
from PIL import Image

from src.base.schema import CustomField, Schema

def get_img_string(path: str):
    buffered = BytesIO()
    Image.open(path).save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue())

# Define Input related Schema
ImageInputSchema = Schema("ImageInput")
ImageInputSchema.append(CustomField(name="url_or_bytestring", type=str, required=True))
ImageInputSchema.url_or_bytestring.set_description("Image URL or Base 64 Encoded Bytestring")

ImageUrlInputSchema = Schema("ImageUrlInput")
ImageUrlInputSchema.append(CustomField(name="url", type=str, required=True))
ImageUrlInputSchema.url.set_description("Image URL")

ImageBytestringInputSchema = Schema("ImageBytestringInput")
ImageBytestringInputSchema.append(CustomField(name="bytestring", type=str, required=True))
ImageBytestringInputSchema.bytestring.set_description("Image Base 64 Encoded Bytestring")

# Define Result Schema
UrlAgeResult = Schema("UrlAgeResult")
UrlAgeResult.append(CustomField(name="age", type=Union[int, str]))
UrlAgeResult.append(CustomField(name="bbox", type=Union[List[int], str]))
UrlAgeResult.age.set_description("Facial Age")
UrlAgeResult.bbox.set_description("Face Bounding Box")

StrAgeResult = Schema("StrAgeResult")
StrAgeResult.append(CustomField(name="age", type=Union[int, str]))
StrAgeResult.append(CustomField(name="bbox", type=Union[List[int], str]))
StrAgeResult.age.set_description("Facial Age")
StrAgeResult.bbox.set_description("Face Bounding Box")

UrlEmotionResult = Schema("UrlEmotionResult")
UrlEmotionResult.append(CustomField(name="emotion", type=str))
UrlEmotionResult.append(CustomField(name="bbox", type=Union[List[int], str]))
UrlEmotionResult.emotion.set_description("Facial Emotion")
UrlEmotionResult.bbox.set_description("Face Bounding Box")

StrEmotionResult = Schema("StrEmotionResult")
StrEmotionResult.append(CustomField(name="emotion", type=str))
StrEmotionResult.append(CustomField(name="bbox", type=Union[List[int], str]))
StrEmotionResult.emotion.set_description("Facial Emotion")
StrEmotionResult.bbox.set_description("Face Bounding Box")

UrlGenderResult = Schema("UrlGenderResult")
UrlGenderResult.append(CustomField(name="gender", type=str))
UrlGenderResult.append(CustomField(name="bbox", type=Union[List[int], str]))
UrlGenderResult.gender.set_description("Facial Gender")
UrlGenderResult.bbox.set_description("Face Bounding Box")

StrGenderResult = Schema("StrGenderResult")
StrGenderResult.append(CustomField(name="gender", type=str))
StrGenderResult.append(CustomField(name="bbox", type=Union[List[int], str]))
StrGenderResult.gender.set_description("Facial Gender")
StrGenderResult.bbox.set_description("Face Bounding Box")

UrlAllResult = Schema("UrlAllResult")
UrlAllResult.append(CustomField(name="age", type=Union[int, str]))
UrlAllResult.append(CustomField(name="emotion", type=str))
UrlAllResult.append(CustomField(name="gender", type=str))
UrlAllResult.append(CustomField(name="bbox", type=Union[List[int], str]))
UrlAllResult.age.set_description("Facial Age")
UrlAllResult.emotion.set_description("Facial Emotion")
UrlAllResult.gender.set_description("Facial Gender")
UrlAllResult.bbox.set_description("Face Bounding Box")

StrAllResult = Schema("StrAllResult")
StrAllResult.append(CustomField(name="age", type=Union[int, str]))
StrAllResult.append(CustomField(name="emotion", type=str))
StrAllResult.append(CustomField(name="gender", type=str))
StrAllResult.append(CustomField(name="bbox", type=Union[List[int], str]))
StrAllResult.age.set_description("Facial Age")
StrAllResult.emotion.set_description("Facial Emotion")
StrAllResult.gender.set_description("Facial Gender")
StrAllResult.bbox.set_description("Face Bounding Box")

# Add examples
# Add image asset
img_url_1 = "https://upload.wikimedia.org/wikipedia/commons/e/eb/%E8%94%A1%E4%BE%9D%E6%9E%97%E5%87%BA%E5%B8%AD%E7%8E%8B%E5%9B%BD%E5%A4%A7%E5%B8%9D%E5%85%A8%E7%90%83%E5%B7%A1%E5%9B%9E%E8%B5%9B%E7%8E%B0%E5%9C%BA_cropped_%28cropped%29.jpg"
img_url_2 = 'https://upload.wikimedia.org/wikipedia/commons/thumb/0/0b/Quentin_Tarantino_by_Gage_Skidmore.jpg/440px-Quentin_Tarantino_by_Gage_Skidmore.jpg'
img_url_3 = 'https://upload.wikimedia.org/wikipedia/commons/thumb/9/95/Christopher_Nolan_Cannes_2018.jpg/440px-Christopher_Nolan_Cannes_2018.jpg'

img_str_1 = get_img_string('src/data/tsai.jpeg')
img_str_2 = get_img_string('src/data/quentine.jpeg')
img_str_3 = get_img_string('src/data/nolan.jpeg')
img_str_multiple = get_img_string('src/data/multiple.jpeg')
img_str_noface = get_img_string('src/data/dog.jpeg')

# Input Examples
image_input_example_url_1 = ImageInputSchema.new_example(
    url_or_bytestring = img_url_1)
image_input_example_url_2 = ImageInputSchema.new_example(
    url_or_bytestring = img_url_2)
image_input_example_url_3 = ImageInputSchema.new_example(
    url_or_bytestring = img_url_3)

image_input_example_str_multiple_1 = ImageInputSchema.new_example(
    url_or_bytestring = img_str_multiple)
image_input_example_str_multiple_2 = ImageInputSchema.new_example(
    url_or_bytestring = img_str_multiple)

image_input_example_str_noface = ImageInputSchema.new_example(
    url_or_bytestring = img_str_noface)

image_input_example_str_1 = ImageInputSchema.new_example(
    url_or_bytestring = img_str_1)
image_input_example_str_2 = ImageInputSchema.new_example(
    url_or_bytestring = img_str_2)
image_input_example_str_3 = ImageInputSchema.new_example(
    url_or_bytestring = img_str_3)

image_input_examples = {"url_1": image_input_example_url_1,
                            "64encoded_1": image_input_example_str_1,
                            "multiple_all": image_input_example_str_multiple_1,
                            "multiple_single": image_input_example_str_multiple_2,
                            "noface": image_input_example_str_noface,
                            "url_2": image_input_example_url_2,
                            "url_3": image_input_example_url_3,                            
                            "64encoded_2": image_input_example_str_2,
                            "64encoded_3": image_input_example_str_3}

image_input_example_url_1 = ImageUrlInputSchema.new_example(
    url = img_url_1)
image_input_example_url_2 = ImageUrlInputSchema.new_example(
    url = img_url_2)
image_input_example_url_3 = ImageUrlInputSchema.new_example(
    url = img_url_3)

image_input_examples_url = {"url_1": image_input_example_url_1,
                            "url_2": image_input_example_url_2,
                            "url_3": image_input_example_url_3}

image_input_example_str_1 = ImageBytestringInputSchema.new_example(
    bytestring = img_str_1)
image_input_example_str_2 = ImageBytestringInputSchema.new_example(
    bytestring = img_str_2)
image_input_example_str_3 = ImageBytestringInputSchema.new_example(
    bytestring = img_str_3)
image_input_example_str_multiple = ImageBytestringInputSchema.new_example(
    bytestring = img_str_multiple)
image_input_example_str_noface = ImageBytestringInputSchema.new_example(
    bytestring = img_str_noface)

image_input_examples_str = {"str_1": image_input_example_str_1,
                            "str_2": image_input_example_str_2,
                            "str_3": image_input_example_str_3,
                            "str_multiple": image_input_example_str_multiple,
                            "str_noface": image_input_example_str_noface}
