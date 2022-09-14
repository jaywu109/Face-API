import base64
from io import BytesIO
from PIL import Image
from typing import List, Union

from src.base.schema import CustomField, Schema


def get_img_string(path: str):
    buffered = BytesIO()
    Image.open(path).save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue())

# Define Input related Schema
ImageUrlInputSchema = Schema("ImageUrlInput")
ImageUrlInputSchema.append(CustomField(name="url", type=str, required=True))
ImageUrlInputSchema.url.set_description("Image URL")

ImageBytestringInputSchema = Schema("ImageBytestringInput")
ImageBytestringInputSchema.append(CustomField(name="bytestring", type=str, required=True))
ImageBytestringInputSchema.bytestring.set_description("Image Base 64 Encoded Bytestring")

# Define Result Schema
UrlBboxResult = Schema("UrlBboxResult")
UrlBboxResult.append(CustomField(name="bbox", type=Union[List[int], str]))
UrlBboxResult.bbox.set_description("Face Bounding Box")

StrBboxResult = Schema("StrBboxResult")
StrBboxResult.append(CustomField(name="bbox", type=Union[List[int], str]))
StrBboxResult.bbox.set_description("Face Bounding Box")

UrlLandmarkResult = Schema("UrlLandmarkResult")
UrlLandmarkResult.append(CustomField(name="landmark", type=Union[List[List[int]], str]))
UrlLandmarkResult.append(CustomField(name="bbox", type=Union[List[int], str]))
UrlLandmarkResult.landmark.set_description("Face Landmark")
UrlLandmarkResult.bbox.set_description("Face Bounding Box")

StrLandmarkResult = Schema("StrLandmarkResult")
StrLandmarkResult.append(CustomField(name="landmark", type=Union[List[List[int]], str]))
StrLandmarkResult.append(CustomField(name="bbox", type=Union[List[int], str]))
StrLandmarkResult.landmark.set_description("Face Landmark")
StrLandmarkResult.bbox.set_description("Face Bounding Box")

UrlAlignCropResult = Schema("UrlAlignCropResult")
UrlAlignCropResult.append(CustomField(name="img_bytestring", type=str))
UrlAlignCropResult.append(CustomField(name="landmark", type=Union[List[List[int]], str]))
UrlAlignCropResult.append(CustomField(name="bbox", type=Union[List[int], str]))
UrlAlignCropResult.img_bytestring.set_description("Return Image")
UrlAlignCropResult.landmark.set_description("Face Landmark")
UrlAlignCropResult.bbox.set_description("Face Bounding Box")

StrAlignCropResult = Schema("StrAlignCropResult")
StrAlignCropResult.append(CustomField(name="img_bytestring", type=str))
StrAlignCropResult.append(CustomField(name="landmark", type=Union[List[List[int]], str]))
StrAlignCropResult.append(CustomField(name="bbox", type=Union[List[int], str]))
StrAlignCropResult.img_bytestring.set_description("Return Image")
StrAlignCropResult.landmark.set_description("Face Landmark")
StrAlignCropResult.bbox.set_description("Face Bounding Box")

UrlPlotBboxLandmarkResult = Schema("UrlPlotBboxLandmarkResult")
UrlPlotBboxLandmarkResult.append(CustomField(name="img_bytestring", type=str))
UrlPlotBboxLandmarkResult.append(CustomField(name="landmark", type=Union[List[List[int]], str]))
UrlPlotBboxLandmarkResult.append(CustomField(name="bbox", type=Union[List[int], str]))
UrlPlotBboxLandmarkResult.img_bytestring.set_description("Return Image")
UrlPlotBboxLandmarkResult.landmark.set_description("Face Landmark")
UrlPlotBboxLandmarkResult.bbox.set_description("Face Bounding Box")

StrPlotBboxLandmarkResult = Schema("StrPlotBboxLandmarkResult")
StrPlotBboxLandmarkResult.append(CustomField(name="img_bytestring", type=str))
StrPlotBboxLandmarkResult.append(CustomField(name="landmark", type=Union[List[List[int]], str]))
StrPlotBboxLandmarkResult.append(CustomField(name="bbox", type=Union[List[int], str]))
StrPlotBboxLandmarkResult.img_bytestring.set_description("Return Image")
StrPlotBboxLandmarkResult.landmark.set_description("Face Landmark")
StrPlotBboxLandmarkResult.bbox.set_description("Face Bounding Box")

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
