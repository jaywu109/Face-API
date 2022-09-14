import base64
from io import BytesIO
from PIL import Image
from typing import Dict, List, Optional, Union

from src.base.schema import CustomField, Schema

def get_img_string(path: str):
    buffered = BytesIO()
    Image.open(path).save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue())

# Define Enroll related Schema
ImageUrlEnrollInput = Schema("ImageUrlEnrollInput")
ImageUrlEnrollInput.append(CustomField(name="url", type=str, required=True))
ImageUrlEnrollInput.append(CustomField(name="name", type=str, required=True))
ImageUrlEnrollInput.append(CustomField(name="label", type=int, required=True))
ImageUrlEnrollInput.url.set_description("Image URL")
ImageUrlEnrollInput.name.set_description("Image Name with File Extension(e.g. tsai.jpg)")
ImageUrlEnrollInput.label.set_description("Image Label ID")

ImageBytestringEnrollInput = Schema("ImageBytestringEnrollInput")
ImageBytestringEnrollInput.append(CustomField(name="bytestring", type=str, required=True))
ImageBytestringEnrollInput.append(CustomField(name="name", type=str, required=True))
ImageBytestringEnrollInput.append(CustomField(name="label", type=int, required=True))
ImageBytestringEnrollInput.bytestring.set_description("Image Base 64 Encoded Bytestring")
ImageBytestringEnrollInput.name.set_description("Image Name with File Extension(e.g. tsai.jpg)")
ImageBytestringEnrollInput.label.set_description("Image Label ID")

# Define Prediction related Schema
ImageUrlPredictInput = Schema("ImageUrlPredictInput")
ImageUrlPredictInput.append(CustomField(name="url", type=str, required=True))
ImageUrlPredictInput.url.set_description("Image URL")

ImageBytestringPredictInput = Schema("ImageBytestringPredictInput")
ImageBytestringPredictInput.append(CustomField(name="bytestring", type=str, required=True))
ImageBytestringPredictInput.bytestring.set_description("Image Base 64 Encoded Bytestring")

# Define Enroll Result Schema
UrlImageEnrollResult = Schema("UrlImageEnrollResult")
UrlImageEnrollResult.append(CustomField(name="bbox", type=Union[List[int], str]))
UrlImageEnrollResult.bbox.set_description("Bounding Box")

StrImageEnrollResult = Schema("StrImageEnrollResult")
StrImageEnrollResult.append(CustomField(name="bbox", type=Union[List[int], str]))
StrImageEnrollResult.bbox.set_description("Bounding Box")

# Define Prediction Result Schema
UrlImagePredictResult = Schema("UrlImagePredictResult")
UrlImagePredictResult.append(CustomField(name="bbox", type=Union[List[int], str]))
UrlImagePredictResult.append(CustomField(name="top1_label", type=Union[int, str]))
UrlImagePredictResult.append(CustomField(name="top1_score", type=Union[float, str]))
UrlImagePredictResult.append(CustomField(name="top2_label", type=Union[int, str]))
UrlImagePredictResult.append(CustomField(name="top2_score", type=Union[float, str]))
UrlImagePredictResult.bbox.set_description("Bounding Box")
UrlImagePredictResult.top1_label.set_description("Top 1 Label ID") 
UrlImagePredictResult.top1_score.set_description("Top 1 Score")
UrlImagePredictResult.top2_label.set_description("Top 2 Label ID") 
UrlImagePredictResult.top2_score.set_description("Top 2 Score")

StrImagePredictResult = Schema("StrImagePredictResult")
StrImagePredictResult.append(CustomField(name="bbox", type=Union[List[int], str]))
StrImagePredictResult.append(CustomField(name="top1_label", type=Union[int, str]))
StrImagePredictResult.append(CustomField(name="top1_score", type=Union[float, str]))
StrImagePredictResult.append(CustomField(name="top2_label", type=Union[int, str]))
StrImagePredictResult.append(CustomField(name="top2_score", type=Union[float, str]))
StrImagePredictResult.bbox.set_description("Bounding Box")
StrImagePredictResult.top1_label.set_description("Top 1 Label ID") 
StrImagePredictResult.top1_score.set_description("Top 1 Score")
StrImagePredictResult.top2_label.set_description("Top 2 Label ID") 
StrImagePredictResult.top2_score.set_description("Top 2 Score")

# Define Database Schemea
DatabaseDeleteSchema = Schema("DatabaseDelete")
DatabaseDeleteSchema.append(CustomField(name="status", type=str))
DatabaseDeleteSchema.append(CustomField(name="valid_records", type=int))
DatabaseDeleteSchema.status.set_description("Delete Status")
DatabaseDeleteSchema.valid_records.set_description("Valids Reference Image Number in Database")

DatabaseResetSchema = Schema("DatabaseReset")
DatabaseResetSchema.append(CustomField(name="status", type=str))
DatabaseResetSchema.status.set_description("Reset Status")

DatabaseGetDataSchema = Schema("DatabaseGetData")
DatabaseGetDataSchema.append(CustomField(name="embedding", type=List[float]))
DatabaseGetDataSchema.append(CustomField(name="bbox", type=List[int]))
DatabaseGetDataSchema.append(CustomField(name="label", type=int))
DatabaseGetDataSchema.embedding.set_description("Image Embedding from Recognition Model")
DatabaseGetDataSchema.bbox.set_description("Bounding Box")
DatabaseGetDataSchema.label.set_description("Reference Face Label")

DatabaseValidStatSchema = Schema("DatabaseValidStat")
DatabaseValidStatSchema.append(CustomField(name="valid_records", type=int))
DatabaseValidStatSchema.append(CustomField(name="valid_names", type=List[str]))
DatabaseValidStatSchema.append(CustomField(name="valid_label_count", type= Dict[int, int]))
DatabaseValidStatSchema.valid_records.set_description("Valid Reference Image Number in Database")
DatabaseValidStatSchema.valid_names.set_description("Valid Reference Image Name List in Database")
DatabaseValidStatSchema.valid_label_count.set_description("Valid Reference Image Label Count")


# Add examples
# Add image asset
img_url_1 = "https://upload.wikimedia.org/wikipedia/commons/e/eb/%E8%94%A1%E4%BE%9D%E6%9E%97%E5%87%BA%E5%B8%AD%E7%8E%8B%E5%9B%BD%E5%A4%A7%E5%B8%9D%E5%85%A8%E7%90%83%E5%B7%A1%E5%9B%9E%E8%B5%9B%E7%8E%B0%E5%9C%BA_cropped_%28cropped%29.jpg"
img_url_2 = 'https://upload.wikimedia.org/wikipedia/commons/thumb/0/0b/Quentin_Tarantino_by_Gage_Skidmore.jpg/440px-Quentin_Tarantino_by_Gage_Skidmore.jpg'
img_url_3 = 'https://upload.wikimedia.org/wikipedia/commons/thumb/9/95/Christopher_Nolan_Cannes_2018.jpg/440px-Christopher_Nolan_Cannes_2018.jpg'

img_str_1 = get_img_string('src/data/tsai.jpeg')
img_str_2 = get_img_string('src/data/quentine.jpeg')
img_str_3 = get_img_string('src/data/nolan.jpeg')

# Predict Examples
predict_single_example_url_1 = ImageUrlPredictInput.new_example(url=img_url_1)
predict_single_example_url_2 = ImageUrlPredictInput.new_example(url=img_url_2)
predict_single_example_str_1 = ImageBytestringPredictInput.new_example(bytestring=img_str_1)
predict_single_example_str_2 = ImageBytestringPredictInput.new_example(bytestring=img_str_2)

predict_single_examples_url = {"url_1": predict_single_example_url_1,
                            "url_2": predict_single_example_url_2}

predict_single_examples_str = {"str_1": predict_single_example_str_1,
                            "str_2": predict_single_example_str_2}

# Enroll Examples
enroll_single_example_url_1 = ImageUrlEnrollInput.new_example(
    url = img_url_1,
    name="tsai.jpeg",
    label=0)
enroll_single_example_url_2 = ImageUrlEnrollInput.new_example(
    url = img_url_2,
    name="quentine.jpeg",
    label=1)
enroll_single_example_url_3 = ImageUrlEnrollInput.new_example(
    url = img_url_3,
    name="nolan.jpeg",
    label=2)
enroll_single_example_str_1 = ImageBytestringEnrollInput.new_example(
    bytestring = img_str_1,
    name="tsai.jpeg",
    label=0)
enroll_single_example_str_2 = ImageBytestringEnrollInput.new_example(
    bytestring = img_str_2,
    name="quentine.jpeg",
    label=1)
enroll_single_example_str_3 = ImageBytestringEnrollInput.new_example(
    bytestring = img_str_3,
    name="nolan.jpeg",
    label=2)

enroll_single_examples_url = {"url_1": enroll_single_example_url_1,
                            "url_2": enroll_single_example_url_2,
                            "url_3": enroll_single_example_url_3}

enroll_single_examples_str = {"str_1": enroll_single_example_str_1,
                            "str_2": enroll_single_example_str_2,
                            "str_3": enroll_single_example_str_3}
