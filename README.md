# Face Web API

Web API for face detection, alignment, recognition, attribute analysis using ***FastAPI*** and other open source project, such as [InsightFace](https://github.com/deepinsight/insightface) and [deepface](https://github.com/serengil/deepface).

Set up environment and start service:
> take face attribute analysis API for example
---
1. Download additional file from [3dmm_data.zip](https://drive.google.com/file/d/1SQsMhvAmpD1O8Hm0yEGom0C0rXtA0qs8/view) and put at following path ***(need to be done for all the api)***:
```
face-attribute/src/model/pipeline_utils/3dmm_data
face-detection/src/model/pipeline_utils/3dmm_data
face-recognition/src/model/pipeline_utils/3dmm_data
```
2. Create environment using:
```
cd face-attribute/deploy/docker
docker build -t attribute_image .
docker run -itd --runtime nvidia --gpus all --name attribute_env -p attribute_image 
```
3. Start the service:
```
uvicorn src.main:app --reload --host 0.0.0.0 --port=8080 --no-access-log --log-level=critical
```