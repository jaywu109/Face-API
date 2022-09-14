APP:
  ENVIRONMENT: development
  DEBUG: false
  ROOT_PATH: {{ .Values.appConfigs.swaggerRootPath }}

MODEL:
  PARSING_MODEL_PATH: "/model/parsing.onnx"
  AGE_MODEL_PATH: "/model/age.onnx"
  EMOTION_MODEL_PATH: "/model/emotion.onnx"
  GENDER_MODEL_PATH: "/model/gender.onnx"

API_LINKS:
  DETECTION_URL: ""
  DETECTION_STR: ""
  ALIGNMENT_URL: ""
  ALIGNMENT_STR: ""