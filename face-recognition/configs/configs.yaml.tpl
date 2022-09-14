APP:
  ENVIRONMENT: development
  DEBUG: false
  ROOT_PATH: {{ .Values.appConfigs.swaggerRootPath }}

MODEL:
  RECOGNITION_MODEL_PATH: "/model/glint360k_r100_final.onnx"
  IMAGEROOT: "/model/custom/ref_images"
  EMBEDDING_ROOT: "/model/custom/ref_embedding.h5"
  ALIGNMENT_URL: ""
  ALIGNMENT_STR: ""

