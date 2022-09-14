APP:
  ENVIRONMENT: development
  DEBUG: false
  ROOT_PATH: {{ .Values.appConfigs.swaggerRootPath }}

MODEL:
  DETECTION_MODEL_PATH: "/model/detection_scrfd_34g.onnx"
  ALIGNMENT_MODEL_PATH: "/model/alignment_synergy.onnx"
  IMAGEROOT: "/model/custom/ref_images"
