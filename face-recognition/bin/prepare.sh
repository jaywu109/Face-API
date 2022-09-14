#!/bin/bash
set -e

NEXTCLD=""
MODELROOT=""
RECOGNITION_MODEL="glint360k_r100_final.onnx"
REFERENCE_EMBEDDING="ref_embedding.h5"
REFERENCE_IMAGES="ref_imgs.tar.gz"

export RECOGNITION_MODEL_PATH="/model/${RECOGNITION_MODEL}"
export IMAGEROOT="/model/custom/ref_images"
export EMBEDDING_ROOT="/model/custom/${REFERENCE_EMBEDDING}"

mkdir -p /model/custom
mkdir -p /model/custom/ref_images

if [ ! -f ${RECOGNITION_MODEL_PATH} ]; then
    echo "Downloading recognition model parameters ${RECOGNITION_MODEL} from nextcloud."
    curl -s -u bigfiles:bigfiles ${NEXTCLD}/${MODELROOT}/${RECOGNITION_MODEL} -o ${RECOGNITION_MODEL_PATH}
fi
