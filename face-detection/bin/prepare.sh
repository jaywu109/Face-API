#!/bin/bash
set -e

NEXTCLD=""
MODELROOT=""
DETECTION_MODEL="detection_scrfd_34g.onnx"
ALIGNMENT_MODEL="alignment_synergy.onnx"

export DETECTION_MODEL_PATH="/model/${DETECTION_MODEL}"
export ALIGNMENT_MODEL_PATH="/model/${ALIGNMENT_MODEL}"

mkdir -p /model
mkdir -p /model/custom
mkdir -p /model/custom/ref_images

if [ ! -f ${DETECTION_MODEL_PATH} ]; then
    echo "Downloading detection model parameters ${DETECTION_MODEL} from nextcloud."
    curl -s -u bigfiles:bigfiles ${NEXTCLD}/${MODELROOT}/${DETECTION_MODEL} -o ${DETECTION_MODEL_PATH}
fi

if [ ! -f ${ALIGNMENT_MODEL_PATH} ]; then
    echo "Downloading alignment model parameters ${ALIGNMENT_MODEL} from nextcloud."
    curl -s -u bigfiles:bigfiles ${NEXTCLD}/${MODELROOT}/${ALIGNMENT_MODEL} -o ${ALIGNMENT_MODEL_PATH}
fi
