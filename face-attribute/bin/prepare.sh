#!/bin/bash
set -e

NEXTCLD=""
MODELROOT="face-attribute-2022"
PARSING_MODEL="parsing.onnx"
AGE_MODEL="age.onnx"
EMOTION_MODEL="emotion.onnx"
GENDER_MODEL="gender.onnx"

export PARSING_MODEL_PATH="/model/${PARSING_MODEL}"
export AGE_MODEL_PATH="/model/${AGE_MODEL}"
export EMOTION_MODEL_PATH="/model/${EMOTION_MODEL}"
export GENDER_MODEL_PATH="/model/${GENDER_MODEL}"

mkdir -p /model

if [ ! -f ${PARSING_MODEL_PATH} ]; then
    echo "Downloading parsing model parameters ${PARSING_MODEL} from nextcloud."
    curl -s -u bigfiles:bigfiles ${NEXTCLD}/${MODELROOT}/${PARSING_MODEL} -o ${PARSING_MODEL_PATH}
fi

if [ ! -f ${AGE_MODEL_PATH} ]; then
    echo "Downloading age model parameters ${AGE_MODEL} from nextcloud."
    curl -s -u bigfiles:bigfiles ${NEXTCLD}/${MODELROOT}/${AGE_MODEL} -o ${AGE_MODEL_PATH}
fi

if [ ! -f ${EMOTION_MODEL_PATH} ]; then
    echo "Downloading emotion model parameters ${EMOTION_MODEL} from nextcloud."
    curl -s -u bigfiles:bigfiles ${NEXTCLD}/${MODELROOT}/${EMOTION_MODEL} -o ${EMOTION_MODEL_PATH}
fi

if [ ! -f ${GENDER_MODEL_PATH} ]; then
    echo "Downloading gender model parameters ${GENDER_MODEL} from nextcloud."
    curl -s -u bigfiles:bigfiles ${NEXTCLD}/${MODELROOT}/${GENDER_MODEL} -o ${GENDER_MODEL_PATH}
fi
