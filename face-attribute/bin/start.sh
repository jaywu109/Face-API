#!/bin/sh
set -e

# test cicd

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source ${SCRIPT_DIR}/prepare.sh
# uvicorn src.main:app --reload --host 0.0.0.0 --port=8080 --no-access-log --log-level=critical