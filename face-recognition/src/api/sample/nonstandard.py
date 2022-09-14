from typing import List

from fastapi import Body

from src.api.model.doc import Doc, examples
from src.api.model.senti_result import SentiResult
from src.base.model import ModelPath
from src.base.router import APIRouter


router = APIRouter(
    prefix="/nonstandard/sample", tags=["Senti"], std_response_handling=False
)  # Close the standard response handling for nonstandard response


@router.post("/single", response_model=SentiResult)
def single(body: Doc = Body(..., examples=examples)):
    """Sample endpoint.

    Args:
        body (Doc, optional): Input Doc class which include docid, content, headline. Defaults to Body(..., examples=examples).

    Returns:
        Dict: Single sample response.
    """
    _ = ModelPath("tiny_model.sample", "weight.sample").open("r")  # ModelPath sample
    res = {
        "score": 0.9769629240036011,
        "scores": {
            "positive": 0.9769629240036011,
            "neutral": 0.012471978552639484,
            "negative": 0.01056508906185627,
        },
        "label": "positive",
    }
    return res


@router.post("/batch", response_model=List[SentiResult])
def batch(body: Doc = Body(..., examples=examples)):
    """Sample endpoint.

    Args:
        body (Doc, optional): Input Doc class which include docid, content, headline. Defaults to Body(..., examples=examples).

    Returns:
        List: Batch sample response.
    """
    _ = ModelPath("tiny_model.sample", "weight.sample").open("r")  # ModelPath sample
    res = [
        {
            "score": 0.9769629240036011,
            "scores": {
                "positive": 0.9769629240036011,
                "neutral": 0.012471978552639484,
                "negative": 0.01056508906185627,
            },
            "label": "positive",
        }
    ]
    return res
