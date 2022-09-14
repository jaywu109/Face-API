from typing import List

from fastapi import Body

from src.api.model.doc import Doc, examples
from src.api.model.senti_result import SentiResult
from src.base.model import ModelPath
from src.base.router import APIRouter
from src.base.schema import APIResponse


router = APIRouter(prefix="/standard/sample", tags=["Senti"])


@router.post("/single", response_model=SentiResult)
def single(body: Doc = Body(..., examples=examples)):
    """Sample endpoint.

    Args:
        body (Doc, optional): Input Doc class which include docid, content, headline. Defaults to Body(..., examples=examples).

    Returns:
        Dict: Single sample response with standard key(retCode, retInfo, retData).
    """
    _ = ModelPath("tiny_model.sample", "weight.sample").open("r")  # Sample of loading model
    # Use APIResponse to package your response data with interface.
    res = APIResponse(
        retCode="W",  # Optional, default S
        retInfo=f"Response example of:\n docid:{body.docid}, headline: {body.headline},content: {body.content}",  # Optional, default OK
        retData={
            "score": 0.9769629240036011,
            "scores": {
                "positive": 0.9769629240036011,
                "neutral": 0.012471978552639484,
                "negative": 0.01056508906185627,
            },
            "label": "positive",
        },
    )
    # or create your own response dictionary
    res = {
        "retCode": "W",
        "retInfo": f"Response example of:\n docid:{body.docid}, headline: {body.headline},content: {body.content}",
        "retData": {
            "score": 0.9769629240036011,
            "scores": {
                "positive": 0.9769629240036011,
                "neutral": 0.012471978552639484,
                "negative": 0.01056508906185627,
            },
            "label": "positive",
        },
    }
    return res


@router.post("/batch", response_model=List[SentiResult])  # Use typing.List
def batch(body: Doc = Body(..., examples=examples)):
    """Sample endpoint.  return a List of SentiResult in key `retData`.

    Args:
        body (Doc, optional): Input Doc class which include docid, content, headline. Defaults to Body(..., examples=examples).

    Returns:
        Dict: Batch sample response with standard key(retCode, retInfo, retData).
    """
    _ = ModelPath("tiny_model.sample", "weight.sample").open("r")  # ModelPath sample
    # Use APIResponse to package your response data with interface.
    res = APIResponse(
        retCode="W",  # Optional, default S
        retInfo=f"Response example of:\n docid:{body.docid}, headline: {body.headline},content: {body.content}",  # Optional, default OK
        retData=[
            {
                "score": 0.9769629240036011,
                "scores": {
                    "positive": 0.9769629240036011,
                    "neutral": 0.012471978552639484,
                    "negative": 0.01056508906185627,
                },
                "label": "positive",
            }
        ],
    )
    # or create your own response dictionary
    res = {
        "retCode": "W",  # Optional, default S
        "retInfo": f"Response example of:\n docid:{body.docid}, headline: {body.headline},content: {body.content}",  # Optional, default OK
        "retData": [
            {
                "score": 0.9769629240036011,
                "scores": {
                    "positive": 0.9769629240036011,
                    "neutral": 0.012471978552639484,
                    "negative": 0.01056508906185627,
                },
                "label": "positive",
            }
        ],
    }
    return res
