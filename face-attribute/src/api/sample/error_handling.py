from src.api.model.doc import Doc
from src.base.exceptions import (
    AttentionRequiredResponse,
    ExternalDependencyException,
    InternalProgramException,
    InvalidInputsException,
    PartialSuccessResponse,
)
from src.base.router import APIRouter


router = APIRouter(prefix="/error_handling/sample", tags=["sample"])


# 200
@router.get("/attention_required_response/")
async def arr_exc_handl_sample():
    raise AttentionRequiredResponse(ret_data={"field": "data"})


@router.get("/partial_success_response/")
async def psr_exc_handl_sample():
    raise PartialSuccessResponse(ret_data={"success": [], "fail": []})


# 400
@router.post("/invalid_inputs_exception/lang/")
async def iie_lang_exc_handl_sample(body: Doc):
    if body.content != "eng":
        raise InvalidInputsException().LanguageInvalid(field_name="content", expect_lang="eng")


@router.post("/invalid_inputs_exception/char/")
async def iie_char_exc_handl_sample(body: Doc):
    not_allow_character = "@"
    if not_allow_character in body.content:
        raise InvalidInputsException().NotAllowCharacter(
            field_name="content",
            error_msg=f"{not_allow_character} should not in content.",
        )  # error_msg is optional


# 500
@router.get("/external_dependency_exception/db/")
async def ede_db_exc_handl_sample():
    dependency_db_crash = True
    if dependency_db_crash:
        raise ExternalDependencyException().DatabaseException(ret_info="Db crash, plz try again later.")


@router.get("/external_dependency_exception/api/")
async def ede_api_exc_handl_sample():
    dependency_api_crash = True
    if dependency_api_crash:
        raise ExternalDependencyException().ExternalAPIException(ret_info="LTP api is crash, plz try again later.")


@router.get("/internal_program_exception/")
async def ipe_exc_handl_sample():
    raise InternalProgramException(ret_info="Error message!")


@router.post("/validation_error/")
async def ve_exc_handl_sample(body: Doc):
    res = Doc()
    return res
