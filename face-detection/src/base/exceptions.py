from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError


class AttentionRequiredResponse(Exception):
    """Success with warning message."""

    def __init__(self, ret_data) -> None:
        self._http_status_code = 200
        self.retCode = "W-1"
        self.retInfo = "Warning"
        self.retData = ret_data


class PartialSuccessResponse(Exception):
    """Partial of the request successed."""

    def __init__(self, ret_data) -> None:
        self._http_status_code = 200
        self.retCode = "W-2"
        self.retInfo = "Warning with partial successed."
        self.retData = ret_data


class InvalidInputsException(Exception):
    """Invalid inputs Exception."""

    def __init__(self) -> None:
        self._http_status_code = 400
        self.retCode = "Sch-2"
        self.retInfo = None
        self.retData = None

    def LanguageInvalid(self, field_name, expect_lang):
        self.retInfo = (
            f"Field's language is invalid, please check:\n\tField name: {field_name}\n\tExpect language:{expect_lang}"
        )
        return self

    def NotAllowCharacter(self, field_name, error_msg=None):
        self.retInfo = f"Catch invalid character in field:\n\t Field name: {field_name}"
        if error_msg:
            self.retInfo += f"\n\t Error message:\n\t{error_msg}"
        return self


class ExternalDependencyException(Exception):
    """External dependency Exceptions."""

    def __init__(self) -> None:
        self._http_status_code = 500
        self.retCode = "Ext-1"
        self.retInfo = None
        self.retData = None

    def DatabaseException(self, ret_info="A database error has occurred."):
        self.retCode = "DB-1"
        self.retInfo = ret_info
        return self

    def ExternalAPIException(self, ret_info="A external api error has occurred."):
        self.retInfo = ret_info
        return self


class InternalProgramException(Exception):
    """Internal program exceptions."""

    def __init__(self, ret_info="Internal program error.") -> None:
        self._http_status_code = 500
        self.retCode = "Pgrm-1"
        self.retInfo = ret_info
        self.retData = None


def basic_exception_handler(req: Request, exc: Exception):
    return JSONResponse(
        status_code=exc._http_status_code,
        content={
            "retCode": exc.retCode,
            "retInfo": exc.retInfo,
            "retData": exc.retData,
        },
    )


def validation_exception_handler(req: Request, exc: ValidationError):
    """Override validation error handling."""
    return JSONResponse(status_code=400, content={"retCode": "Sch-1", "retInfo": str(exc), "retData": None})


def add_exception_handler(app: FastAPI) -> None:
    basic_exc_list = [
        AttentionRequiredResponse,
        PartialSuccessResponse,
        InvalidInputsException,
        ExternalDependencyException,
        InternalProgramException,
    ]
    validate_exc_list = [ValidationError, RequestValidationError]
    for exc in basic_exc_list:
        app.add_exception_handler(exc, basic_exception_handler)
    for val_exc in validate_exc_list:
        app.add_exception_handler(val_exc, validation_exception_handler)
