import collections
import json
import socket
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, Tuple, Union

from fastapi import Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute
from loguru import logger as _logger
from user_agents import parse

from .exceptions import InternalProgramException


_logger_format = "{time:YYYY-MM-DD HH:mm:ss:sss!UTC} | {level} | {extra[hostname]} | {extra[remote_address]} | {extra[method]} | {extra[path]} | {extra[agent]} | {extra[status]} | {extra[ret_code]} | {extra[process_time]} | {message} | {extra[curl]}"

_logger.add(
    sink=sys.stdout,
    level="INFO",
    format=_logger_format,
)

_logger.remove(handler_id=0)


@dataclass
class Extra(collections.abc.Mapping):
    remote_address: str = ""
    hostname: str = socket.gethostname()
    method: str = ""
    path: str = ""
    scheme: str = ""
    agent: str = ""
    status: int = 200
    ret_code: str = ""
    process_time: str = ""
    curl: str = ""

    @classmethod
    def pre_requrest(cls, request: Request) -> "Extra":
        user_agent = parse(request.headers["user-agent"])
        if user_agent.browser.family != "Other":
            agent = "%s/%s" % (
                user_agent.browser.family,
                user_agent.browser.version_string,
            )
        else:
            agent = request.headers.get("User-Agent")
        obj = cls(
            remote_address=request.client.host,
            hostname=socket.gethostname(),
            method=request.method,
            path=request.url.path,
            scheme=request.url.scheme,
            agent=agent,
            curl="",
            ret_code="",
        )
        return obj

    async def raise_exception(self, request: Request, start: datetime) -> None:
        self.process_time = int((datetime.now() - start).microseconds / 1000)
        command_template = (
            "curl -X '{method}' '{uri}' -H 'Content-Type: application/json' " "-H 'accept: application/json' -d '{payload}'"
        )
        uri = request.url._url
        body = await request.body()
        payload = body.decode("utf8")
        self.curl = command_template.format(self.method, uri, payload)

    def __setitem__(self, __name: str, __value: Any) -> None:
        self.__dict__[__name] = __value

    def __getitem__(self, __name: str) -> Any:
        return self.__dict__[__name]

    def __len__(self) -> int:
        return len(self.__dict__)

    def __iter__(self) -> str:
        for key in self.__dict__:
            yield key


class Logger:
    """Logger with standard template."""

    @staticmethod
    def trace(message, extra: Union[Extra, Dict] = Extra(), **kwargs):
        """Log trace level message with status 200.

        Attriburtes:
            msg: Message.
        """
        _logger.trace(message, **extra, **kwargs)

    @staticmethod
    def debug(message, extra: Union[Extra, Dict] = Extra(), **kwargs):
        """Log debug level message with status 200.

        Attriburtes:
            msg: Message.
        """
        _logger.debug(message, **extra, **kwargs)

    @staticmethod
    def info(message, extra: Union[Extra, Dict] = Extra(), **kwargs):
        """Log info level message with status 200.

        Attriburtes:
            msg: Message.
        """
        _logger.info(message, **extra, **kwargs)

    @staticmethod
    def success(message, extra: Union[Extra, Dict] = Extra(), **kwargs):
        """Log success level message with status 200.

        Attriburtes:
            msg: Message.
        """
        _logger.success(message, **extra, **kwargs)

    @staticmethod
    def warning(message, extra: Union[Extra, Dict] = Extra(), **kwargs):
        """Log warning level message with status 200.

        Attriburtes:
            msg: Message.
        """
        _logger.warning(message, **extra, **kwargs)

    @staticmethod
    def error(message, extra: Union[Extra, Dict] = Extra(), **kwargs):
        """Log error level message with status 500.

        Attriburtes:
            msg: Message.
        """
        _logger.error(message, **extra, **kwargs)

    @staticmethod
    def critical(message, extra: Union[Extra, Dict] = Extra(), **kwargs):
        """Log critical level message with status 500.

        Attriburtes:
            msg: Message.
        """
        _logger.critical(message, **extra, **kwargs)


class LoggingRoute(APIRoute):
    def get_route_handler(self) -> Callable:
        original_route_handler = super().get_route_handler()

        async def custom_route_handler(request: Request) -> Tuple[Response, int, str, str, str]:
            msg = ""
            extra = Extra.pre_requrest(request)
            logger = Logger
            start = datetime.now()
            try:
                response = await original_route_handler(request)
                extra.process_time = int((datetime.now() - start).microseconds / 1000)
                extra.status = response.status_code
                try:
                    extra.ret_code = json.loads(response.body)["retCode"]
                except (AttributeError, KeyError, TypeError):
                    extra.ret_code = ""
                logger.info("{msg}", msg=msg, extra=extra)
                return response
            except RequestValidationError as e:
                extra.raise_exception(request, start)
                extra.status = 400
                msg = traceback.format_exc()
                logger.error("{msg}", msg=msg, extra=extra)
                raise e
            except Exception as e:
                extra.raise_exception(request, start)
                if getattr(e, "_http_status_code", None):
                    # Standard Exception of project template
                    extra.status = e._http_status_code
                    extra.ret_code = e.retCode
                    msg = e.retInfo
                    logger.error("{msg}", msg=msg, extra=extra)
                    raise e
                else:
                    # Unexpected Exception
                    extra.status = 500
                    msg = traceback.format_exc()
                    logger.error("{msg}", msg=msg, extra=extra)
                    raise InternalProgramException

        return custom_route_handler
