import typing
from typing import Any, Callable, Coroutine, Dict, List, Optional, Sequence, Type, Union

from fastapi import APIRouter as _APIRouter
from fastapi import FastAPI as _FastAPI
from fastapi import params
from fastapi import routing as _routing
from fastapi.datastructures import Default
from fastapi.routing import APIRoute as _APIRoute
from fastapi.types import DecoratedCallable
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import BaseRoute

from src.base._schema import ResponseSchema
from src.base.configs import get_conf

from .logger import LoggingRoute


app_conf = get_conf("APP")


class APIRouter(_APIRouter):
    def __init__(
        self,
        *,
        prefix: str = "",
        tags: Optional[List[str]] = None,
        dependencies: Optional[Sequence[params.Depends]] = None,
        default_response_class: Type[_routing.Response] = Default(_routing.JSONResponse),
        responses: Optional[Dict[Union[int, str], Dict[str, Any]]] = None,
        callbacks: Optional[List[_routing.BaseRoute]] = None,
        routes: Optional[List[_routing.BaseRoute]] = None,
        redirect_slashes: bool = True,
        default: Optional[_routing.ASGIApp] = None,
        dependency_overrides_provider: Optional[Any] = None,
        route_class: Type[_APIRoute] = LoggingRoute,
        on_startup: Optional[Sequence[Callable[[], Any]]] = None,
        on_shutdown: Optional[Sequence[Callable[[], Any]]] = None,
        deprecated: Optional[bool] = None,
        include_in_schema: bool = True,
        std_response_handling: bool = True,
    ):
        self.std_response_handling = std_response_handling
        super().__init__(
            prefix=prefix,
            tags=tags,
            dependencies=dependencies,
            default_response_class=default_response_class,
            responses=responses,
            callbacks=callbacks,
            routes=routes,
            redirect_slashes=redirect_slashes,
            default=default,
            dependency_overrides_provider=dependency_overrides_provider,
            route_class=route_class,
            on_startup=on_startup,
            on_shutdown=on_shutdown,
            deprecated=deprecated,
            include_in_schema=include_in_schema,
        )

    def api_route(
        self, path, *, response_model: Optional[Type[Any]] = None, response_model_is_list: bool = False, **kwargs
    ) -> Callable[[DecoratedCallable], DecoratedCallable]:
        if response_model and self.std_response_handling:  # Remodel the ResponseSchema
            if (
                type(response_model) is typing._GenericAlias and response_model.__origin__ is list
            ):  # Check if type in List[Class]
                response_model = response_model.__args__[0]
                response_model_is_list = True
            response_model = ResponseSchema(
                f"{self.prefix+path} Response Schema", ret_data=response_model, is_list=response_model_is_list
            )
        return super().api_route(path, response_model=response_model, **kwargs)


class FastAPI(_FastAPI):
    def __init__(
        self,
        *,
        debug: bool = app_conf["DEBUG"],
        routes: Optional[List[BaseRoute]] = None,
        title: str = "FastAPI",
        description: str = "",
        version: str = "0.1.0",
        openapi_url: Optional[str] = "/openapi.json",
        openapi_tags: Optional[List[Dict[str, Any]]] = None,
        servers: Optional[List[Dict[str, Union[str, Any]]]] = None,
        dependencies: Optional[Sequence[params.Depends]] = None,
        default_response_class: Type[Response] = Default(JSONResponse),
        docs_url: Optional[str] = "/docs",
        redoc_url: Optional[str] = "/redoc",
        swagger_ui_oauth2_redirect_url: Optional[str] = "/docs/oauth2-redirect",
        swagger_ui_init_oauth: Optional[Dict[str, Any]] = None,
        middleware: Optional[Sequence[Middleware]] = None,
        exception_handlers: Optional[
            Dict[
                Union[int, Type[Exception]],
                Callable[[Request, Any], Coroutine[Any, Any, Response]],
            ]
        ] = None,
        on_startup: Optional[Sequence[Callable[[], Any]]] = None,
        on_shutdown: Optional[Sequence[Callable[[], Any]]] = None,
        openapi_prefix: str = "",
        root_path: str = app_conf["ROOT_PATH"],
        root_path_in_servers: bool = True,
        responses: Optional[Dict[Union[int, str], Dict[str, Any]]] = None,
        callbacks: Optional[List[BaseRoute]] = None,
        deprecated: Optional[bool] = None,
        include_in_schema: bool = True,
        **extra: Any,
    ) -> None:
        super().__init__(
            debug=debug,
            routes=routes,
            title=title,
            description=description,
            version=version,
            openapi_url=openapi_url,
            openapi_tags=openapi_tags,
            servers=servers,
            dependencies=dependencies,
            default_response_class=default_response_class,
            docs_url=docs_url,
            redoc_url=redoc_url,
            swagger_ui_oauth2_redirect_url=swagger_ui_oauth2_redirect_url,
            swagger_ui_init_oauth=swagger_ui_init_oauth,
            middleware=middleware,
            exception_handlers=exception_handlers,
            on_startup=on_startup,
            on_shutdown=on_shutdown,
            openapi_prefix=openapi_prefix,
            root_path=root_path,
            root_path_in_servers=root_path_in_servers,
            responses=responses,
            callbacks=callbacks,
            deprecated=deprecated,
            include_in_schema=include_in_schema,
            **extra,
        )
