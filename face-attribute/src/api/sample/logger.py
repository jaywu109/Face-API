from src.base import APIRouter, Logger


router = APIRouter(prefix="/logger/sample", tags=["Logging"])


@router.get("/trace")
def test_trace():
    Logger.trace("Trace log with default `method`, `path`.")


@router.get("/debug")
def test_debug():
    Logger.debug("Debug log with default `method`, `path`.")


@router.get("/info")
def test_info():
    Logger.info("Info log with default `method`, `path`.")


@router.get("/success")
def test_success():
    Logger.success("Success log with default `method`, `path`.")


@router.get("/warning")
def test_warning():
    Logger.warning("Warning log with default `method`, `path`.")


@router.get("/error")
def test_error():
    Logger.error("Error log with default `method`, `path`.")


@router.get("/critical")
def test_critical():
    Logger.critical("Critical log with default `method`, `path`.")
