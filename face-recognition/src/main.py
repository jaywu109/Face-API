from fastapi.responses import JSONResponse

from src.api.sample import error_handling, logger, nonstandard, standard
from src.base.configs import get_conf
from src.base.exceptions import add_exception_handler
from src.base.logger import LoggingRoute
from src.base.router import FastAPI
from src.api.routers import recognition

app_conf = get_conf("APP")

app = FastAPI(
    title="Face Recognition API",
    description="""
# Description

**General Face Recognition** predicts face label and its similarity score from the user-defined database by query an image with single/multipe face.

The model is based on a pipeline composed of three models, which is Face Detection Model, Face Alignment Model and Face Recgonition Model.

# API Usage

## Database Process

- Get current valid face reference image number enrolled in database using `/database/valid_records`
- Reset database to clear face reference images and embeddings saved using `/database/reset`
- Download the image using `/database/get`
- Delete the database image using `/database/delete`

### Additional Note

- It's necessary to **reset** the database whenever the reference dataset changed to **different domain**.

---

## Enrollment Process

- Enroll a single reference face image with url using `/enroll/single_url`
- Enroll a single reference face image with base64 encoded bytestring using `/enroll/single_bytestring`

### Additional Note

- Each reference image **should not contain more than one face**, as it would be confused to choose which face should be the label if it contains multiple faces.   
- Label should be a **non-negative interger** value.   
- If enroll image with the name same as the one already stored in database, the new-comer will **replace** the one saved in database.   

---

## Prediction Process

- Predict face label of a single image with url using `/predict/single_url`
- Predict face label of a single image with base64 encoded bytestring using `/predict/single_bytestring`

### Additional Note

- It's supported to predict face label with image containing **multiple faces**.

""",
)

app.router.route_class = LoggingRoute
app.include_router(recognition.database_router)
app.include_router(recognition.enroll_router)
app.include_router(recognition.predict_router)
add_exception_handler(app)

# Health check =====
# If alived, return status code 200 and message "I'm"


@app.get("/healthz")
async def health_check():
    """Check the server is alived."""
    return JSONResponse(status_code=200, content={"status": "OK"})


# Test example =====
@app.post("/test")
def post_(json_dict: dict):
    """Test endpoint for loadtest."""
    resp = JSONResponse(content=json_dict, media_type="application/json", status_code=200)
    return resp


if __name__ == "__main__":
    import asyncio

    import uvloop
    from hypercorn.asyncio import serve
    from hypercorn.config import Config

    config = Config()
    config.bind = ["0.0.0.0:8080"]

    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    loop.run_until_complete(serve(app, config))
