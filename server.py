from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import asyncio
from PIL import Image
import io
from base64 import b64encode
import logging

from onnx_optimizer import ONNXOptimizer
from ship_detection import ShipDetector

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

OPTIMIZED = True

if OPTIMIZED:
    ONNXOptimizer.optimize("/workspaces/prototype/deployment_files")

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="views")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(name="index.html", context={"request": request})


@app.post("/process/")
async def process_image(request: Request, image: UploadFile = File(...)):
    image = await image.read()
    image = Image.open(io.BytesIO(image))

    original_scale_image, double_scale_image, quad_scale_image = await asyncio.gather(
        handle_image(image, sr_scale=1),
        handle_image(image, sr_scale=2),
        handle_image(image, sr_scale=4),
    )

    return templates.TemplateResponse(
        name="image.html",
        context={
            "request": request,
            "original_scale_image": original_scale_image,
            "double_scale_image": double_scale_image,
            "quad_scale_image": quad_scale_image,
        },
    )


async def handle_image(image, sr_scale):
    result = ShipDetector.run(image, sr_scale=sr_scale, optimized=OPTIMIZED)

    result_io = io.BytesIO()
    result.save(result_io, format="PNG")
    result_io.seek(0)
    result_io = result_io.getvalue()

    b64_result = b64encode(result_io).decode("utf-8")

    return f"data:image/png;base64,{b64_result}"
