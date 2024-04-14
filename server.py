import asyncio
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from models.oriented_rcnn import Detector
from models.rsinet import SuperResolver

from PIL import Image
import io

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

    original, double_scale, quad_scale = await asyncio.gather(
        process_image(image),
        process_image(image, sr_scale=2),
        process_image(image, sr_scale=4),
    )

    return templates.TemplateResponse(
        name="image.html",
        context={
            "request": request,
            "original_image": original,
            "2x_image": double_scale,
            "4x_image": quad_scale,
        },
    )


async def process_image(image, sr_scale=0):
    image = image.copy()

    if sr_scale > 0:
        image = resolve_image(image, sr_scale)

    detector = build_detector()
    bboxes, classes = detector.run(image)
    img_with_results = detector.visualize_results(image, bboxes, classes)

    img_io = io.BytesIO()
    img_with_results.save(img_io, format="PNG")
    img_io.seek(0)
    return img_io.getvalue()


def build_detector():
    return Detector(model_path="/workspaces/prototype/deployment_files/oriented_rcnn")


def build_super_resolver(scale):
    if scale == 2:
        return SuperResolver(
            model_path="/workspaces/prototype/deployment_files/sr/2x.onnx"
        )
    elif scale == 4:
        return SuperResolver(
            model_path="/workspaces/prototype/deployment_files/sr/4x.onnx"
        )
    
def resolve_image(image, scale):
    super_resolver = build_super_resolver(scale)
    return super_resolver.run(image)