from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from PIL import Image
import base64
import io

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="views")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(name="index.html", context={"request": request})


@app.post("/process/")
async def process_image(request: Request, image: UploadFile = File(...)):
    contents = await image.read()

    with Image.open(io.BytesIO(contents)) as img:
        img = img.rotate(90)
        img_io = io.BytesIO()
        img.save(img_io, format="PNG")
        img_io.seek(0)
        base64_image = base64.b64encode(img_io.getvalue()).decode("utf-8")

    return templates.TemplateResponse(
        name="image.html", context={ "request": request, "base64_image": base64_image }
    )
