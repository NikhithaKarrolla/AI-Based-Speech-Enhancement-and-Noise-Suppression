from pathlib import Path
import shutil
import uuid

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from webapp.inference import SpeechEnhancer


BASE_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


app = FastAPI(
    title="Deep Learning Speech Enhancement"
)


app.mount(
    "/static",
    StaticFiles(directory=BASE_DIR / "static"),
    name="static"
)

templates = Jinja2Templates(
    directory=BASE_DIR / "templates"
)


enhancer = SpeechEnhancer()


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


@app.post("/enhance")
async def enhance_audio(
    file: UploadFile = File(...)
):
    file_id = uuid.uuid4().hex

    input_path = UPLOAD_DIR / f"{file_id}_{file.filename}"
    output_path = OUTPUT_DIR / f"{file_id}_enhanced.wav"

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(
            file.file,
            buffer
        )

    enhancer.enhance(
        str(input_path),
        str(output_path)
    )

    return {
        "message": "Audio enhanced successfully",
        "output_file": f"/download/{output_path.name}"
    }


@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = OUTPUT_DIR / filename

    if not file_path.exists():
        return {
            "error": "File not found"
        }

    return FileResponse(
        file_path,
        media_type="audio/wav",
        filename=filename
    )