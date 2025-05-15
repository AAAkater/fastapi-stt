from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse

router = APIRouter(tags=["audio"])


@router.post(path="/transcribe", summary="stt")
def transcribe_audio(file: UploadFile = File(...)):
    return JSONResponse(content={"filename": file.filename})
