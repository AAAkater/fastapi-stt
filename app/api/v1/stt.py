from io import BytesIO

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.models.response import ResponseBase, SttItem
from app.utils.logger import logger
from app.utils.speech_to_text import model

router = APIRouter(tags=["audio"])


@router.post(path="/transcribe", summary="stt")
async def transcribe_audio_to_text(file: UploadFile = File(...)):
    # 检查文件类型
    if file.content_type and not file.content_type.startswith("audio/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="参数错误"
        )
    try:
        # 读取上传的音频文件
        audio_content = await file.read()

        # 调用 transcribe_audio 进行转录
        text = model.stt(
            audio_file=BytesIO(audio_content),
            lang="zh",  # 可以根据需求设置语言
        )
        logger.success(f"转录成功:{text}")
    except Exception as e:
        logger.error(f"转录失败:\n{e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="转录失败",
        )
    return ResponseBase[SttItem](data=SttItem(text=text))
