import io
import os
import re
import sys
import time
import uuid
import base64
import logging
import argparse
from typing import Optional, Literal, Type, AsyncGenerator, Any

import torch
import torchaudio
from typing_extensions import Annotated
from pydantic import BaseModel, Field, ValidationError
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from utils import convert_audio_tensor_to_bytes, load_audio_from_bytes

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{ROOT_DIR}/../../..')
sys.path.append(f'{ROOT_DIR}/../../../third_party/Matcha-TTS')

log_level = logging.INFO
if os.getenv("MOCK_ENABLED", "0") == "1":
    from mock_cosyvoice import MockAsyncCosyVoice as AsyncCosyVoice2
    # Mock 时开启 Debug 日志
    log_level = logging.DEBUG
else:
    from async_cosyvoice.async_cosyvoice import AsyncCosyVoice2

logging.basicConfig(level=log_level,
                    format='%(asctime)s %(levelname)s %(message)s')

cosyvoice: AsyncCosyVoice2 | None = None


app = FastAPI()

# 配置CORS
app.add_middleware(
    CORSMiddleware,          # noqa
    allow_origins=["*"],     # 允许所有源，生产环境应限制为具体域名
    allow_credentials=True,
    allow_methods=["*"],     # 允许所有方法（如GET、POST等）
    allow_headers=["*"],     # 允许所有请求头
)

class VoiceUploadResponse(BaseModel):
    """音频上传响应参数"""
    uri: str = Field(...,
                     examples=["speech:your-voice-name:xxx:xxx"],
                     description="音色对应的URI")

# noinspection PyMethodParameters
class SpeechRequest(BaseModel):
    """语音合成请求参数"""
    input: str = Field(
        ...,
        max_length=4096,
        examples=["你好，欢迎使用语音合成服务！"],        description="需要转换为语音的文本内容"
    )
    voice: str = Field(
        ...,
        examples=[
            "001",
            "speech:voice-name:xxx:xxx",
        ],
        description="音色选择"
    )
    response_format: Optional[Literal["mp3", "wav", "pcm"]] = Field(
        "mp3",
        examples=["mp3", "wav", "pcm"],
        description="输出音频格式"
    )
    sample_rate: Optional[int] = Field(
        24000,
        description="采样率，目前不支持设置，默认为返回 24000 Hz音频数据"
    )
    stream: Optional[bool] = Field(
        False,
        description="开启流式返回。"
    )
    speed: Annotated[Optional[float], Field(strict=True, ge=0.25, le=4.0)] = Field(
        1.0,
        description="语速控制[0.25-4.0]"
    )

def save_voice_data(customName: str, audio_data: bytes, text: str) -> str:
    """保存音频数据并生成音色对应的URI"""
    user_id = "xxx"
    voice_id = str(uuid.uuid4())[:8]
    uri = f"speech:{customName}:{user_id}:{voice_id}"
    prompt_speech_16k = load_audio_from_bytes(audio_data, 16000)
    cosyvoice.frontend.generate_spk_info(
        uri,
        text,
        prompt_speech_16k,
        24000,
        customName
    )
    return uri

async def generator_wrapper(audio_data_generator: AsyncGenerator[dict, None]) -> AsyncGenerator[torch.Tensor, None]:
    async for chunk in audio_data_generator:
        yield chunk["tts_speech"]

async def generate_audio_content(request: SpeechRequest) -> AsyncGenerator[bytes, Any] | None:
    """生成音频内容（示例实现）"""
    tts_text = request.input
    spk_id = request.voice

    try:
        end_of_prompt_index = tts_text.find("<|endofprompt|>")
        if end_of_prompt_index != -1:
            instruct_text = tts_text[: end_of_prompt_index + len("<|endofprompt|>")]
            tts_text = tts_text[end_of_prompt_index + len("<|endofprompt|>") :]

            audio_tensor_data_generator = generator_wrapper(cosyvoice.inference_instruct2_by_spk_id(
                tts_text,
                instruct_text,
                spk_id,
                stream=request.stream,
                speed=request.speed,
                text_frontend=True,
            ))
        else:
            audio_tensor_data_generator = generator_wrapper(cosyvoice.inference_zero_shot_by_spk_id(
                tts_text,
                spk_id,
                stream=request.stream,
                speed=request.speed,
                text_frontend=True,
            ))

        audio_bytes_data_generator = convert_audio_tensor_to_bytes(
            audio_tensor_data_generator,
            request.response_format,
            sample_rate=request.sample_rate,
            stream=request.stream,
        )
        return audio_bytes_data_generator
    except Exception as e:
        logging.error(f"Processing failed: {str(e)}", exc_info=True)

def get_content_type(fmt: str, sample_rate: int) -> str:
    """获取对应格式的Content-Type"""
    if fmt == "pcm":
        return f"audio/L16; rate={sample_rate}; channels=1"
    return {
        "mp3": "audio/mpeg",
        "opus": "audio/opus",
        "wav": "audio/wav"
    }[fmt]

@app.post("/v1/audio/speech")
async def text_to_speech(request: SpeechRequest):
    """## 文本转语音接口"""
    try:
        # 构建响应头
        content_type = get_content_type(
            request.response_format,
            request.sample_rate
        )
        filename = f"audio.{request.response_format}"

        # 返回流式响应
        return StreamingResponse(
            content=await generate_audio_content(request),
            media_type=content_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as e:
        raise HTTPException(500, detail=str(e))

@app.post("/v1/uploads/audio/voice", response_model=VoiceUploadResponse)
async def upload_voice(
    customName: str = Form(...),
    text: str = Form(...),
    file: UploadFile = File(...)
):
    """## 增加用户自定义音色"""
    try:
        audio_data = await file.read()
        uri = save_voice_data(customName, audio_data, text)
        return VoiceUploadResponse(uri=uri)
    except ValidationError as ve:
        raise HTTPException(422, detail=ve.errors())
    except Exception as e:
        logging.error(f"上传失败: {str(e)}")
        raise HTTPException(500, detail=str(e))

def main(args):
    global cosyvoice
    cosyvoice = AsyncCosyVoice2(args.model_dir, load_jit=args.load_jit, load_trt=args.load_trt, fp16=args.fp16)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8022)
    parser.add_argument('--model_dir', type=str,
                        default='../../../pretrained_models/CosyVoice2-0.5B',
                        help='local path or modelscope repo id')
    parser.add_argument('--load_jit', action='store_true', help='load jit model')
    parser.add_argument('--load_trt', action='store_true', help='load tensorrt model')
    parser.add_argument('--fp16', action='store_true', help='use fp16')
    args = parser.parse_args()
    main(args)

    # python server.py --load_jit --load_trt --fp16
