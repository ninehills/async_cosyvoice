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
from pydantic_settings import BaseSettings, SettingsConfigDict
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Depends, status
from pydantic import RedisDsn
import redis.asyncio as redis

from utils import convert_audio_tensor_to_bytes, load_audio_from_bytes
from voice_storage import VoiceStorage
from model import VoiceMeta

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{ROOT_DIR}/../../..')
sys.path.append(f'{ROOT_DIR}/../../../third_party/Matcha-TTS')
CACHE_DIR = os.path.join(ROOT_DIR, 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)

log_level = logging.INFO
if os.getenv("MOCK_ENABLED", "0") == "1":
    from mock_cosyvoice import MockAsyncCosyVoice as AsyncCosyVoice2
    # Mock 时开启 Debug 日志
    log_level = logging.DEBUG
else:
    from async_cosyvoice.async_cosyvoice import AsyncCosyVoice2

logging.basicConfig(level=log_level,
                    format='%(asctime)s %(levelname)s %(message)s')

# 配置
class UserAuth(BaseModel):
    user_id: str
    access_key: str


class Settings(BaseSettings):
    # 预设用户和AccessKey，默认是 abc:abc
    preset_users: list[UserAuth] = [UserAuth(user_id="abc", access_key="abc")]
    # redis 地址
    redis_dsn: RedisDsn = 'redis://user:pass@localhost:6379/0'

    # 配置
    thread_count: int = 4, # token2wav threads
    peer_chunk_token_num: int = 60, # 流式请求时，初始的每个chunk处理语音token的数量。越小则首字节延迟越低，但性能越差。
    estimator_count: int = 4, # flow 的 estimator 的数量

    model_config = SettingsConfigDict(env_file=".env")


# 全局实例
cosyvoice: AsyncCosyVoice2 | None = None
settings = Settings()
# Redis
redis_client = redis.Redis.from_url(str(settings.redis_dsn))
# 音色存储
voice_storage = VoiceStorage(redis_client=redis_client)
# 鉴权
access_keys = {user.access_key: user.user_id for user in settings.preset_users}
security = HTTPBearer()

# 依赖项：获取当前用户ID
async def get_current_user_id(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
) -> str:
    access_key = credentials.credentials
    user_id = access_keys.get(access_key)
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid access key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user_id


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

class VoiceListResponse(BaseModel):
    """音频列表响应参数"""
    results: list[VoiceMeta] = Field(
        ...,
        description="音色列表"
    )


class VoiceDeletionRequest(BaseModel):
    """音频删除请求参数"""
    uri: str = Field(
        ...,
        examples=["speech:your-voice-name:xxx:xxx"],
        description="要删除的音色URI"
    )


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
    model: Optional[str] = Field(
        default="FunAudioLLM/CosyVoice2-0.5B",
        description="模型选择，目前支持FunAudioLLM/CosyVoice2-0.5B"
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

async def save_voice_data(customName: str, audio_data: bytes, text: str, user_id: str, model: str) -> str:
    """保存音频数据并生成音色对应的URI"""
    voice_id = str(uuid.uuid4())[:8]
    uri = f"speech:{user_id}:{customName}:{voice_id}"
    # TODO: 目前上传音频是同步阻塞的，会导致服务阻塞大约2-3s，此处应该使用异步方式。
    prompt_speech_16k = load_audio_from_bytes(audio_data, 16000)
    voice_data = cosyvoice.frontend.generate_spk_info(
        uri,
        text,
        prompt_speech_16k,
        24000,
        customName
    )
    voice_meta = VoiceMeta(
        model=model,
        customName=customName,
        text=text,
        uri=uri,
    )
    await voice_storage.save_voice(uri, voice_meta, voice_data)
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
async def text_to_speech(request: SpeechRequest, user_id: Annotated[str, Depends(get_current_user_id)]):
    """## 文本转语音接口"""
    # 判断 CacheDir 中是否有目标 pt 文件，没有则从 Redis 中下载（仅判断用speech开头的，不判断系统音色）
    if request.voice.startswith("speech:"):
        spk_info_path = os.path.join(CACHE_DIR, request.voice + '.pt')
        if not os.path.exists(spk_info_path):
            logging.info(f"voice {request.voice} not in cache dir {spk_info_path}, download it.")
            tmp_file = f"{spk_info_path}.{uuid.uuid4().hex}"
            try:
                spk_info = await voice_storage.get_voice(request.voice)
                if spk_info:
                    with open(tmp_file, 'wb') as f:
                        f.write(spk_info)
                    if not os.path.exists(spk_info_path):
                        logging.info(f"download {request.voice} spk info file, and move to {spk_info_path}.")
                        os.rename(tmp_file, spk_info_path)
                    else:
                        logging.info(f"spk info file {spk_info_path} already download by other thread, skip move it.")
            except KeyError as e:
                logging.warning(f"voice {request.voice} not found in redis, please upload it first: {e}")
            finally:
                if os.path.exists(tmp_file):
                    os.remove(tmp_file)
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
        logging.error("TTS failed: {e}", exc_info=True)
        raise HTTPException(500, detail=str(e))

@app.post("/v1/uploads/audio/voice", response_model=VoiceUploadResponse)
async def upload_voice(
    user_id: Annotated[str, Depends(get_current_user_id)],
    model: Optional[str] = Form(default="FunAudioLLM/CosyVoice2-0.5B"),
    customName: str = Form(..., regex="^[a-zA-Z0-9_-]{1,30}$", description="仅支持字母、数字、下划线、横线，最大长度30个字符"),
    text: str = Form(...),
    file: UploadFile = File(...),
):
    """增加用户自定义音色"""
    try:
        audio_data = await file.read()
        uri = await save_voice_data(customName, audio_data, text, user_id, model)
        return VoiceUploadResponse(uri=uri)
    except ValidationError as ve:
        logging.error(f"Upload voice validation error: {ve}")
        raise HTTPException(422, detail=ve.errors())
    except Exception as e:
        logging.error(f"上传失败: {str(e)}", exc_info=True)
        raise HTTPException(500, detail=str(e))


@app.get("/v1/audio/voice/list", response_model=VoiceListResponse)
async def list_voices(user_id: Annotated[str, Depends(get_current_user_id)]):
    """获取用户的参考音频列表"""
    try:
        voices = await voice_storage.list_voices(user_id)
        return VoiceListResponse(results=voices)
    except Exception as e:
        logging.error(f"获取音色列表失败: {str(e)}", exc_info=True)
        raise HTTPException(500, detail=str(e))

@app.post("/v1/audio/voice/deletions")
async def delete_voice(request: VoiceDeletionRequest, user_id: Annotated[str, Depends(get_current_user_id)]):
    """删除用户的参考音频"""
    try:
        # 验证URI是否属于当前用户
        uri_parts = request.uri.split(":")
        if len(uri_parts) < 3 or uri_parts[1] != user_id:
            raise HTTPException(403, detail="无权删除此音色")
        
        await voice_storage.delete_voice(request.uri)
        # 从frontend 缓存中删除音色
        cosyvoice.frontend.delete_spk_info(request.uri)
        return {}
    except HTTPException as he:
        raise he
    except Exception as e:
        logging.error(f"删除音色失败: {str(e)}", exc_info=True)
        raise HTTPException(500, detail=str(e))

@app.get("/auth/me")
async def read_current_user(user_id: Annotated[str, Depends(get_current_user_id)]):
    """返回当前用户信息
    
        Test: curl --header 'Authorization: Bearer 123456' http://127.0.0.1:8022/auth/me
    """
    return {"user_id": user_id}


def main(args):
    global cosyvoice
    logging.info("Init AsyncCosyVoice2 with args: %s, settings: %s", args, settings)
    cosyvoice = AsyncCosyVoice2(
        args.model_dir,
        load_jit=args.load_jit,
        load_trt=args.load_trt,
        fp16=args.fp16,
        cache_dir=CACHE_DIR,
        model_configs=dict(
            thread_count=settings.thread_count,
            peer_chunk_token_num=settings.peer_chunk_token_num,
            estimator_count=settings.estimator_count,
        ),
    )

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
