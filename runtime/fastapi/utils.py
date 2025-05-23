import struct
from io import BytesIO
from typing import AsyncGenerator, Any, AsyncIterator

import lameenc
import asyncio
import time
import logging
import numpy as np
import torch
import torchaudio
from starlette.concurrency import run_in_threadpool

def load_wav(wav, target_sr):
    speech, sample_rate = torchaudio.load(wav, backend='soundfile')
    speech = speech.mean(dim=0, keepdim=True)
    if sample_rate != target_sr:
        assert sample_rate > target_sr, 'wav sample rate {} must be greater than {}'.format(sample_rate, target_sr)
        speech = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)(speech)
    return speech

def load_audio_from_bytes(audio_data, target_sr):
    # 将字节数据包装成文件对象
    buffer = BytesIO(audio_data)
    # 使用soundfile后端加载音频
    speech, sample_rate = torchaudio.load(buffer, backend='soundfile')
    # 多声道转单声道（取均值）
    speech = speech.mean(dim=0, keepdim=True)

    # 检查并调整采样率
    if sample_rate != target_sr:
        if sample_rate < target_sr:
            raise ValueError(f"原始采样率 {sample_rate}Hz 必须不低于目标采样率 {target_sr}Hz")
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=target_sr
        )
        speech = resampler(speech)
    return speech

class AsyncWrapper:
    def __init__(self, obj):
        self.obj = obj

    async def __aiter__(self):
        for item in self.obj:
            yield item

def _tensor_to_bytes(tensor: torch.Tensor, format: str = None, sample_rate=24000):
    # 统一Tensor形状为 (channels, samples)
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    elif tensor.dim() == 2:
        if tensor.size(0) > tensor.size(1):  # 假设输入为 (samples, channels)
            tensor = tensor.permute(1, 0)
    else:
        raise ValueError("Invalid tensor shape")

    match format:
        case "mp3":
            return _encode_mp3(tensor, sample_rate)
        case "wav":
            # bits_per_sample: PCM/WAV的量化位数（16或32）
            return _encode_wav(tensor, sample_rate, 16)
        case "pcm":
            return _encode_pcm(tensor, 16)
        case None:
            return tensor.numpy().astype(np.float32).tobytes()
        case _:
            raise ValueError(f"Unsupported format: {format}")

def convert_audio_bytes_to_tensor(raw_audio: bytes) -> torch.Tensor:
    """同步音频转换方法"""
    return torch.from_numpy(np.array(np.frombuffer(raw_audio, dtype=np.float32))).unsqueeze(0)


async def convert_audio_tensor_to_bytes(
        tensor_generator: torch.Tensor| AsyncGenerator[torch.Tensor, Any],
        format: str = None, sample_rate=24000,
        stream=False) -> AsyncGenerator[bytes, Any]:
    """将音频Tensor转换为指定格式的字节流

    Args:
        tensor_generator: 输入音频Tensor，形状需为 (channels, samples) 或 (samples,)
        format: 目标格式，支持 'wav', 'pcm', 'mp3'
        sample_rate: 采样率（默认16000）
        stream: 是否以流式方式返回音频数据

    Returns:
        bytes: 编码后的音频字节流
        AsyncGenerator[bytes]: 流式返回音频数据
    """
    if isinstance(tensor_generator, torch.Tensor):
        tensor_generator = AsyncWrapper([tensor_generator])
    if not stream:
        tensor: torch.Tensor|None = None
        async for chunk in tensor_generator:
            if tensor is not None:
                tensor = torch.concat([tensor, chunk], dim=1)
            else:
                tensor = chunk

        yield await run_in_threadpool(_tensor_to_bytes, tensor, format, sample_rate)
    else:
        channels, bit_depth = 1, 16
        bitrate = 128
        if format == 'mp3':
            async for data in _encode_mp3_stream(tensor_generator, sample_rate, channels, bitrate):
                yield data
        elif format == 'pcm':
            # 无头，直接输出原始PCM
            async for chunk in tensor_generator:
                yield _pcm_to_bytes(chunk, bit_depth)
        elif format == 'wav':
            # 生成WAV头
            yield _generate_wav_header(sample_rate, channels, bit_depth)
            # 直接输出PCM数据
            async for chunk in tensor_generator:
                yield _pcm_to_bytes(chunk, bit_depth)
        else:
            raise ValueError(f"不支持的格式：{format}")

def _encode_wav(tensor: torch.Tensor, sr: int, bits: int) -> bytes:
    """编码WAV格式"""
    if bits == 16:
        encoding = "PCM_S"
        bits_depth = 16
    elif bits == 32:
        encoding = "PCM_F"
        bits_depth = 32
    else:
        raise ValueError("Only 16/32-bit WAV supported")
    buffer = BytesIO()
    torchaudio.save(
        buffer,
        tensor,
        sr,
        format="wav",
        encoding=encoding,
        bits_per_sample=bits_depth,
    )
    buffer.seek(0)
    return buffer.getvalue()

def _encode_pcm(tensor: torch.Tensor, bits: int) -> bytes:
    """编码原始PCM数据"""
    assert tensor.dtype == torch.float32 or tensor.dtype == torch.float64, "输入张量应为浮点类型"
    np_array = tensor.cpu().numpy()
    np_array = np.clip(np_array, -1.0, 1.0)

    # 量化到目标位深
    if bits == 16:
        np_array = (np_array * 32767.0).astype(np.int16)
    elif bits == 32:
        np_array = (np_array * 2147483647.0).astype(np.int32)
    else:
        raise ValueError("Only 16/32-bit PCM supported")
    return np_array.tobytes()

def _encode_mp3(tensor: torch.Tensor, sr: int) -> bytes:
    """编码MP3格式"""
    buffer = BytesIO()

    # 注意：需要安装支持MP3编码的后端（如ffmpeg, libsox）
    torchaudio.save(
        buffer,
        tensor,
        sr,
        format="mp3",
        encoding="MP3",
    )
    buffer.seek(0)
    return buffer.getvalue()

def _generate_wav_header(sample_rate: int, channels: int = 1, bit_depth: int = 16) -> bytes:
    """生成适用于流式传输的WAV头，使用0xFFFFFFFF表示无限长度。

    Args:
        sample_rate (int): 采样率
        channels (int, optional): 通道数. Defaults to 1.
        bit_depth (int, optional): 位深（16或32位）。 Defaults to 16.
    """
    byte_rate = sample_rate * channels * (bit_depth // 8)
    block_align = channels * (bit_depth // 8)

    header = b'RIFF' + \
        struct.pack('<I', 0xFFFFFFFF) + \
        b'WAVE' + \
        b'fmt ' + \
        struct.pack('<IHHIIHH', 16, 1, channels, sample_rate, byte_rate, block_align, bit_depth) + \
        b'data' + \
        struct.pack('<I', 0xFFFFFFFF)  # 数据大小（未知）
    return header

def _pcm_to_bytes(pcm_data: torch.Tensor, bit_depth: int = 16) -> bytes:
    """将PCM tensor转换为字节"""
    if bit_depth not in (16, 32):
        raise ValueError(f"不支持的位深度：{bit_depth}")
    if bit_depth == 16:
        if pcm_data.dtype != torch.int16:
            pcm_data = pcm_data.to(torch.float32)
            pcm_data = (pcm_data * 32767.0).clamp(-32768, 32767).to(torch.int16)
        return pcm_data.numpy().tobytes()
    elif bit_depth == 32:
        if pcm_data.dtype != torch.float32:
            pcm_data = pcm_data.to(torch.float32) / 32768.0
        return pcm_data.numpy().tobytes()

async def _encode_mp3_stream(
        audio_chunks: AsyncIterator[torch.Tensor],
        sample_rate: int = 24000,
        channels: int = 1,
        bitrate: int = 128,
) -> AsyncGenerator[bytes, None]:
    """MP3编码实现"""
    encoder = lameenc.Encoder()
    encoder.set_channels(channels)
    encoder.set_bit_rate(bitrate * 1000)
    encoder.set_in_sample_rate(sample_rate)
    encoder.set_out_sample_rate(sample_rate)
    encoder.set_quality(9)
    encoder.set_vbr(4)
    encoder.set_vbr_quality(9)
    first_mp3 = True
    first_mp3_yield_time = None
    async for chunk in audio_chunks:
        pcm_bytes = _pcm_to_bytes(chunk, 16)  # MP3通常使用16位
        if len(pcm_bytes) > 0 and first_mp3:
            first_mp3_yield_time = time.perf_counter()
        mp3_data = encoder.encode(pcm_bytes)
        # mp3_data = await asyncio.to_thread(encoder.encode, pcm_bytes)
        if mp3_data:
            if first_mp3_yield_time is not None and first_mp3:
                elapsed_time = time.perf_counter() - first_mp3_yield_time
                first_mp3 = False
                logging.debug(f"mp3 first packet elapsed time: {elapsed_time:.10f} seconds")
            yield bytes(mp3_data)
    # 刷新编码器缓冲区
    final_data = encoder.flush()
    #final_data = await asyncio.to_thread(encoder.flush)
    if final_data:
        yield bytes(final_data)