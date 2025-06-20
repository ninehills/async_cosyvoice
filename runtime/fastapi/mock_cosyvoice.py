import os
import time
import logging
import asyncio
import torch
import torchaudio
from typing import AsyncGenerator, Dict, Any, Callable


class MockFrontend:
    def __init__(self, binary: bytes):
        self.binary = binary
    def generate_spk_info(self, *args, **kwargs) -> bytes:
        return self.binary
    def delete_spk_info(self, *args, **kwargs):
        return

class MockAsyncCosyVoice:
    """
    CosyVoice的Mock实现，用于测试和开发环境
    提供与原始CosyVoice相同的接口，但返回预定义的音频数据
    """
    def __init__(self, *args, **kwargs):
        self.sample_rate = 24000  # 采样率
        self.mock_audio = torch.zeros((5, self.sample_rate))  # 默认5秒静音
        self.spk_info_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mock_spk_info")
        
        # 确保存储目录存在
        os.makedirs(self.spk_info_dir, exist_ok=True)
        
        # 尝试加载 sample.wav 文件
        sample_wav_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample.wav")
        if os.path.exists(sample_wav_path):
            try:
                self.mock_audio, loaded_sample_rate = torchaudio.load(sample_wav_path)
                # 如果采样率不匹配，进行重采样
                if loaded_sample_rate != self.sample_rate:
                    resampler = torchaudio.transforms.Resample(
                        orig_freq=loaded_sample_rate,
                        new_freq=self.sample_rate
                    )
                    self.mock_audio = resampler(self.mock_audio)
                # 确保音频是单声道
                if self.mock_audio.shape[0] > 1:
                    self.mock_audio = self.mock_audio.mean(dim=0, keepdim=True)
                logging.info(f"Loaded sample.wav with shape {self.mock_audio.shape}")
            except Exception as e:
                logging.error(f"Error loading sample.wav: {e}")
                raise Exception(f"Error loading sample.wav: {e}")
        else:
            logging.warning(f"sample.wav not found at {sample_wav_path}, using empty audio")
        
        with open(sample_wav_path, "rb") as f:
            self.frontend = MockFrontend(binary=f.read())
        logging.info(f"MockCosyVoice initialized with sample rate {self.sample_rate}")
    
    def generate_spk_info(self, uri: str, text: str, prompt_speech: torch.Tensor, 
                         sample_rate: int, custom_name: str) -> None:
        """
        保存一个假的说话人信息文件到指定目录
        
        Args:
            uri: 说话人URI
            text: 提示文本
            prompt_speech: 提示音频
            sample_rate: 采样率
            custom_name: 自定义名称
        """
        # 创建一个简单的文本文件来模拟说话人信息
        spk_info_path = os.path.join(self.spk_info_dir, f"{custom_name}.txt")
        with open(spk_info_path, "w") as f:
            f.write(f"URI: {uri}\n")
            f.write(f"Text: {text}\n")
            f.write(f"Sample Rate: {sample_rate}\n")
            f.write(f"Custom Name: {custom_name}\n")
            f.write(f"Timestamp: {time.time()}\n")
        
        # 保存一个示例音频文件
        audio_path = os.path.join(self.spk_info_dir, f"{custom_name}.wav")
        torchaudio.save(audio_path, prompt_speech, sample_rate)
        
        logging.info(f"Mock speaker info saved for {custom_name} at {spk_info_path}")
        return
    
    async def _process_audio(self, stream: bool = False) -> AsyncGenerator[Dict[str, Any], None]:
        """
        处理音频的共享方法，根据是否流式返回不同的结果
        
        Args:
            stream: 是否流式输出
            
        Returns:
            包含音频数据的异步生成器
        """
        if stream:
            # 流式模式：分块返回音频，总共5秒左右完成
            # 计算每个块的大小，假设我们想要10个块，每个块间隔0.5秒
            total_chunks = 10
            chunk_size = self.mock_audio.shape[1] // total_chunks
            
            for i in range(total_chunks):
                # 创建音频块
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, self.mock_audio.shape[1])
                audio_chunk = self.mock_audio[:, start_idx:end_idx]
                await asyncio.sleep(0.5)  # 每块间隔0.5秒，总共5秒
                yield {"tts_speech": audio_chunk}
        else:
            # 非流式模式：等待3秒后一次性返回完整音频
            await asyncio.sleep(3.0)  # 模拟处理延迟3秒
            yield {"tts_speech": self.mock_audio}
    
    async def inference_instruct2_by_spk_id(self, tts_text: str, instruct_text: str, 
                                           spk_id: str, stream: bool = False, 
                                           speed: float = 1.0, text_frontend: bool = True) -> AsyncGenerator[Dict[str, Any], None]:
        """
        使用指令和说话人ID执行推理的Mock实现
        
        Args:
            tts_text: 要合成的文本
            instruct_text: 指令文本
            spk_id: 说话人ID
            stream: 是否流式输出
            speed: 语速
            text_frontend: 是否使用文本前端
            
        Returns:
            包含音频数据的异步生成器
        """
        logging.info(f"Mock inference_instruct2_by_spk_id called with text: {tts_text[:30]}...")
        logging.info(f"Instruct: {instruct_text[:30]}..., Speaker ID: {spk_id}")
        
        # 使用共享方法处理音频
        async for result in self._process_audio(stream):
            yield result
    
    async def inference_zero_shot_by_spk_id(self, tts_text: str, spk_id: str, 
                                          stream: bool = False, speed: float = 1.0, 
                                          text_frontend: bool = True) -> AsyncGenerator[Dict[str, Any], None]:
        """
        使用说话人ID执行zero-shot推理的Mock实现
        
        Args:
            tts_text: 要合成的文本
            spk_id: 说话人ID
            stream: 是否流式输出
            speed: 语速
            text_frontend: 是否使用文本前端
            
        Returns:
            包含音频数据的异步生成器
        """
        logging.info(f"Mock inference_zero_shot_by_spk_id called with text: {tts_text[:30]}...")
        logging.info(f"Speaker ID: {spk_id}")
        
        # 使用共享方法处理音频
        async for result in self._process_audio(stream):
            yield result
    