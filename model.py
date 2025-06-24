# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import logging
import os
import queue
import threading
from typing import AsyncGenerator, Generator, List, Union
import torch
import numpy as np
import time
from torch.nn import functional as F
from contextlib import nullcontext
import uuid
from concurrent.futures import ThreadPoolExecutor

from cosyvoice.flow.flow import CausalMaskedDiffWithXvec
from cosyvoice.flow.flow_matching import EstimatorWrapper
from cosyvoice.hifigan.generator import HiFTGenerator
from cosyvoice.utils.common import fade_in_out
from cosyvoice.utils.file_utils import convert_onnx_to_trt

# 启用vllm V1版本
os.environ["VLLM_USE_V1"] = '1'
from vllm import  AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams

from vllm import ModelRegistry
from async_cosyvoice.config import ENGINE_ARGS, SAMPLING_PARAMS, ESTIMATOR_COUNT

from async_cosyvoice.vllm_use_cosyvoice2_model import CosyVoice2Model as CosyVoice2LLM
ModelRegistry.register_model("CosyVoice2Model", CosyVoice2LLM)

class AsyncWrapper:
    """将一个同步生成器包装为异步生成器"""
    def __init__(self, obj):
        self.obj = obj

    async def __aiter__(self):
        for item in self.obj:
            yield item

def tensor_to_list(tensor: torch.tensor):
    return tensor.view(-1).cpu().numpy().tolist()


class CosyVoice2Model:

    def __init__(self,
         model_dir: str,
         flow: CausalMaskedDiffWithXvec | torch.nn.Module,
         hift: HiFTGenerator | torch.nn.Module,
         fp16: bool,
         mix_ratio: List[int] = None,
         thread_count: int = 4, # token2wav threads
         peer_chunk_token_num: int = 60, # 流式请求时，初始的每个chunk处理语音token的数量。越小则首字节延迟越低，但性能越差。
         estimator_count: int = ESTIMATOR_COUNT, # flow 的 estimator 的数量，默认为 config.py 中的ESTIMATOR_COUNT
    ):
        # vllm engine 的参数配置
        engine_args = AsyncEngineArgs(
            model=model_dir,
            **ENGINE_ARGS,
        )
        self.llm_engine: AsyncLLMEngine = AsyncLLMEngine.from_engine_args(engine_args)
        self.thread_count = thread_count
        self.peer_chunk_token_num = peer_chunk_token_num
        self.estimator_count = estimator_count
        self.thread_executor = ThreadPoolExecutor(max_workers=self.thread_count)
        self.thread_local = threading.local()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.stream_pool = queue.Queue()
        for _ in range(self.thread_count):
            stream = torch.cuda.Stream(self.device)
            self.stream_pool.put(stream)

        self.flow = flow
        self.hift = hift
        self.fp16 = fp16
        self.flow.fp16 = fp16
        if self.fp16 is True:
            self.flow.half()
        self.token_hop_len = 2 * self.flow.input_frame_rate
        # here we fix flow encoder/decoder decoding_chunk_size, in the future we will send it as arguments, or use cache
        self.flow.encoder.static_chunk_size = 2 * self.flow.input_frame_rate
        self.flow.decoder.estimator.static_chunk_size = 2 * self.flow.input_frame_rate * self.flow.token_mel_ratio
        # hift cache
        self.mel_cache_len = 8
        self.source_cache_len = int(self.mel_cache_len * 480)
        # speech fade in out
        self.speech_window = np.hamming(2 * self.source_cache_len)
        # rtf and decoding related
        self.stream_scale_factor = 1
        self.llm_context = torch.cuda.stream(torch.cuda.Stream(self.device)) if torch.cuda.is_available() else nullcontext()

        self.mix_ratio = mix_ratio or [5, 15]

        self.lock = asyncio.Lock()  # 改为异步锁

        # dict used to store session related variable
        self.tts_speech_token_dict = {}
        self.llm_end_dict = {}
        self.hift_cache_dict = {}

        # 与vllm中的模型保持一致
        self.speech_token_size = 6564
        self.llm_token_size = 151936
        self.sos_eos_token_id = self.speech_token_size + self.llm_token_size + 1
        self.task_token_id = self.sos_eos_token_id + 1
        self.zero_token_id = self.task_token_id + 1

        # vllm 的推理任务需要在一个固定的事件循环中，因此启动一个后台线程专用于推理任务
        self.background_loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self.loop_thread.start()

    def get_stream_for_thread(self):
        """获取当前线程绑定的 stream"""
        if not hasattr(self.thread_local, 'stream'):
            self.thread_local.stream = self.stream_pool.get()
        return self.thread_local.stream

    def _run_event_loop(self):
        asyncio.set_event_loop(self.background_loop)
        self.background_loop.run_forever()

    def load(self, flow_model, hift_model):
        self.flow.load_state_dict(torch.load(flow_model, weights_only=True, map_location=self.device), strict=True)
        self.flow.to(self.device).eval()
        # in case hift_model is a hifigan model
        hift_state_dict = {k.replace('generator.', ''): v for k, v in torch.load(hift_model, weights_only=True, map_location=self.device).items()}
        self.hift.load_state_dict(hift_state_dict, strict=True)
        self.hift.to(self.device).eval()

    def load_jit(self, flow_encoder_model):
        flow_encoder = torch.jit.load(flow_encoder_model, map_location=self.device)
        self.flow.encoder = flow_encoder

    def load_trt(self, flow_decoder_estimator_model, flow_decoder_onnx_model, fp16):
        assert torch.cuda.is_available(), 'tensorrt only supports gpu!'
        if not os.path.exists(flow_decoder_estimator_model):
            convert_onnx_to_trt(flow_decoder_estimator_model, flow_decoder_onnx_model, fp16)
        if os.path.getsize(flow_decoder_estimator_model) == 0:
            raise ValueError('{} is empty file, delete it and export again!'.format(flow_decoder_estimator_model))
        del self.flow.decoder.estimator
        import tensorrt as trt
        with open(flow_decoder_estimator_model, 'rb') as f:
            self.flow.decoder.estimator_engine = trt.Runtime(trt.Logger(trt.Logger.INFO)).deserialize_cuda_engine(f.read())
        if self.flow.decoder.estimator_engine is None:
            raise ValueError('failed to load trt {}'.format(flow_decoder_estimator_model))
        self.flow.decoder.estimator = EstimatorWrapper(self.flow.decoder.estimator_engine, estimator_count=self.estimator_count)

    async def background_llm_inference(self, out_queue, prompt_token_ids, request_id, stop_token_ids, max_tokens):
        sampling_params = SamplingParams(**SAMPLING_PARAMS)
        sampling_params.stop_token_ids = stop_token_ids or [6561, 6563]
        if max_tokens:
            sampling_params.max_tokens = max_tokens
        async for output in self.llm_engine.generate(
                {
                    "prompt_token_ids": prompt_token_ids,
                },
                sampling_params=sampling_params,
                request_id=request_id or f"{time.time()}",
        ):
            out_queue.put((output.outputs[0], output.finished))

    async def llm_inference(self, prompt_token_ids: List[int], request_id: str=None, stop_token_ids=None, max_tokens=None):
        out_queue = queue.Queue()
        asyncio.run_coroutine_threadsafe(
            self.background_llm_inference(out_queue, prompt_token_ids, request_id, stop_token_ids, max_tokens), self.background_loop
        )
        # 接收 out_queue 返回的结果
        finished = False
        while not finished:
            if not out_queue.empty():
                (output, finished) = out_queue.get_nowait()
                yield output
            else:
                # 主动等待5ms
                await asyncio.sleep(0.005)

    async def llm_job(self, text, prompt_text, llm_prompt_speech_token, llm_embedding, uuid):
        prompt_text = tensor_to_list(prompt_text + torch.tensor(6564))
        llm_prompt_speech_token = tensor_to_list(llm_prompt_speech_token)

        start_time = time.time()
        if isinstance(text, Union[Generator,AsyncGenerator]):
            if isinstance(text, Generator):
                text = AsyncWrapper(text)
            last_tokens = []
            prompt_token_ids = [self.sos_eos_token_id]
            text_tokens_cache = prompt_text
            async for this_text in text:
                this_text = tensor_to_list(this_text + torch.tensor(6564))
                # text need tokens
                text_tokens_cache += this_text
                while len(llm_prompt_speech_token) != 0:
                    if len(text_tokens_cache) >= self.mix_ratio[0]:
                        text_input_token = text_tokens_cache[:self.mix_ratio[0]]
                        speech_input_token = llm_prompt_speech_token[:self.mix_ratio[1]]
                        prompt_token_ids += text_input_token + speech_input_token
                        # reset the last cache
                        text_tokens_cache = text_tokens_cache[self.mix_ratio[0]:]
                        llm_prompt_speech_token = llm_prompt_speech_token[self.mix_ratio[1]:]
                    else:
                        break
                if len(llm_prompt_speech_token) == 0:
                    if (len(last_tokens) > 0 and last_tokens[-1] == 6563) or len(prompt_token_ids) == 1:
                        if len(text_tokens_cache) >= self.mix_ratio[0]:
                            text_tokens_temp = text_tokens_cache[:self.mix_ratio[0]]
                            prompt_token_ids += text_tokens_temp
                            text_tokens_cache = text_tokens_cache[self.mix_ratio[0]:]
                        else:
                            continue
                    async for output in self.llm_inference(prompt_token_ids, request_id=uuid, stop_token_ids=[6563]):
                        last_tokens = output.token_ids
                        if last_tokens[-1] >= 6561:
                            need_add_tokens = last_tokens[:-1]
                        else:
                            need_add_tokens = last_tokens
                        self.tts_speech_token_dict[uuid].extend(need_add_tokens)
                        prompt_token_ids.extend(need_add_tokens)

            prompt_token_ids += text_tokens_cache + [self.task_token_id]
            async for output in self.llm_inference(prompt_token_ids, request_id=uuid, stop_token_ids=[6561, 6563]):
                if output.token_ids[-1] >= 6561:
                    need_add_tokens = last_tokens[:-1]
                else:
                    need_add_tokens = output.token_ids
                self.tts_speech_token_dict[uuid].extend(need_add_tokens)
        else:
            text = tensor_to_list(text + torch.tensor(6564))
            prompt_token_ids = [self.sos_eos_token_id] + prompt_text + text + \
                               [self.task_token_id] + llm_prompt_speech_token
            max_tokens = len(text) * 20
            async for output in self.llm_inference(
                    prompt_token_ids,
                    request_id=uuid,
                    stop_token_ids=[6561, 6563],
                    max_tokens=max_tokens,
            ):
                if output.token_ids[-1] >= 6561:
                    need_add_tokens = output.token_ids[:-1]
                else:
                    need_add_tokens = output.token_ids
                self.tts_speech_token_dict[uuid].extend(need_add_tokens)

        self.llm_end_dict[uuid] = True
        logging.info(f'llm job done, generated {len(self.tts_speech_token_dict[uuid]):>4} tokens, time cost: {time.time() - start_time:.3f}s')
        logging.debug(f'speech_tokens: len: {len(self.tts_speech_token_dict[uuid])}  data: {self.tts_speech_token_dict[uuid]}')
        # 记录 prompt_token_ids self.tts_speech_token_dict[uuid] 数据用于后续的训练，与flow推理测试


    def token2wav(self, token, prompt_token, prompt_feat, embedding, uuid, token_offset, finalize=False, speed=1.0):
        """在线程池中执行，需要特别注意并发安全问题"""
        # torch.cuda.current_stream().synchronize() # 将当前流进行同步了再处理后续逻辑（正常来说不需要，因为每次都有回收）
        stream = self.get_stream_for_thread() # 获取线程绑定的stream，每个线程的stream不变。
        try:
            with torch.cuda.stream(stream):
                tts_mel, _ = self.flow.inference(token=token.to(self.device),
                                                token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(self.device),
                                                prompt_token=prompt_token.to(self.device),
                                                prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32).to(self.device),
                                                prompt_feat=prompt_feat.to(self.device),
                                                prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(self.device),
                                                embedding=embedding.to(self.device),
                                                finalize=finalize)
                tts_mel = tts_mel[:, :, token_offset * self.flow.token_mel_ratio:]
                # append hift cache，为避免冲突 hift_cache_dit 只保存 cpu 上的元素。
                if self.hift_cache_dict[uuid] is not None:
                    hift_cache_mel, hift_cache_source = self.hift_cache_dict[uuid]['mel'].to(self.device), self.hift_cache_dict[uuid]['source'].to(self.device)
                    tts_mel = torch.concat([hift_cache_mel, tts_mel], dim=2)
                else:
                    hift_cache_source = torch.zeros(1, 1, 0).to(self.device)
                # keep overlap mel and hift cache
                if finalize is False:
                    tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)
                    if self.hift_cache_dict[uuid] is not None:
                        tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'].to(self.device), self.speech_window)
                    self.hift_cache_dict[uuid] = {'mel': tts_mel[:, :, -self.mel_cache_len:].to('cpu'),
                                                'source': tts_source[:, :, -self.source_cache_len:].to('cpu'),
                                                'speech': tts_speech[:, -self.source_cache_len:].to('cpu')}
                    tts_speech = tts_speech[:, :-self.source_cache_len]
                else:
                    if speed != 1.0:
                        assert self.hift_cache_dict[uuid] is None, 'speed change only support non-stream inference mode'
                        tts_mel = F.interpolate(tts_mel, size=int(tts_mel.shape[2] / speed), mode='linear')
                    tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)
                    if self.hift_cache_dict[uuid] is not None:
                        tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'].to(self.device), self.speech_window)
                stream.synchronize()
                return tts_speech
        except Exception as e:
            if "an illegal memory access was encountered" in str(e):
                logging.error(f"CUDA error: {e}")
                import sys
                sys.exit(1)
            raise e

    async def async_tts(self, text, flow_embedding, llm_embedding=torch.zeros(0, 192),
            prompt_text=torch.zeros(1, 0, dtype=torch.int32),
            llm_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            flow_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            prompt_speech_feat=torch.zeros(1, 0, 80), stream=False, speed=1.0, **kwargs):
        
        prompt_text = prompt_text.cpu()
        llm_prompt_speech_token = llm_prompt_speech_token.cpu()
        flow_prompt_speech_token = flow_prompt_speech_token.cpu()
        prompt_speech_feat = prompt_speech_feat.cpu()
        # this_uuid is used to track variables related to this inference thread
        this_uuid = str(uuid.uuid1())
        async with self.lock:
            self.tts_speech_token_dict[this_uuid] = []
            self.llm_end_dict[this_uuid] = False
            self.hift_cache_dict[this_uuid] = None
        # queue: asyncio.Queue[int|None] = asyncio.Queue()
        llm_task = asyncio.create_task(self.llm_job(text, prompt_text, llm_prompt_speech_token, llm_embedding, this_uuid))
        if stream is True:
            token_offset = 0
            peer_chunk_token_num = self.peer_chunk_token_num
            await asyncio.sleep(0.05)
            loop = asyncio.get_event_loop()
            start_time = time.time()
            chunk_index = 0
            while True:
                if (pending_num:= len(self.tts_speech_token_dict[this_uuid]) - token_offset) >= (peer_chunk_token_num + self.flow.pre_lookahead_len):
                    this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid][:token_offset + peer_chunk_token_num + self.flow.pre_lookahead_len]).unsqueeze(dim=0)
                    this_tts_speech = await loop.run_in_executor(self.thread_executor,
                        self.token2wav,
                        this_tts_speech_token,
                            flow_prompt_speech_token,
                            prompt_speech_feat,
                            flow_embedding,
                            this_uuid,
                            token_offset,
                            False
                    )
                    token_offset += peer_chunk_token_num
                    yield {'tts_speech': this_tts_speech.cpu()}
                    chunk_index += 1
                    cost_time = time.time() - start_time
                    # 动态增大 peer_chunk_token_num，以减少调用 token2wav 的次数
                    duration = token_offset/self.flow.input_frame_rate
                    if (multiples:= (duration - cost_time)/(cost_time/chunk_index)) > 4:
                        if self.llm_end_dict[this_uuid] is True:   # 直接一次性推理 token2wav 返回剩余的语音
                            break
                        else:
                            # 有较多的计算时间，还可以等待llm生成更多的token，用于下次 token2wav 推理
                            peer_chunk_token_num = (pending_num // 15 + 1) * 15
                    elif multiples > 2:
                        peer_chunk_token_num = (pending_num // 15) * 15
                    logging.debug(f'the multiples: {multiples:.2f}, next chunk_token_num: {peer_chunk_token_num}')
                else:
                    if self.llm_end_dict[this_uuid] is True:
                        break
                    else:
                        await asyncio.sleep(0.02)
            await llm_task
            # deal with remain tokens, make sure inference remain token len equals token_hop_len when cache_speech is not None
            this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
            this_tts_speech = await loop.run_in_executor(self.thread_executor,
                                                         self.token2wav,
                                                         this_tts_speech_token,
                                                         flow_prompt_speech_token,
                                                         prompt_speech_feat,
                                                         flow_embedding,
                                                         this_uuid,
                                                         token_offset,
                                                         True,
                                                         )
            if this_tts_speech.shape[1] == 0:
                logging.debug(f'no tts_speech_token shape: {this_tts_speech.shape}, data: {this_tts_speech}')
                return
            yield {'tts_speech': this_tts_speech.cpu()}
        else:
            # deal with all tokens
            await llm_task
            this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
            loop = asyncio.get_event_loop()
            this_tts_speech = await loop.run_in_executor(self.thread_executor,
                                                         self.token2wav,
                                                         this_tts_speech_token,
                                                         flow_prompt_speech_token,
                                                         prompt_speech_feat,
                                                         flow_embedding,
                                                         this_uuid,
                                                         0,
                                                         True,
                                                         speed,
                                                         )
            yield {'tts_speech': this_tts_speech.cpu()}
        async with self.lock:
            self.tts_speech_token_dict.pop(this_uuid)
            self.llm_end_dict.pop(this_uuid)
            self.hift_cache_dict.pop(this_uuid)
