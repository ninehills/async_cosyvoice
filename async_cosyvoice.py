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
import os
import time
import yaml
from typing import Generator, Union, AsyncGenerator, Callable
os.environ["VLLM_USE_V1"] = '1'

import torch
from async_cosyvoice.frontend import CosyVoiceFrontEnd
from async_cosyvoice.model import CosyVoice2Model
from cosyvoice.utils.file_utils import logging
from hyperpyyaml import load_hyperpyyaml
from modelscope import snapshot_download
from tqdm import tqdm


class AsyncCosyVoice2:
    def __init__(self, model_dir, load_jit=False, load_trt=False, fp16=False, cache_dir='./cache',
        model_configs: dict = None,
    ):
        self.instruct = True if '-Instruct' in model_dir else False
        self.model_dir = model_dir
        self.fp16 = fp16
        if not os.path.exists(model_dir):
            model_dir = snapshot_download(model_dir)
        with open('{}/cosyvoice2.yaml'.format(model_dir), 'r') as f:
            configs = load_hyperpyyaml(f, overrides={'model_path': model_dir})
        self.frontend = CosyVoiceFrontEnd(configs['get_tokenizer'],
                                          configs['feat_extractor'],
                                          '{}/campplus.onnx'.format(model_dir),
                                          '{}/speech_tokenizer_v2.onnx'.format(model_dir),
                                          '{}/spk2info.pt'.format(model_dir),
                                          configs['allowed_special'],
                                          cache_dir)
        self.sample_rate = configs['sample_rate']
        if torch.cuda.is_available() is False and (load_jit is True or load_trt is True or fp16 is True):
            load_jit, load_trt, fp16 = False, False, False
            logging.warning('no cuda device, set load_jit/load_trt/fp16 to False')
        model_configs = model_configs or {}
        model_configs["load_jit"] = load_jit
        model_configs["load_trt"] = load_trt
        self.model = CosyVoice2Model(
            model_dir,
            '{}/cosyvoice2.yaml'.format(model_dir),
            fp16,
            **model_configs,
        )
        del configs

    def list_available_spks(self):
        spks = list(self.frontend.spk2info.keys())
        return spks

    async def add_spk_info(self, spk_id, spk_info):
        self.frontend.add_spk_info(spk_id, spk_info)

    async def inference_sft(self, tts_text, spk_id, stream=False, speed=1.0, text_frontend=True):
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            model_input = self.frontend.frontend_sft(i, spk_id)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            async for model_output in self.model.async_tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}, time cost: {}'.format(speech_len, (time.time() - start_time) / speech_len, (time.time() - start_time)))
                yield model_output
                start_time = time.time()

    async def inference_cross_lingual(self, tts_text, prompt_speech_16k, stream=False, speed=1.0, text_frontend=True):
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            model_input = self.frontend.frontend_cross_lingual(i, prompt_speech_16k, self.sample_rate)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            async for model_output in self.model.async_tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

    async def inference_zero_shot(self, tts_text, prompt_text, prompt_speech_16k, stream=False, speed=1.0, text_frontend=True):
        prompt_text = self.frontend.text_normalize(prompt_text, split=False, text_frontend=text_frontend)
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            if (not isinstance(i, Union[Generator, AsyncGenerator])) and len(i) < 0.5 * len(prompt_text):
                logging.warning('synthesis text {} too short than prompt text {}, this may lead to bad performance'.format(i, prompt_text))
            model_input = self.frontend.frontend_zero_shot(i, prompt_text, prompt_speech_16k, self.sample_rate)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            async for model_output in self.model.async_tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

    async def inference_instruct2(self, tts_text, instruct_text, prompt_speech_16k, stream=False, speed=1.0, text_frontend=True):
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            model_input = self.frontend.frontend_instruct2(i, instruct_text, prompt_speech_16k, self.sample_rate)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            async for model_output in self.model.async_tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

    async def inference_instruct2_by_spk_id(self, tts_text, instruct_text, spk_id, stream=False, speed=1.0, text_frontend=True):
        for i in self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend):
            model_input = self.frontend.frontend_instruct2_by_spk_id(i, instruct_text, spk_id)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            async for model_output in self.model.async_tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

    async def inference_zero_shot_by_spk_id(self, tts_text, spk_id, stream=False, speed=1.0, text_frontend=True):
        """使用预定义的说话人执行 zero_shot 推理"""
        infos = {
            "tts_text": tts_text,
            "spk_id": spk_id,
        }
        s_time = time.time()
        for order, i in enumerate(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            model_input = self.frontend.frontend_zero_shot_by_spk_id(i, spk_id)
            start_time = time.time()
            last_time = start_time
            chunk_index = 0
            logging.info('synthesis text {}'.format(i))
            infos[f"s{order}_text_len"] = len(i)
            async for model_output in self.model.async_tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                prefix = f"s{order}_c{chunk_index}"
                infos[f"{prefix}_speech_len"] = speech_len
                infos[f"{prefix}_rtf"] = (time.time()-last_time)/speech_len
                infos[f"{prefix}_cost"] = time.time()-last_time
                infos[f"s{order}_cost"] = time.time()-start_time
                logging.info('yield speech index:{}, len {:.2f}, rtf {:.3f},  cost {:.3f}s,  all cost time {:.3f}s'.format(
                    chunk_index, speech_len,  (time.time()-last_time)/speech_len, time.time()-last_time, time.time()-start_time))
                yield model_output
                last_time = time.time()
                chunk_index += 1
            infos["total_cost"] = time.time()-s_time
            info_kvs = sorted([f"{i}:{infos[i]}" for i in infos])
            logging.info(f"trace: {'|'.join(info_kvs)}")
