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
import re
import json
import time
import inflect
from functools import partial
from collections import OrderedDict
from typing import Generator, Optional, AsyncGenerator, Union, Callable

import onnxruntime
import torch
import numpy as np
import whisper
import torchaudio
import torchaudio.compliance.kaldi as kaldi

from pydantic import BaseModel, ConfigDict

from async_cosyvoice.config import OVERWRITE_NORMALIZER_CACHE
from async_cosyvoice.config import REGISTER_SPK_INFO_DICT

try:
    import ttsfrd
    use_ttsfrd = True
except ImportError:
    print("failed to import ttsfrd, use WeTextProcessing instead")
    from tn.chinese.normalizer import Normalizer as ZhNormalizer
    from tn.english.normalizer import Normalizer as EnNormalizer
    use_ttsfrd = False
from cosyvoice.utils.file_utils import logging
from cosyvoice.utils.frontend_utils import contains_chinese, replace_blank, replace_corner_mark, remove_bracket, spell_out_number, split_paragraph, is_only_punctuation


class AsyncTextGeneratorWrapper:
    def __init__(self, obj):
        self.obj = obj
        self.is_async_generator = isinstance(obj, AsyncGenerator)
        self.str_num = 0
        self.min_str_num = 60
        self.split_threshold = 60
        self.buffer = ""  # 用于存储未发送的文本
        self.finished = False
        self._len = 1

    def __len__(self):
        return self._len

    def _find_split_point(self, text, max_split_len=None):
        if max_split_len is None:
            max_split_len = self.split_threshold
        """在文本中找到合适的分割点"""
        split_chars = {'。', '！', '？', '.', '!', '?', '，', '、', '；', ';', ','}
        for i, t in enumerate(text):
            if t in split_chars:
                return i
            elif i >= max_split_len:
                # 如果超过最大分割长度，则返回当前位置，强制分隔
                return i
        return -1

    async def _async_generator(self):
        """异步生成器，处理文本分块"""
        self.str_num = 0
        # 发送缓冲区数据
        if self.buffer:
            if len(self.buffer) >= self.min_str_num:
                yield self.buffer[:self.min_str_num]
                self.str_num = self.min_str_num
                self.buffer = self.buffer[self.min_str_num:]
                # 在 buffer 中寻找合适的切分点
                split_pos = self._find_split_point(self.buffer)
                if split_pos >= 0:
                    yield self.buffer[:split_pos]
                    self.buffer = self.buffer[split_pos:]
                    return
                else:
                    yield self.buffer
                    self.str_num += len(self.buffer)
                    self.buffer = ""
            else:
                yield self.buffer
                self.str_num = len(self.buffer)
                self.buffer = ""
        chunk = ""
        while True:
            # 获取新数据
            next_chunk :str = ""
            try:
                if self.is_async_generator:
                    next_chunk = await self.obj.__anext__()
                else:
                    next_chunk = self.obj.__next__()
            except (StopIteration, StopAsyncIteration):
                self.finished = True
            except Exception as e:
                raise f"Error in AsyncTextGeneratorWrapper: {e}"

            if not self.finished:
                # 如果新数据长度小于最小长度，则直接返回
                if (num := self.str_num + len(chunk)) < self.min_str_num:
                    self.str_num = num
                    yield chunk
                # 确保切分后还没有结束传入新的字符，避免出现生成器只生成一个空字符的情况（合成错误音频）
                # 切分输入，并保存更新 buffer
                elif split_pos := self._find_split_point(chunk) >= 0:
                    yield chunk[:split_pos]
                    self.buffer = chunk[split_pos:] + next_chunk
                    return
                # 如果长度超过最小长度+阈值，则强制进行切分
                elif num > self.min_str_num + self.split_threshold:
                    index = self.min_str_num + self.split_threshold - self.str_num
                    yield chunk[:index]
                    self.buffer = chunk[index:] + next_chunk
                    return
                else:
                    self.str_num = num
                    yield chunk

                chunk = next_chunk
            else:
                # 如果已经结束，则直接返回剩余的 buffer
                yield self.buffer + chunk
                return

    def __iter__(self):
        """同步生成器"""
        while not self.finished:
            # 返回异步生成器
            yield self._async_generator()
            # self._len += 1

class SpeakerInfo(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: Optional[str] = None
    spk_id: str
    prompt_text: str
    prompt_text_token: torch.Tensor
    speech_feat: torch.Tensor
    speech_token: torch.Tensor
    embedding: torch.Tensor


class LRUCache(OrderedDict):
    """LRU缓存容器，继承自OrderedDict"""

    def __init__(self, max_size=100_000):
        super().__init__()
        self.max_size = max_size

    def __getitem__(self, key):
        # 访问时移动到末尾（表示最新）
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def __setitem__(self, key, value):
        # 插入时检查容量，超限则移除最旧项
        if key in self:
            self.move_to_end(key)
        else:
            if len(self) >= self.max_size:
                self.popitem(last=False)
        super().__setitem__(key, value)

class CosyVoiceFrontEnd:

    def __init__(self,
                 get_tokenizer: Callable,
                 feat_extractor: Callable,
                 campplus_model: str,
                 speech_tokenizer_model: str,
                 spk2info: str = '',
                 allowed_special: str = 'all',
                 spk2info_path: str = "./spk2info"):
        self.tokenizer = get_tokenizer()
        self.feat_extractor = feat_extractor
        self.device = 'cpu'
        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = 8
        #self.campplus_session = onnxruntime.InferenceSession(campplus_model, sess_options=option, providers=["CUDAExecutionProvider" if torch.cuda.is_available() else
        #                                                                        "CPUExecutionProvider"])
        self.campplus_session = onnxruntime.InferenceSession(campplus_model, sess_options=option, providers=["CPUExecutionProvider"])
        #self.speech_tokenizer_session = onnxruntime.InferenceSession(speech_tokenizer_model, sess_options=option,
        #                                                             providers=["CUDAExecutionProvider" if torch.cuda.is_available() else
        #                                                                        "CPUExecutionProvider"])
        self.speech_tokenizer_session = onnxruntime.InferenceSession(speech_tokenizer_model, sess_options=option, providers=["CPUExecutionProvider"])
        self.spk2info = LRUCache(max_size=10000)

        if os.path.exists(spk2info):
            spk_infos = torch.load(spk2info, map_location=self.device, weights_only=False)
            for spk_id, info in spk_infos.items():
                self.spk2info[spk_id] = info

        self.spk2info_path = spk2info_path
        os.makedirs(self.spk2info_path, exist_ok=True)
        self.allowed_special = allowed_special
        self.use_ttsfrd = use_ttsfrd
        if self.use_ttsfrd:
            self.frd = ttsfrd.TtsFrontendEngine()
            ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
            assert self.frd.initialize('{}/../pretrained_models/CosyVoice-ttsfrd/resource'.format(ROOT_DIR)) is True, \
                'failed to initialize ttsfrd resource'
            self.frd.set_lang_type('pinyinvg')
        else:
            # self.zh_tn_model = ZhNormalizer(remove_erhua=False, full_to_half=False, overwrite_cache=True)
            self.zh_tn_model = ZhNormalizer(remove_erhua=False, full_to_half=False, overwrite_cache=OVERWRITE_NORMALIZER_CACHE)
            self.en_tn_model = EnNormalizer()
            self.inflect_parser = inflect.engine()

        if REGISTER_SPK_INFO_DICT:
            for spk_id, info in REGISTER_SPK_INFO_DICT.items():
                try:
                    prompt_text = info['prompt_text']
                    prompt_audio_path = info['prompt_audio_path']
                    if not os.path.exists(prompt_audio_path):
                        logging.error(f"doesn't exist prompt_audio_path: {prompt_audio_path}")
                        continue
                    logging.info(f"Generate spk_info for {spk_id}....")
                    prompt_audio, sr = torchaudio.load(prompt_audio_path)
                    prompt_audio = prompt_audio
                    if sr != 16000:
                        prompt_audio = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(prompt_audio)
                    self.generate_spk_info(spk_id, prompt_text, prompt_audio, self.feat_extractor.sampling_rate)  # noqa
                    logging.info(f"Generate spk_info for {spk_id} success!")
                except Exception as e:
                    logging.error(f"Generate spk_info for {spk_id} failed!")
                    logging.error(e)

    def _extract_text_token(self, text):
        if isinstance(text, Generator):
            logging.info('get tts_text generator, will return _extract_text_token_generator!')
            # NOTE add a dummy text_token_len for compatibility
            return self._extract_text_token_generator(text), torch.tensor([0], dtype=torch.int32)
        elif isinstance(text, Union[AsyncGenerator, AsyncTextGeneratorWrapper]):
            logging.info('get tts_text async generator, will return _async_extract_text_token_generator!')
            # NOTE add a dummy text_token_len for compatibility
            return self._async_extract_text_token_generator(text), torch.tensor([0], dtype=torch.int32)
        else:
            text_token = self.tokenizer.encode(text, allowed_special=self.allowed_special)
            text_token = torch.tensor([text_token], dtype=torch.int32)
            text_token_len = torch.tensor([text_token.shape[1]], dtype=torch.int32)
            return text_token, text_token_len

    async def _async_extract_text_token_generator(self, text_generator):
        async for text in text_generator:
            text_token, _ = self._extract_text_token(text)
            for i in range(text_token.shape[1]):
                yield text_token[:, i: i + 1]

    def _extract_text_token_generator(self, text_generator):
        for text in text_generator:
            text_token, _ = self._extract_text_token(text)
            for i in range(text_token.shape[1]):
                yield text_token[:, i: i + 1]

    def _extract_speech_token(self, speech):
        logging.debug('_extract_speech_token start')
        assert speech.shape[1] / 16000 <= 30, 'do not support extract speech token for audio longer than 30s'
        feat = whisper.log_mel_spectrogram(speech, n_mels=128)
        logging.debug('_extract_speech_token log_mel_spectrogram finished')
        speech_token_input = {self.speech_tokenizer_session.get_inputs()[0].name:
                                                          feat.detach().cpu().numpy(),
                                                          self.speech_tokenizer_session.get_inputs()[1].name:
                                                          np.array([feat.shape[2]], dtype=np.int32)}
        logging.debug('_extract_speech_token speech_token_input prepared')
        speech_token_output = self.speech_tokenizer_session.run(None, speech_token_input)
        logging.debug('_extract_speech_token speech_tokenizer_session run')
        speech_token = speech_token_output[0].flatten().tolist()
        logging.debug('_extract_speech_token speech_tokenizer_session finished')
        speech_token = torch.tensor([speech_token], dtype=torch.int32)
        speech_token_len = torch.tensor([speech_token.shape[1]], dtype=torch.int32)
        return speech_token, speech_token_len

    def _extract_spk_embedding(self, speech):
        logging.debug('_extract_spk_embedding start')
        feat = kaldi.fbank(speech,
                           num_mel_bins=80,
                           dither=0,
                           sample_frequency=16000)
        logging.debug('_extract_spk_embedding fbank finished')
        feat = feat - feat.mean(dim=0, keepdim=True)
        embedding = self.campplus_session.run(None,
                                              {self.campplus_session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()})[0].flatten().tolist()
        logging.debug('_extract_spk_embedding campplus_session finished')
        embedding = torch.tensor([embedding]).to(self.device)
        return embedding

    def _extract_speech_feat(self, speech):
        speech_feat = self.feat_extractor(speech).squeeze(dim=0).transpose(0, 1).to(self.device)
        speech_feat = speech_feat.unsqueeze(dim=0)
        speech_feat_len = torch.tensor([speech_feat.shape[1]], dtype=torch.int32).to(self.device)
        return speech_feat, speech_feat_len

    def text_normalize(self, text, split=True, text_frontend=True):
        if isinstance(text, Union[Generator, AsyncGenerator]):
            logging.info('get tts_text generator, will skip text_normalize!')
            return AsyncTextGeneratorWrapper(text)

        if text_frontend is False:
            return [text] if split is True else text

        start = time.perf_counter()
        text = text.strip()
        text_len = len(text)
        if self.use_ttsfrd:
            texts = [i["text"] for i in json.loads(self.frd.do_voicegen_frd(text))["sentences"]]
            text = ''.join(texts)
        else:
            if contains_chinese(text):
                text = self.zh_tn_model.normalize(text)
                text = text.replace("\n", "")
                text = replace_blank(text)
                text = replace_corner_mark(text)
                text = text.replace(".", "。")
                text = text.replace(" - ", "，")
                text = remove_bracket(text)
                text = re.sub(r'[，,、]+$', '。', text)
                if not split:
                    return text
                texts = list(split_paragraph(text, partial(self.tokenizer.encode, allowed_special=self.allowed_special), "zh", token_max_n=80,
                                             token_min_n=60, merge_len=20, comma_split=False))
            else:
                text = self.en_tn_model.normalize(text)
                text = spell_out_number(text, self.inflect_parser)
                if not split:
                    return text
                texts = list(split_paragraph(text, partial(self.tokenizer.encode, allowed_special=self.allowed_special), "en", token_max_n=80,
                                             token_min_n=60, merge_len=20, comma_split=False))
        texts = [i for i in texts if not is_only_punctuation(i)]
        logging.debug(f'text_normalize text_len {text_len} cost {time.perf_counter() - start:.6f}s')
        return texts if split is True else text

    def frontend_sft(self, tts_text, spk_id):
        tts_text_token, tts_text_token_len = self._extract_text_token(tts_text)
        self.load_spk_info(spk_id)
        embedding = self.spk2info[spk_id]['embedding']
        assert embedding is not None
        model_input = {'text': tts_text_token, 'text_len': tts_text_token_len, 'llm_embedding': embedding, 'flow_embedding': embedding}
        return model_input

    def frontend_zero_shot(self, tts_text, prompt_text, prompt_speech_16k, resample_rate):
        tts_text_token, tts_text_token_len = self._extract_text_token(tts_text)
        prompt_text_token, prompt_text_token_len = self._extract_text_token(prompt_text)
        prompt_speech_resample = torchaudio.transforms.Resample(orig_freq=16000, new_freq=resample_rate)(prompt_speech_16k)
        speech_feat, speech_feat_len = self._extract_speech_feat(prompt_speech_resample)
        speech_token, speech_token_len = self._extract_speech_token(prompt_speech_16k)
        if resample_rate == 24000:
            # cosyvoice2, force speech_feat % speech_token = 2
            token_len = min(int(speech_feat.shape[1] / 2), speech_token.shape[1])
            speech_feat, speech_feat_len[:] = speech_feat[:, :2 * token_len], 2 * token_len
            speech_token, speech_token_len[:] = speech_token[:, :token_len], token_len
        embedding = self._extract_spk_embedding(prompt_speech_16k)
        model_input = {'text': tts_text_token, 'text_len': tts_text_token_len,
                       'prompt_text': prompt_text_token, 'prompt_text_len': prompt_text_token_len,
                       'llm_prompt_speech_token': speech_token, 'llm_prompt_speech_token_len': speech_token_len,
                       'flow_prompt_speech_token': speech_token, 'flow_prompt_speech_token_len': speech_token_len,
                       'prompt_speech_feat': speech_feat, 'prompt_speech_feat_len': speech_feat_len,
                       'llm_embedding': embedding, 'flow_embedding': embedding}
        return model_input

    def frontend_cross_lingual(self, tts_text, prompt_speech_16k, resample_rate):
        model_input = self.frontend_zero_shot(tts_text, '', prompt_speech_16k, resample_rate)
        # in cross lingual mode, we remove prompt in llm
        del model_input['prompt_text']
        del model_input['prompt_text_len']
        del model_input['llm_prompt_speech_token']
        del model_input['llm_prompt_speech_token_len']
        return model_input

    def frontend_instruct(self, tts_text, spk_id, instruct_text):
        model_input = self.frontend_sft(tts_text, spk_id)
        # in instruct mode, we remove spk_embedding in llm due to information leakage
        del model_input['llm_embedding']
        instruct_text_token, instruct_text_token_len = self._extract_text_token(instruct_text + '<endofprompt>')
        model_input['prompt_text'] = instruct_text_token
        model_input['prompt_text_len'] = instruct_text_token_len
        return model_input

    def frontend_instruct2(self, tts_text, instruct_text, prompt_speech_16k, resample_rate):
        model_input = self.frontend_zero_shot(tts_text, instruct_text + '<|endofprompt|>', prompt_speech_16k, resample_rate)
        del model_input['llm_prompt_speech_token']
        del model_input['llm_prompt_speech_token_len']
        return model_input

    def frontend_vc(self, source_speech_16k, prompt_speech_16k, resample_rate):
        prompt_speech_token, prompt_speech_token_len = self._extract_speech_token(prompt_speech_16k)
        prompt_speech_resample = torchaudio.transforms.Resample(orig_freq=16000, new_freq=resample_rate)(prompt_speech_16k)
        prompt_speech_feat, prompt_speech_feat_len = self._extract_speech_feat(prompt_speech_resample)
        embedding = self._extract_spk_embedding(prompt_speech_16k)
        source_speech_token, source_speech_token_len = self._extract_speech_token(source_speech_16k)
        model_input = {'source_speech_token': source_speech_token, 'source_speech_token_len': source_speech_token_len,
                       'flow_prompt_speech_token': prompt_speech_token, 'flow_prompt_speech_token_len': prompt_speech_token_len,
                       'prompt_speech_feat': prompt_speech_feat, 'prompt_speech_feat_len': prompt_speech_feat_len,
                       'flow_embedding': embedding}
        return model_input

    def generate_spk_info(self, spk_id: str, prompt_text: str, prompt_speech_16k: torch.Tensor, resample_rate:int=24000, name: str=None) -> bytes:
        assert isinstance(spk_id, str)
        logging.debug(f"generate_spk_info {spk_id} start: prompt_text: {prompt_text}, name: {name}, resample_rate: {resample_rate}")
        prompt_text_token, _ = self._extract_text_token(prompt_text)
        logging.debug(f"generate_spk_info {spk_id} prompt_text_token finished")
        prompt_speech_resample = torchaudio.transforms.Resample(orig_freq=16000, new_freq=resample_rate)(prompt_speech_16k)
        logging.debug(f"generate_spk_info {spk_id} Resample finished")
        speech_feat, _ = self._extract_speech_feat(prompt_speech_resample)
        logging.debug(f"generate_spk_info {spk_id} _extract_speech_feat finished")
        speech_token, speech_token_len = self._extract_speech_token(prompt_speech_16k)
        logging.debug(f"generate_spk_info {spk_id} _extract_speech_token finished")
        if resample_rate == 24000:
            # cosyvoice2, force speech_feat % speech_token = 2
            token_len = min(int(speech_feat.shape[1] / 2), speech_token.shape[1])
            speech_feat = speech_feat[:, :2 * token_len]
            speech_token = speech_token[:, :token_len]
        embedding = self._extract_spk_embedding(prompt_speech_16k)
        logging.debug(f"generate_spk_info {spk_id} _extract_spk_embedding finished")
        spk_info = SpeakerInfo(
            name=name,
            spk_id=spk_id,
            prompt_text=prompt_text,
            prompt_text_token=prompt_text_token,
            speech_feat=speech_feat,
            speech_token=speech_token,
            embedding=embedding,
        )
        return self.add_spk_info(spk_id, spk_info)

    def add_spk_info(self, spk_id: str, spk_info: Union[dict, SpeakerInfo]) -> bytes:
        if isinstance(spk_info, BaseModel):
            spk_info = spk_info.model_dump()
        self.spk2info[spk_id] = spk_info
        spk_info_path = os.path.join(self.spk2info_path, spk_id + '.pt')
        torch.save(self.spk2info[spk_id], spk_info_path)
        with open(spk_info_path, 'rb') as f:
            return f.read()

    def load_spk_info(self, spk_id: str):
        if spk_id not in self.spk2info:
            spk_info_path = os.path.join(self.spk2info_path, spk_id + '.pt')
            if not os.path.exists(spk_info_path):
                raise ValueError(f'not found spk2info: {spk_id}')
            spk_info = torch.load(spk_info_path, map_location=self.device, weights_only=False)
            self.spk2info[spk_id] = spk_info

    def delete_spk_info(self, spk_id: str):
        if spk_id in self.spk2info:
            del self.spk2info[spk_id]
        if os.path.exists(os.path.join(self.spk2info_path, spk_id + '.pt')):
            os.remove(os.path.join(self.spk2info_path, spk_id + '.pt'))

    def frontend_instruct2_by_spk_id(self, tts_text, instruct_text, spk_id):
        self.load_spk_info(spk_id)
        tts_text_token, _ = self._extract_text_token(tts_text)
        prompt_text_token, _ = self._extract_text_token(instruct_text + '<|endofprompt|>')
        model_input = {'text': tts_text_token,
                       'prompt_text': prompt_text_token,
                       'flow_prompt_speech_token': self.spk2info[spk_id]['speech_token'],
                       'prompt_speech_feat': self.spk2info[spk_id]['speech_feat'],
                       'llm_embedding': self.spk2info[spk_id]['embedding'],
                       'flow_embedding': self.spk2info[spk_id]['embedding'],
        }
        return model_input

    def frontend_zero_shot_by_spk_id(self, tts_text, spk_id):
        self.load_spk_info(spk_id)
        tts_text_token, _ = self._extract_text_token(tts_text)
        model_input = {'text': tts_text_token,
                       'prompt_text': self.spk2info[spk_id]['prompt_text_token'],
                       'llm_prompt_speech_token': self.spk2info[spk_id]['speech_token'],
                       'flow_prompt_speech_token': self.spk2info[spk_id]['speech_token'],
                       'prompt_speech_feat': self.spk2info[spk_id]['speech_feat'],
                       'llm_embedding': self.spk2info[spk_id]['embedding'],
                       'flow_embedding': self.spk2info[spk_id]['embedding']
        }
        return model_input