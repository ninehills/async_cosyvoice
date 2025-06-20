# vllm settings
# from vllm.engine.arg_utils import AsyncEngineArgs
# AsyncEngineArgs
ENGINE_ARGS = {
    # "enforce_eager": True,
    "gpu_memory_utilization": 0.4,
    "max_num_batched_tokens": 1024,
    "max_model_len": 2048,
    "max_num_seqs": 256,
    "disable_log_requests": True,
    "disable_log_stats": True,
    "dtype": "float16",
}

from vllm.sampling_params import RequestOutputKind
# SamplingParams
SAMPLING_PARAMS = {
    "temperature": 1,  # 不能低于0.8, 否则会生成非常多的空音频，或者无法正常生成语音Token
    "top_p": 1,       # 不能低于0.8, 否则会生成非常多的空音频，或者无法正常生成语音Token
    "top_k": 25,
    # "min_tokens": 80,       # 不支持设置最小的tokens数量设置，开启后vllm直接崩溃，无法启动
    # "presence_penalty": 1.0,    # 不支持设置
    # "frequency_penalty": 0.0,   # 不支持设置
    "max_tokens": 2048,
    "detokenize": False,          # 目前 vllm 0.7.3 v1版本中设置无效，待后续版本更新后减少计算
    "ignore_eos": False,
    "output_kind": RequestOutputKind.DELTA  # 设置为DELTA，如调整该参数，请同时调整llm_inference的处理代码
}

# 设置frontend中 ZhNormalizer 的 overwrite_cache 参数
# 首次运行时，需要设置 True 正确生成缓存，避免 frontend 过滤掉儿化音。
# 后续可以设置为 False 可避免后续运行时重复生成。
OVERWRITE_NORMALIZER_CACHE = True

# 限制 estimator 内存方法  由 @hexisyztem 提供
# 原本代码编译后的flow trt模型 显存占用4.6G过大，修改为 1.6G，便于启动多个 estimator 实例，并发推理。
# 修改 cosyvoice/utils/file_utils.py:64
#     # config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 33)  # 8GB
#     config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
# 删除已经编译的模型./pretrained_models/CosyVoice2-0.5B/flow.decoder.estimator.fp16.mygpu.plan
# 重新运行服务器代码  --trt 将重新编译模型，到时将生成新的模型，并单个 estimator 只占用 1.6GB 显存
# 根据GPU显存大小量及性能设置合适的 ESTIMATOR_COUNT
ESTIMATOR_COUNT = 8


# 注册音色信息，用于frontend中生成音色信息, 请根据自己的实际情况进行使用
# key 为 spk_id, value 为参考音频对应的文本、音频路径， 必填
# 可以一次性注册多个音色信息，之前注册的音色信息，后续可以再次注册，将覆盖之前的音色信息
REGISTER_SPK_INFO_DICT = {
    "new_spk_id": {
        "prompt_text": "这是参考音频对应的文本",
        "prompt_audio_path": "/home/none/test.wav", # 请尽量是16000Hz的音频，并以绝对路径提供
    }
}

# 编写一个代码检查是否有 private_config.py 文件，如果有则读取该文件，并【完全】覆盖ENGINE_ARGS 和 SAMPLING_PARAMS 的值
try:
    from async_cosyvoice.private_config import *
    import logging
    logging.info("Loaded private_config.py")
except ImportError:
    pass
