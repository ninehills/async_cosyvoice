# API 文档

## 上传参考音频

### 接口

- **URL**: `/v1/uploads/audio/voice`
- **方法**: `POST`
- **权限**: 需要用户认证

### 请求参数（application/form）

- `model` (字符串, 可选): 模型选择，目前支持"FunAudioLLM/CosyVoice2-0.5B"，默认为"FunAudioLLM/CosyVoice2-0.5B"
- `customName` (字符串, 必需): 自定义音色名称，注意仅能使用字母、数字、下划线、横线，且不能超过30个字符
- `text` (字符串, 必需): 参考文本
- `file` (文件, 必需): 参考音频文件

### 响应

- **状态码**: `200 OK`
- **内容**:
  ```json
  {
    "uri": "speech:xxx:xxx:xxx"
  }
  ```

### 示例

```bash
curl -X POST \
  -H "Authorization: Bearer your_access_key" \
  -F "customName=your-voice-name" \
  -F "text=这是一个示例文本" \
  -F "file=@/path/to/audio.wav" \
  http://localhost:8022/v1/uploads/audio/voice
```

### 注意事项

提供参考音频的高质量样本可以提升语音克隆效果。

- 仅限单一说话人。
- 参考音频的语音内容必须和提供的文本内容一致。
- 吐字清晰、稳定的音量、音调和情绪。
- 简短的停顿（建议 0.5 秒）。
- 理想情况：无背景噪音、专业录音质量、无房间回声。
- 建议时间8～10s左右。
- 文件格式
  - 支持格式：mp3, wav, pcm, opus。
  - 推荐使用 192kbps 以上的 mp3 或 wav 格式以避免质量损失。

## 创建文本转语音请求

### 接口

- **URL**: `/v1/audio/speech`
- **方法**: `POST`
- **权限**: 需要用户认证

### 请求参数（application/json）

```json
{
  "input": "你好，欢迎使用语音合成服务！",
  "voice": "speech:xxx:xxx:xxx",
  "response_format": "mp3",
  "sample_rate": 24000,
  "stream": false,
  "speed": 1.0
}
```

- `model` (字符串, 可选): 模型选择，目前支持"FunAudioLLM/CosyVoice2-0.5B"，默认为"FunAudioLLM/CosyVoice2-0.5B"
- `input` (字符串, 必需): 需要转换为语音的文本内容，最大长度4096字符
- `voice` (字符串, 必需): 音色选择，可以是预设音色ID（001）或自定义音色URI
- `response_format` (字符串, 可选): 输出音频格式，支持 "mp3"、"wav"、"pcm"，默认为 "mp3"
- `sample_rate` (整数, 可选): 采样率，目前不支持设置，默认为24000 Hz
- `stream` (布尔值, 可选): 是否开启流式返回，默认为 false
- `speed` (浮点数, 可选): 语速控制，范围[0.25-4.0]，默认为1.0

### 响应

- **状态码**: `200 OK`
- **内容类型**: 根据请求的`response_format`返回对应的音频数据
  - mp3: `audio/mpeg`
  - wav: `audio/wav`
  - pcm: `audio/L16; rate={sample_rate}; channels=1`

### 示例

```bash
curl -X POST \
  -H "Authorization: Bearer your_access_key" \
  -H "Content-Type: application/json" \
  -d '{"input":"你好，欢迎使用语音合成服务！","voice":"speech:xxx:xxx:xxx","response_format":"mp3"}' \
  --output audio.mp3 \
  http://localhost:8022/v1/audio/speech
```

### 注意事项

- 如果文本中包含 `<|endofprompt|>` 标记，系统将使用指令模式进行合成
- 语速参数可以调整语音的播放速度，但不会影响音质
- 流式返回模式适合长文本实时合成场景

## 参考音频列表获取

### 接口

- **URL**: `/v1/audio/voice/list`
- **方法**: `GET`
- **权限**: 需要用户认证

### 请求参数

无

### 响应

- **状态码**: `200 OK`
- **内容**:
  ```json
    {
        "results": [
            {
            "model": "xxxxx",
            "customName": "your-voice-name",
            "text": "在一无所知中, 梦里的一天结束了，一个新的轮回便会开始",
            "uri": "speech:xxx:xxx:xxx"
            }
        ]
    }
  ```

### 示例

```bash
curl -X GET \
  -H "Authorization: Bearer your_access_key" \
  http://localhost:8022/v1/audio/voice/list
```


## 删除参考音频

### 接口

- **URL**: `/v1/audio/voice/deletions`
- **方法**: `POST`
- **权限**: 需要用户认证

### 请求参数（application/json）

```json
{
    "uri": "speech:xxx:xxx:xxx"
}
```

### 响应

- **状态码**: `200 OK`
- **内容**:
  ```json
  {}
  ```

### 示例

```bash
curl -X POST \
  -H "Authorization: Bearer your_access_key" \
  -H "Content-Type: application/json" \
  -d '{"uri": "speech:xxx:xxx:xxx"}' \
  http://localhost:8022/v1/audio/voice/deletions
```

