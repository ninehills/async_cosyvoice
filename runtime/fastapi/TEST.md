# 测试指南

## 环境依赖

硬件：
- CPU: x86
- GPU: GTX4090

Python:

todo

Redis:

```bash
docker run -d --name redis -p 6379:6379 redis:6
```

## Mock 环境启动

```bash
pip install soundfile==0.12.1 torch torchaudio
pip install -r requirements.txt
MOCK_ENABLED=1 python ./server.py
```

## 正式环境启动


## 功能测试
```bash
export SERVER_URL="http://182.18.34.67:8022"
```

### 1. 上传参考音频

```bash
curl -X POST \
  -H "Authorization: Bearer 123456" \
  -F "customName=tyzr" \
  -F "text=对，这就是我，万人敬仰的太乙真人，虽然有点婴儿肥，但也掩不住我逼人的帅气。" \
  -F "file=@sample.wav" \
  $SERVER_URL/v1/uploads/audio/voice
```

`{"uri":"speech:1:tyzr:ac63bf50"}`

### 2. 列出参考音频列表

```bash
curl -X GET \
  -H "Authorization: Bearer 123456" \
  $SERVER_URL/v1/audio/voice/list
```
`{"results":[{"model":"FunAudioLLM/CosyVoice2-0.5B","customName":"tyzr","text":"对，这就是我，万人敬仰的太乙真人，虽然有点婴儿肥，但也掩不住我逼人的帅气。","uri":"speech:1:tyzr:b164d612"}]}`

### 3. 创建文本转语音请求

```bash
curl -X POST \
  -H "Authorization: Bearer 123456" \
  -H "Content-Type: application/json" \
  -d '{"input":"突然，身边一阵笑声。我看着他们，意气风发地挺直了胸膛，甩了甩那稍显肉感的双臂，轻笑道：\"我身上的肉，是为了掩饰我爆棚的魅力，否则，岂不吓坏了你们呢？\"", "voice":"speech:1:tyzr:ac63bf50", "response_format":"mp3"}' \
  $SERVER_URL/v1/audio/speech -o audio.mp3
```

### 4. 删除参考音频

```bash
curl -X POST \
  -H "Authorization: Bearer 123456" \
  -H "Content-Type: application/json" \
  -d '{"uri": "speech:1:tyzr:b164d612"}' \
  $SERVER_URL/v1/audio/voice/deletions
```

`{}`