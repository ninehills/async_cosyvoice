from pydantic import BaseModel


class VoiceMeta(BaseModel):
    """音色元信息"""
    model: str
    customName: str
    text: str
    uri: str
