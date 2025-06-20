import redis.asyncio as redis

from model import VoiceMeta

class VoiceStorage():
    """音色存储服务，将元信息和音色文件保存到 Redis 中。

    而在 frontend.py 中，自带内存/磁盘缓存。

    音色元信息和文件分别使用不同的 Key 进行存储，Key 格式为：
    - 音色元信息：`voice:speech:<user_id>:xxx`
    - 音色文件：`voice_file:speech:<user_id>:xxx`
    
    其中 `speech:<user_id>:xxx` 是 uri
    """
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client

    def get_voice_meta_key(self, uri: str):
        return f'voice:{uri}'
    
    def get_voice_file_key(self, uri: str):
        return f'voice_file:{uri}'
    
    async def get_voice(self, uri: str) -> bytes:
        """获取指定 URI 的音色文件
        
        注意：因为是先从缓存中获取，所以目前 get_voice 可能会获得已经删除的 voice
        Args:
            uri: 音色 URI
        
        Exception:
            KeyError: 音色不存在
        """
        # 从Redis获取数据
        data = await self.redis_client.get(self.get_voice_file_key(uri))
        if not data:
            raise KeyError(f'Voice {uri} not found')
        return data

    async def save_voice(self, uri: str, voice_meta: VoiceMeta, voice_data: bytes):
        """保存一个音色文件

        Args:
            uri: 音色 URI
            voice_meta: 音色元信息
            voice_data: 音色文件数据
        """
        # 使用事务的方式写入Redis
        async with self.redis_client.pipeline() as pipe:
            pipe.set(self.get_voice_meta_key(uri), voice_meta.model_dump_json())
            pipe.set(self.get_voice_file_key(uri), voice_data)
            await pipe.execute()

    async def list_voices(self, user_id: str) -> list[VoiceMeta]:
        """列出用户的所有音色元信息

        Args:
            user_id: 用户ID
        
        Exception:
            KeyError: 用户不存在
        
        Returns:
            List[VoiceData]: 音色元信息列表。每个元素都是一个 VoiceData 类的实例，包含音色的元信息。
        """
        # TODO: 如果用户下的音色过多，可能会导致性能瓶颈。更好的方法是使用分页获取的方式。
        key_pattern = f'voice:speech:{user_id}:*'
        keys = await self.redis_client.keys(key_pattern)
        values = await self.redis_client.mget(keys)
        voice_data_list = []
        for value in values:
            if value:
                voice_data_list.append(VoiceMeta.model_validate_json(value))
        return voice_data_list

    async def delete_voice(self, uri: str):
        """删除一个音色文件

        Args:
            uri: 音色 URI
        """
        # 使用事务从redis 中删除 Meta 和 Voice
        async with self.redis_client.pipeline() as pipe:
            pipe.delete(self.get_voice_meta_key(uri))
            pipe.delete(self.get_voice_file_key(uri))
            await pipe.execute()
