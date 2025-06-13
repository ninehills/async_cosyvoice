from typing import Optional
import os
import aiofiles

import redis.asyncio as redis

class AsyncMixCache():
    def __init__(self, disk_dir: str, redis_client: Optional[redis.Redis] = None, redis_key_prefix: str = 'cache'):
        self.disk_dir = disk_dir
        self.redis_key_prefix = redis_key_prefix
        self.redis_client = redis_client

    def disk_cache_file_path(self, key: str):
        # TODO: 拆分Key的子目录，以减少单个目录的文件数量
        return os.path.join(self.disk_dir, key)
    
    def gen_redis_key(self, key):
        return f'{self.redis_key_prefix}:{key}'
    
    async def get(self, key):
        if os.path.exists(self.disk_cache_file_path(key)):
            async with aiofiles.open(self.disk_cache_file_path(key), 'rb') as f:
                return await f.read()

        if not self.redis_client:
            return KeyError(f'Key {key} not found in disk cache and redis is not set')

        # 从Redis获取数据
        data = await self.redis_client.get(self.gen_redis_key(key))
        
        # 如果获取到数据，将其缓存到本地
        if data:
            # 确保目录存在
            cache_file_path = self.disk_cache_file_path(key)
            os.makedirs(os.path.dirname(cache_file_path), exist_ok=True)
            
            # 异步写入文件
            async with aiofiles.open(cache_file_path, 'wb') as f:
                await f.write(data)
        
        return data
    
    async def set(self, key, value):
        # 先写入本地
        cache_file_path = self.disk_cache_file_path(key)
        os.makedirs(os.path.dirname(cache_file_path), exist_ok=True)
        async with aiofiles.open(cache_file_path, 'wb') as f:
            await f.write(value)
        # 再写入Redis
        if self.redis_client:
            await self.redis_client.set(self.gen_redis_key(key), value)

    async def delete(self, key):
        # TODO: 只从单机删除，并没有从其他节点的 MemoryCache/DiskCache 中删除。
        # 从本地删除
        if os.path.exists(self.disk_cache_file_path(key)):
            os.remove(self.disk_cache_file_path(key))
        # 从Redis删除
        if self.redis_client:
            await self.redis_client.delete(self.gen_redis_key(key))
