import logging
import pickle
import inspect

import aiohttp
import psutil
from aiohttp import web

logger = logging.getLogger(__name__)


class RemoteObjectDaemon:
    def __init__(self, host, port, instance_strategy: str, max_memory: int):
        self.host = host
        self.port = port
        self.app = web.Application()
        self.app.router.add_post('/rpc', self.handle_request)
        self.registered_classes = {}
        self.instance_strategy = instance_strategy
        self.persistent_instances = {}
        self.max_memory = max_memory

    def register_class(self, cls):
        self.registered_classes[cls.__name__] = cls

    def _get_or_create_instance(self, cls, class_name, class_args, class_kwargs):
        instance_key = (class_name, pickle.dumps((class_args, class_kwargs)))
        if instance_key not in self.persistent_instances:
            logger.debug(f'Creating new instance for {class_name}')
            self.persistent_instances[instance_key] = cls(*class_args, **class_kwargs)
        logger.debug(f'Current total instance count: {len(self.persistent_instances.items())}')
        return self.persistent_instances[instance_key]

    async def handle_request(self, request):
        data = await request.read()
        try:
            class_name, method_name, args, kwargs, class_args, class_kwargs = pickle.loads(data)
            cls = self.registered_classes.get(class_name)
            if cls:
                # get class instance
                if self.instance_strategy == 'per_request':
                    logger.debug(f'Creating new instance for {class_name}')
                    instance = cls(*class_args, **class_kwargs)
                elif self.instance_strategy == "persistent":
                    instance = self._get_or_create_instance(cls, class_name, class_args, class_kwargs)
                else:  # memory_limit
                    process = psutil.Process()
                    memory_info = process.memory_info()
                    if memory_info.rss > self.max_memory:
                        self.persistent_instances.clear()
                    instance = self._get_or_create_instance(cls, class_name, class_args, class_kwargs)

                # get and call method
                method = getattr(instance, method_name)
                if inspect.iscoroutinefunction(method):
                    result = await method(*args, **kwargs)
                else:
                    result = method(*args, **kwargs)
                return web.Response(body=pickle.dumps(result))
            else:
                return web.Response(status=404, text=f"Class '{class_name}' not found.")
        except Exception as e:
            return web.Response(status=500, text=str(e))

    def serve(self):
        web.run_app(self.app, host=self.host, port=self.port, access_log=None)


class RemoteObjectProxy:
    def __init__(self, class_name, host, port, instance_args=None, instance_kwargs=None):
        self.class_name = class_name
        self.host = host
        self.port = port
        self.instance_args = instance_args or ()
        self.instance_kwargs = instance_kwargs or {}

    def __getattr__(self, name):
        if name == '__slots__':
            return ['class_name', 'host', 'port', 'instance_args', 'instance_kwargs']

        async def remote_method(*args, **kwargs):
            data = pickle.dumps((self.class_name, name, args, kwargs, self.instance_args, self.instance_kwargs))
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as session:
                async with session.post(f'http://{self.host}:{self.port}/rpc', data=data) as response:
                    if response.status == 200:
                        result = await response.read()
                        return pickle.loads(result)
                    else:
                        error_message = await response.text()
                        raise Exception(f"Remote method call failed: {error_message}")

        return remote_method

    def __str__(self):
        return f"RemoteObjectProxy(class_name={self.class_name}, host={self.host}, port={self.port}, instance_args={self.instance_args}, instance_kwargs={self.instance_kwargs})"


def create_remote_object_daemon(
        host='0.0.0.0',
        port=8008,
        instance_strategy='memory_limit',
        max_memory: int = 8 * 1024 * 1024 * 1024
):
    return RemoteObjectDaemon(host, port, instance_strategy, max_memory)


def remote_object(class_name, host, port, *args, **kwargs):
    return RemoteObjectProxy(class_name, host, port, args, kwargs)