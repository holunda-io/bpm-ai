import asyncio
import json
import logging
import os
import pickle
from typing import Any

logger = logging.getLogger(__name__)


def remote_object_init():
    import Pyro5
    from Pyro5.api import register_class_to_dict, register_dict_to_class

    def _pickle_class_to_dict(obj: Any) -> dict:
        return {"__class__": "object", "pickle": pickle.dumps(obj)}

    def _pickle_dict_to_class(_, obj_dict: dict) -> Any:
        return pickle.loads(obj_dict["pickle"])

    register_class_to_dict(object, _pickle_class_to_dict)
    register_dict_to_class("object", _pickle_dict_to_class)

    Pyro5.config.SERIALIZER = "marshal"


def create_object_identifier(name: str, *args, **kwargs):
    args_str = '_'.join(str(arg) for arg in args)
    kwargs_str = json.dumps(kwargs, sort_keys=True).replace('"', '').replace(' ', '')
    return f"{name}:{args_str}_{kwargs_str}"


class ObjectDaemon:
    from Pyro5.server import expose

    def __init__(self, daemon, classes: list = None, instance_mode: str = None):
        self.daemon = daemon
        self.classes = classes or []
        self.instances_by_class_name = {}

        self.instance_mode = instance_mode or os.environ.get("INSTANCE_MODE", "per_class")

    @property
    def _class_dict(self):
        return {c.__name__: c for c in self.classes}

    def register_class(self, clazz):
        from Pyro5.server import expose

        def create_sync_method(async_method):
            def sync_method(self, *args, **kwargs):
                return asyncio.run(async_method(self, *args, **kwargs))

            return sync_method

        def add_sync_methods(cls):
            for attr_name in dir(cls):
                attr = getattr(cls, attr_name)
                if callable(attr) and asyncio.iscoroutinefunction(attr):
                    sync_method_name = f"{attr_name}_sync"
                    sync_method = create_sync_method(attr)
                    setattr(cls, sync_method_name, sync_method)
            return cls

        add_sync_methods(clazz)
        expose(clazz)
        self.classes.append(clazz)

    @expose
    def create_instance(self, class_name: str, *args, **kwargs) -> str:
        object_id = create_object_identifier(class_name, *args, **kwargs)

        if object_id in self.daemon.objectsById:
            return object_id
        else:
            if self.instance_mode == "per_call":
                for obj in self.instances_by_class_name:
                    logging.info(f"Deleting old object for class {class_name}")
                    self.daemon.unregister(obj)
                    del obj
            elif self.instance_mode == "per_class" and class_name in self.instances_by_class_name:
                obj = self.instances_by_class_name[class_name]
                logging.info(f"Deleting old object for class {class_name}")
                self.daemon.unregister(obj)
                del obj
            else:  # per_params
                pass

            clazz = self._class_dict[class_name]
            instance = clazz(*args, **kwargs)
            self.daemon.register(instance, objectId=object_id)
            self.instances_by_class_name[class_name] = instance

            logging.info(f"Created new object for class {class_name}")

            return object_id

    def serve(self):
        self.daemon.requestLoop()


def create_remote_object_daemon(host: str = "localhost", port: int = 1337) -> ObjectDaemon:
    import Pyro5.api
    remote_object_init()
    pyro_daemon = Pyro5.api.Daemon(host=host, port=port)
    daemon = ObjectDaemon(pyro_daemon)
    pyro_daemon.register(daemon, objectId="__object_creator__")
    logging.info(f"Object daemon created for {host}:{port}")
    return daemon


def remote_object(name: str, host: str = "localhost", port: int = 6666, *args, **kwargs) -> Any:
    import Pyro5.errors
    remote_object_init()
    try:
        object_id = create_object_identifier(name, *args, **kwargs)
        return _get_proxy(object_id, host, port, make_async=True)
    except Pyro5.errors.CommunicationError:
        logging.info(f"Remote object for {name} not found, creating a new one...")
        creator = _get_proxy("__object_creator__", host, port)
        object_id = creator.create_instance(name, *args, **kwargs)
        return _get_proxy(object_id, host, port, make_async=True)


def _get_proxy(object_id, host, port, make_async=False):
    import Pyro5.api
    from Pyro5.client import Proxy

    class AsyncProxy(Pyro5.api.Proxy):
        def __getattr__(self, name):
            attr = super().__getattr__(name + "_sync")
            return self._sync_to_async_wrapper(attr)

        def _sync_to_async_wrapper(self, sync_func):
            s = super()

            def ownership_wrapper(f, *args, **kwargs):
                # claim ownership for object as we are running in a new thread
                s._pyroClaimOwnership()
                return f(*args, **kwargs)

            async def async_wrapper(*args, **kwargs):
                coro = asyncio.to_thread(ownership_wrapper, sync_func, *args, **kwargs)
                return await coro

            return async_wrapper

    ProxyClass = AsyncProxy if make_async else Proxy
    proxy = ProxyClass(f"PYRO:{object_id}@{host}:{port}")
    proxy._pyroBind()
    return proxy
