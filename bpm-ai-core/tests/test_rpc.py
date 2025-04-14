import pytest

from bpm_ai_core.util.rpc import RemoteObjectDaemon, RemoteObjectProxy, create_remote_object_daemon, remote_object


class MyContainer:
    def __init__(self, value):
        self.value = value

class MyClass:

    def __init__(self, y, z, debug: bool = False):
        self.debug = debug
        self.y = y
        self.z = z

    async def do_something(self, x: int):
        return MyContainer(f"{self.y} {self.z}: Hello World! {x} - debug: {self.debug}")

@pytest.mark.skip("for manual testing")
def test_rpc():
    daemon = create_remote_object_daemon()
    daemon.register_class(MyClass)
    daemon.serve()

@pytest.mark.skip("for manual testing")
async def test_rpc_client():
    proxy: MyClass = remote_object("MyClass", "0.0.0.0", 8008, 2, 221, debug=True)
    result = await proxy.do_something(1338)

    print(result.value)
