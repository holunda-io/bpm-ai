from bpm_ai_core.llm.common.blob import Blob


def test_blob_path():
    blob = Blob.from_path_or_url('test.mp3')

    assert blob.path.endswith('/test.mp3')
    assert blob.mimetype == 'audio/mpeg'
    assert blob.is_audio() is True
    assert blob.data is None


def test_blob_path2():
    blob = Blob.from_path_or_url('example.jpg')

    assert blob.path.endswith('/example.jpg')
    assert blob.mimetype == 'image/jpeg'
    assert blob.is_image() is True
    assert blob.data is None
