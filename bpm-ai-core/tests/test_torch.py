
def test_torch():
    import torch
    x = torch.rand(3, 3)
    assert x.dim() == 2
