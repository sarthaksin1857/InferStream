from inferstream.inference.engine import sample_next_token
import torch


def test_sample_next_token_shape():
    logits = torch.randn(2, 100)
    next_token = sample_next_token(logits, temperature=1.0, top_p=0.95)
    assert next_token.shape == (2, 1)
