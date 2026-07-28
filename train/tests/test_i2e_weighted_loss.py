import pytest

pytest.importorskip("torch")
from types import SimpleNamespace

import torch


from training.trainer import QwenTrainer


class DummyTokenizer:
    mapping = {
        "<think>": [1],
        "</think>": [2],
        "<explicit>": [3, 4, 5],
        "</explicit>": [6, 4, 5],
        "{<": [9],
        ">}": [10],
        "<answer>": [7],
        "</answer>": [8],
    }

    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return SimpleNamespace(input_ids=self.mapping[text])


def test_i2e_token_weights_cover_format_and_bbox_suffix() -> None:
    trainer = object.__new__(QwenTrainer)
    trainer.processor = SimpleNamespace(tokenizer=DummyTokenizer())
    trainer.args = SimpleNamespace(
        i2e_answer_token_weight=5.0,
        i2e_format_token_weight=2.0,
    )
    labels = torch.tensor(
        [[-100, 1, 30, 2, 3, 4, 5, 31, 6, 4, 5, 7, 40, 41, 8, 99]]
    )
    weights = trainer._i2e_token_weights(labels)

    assert weights[0, 1].item() == 2.0
    assert weights[0, 4].item() == 2.0
    assert weights[0, 7].item() == 1.0
    assert weights[0, 11].item() == 5.0
    assert weights[0, 12].item() == 5.0
    assert weights[0, 14].item() == 5.0
    assert weights[0, 15].item() == 1.0

def test_i2e_token_weights_cover_answer_only_bbox() -> None:
    trainer = object.__new__(QwenTrainer)
    trainer.processor = SimpleNamespace(tokenizer=DummyTokenizer())
    trainer.args = SimpleNamespace(
        i2e_answer_token_weight=5.0,
        i2e_format_token_weight=2.0,
    )
    labels = torch.tensor([[-100, 9, 40, 41, 42, 43, 10, 99]])
    weights = trainer._i2e_token_weights(labels)

    assert weights[0, 1].item() == 5.0
    assert weights[0, 5].item() == 5.0
    assert weights[0, 6].item() == 5.0
    assert weights[0, 7].item() == 1.0
