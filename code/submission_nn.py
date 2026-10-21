"""Listwise NN selector submission (numpy-only inference). checkpoint dir: checkpoint_nn/ (nn.npz + config.json)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'method'))
import nn_infer  # noqa: E402


def predict(bank, arrays, checkpoint=None):
    return nn_infer.predict_nn(bank, arrays['embeddings'], Path(checkpoint))
