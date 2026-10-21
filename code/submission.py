"""Final method: listwise DeepSets NN ensemble over hand+cross features and the full embedding (numpy-only inference).
Checkpoint dir: checkpoint/ (nn.npz + config.json). LR ablation: submission_lr.py + checkpoint_lr/."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'method'))
import nn_infer  # noqa: E402


def predict(bank, arrays, checkpoint=None):
    return nn_infer.predict_nn(bank, arrays['embeddings'], Path(checkpoint))
