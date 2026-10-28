"""v2 (submitted until 2026-10-28, now an ablation): listwise DeepSets NN ensemble (K=5) over hand+cross features and the full embedding (numpy-only inference).
Checkpoint dir: checkpoint_v2/ (nn.npz + config.json). LR ablation: submission_lr.py + checkpoint_lr/."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'method'))
import nn_infer  # noqa: E402


def predict(bank, arrays, checkpoint=None):
    return nn_infer.predict_nn(bank, arrays['embeddings'], Path(checkpoint))
