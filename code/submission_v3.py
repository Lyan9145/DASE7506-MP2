"""v3 experiment: listwise DeepSets ensemble (K=10) over v1 hand+cross features, v3 cross-candidate text
features (features_v3.py: do the other candidates' final answers occur inside this candidate's reasoning, and
vice versa) and the full embedding; numpy-only inference, cluster-sum decode.
Checkpoint dir: checkpoint_v3/ (nn.npz + config.json). Experiment, NOT the submitted method (see README).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'method'))
import nn3_infer  # noqa: E402


def predict(bank, arrays, checkpoint=None):
    return nn3_infer.predict(bank, arrays['embeddings'], Path(checkpoint))
