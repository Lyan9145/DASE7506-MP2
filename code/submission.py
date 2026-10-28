"""Final method (v4): listwise DeepSets ensemble (K=10) over v1 hand+cross features, v3 cross-candidate text
features (features_v3.py: do the other candidates' final answers occur inside this candidate's reasoning, and
vice versa), arithmetic-repair features (features_repair.py; idea credited to github.com/Devin-Lo1/dase-7506-mp2,
independent implementation) and the full embedding; numpy-only inference, cluster-sum decode.
Checkpoint dir: checkpoint/ (nn.npz + config.json). Ablations: submission_v3.py + checkpoint_v3/ (no repair),
submission_v2.py + checkpoint_v2/ (v1 features only), submission_lr.py + checkpoint_lr/ (logistic regression).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'method'))
import nn3_infer  # noqa: E402


def predict(bank, arrays, checkpoint=None):
    return nn3_infer.predict(bank, arrays['embeddings'], Path(checkpoint))
