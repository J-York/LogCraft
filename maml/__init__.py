import sys
sys.path.append("../")
from maml.maml_preprocess import FeatureExtractor
from maml.maml_utils import set_device
from maml.maml_utils import seed_everything
# from models.DeepLog.deeploglizer.models.base_model import Embedder
from models import LSTM