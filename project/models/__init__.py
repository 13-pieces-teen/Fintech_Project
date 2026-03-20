from .lstm import LSTMModel, GRUModel
from .tcn import TCNModel
from .transformer import TransformerModel
from .patchtst import PatchTSTModel
from .mst import MarketStructureTransformer

MODEL_REGISTRY = {
    "lstm": LSTMModel,
    "gru": GRUModel,
    "tcn": TCNModel,
    "transformer": TransformerModel,
    "patchtst": PatchTSTModel,
    "mst": MarketStructureTransformer,
}


def build_model(cfg):
    """Instantiate a model from config."""
    cls = MODEL_REGISTRY[cfg.model.name]
    return cls(cfg.model)
