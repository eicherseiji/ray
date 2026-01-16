import ray.data.preprocessors as preprocessors_module
from ray.anyscale.data.preprocessors import TurboPreprocessor

# Store original reference before potential patching
_OriginalStandardScaler = preprocessors_module.StandardScaler


class StandardScaler(_OriginalStandardScaler, TurboPreprocessor):
    pass
