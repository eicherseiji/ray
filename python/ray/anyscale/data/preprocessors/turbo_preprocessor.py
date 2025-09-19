from typing import TYPE_CHECKING

from ray.data.preprocessor import Preprocessor


if TYPE_CHECKING:
    from ray.data import Dataset


class TurboPreprocessor(Preprocessor):
    def __init__(self):
        super().__init__()
        self.is_chain = False

    def _fit_execute(self, dataset: "Dataset"):
        if self.is_chain:
            return
        return super()._fit_execute(dataset)
