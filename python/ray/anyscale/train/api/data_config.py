from typing import Dict, List, Literal, Optional, Union

from ray.anyscale.data.checkpoint.interfaces import TrainingIngestCheckpointConfig
from ray.data import ExecutionOptions
from ray.train._internal.data_config import DataConfig as RayDataConfig


# Alias for the TrainingIngestCheckpointConfig to expose to users.
DatasetCheckpointConfig = TrainingIngestCheckpointConfig


class DataConfig(RayDataConfig):
    """Configuration for using Ray Data for training data ingestion."""

    def __init__(
        self,
        datasets_to_split: Union[Literal["all"], List[str]] = "all",
        execution_options: Optional[ExecutionOptions] = None,
        enable_shard_locality: bool = True,
        dataset_checkpoint_configs: Optional[Dict[str, DatasetCheckpointConfig]] = None,
    ):
        """Construct a DataConfig.

        Args:
            datasets_to_split: Specifies which datasets should be split among workers.
                Can be set to "all" or a list of dataset names. Defaults to "all",
                i.e. split all datasets.
            execution_options: The execution options to pass to Ray Data. By default,
                the options will be optimized for data ingest. When overriding this,
                base your options off of `DataConfig.default_ingest_options()`.
            enable_shard_locality: If true, when sharding the datasets across Train
                workers, locality will be considered to minimize cross-node data transfer.
                This is on by default.
            dataset_checkpoint_configs: A dictionary of dataset names to
                dataset checkpoint configs. Providing the configs will enable
                checkpointing iterator state and mid-epoch resumption for the
                specified datasets.
        """
        super().__init__(
            datasets_to_split=datasets_to_split,
            execution_options=execution_options,
            enable_shard_locality=enable_shard_locality,
        )
        self.dataset_checkpoint_configs: Dict[str, DatasetCheckpointConfig] = (
            dataset_checkpoint_configs or {}
        )
