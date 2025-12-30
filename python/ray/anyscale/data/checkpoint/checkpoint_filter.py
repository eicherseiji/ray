import logging

import numpy
import pyarrow
import pyarrow.compute as pc

import ray
from ray.anyscale.data.checkpoint.interfaces import CheckpointConfig
from ray.anyscale.data.checkpoint.util import (
    GeneratedIdFieldIndex,
    get_struct_field_index,
    FILE_NAME_FIELD,
    GENERATED_ID_FIELD_MAPPING,
    PATH_PREFIX_FIELD,
    NUM_FRAGMENTS_FIELD,
    FRAGMENT_FIELD,
    NUM_ROWS_FIELD,
    ROW_ID_FIELD,
    CHECKPOINTED_GENERATED_ID_COLUMN_TABLE_SCHEMA,
    CHECKPOINTED_FRAGMENT_TYPE,
    CHECKPOINTED_FILE_COLUMN_NAME,
    CHECKPOINTED_FILE_FRAGMENT_ID_FIELD,
    CHECKPOINTED_FILE_FRAGMENT_NUM_ROWS_FIELD,
    CHECKPOINTED_FILE_FRAGMENT_NUM_CHECKPOINTED_ROWS_FIELD,
    CHECKPOINTED_FILE_FRAGMENT_CHECKPOINTED_ROW_IDS_FIELD,
    CHECKPOINTED_FILE_FRAGMENTS_NUM_FRAGMENTS_FIELD,
    CHECKPOINTED_FILE_FULLY_CHECKPOINTED_FIELD,
    CHECKPOINTED_FILE_FRAGMENTS_INFO_FIELD,
)
from ray.data.block import Block, BlockMetadata, Schema
from ray.data.checkpoint.checkpoint_filter import (
    CheckpointFilter as OSSCheckpointFilter,
    CheckpointLoader,
    BatchBasedCheckpointFilter as OSSBatchBasedCheckpointFilter,
)
from ray.data.context import ShuffleStrategy
from ray.types import ObjectRef

logger = logging.getLogger(__name__)


class CheckpointFilter(OSSCheckpointFilter):
    """Extends OSS CheckpointFilter with generated ID column support."""

    def __init__(self, config: CheckpointConfig):
        super().__init__(config)
        self.generated_id_column = self.ckpt_config.generated_id_column


class GeneratedIdColumnCheckpointLoader(CheckpointLoader):
    """Loader for generated ID columns."""

    def _extract_grouping_fields(self, batch: pyarrow.Table) -> pyarrow.Table:
        """Extract the fields needed for grouping from the checkpoint data.

        Args:
            batch: PyArrow table containing checkpoint data with ID column

        Returns:
            PyArrow table with extracted grouping fields and ID column
        """
        id_col: pyarrow.ChunkedArray = batch[self.id_column]

        # Extract path_prefix
        path_prefix_idx: int = get_struct_field_index(id_col, PATH_PREFIX_FIELD)
        path_prefix = pc.struct_field(id_col, [path_prefix_idx])

        # Extract file_name
        file_name_idx: int = get_struct_field_index(id_col, FILE_NAME_FIELD)
        file_name = pc.struct_field(id_col, [file_name_idx])

        return pyarrow.Table.from_arrays(
            [
                path_prefix.cast(pyarrow.large_string()),
                file_name.cast(pyarrow.large_string()),
                batch[self.id_column],
            ],
            names=[
                PATH_PREFIX_FIELD,
                FILE_NAME_FIELD,
                self.id_column,
            ],
        )

    def _process_file_group(self, file_group_batch: pyarrow.Table) -> pyarrow.Table:
        """Process a single group (file) to create a single row with
        CHECKPOINTED_GENERATED_ID_COLUMN_TABLE_SCHEMA.

        Args:
            file_group_batch: PyArrow table containing rows from a single group,
                i.e.: path_prefix/file_name

        Returns:
            PyArrow table with 1 row containing the CHECKPOINTED_GENERATED_ID_COLUMN_TABLE_SCHEMA
        """
        # Get file path information
        path_prefix = file_group_batch[PATH_PREFIX_FIELD][0].as_py()
        file_name = file_group_batch[FILE_NAME_FIELD][0].as_py()
        file_path = f"{path_prefix}/{file_name}"

        # Extract all fields from the generated ID column
        id_columns = file_group_batch[self.id_column]

        # Extract NUM_FRAGMENTS_FIELD (should be the same for all rows in a file)
        num_fragments_field_idx = get_struct_field_index(
            id_columns[0], NUM_FRAGMENTS_FIELD
        )
        num_fragments = pc.struct_field(id_columns, [num_fragments_field_idx])[
            0
        ].as_py()

        # Extract FRAGMENT_FIELD for all rows
        fragment_field_idx = get_struct_field_index(id_columns[0], FRAGMENT_FIELD)
        fragments_array = pc.struct_field(id_columns, [fragment_field_idx])

        # Extract NUM_ROWS_FIELD for all rows
        num_rows_field_idx = get_struct_field_index(id_columns[0], NUM_ROWS_FIELD)
        num_rows_array = pc.struct_field(id_columns, [num_rows_field_idx])

        # Extract ROW_ID_FIELD for all rows
        row_id_field_idx = get_struct_field_index(id_columns[0], ROW_ID_FIELD)
        row_ids_array = pc.struct_field(id_columns, [row_id_field_idx])

        # Group by fragment using PyArrow operations
        # Create a table with fragment and row_id columns for grouping
        FRAGMENT_COLUMN_NAME = "fragment"
        ROW_ID_COLUMN_NAME = "row_id"
        NUM_ROWS_COLUMN_NAME = "num_rows"
        fragment_table = pyarrow.table(
            {
                FRAGMENT_COLUMN_NAME: fragments_array,
                ROW_ID_COLUMN_NAME: row_ids_array,
                NUM_ROWS_COLUMN_NAME: num_rows_array,
            }
        )

        # Group by fragment and aggregate row_ids
        grouped = fragment_table.group_by(FRAGMENT_COLUMN_NAME).aggregate(
            [
                (ROW_ID_COLUMN_NAME, "list"),
                # num_rows should be the same for all rows in a fragment, use min
                (NUM_ROWS_COLUMN_NAME, "min"),
            ]
        )

        # Process fragments
        fragments_array = grouped[FRAGMENT_COLUMN_NAME]
        row_ids_lists = grouped[
            f"{ROW_ID_COLUMN_NAME}_list"
        ]  # PyArrow adds "_list" suffix
        num_rows_array = grouped[
            f"{NUM_ROWS_COLUMN_NAME}_min"
        ]  # PyArrow adds "_min" suffix

        # Calculate checkpointed row counts (length of each row_ids list)
        checkpointed_row_counts = pc.cast(
            pc.list_value_length(row_ids_lists), pyarrow.int32()
        )

        # Determine which fragments are fully checkpointed
        fully_checkpointed_mask = pc.equal(checkpointed_row_counts, num_rows_array)
        num_fragments_fully_checkpointed = pc.sum(fully_checkpointed_mask).as_py()

        # Create boolean arrays for checkpointed row IDs
        checkpointed_row_ids_arrays = []

        for i in range(len(grouped)):
            num_rows = num_rows_array[i].as_py()
            checkpointed_row_count = checkpointed_row_counts[i].as_py()
            row_ids_list = row_ids_lists[i]

            if checkpointed_row_count == num_rows:
                # All rows checkpointed - use empty list for efficiency
                checkpointed_row_ids_col = pyarrow.array(
                    [[]], type=pyarrow.large_list(pyarrow.bool_())
                )
            else:
                # Create boolean array where True = checkpointed, False = not checkpointed
                row_indices = pyarrow.array(
                    numpy.arange(num_rows), type=pyarrow.int32()
                )
                # Convert list scalar to array and sort for pc.is_in requirement
                row_ids_array = row_ids_list.values
                sorted_indices = pc.sort_indices(row_ids_array)
                sorted_row_ids = row_ids_array.take(sorted_indices)
                boolean_array = pc.is_in(row_indices, sorted_row_ids)

                # Wrap as LargeList<bool>: offsets [0, N]
                offsets = pyarrow.array([0, len(boolean_array)], type=pyarrow.int64())
                checkpointed_row_ids_col = pyarrow.LargeListArray.from_arrays(
                    offsets, boolean_array
                )

            checkpointed_row_ids_arrays.append(checkpointed_row_ids_col)

        # Create fragment structs
        # Create a proper LargeListArray from the checkpointed_row_ids_arrays
        if checkpointed_row_ids_arrays:
            # Concatenate all the LargeListArray objects into a single array
            checkpointed_row_ids_array = pyarrow.concat_arrays(
                checkpointed_row_ids_arrays
            )
        else:
            # Create an empty LargeListArray with the correct type
            checkpointed_row_ids_array = pyarrow.array(
                [[]], type=pyarrow.large_list(pyarrow.bool_())
            )

        fragment_structs = pyarrow.StructArray.from_arrays(
            [
                # fragment_id
                pc.cast(fragments_array.combine_chunks(), pyarrow.int32()),
                # num_rows
                pc.cast(num_rows_array.combine_chunks(), pyarrow.int32()),
                # num_checkpointed_rows
                checkpointed_row_counts.combine_chunks(),
                # checkpointed_row_ids
                checkpointed_row_ids_array,
            ],
            fields=[
                pyarrow.field(
                    CHECKPOINTED_FILE_FRAGMENT_ID_FIELD, pyarrow.int32(), nullable=False
                ),
                pyarrow.field(
                    CHECKPOINTED_FILE_FRAGMENT_NUM_ROWS_FIELD,
                    pyarrow.int32(),
                    nullable=False,
                ),
                pyarrow.field(
                    CHECKPOINTED_FILE_FRAGMENT_NUM_CHECKPOINTED_ROWS_FIELD,
                    pyarrow.int32(),
                    nullable=False,
                ),
                pyarrow.field(
                    CHECKPOINTED_FILE_FRAGMENT_CHECKPOINTED_ROW_IDS_FIELD,
                    pyarrow.large_list(pyarrow.bool_()),
                    nullable=True,
                ),
            ],
        )

        # Create the checkpointed fragment column with CHECKPOINTED_FILE_FRAGMENTS_TYPE
        # Create a proper LargeListArray from the fragment structs
        if len(fragment_structs) > 0:
            # Create offsets for the list: [0, len(fragment_structs)]
            offsets = pyarrow.array([0, len(fragment_structs)], type=pyarrow.int64())
            fragments_list = pyarrow.LargeListArray.from_arrays(
                offsets, fragment_structs
            )
        else:
            # Empty list
            fragments_list = pyarrow.array(
                [[]], type=pyarrow.large_list(CHECKPOINTED_FRAGMENT_TYPE)
            )
        fully_checkpointed = num_fragments_fully_checkpointed == num_fragments
        checkpointed_fragment_col = pyarrow.StructArray.from_arrays(
            [
                # Number of fragments for this file
                pyarrow.array([len(fragment_structs)], type=pyarrow.int32()),
                # Whether all fragments in the file are checkpointed
                pyarrow.array([fully_checkpointed], type=pyarrow.bool_()),
                # List of checkpointed fragment structs
                fragments_list,  # already a proper LargeListArray
            ],
            fields=[
                pyarrow.field(
                    CHECKPOINTED_FILE_FRAGMENTS_NUM_FRAGMENTS_FIELD,
                    pyarrow.int32(),
                    nullable=False,
                ),
                pyarrow.field(
                    CHECKPOINTED_FILE_FULLY_CHECKPOINTED_FIELD,
                    pyarrow.bool_(),
                    nullable=False,
                ),
                pyarrow.field(
                    CHECKPOINTED_FILE_FRAGMENTS_INFO_FIELD,
                    pyarrow.large_list(
                        pyarrow.struct(
                            [
                                pyarrow.field(
                                    CHECKPOINTED_FILE_FRAGMENT_ID_FIELD,
                                    pyarrow.int32(),
                                    nullable=False,
                                ),
                                pyarrow.field(
                                    CHECKPOINTED_FILE_FRAGMENT_NUM_ROWS_FIELD,
                                    pyarrow.int32(),
                                    nullable=False,
                                ),
                                pyarrow.field(
                                    CHECKPOINTED_FILE_FRAGMENT_NUM_CHECKPOINTED_ROWS_FIELD,
                                    pyarrow.int32(),
                                    nullable=False,
                                ),
                                pyarrow.field(
                                    CHECKPOINTED_FILE_FRAGMENT_CHECKPOINTED_ROW_IDS_FIELD,
                                    pyarrow.large_list(pyarrow.bool_()),
                                    nullable=True,
                                ),
                            ]
                        )
                    ),
                    nullable=True,
                ),
            ],
        )

        logger.debug(
            "Group processing stats - File: %s, Num Fragments: %d, Num Fragments Fully Checkpointed: %d, Fully Checkpointed: %s",
            file_path,
            num_fragments,
            num_fragments_fully_checkpointed,
            fully_checkpointed,
        )

        # Return exactly 1 row per group with CHECKPOINTED_GENERATED_ID_COLUMN_TABLE_SCHEMA
        return pyarrow.Table.from_arrays(
            [
                # Checkpointed file path
                pyarrow.array([file_path]),
                # Checkpointed file fragments
                checkpointed_fragment_col,
            ],
            schema=CHECKPOINTED_GENERATED_ID_COLUMN_TABLE_SCHEMA,
        )

    def _preprocess_data_pipeline(
        self, checkpoint_ds: ray.data.Dataset
    ) -> ray.data.Dataset:
        """Pre-process generated ID columns with grouping and sorting.

        Args:
            checkpoint_ds: The checkpoint dataset to pre-process

        Returns:
            The pre-processed checkpoint dataset
        """
        # Map batches to extract grouping fields.
        checkpoint_ds = checkpoint_ds.map_batches(
            self._extract_grouping_fields,
            batch_format="pyarrow",
            batch_size=None,
        )
        # Set shuffle strategy to hash shuffle for groupby
        checkpoint_ds.context._shuffle_strategy = ShuffleStrategy.HASH_SHUFFLE

        # Group by path_prefix and file_name
        checkpoint_ds = checkpoint_ds.groupby(
            [
                GENERATED_ID_FIELD_MAPPING[GeneratedIdFieldIndex.PATH_PREFIX],
                GENERATED_ID_FIELD_MAPPING[GeneratedIdFieldIndex.FILE_NAME],
            ]
        )

        # Process each group and reduce to a single row per group
        checkpoint_ds = checkpoint_ds.map_groups(
            self._process_file_group, batch_format="pyarrow"
        )

        # Sort by the file path
        checkpoint_ds = checkpoint_ds.sort(CHECKPOINTED_FILE_COLUMN_NAME)

        return checkpoint_ds

    def _validate_loaded_checkpoint(
        self, schema: Schema, metadata: BlockMetadata
    ) -> None:
        """Validate schema for generated ID columns."""
        if metadata.num_rows > 0:
            assert (
                schema == CHECKPOINTED_GENERATED_ID_COLUMN_TABLE_SCHEMA
            ), f"Schema mismatch: {schema} != {CHECKPOINTED_GENERATED_ID_COLUMN_TABLE_SCHEMA}"


class BatchBasedCheckpointFilter(OSSBatchBasedCheckpointFilter, CheckpointFilter):
    """CheckpointFilter for batch-based backends with generated ID column support.

    Extends OSS BatchBasedCheckpointFilter with support for generated ID columns.
    """

    def load_checkpoint(self) -> ObjectRef[Block]:
        """Load checkpointed ids as a sorted block.

        Returns:
            ObjectRef[Block]: ObjectRef to the checkpointed IDs block.
        """
        if self.generated_id_column:
            loader = GeneratedIdColumnCheckpointLoader(
                checkpoint_path=self.checkpoint_path,
                filesystem=self.filesystem,
                id_column=self.id_column,
                checkpoint_path_partition_filter=self.ckpt_config.checkpoint_path_partition_filter,
            )
            return loader.load_checkpoint()
        # For regular ID columns, use the parent implementation
        return super().load_checkpoint()

    def filter_rows_for_block(
        self,
        block: Block,
        checkpointed_ids: Block,
    ) -> Block:
        """For the given block, filter out rows that have already
        been checkpointed, and return the resulting block.

        Args:
            block: The input block to filter.
            checkpointed_ids: A block containing IDs of all rows that have
                been checkpointed.
        Returns:
            A new block with rows that have not been checkpointed.
        """
        if len(checkpointed_ids) == 0 or len(block) == 0:
            return block

        if self.generated_id_column:
            # For generated id column, filter out rows that have already been checkpointed
            # in the read task.
            return block

        # For regular ID columns, use the parent implementation
        return super().filter_rows_for_block(block, checkpointed_ids)
