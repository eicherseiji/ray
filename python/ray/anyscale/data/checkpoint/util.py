from typing import Union, Iterable, Optional
import os
import urllib.parse
from enum import IntEnum
from dataclasses import dataclass

import pyarrow as pa
import pyarrow.compute as pc
import numpy as np

from ray.anyscale.data.checkpoint.interfaces import (
    CheckpointConfig,
)
from ray.data._internal.execution.interfaces.task_context import TaskContext
from ray.data.block import Block, BlockAccessor, DataBatch
from ray.data._internal.arrow_block import ArrowBlockAccessor


# Checkpoint keyword argument name
CHECKPOINTED_IDS_KWARG_NAME = "checkpointed_ids"


#
# Schema definition for `generated_id_column`
#

# Generated ID string format
GENERATED_ID_STRING_FORMAT = "{path_prefix}/{file_name}/row_group={row_group}/num_rows={num_rows}/row_id={row_id}"

# Field names for `generated_id_column` schema
PATH_PREFIX_FIELD = "path_prefix"
FILE_NAME_FIELD = "file_name"
ROW_GROUP_FIELD = "row_group"
NUM_ROWS_FIELD = "num_rows"
ROW_ID_FIELD = "row_id"

# Field types for `generated_id_column` schema
# Note: Dictionary encoding is used to reduce the size of the generated id column
# because of low cardinality of the fields.
GENERATED_ID_COLUMN_FIELDS = {
    PATH_PREFIX_FIELD: pa.dictionary(pa.int32(), pa.string()),
    FILE_NAME_FIELD: pa.dictionary(pa.int32(), pa.string()),
    ROW_GROUP_FIELD: pa.dictionary(pa.int32(), pa.int32()),
    NUM_ROWS_FIELD: pa.dictionary(pa.int32(), pa.int32()),
    ROW_ID_FIELD: pa.int32(),
}

# Fields order
GENERATED_ID_COLUMN_FIELD_NAMES = [
    PATH_PREFIX_FIELD,
    FILE_NAME_FIELD,
    ROW_GROUP_FIELD,
    NUM_ROWS_FIELD,
    ROW_ID_FIELD,
]


# Field indices for GENERATED_ID_COLUMN_TYPE struct
class GeneratedIdFieldIndex(IntEnum):
    PATH_PREFIX = 0
    FILE_NAME = 1
    ROW_GROUP = 2
    NUM_ROWS = 3
    ROW_ID = 4


# Explicit mapping from enum to field names
GENERATED_ID_FIELD_MAPPING = {
    GeneratedIdFieldIndex.PATH_PREFIX: PATH_PREFIX_FIELD,
    GeneratedIdFieldIndex.FILE_NAME: FILE_NAME_FIELD,
    GeneratedIdFieldIndex.ROW_GROUP: ROW_GROUP_FIELD,
    GeneratedIdFieldIndex.NUM_ROWS: NUM_ROWS_FIELD,
    GeneratedIdFieldIndex.ROW_ID: ROW_ID_FIELD,
}

# Reverse mapping from field names to enum
GENERATED_ID_FIELD_INDICES = {
    field_name: enum_value
    for enum_value, field_name in GENERATED_ID_FIELD_MAPPING.items()
}

# PyArrow struct type for `generated_id_column`
GENERATED_ID_COLUMN_TYPE = pa.struct(
    [
        pa.field(field_name, GENERATED_ID_COLUMN_FIELDS[field_name])
        for field_name in GENERATED_ID_COLUMN_FIELD_NAMES
    ]
)

#
# Generated ID column Checkpoint table schema and type definitions
#

# Checkpointed fragment column name
CHECKPOINTED_FRAGMENT_COLUMN_NAME = "checkpointed_fragment"

# Checkpointed row count column name
CHECKPOINTED_ROW_COUNT_COLUMN_NAME = "checkpointed_row_count"

# Checkpointed row ids column name
CHECKPOINTED_ROW_IDS_COLUMN_NAME = "checkpointed_row_ids"

# Schema for the checkpointed generated id column table
CHECKPOINTED_GENERATED_ID_COLUMN_TABLE_SCHEMA = pa.schema(
    [
        pa.field(CHECKPOINTED_FRAGMENT_COLUMN_NAME, pa.string()),
        pa.field(NUM_ROWS_FIELD, pa.int32()),
        pa.field(CHECKPOINTED_ROW_COUNT_COLUMN_NAME, pa.int32()),
        pa.field(CHECKPOINTED_ROW_IDS_COLUMN_NAME, pa.list_(pa.bool_())),
    ]
)


def filter_checkpointed_rows_for_blocks(
    blocks: Iterable[Block],
    task_context: TaskContext,
    checkpoint_config: CheckpointConfig,
) -> Iterable[Block]:
    """For each block, filter rows that have already been checkpointed
    and yield the resulting block."""
    if checkpoint_config.is_batch_based():
        from ray.anyscale.data.checkpoint.checkpoint_filter import (
            BatchBasedCheckpointFilter,
        )

        ckpt_filter = BatchBasedCheckpointFilter(checkpoint_config)
        checkpointed_ids = task_context.kwargs[CHECKPOINTED_IDS_KWARG_NAME]

        def filter_fn(block: Block) -> Block:
            return ckpt_filter.filter_rows_for_block(
                block=block,
                checkpointed_ids=checkpointed_ids,
            )

    else:
        from ray.anyscale.data.checkpoint.checkpoint_filter import (
            RowBasedCheckpointFilter,
        )

        ckpt_filter = RowBasedCheckpointFilter.create(checkpoint_config)

        def filter_fn(block: Block) -> Block:
            return ckpt_filter.filter_rows_for_block(block)

    for block in blocks:
        filtered_block = filter_fn(block)
        ba = BlockAccessor.for_block(filtered_block)
        if ba.num_rows() > 0:
            yield filtered_block


def filter_checkpointed_rows_for_batches(
    batches: Iterable[DataBatch],
    task_context: TaskContext,
    checkpoint_config: CheckpointConfig,
) -> Iterable[DataBatch]:
    """For each batch, filter rows that have already been checkpointed
    and yield the resulting batches."""
    if checkpoint_config.is_batch_based():
        from ray.anyscale.data.checkpoint.checkpoint_filter import (
            BatchBasedCheckpointFilter,
        )

        ckpt_filter = BatchBasedCheckpointFilter(checkpoint_config)
        checkpointed_ids = task_context.kwargs[CHECKPOINTED_IDS_KWARG_NAME]

        def filter_fn(batch: DataBatch) -> DataBatch:
            return ckpt_filter.filter_rows_for_batch(
                batch=batch,
                checkpointed_ids=checkpointed_ids,
            )

    else:
        from ray.anyscale.data.checkpoint.checkpoint_filter import (
            RowBasedCheckpointFilter,
        )

        ckpt_filter = RowBasedCheckpointFilter.create(checkpoint_config)

        def filter_fn(batch: DataBatch) -> DataBatch:
            return ckpt_filter.filter_rows_for_batch(batch)

    for batch in batches:
        filtered_batch = filter_fn(batch)
        yield filtered_batch


@dataclass
class CheckpointedFragmentInfo:
    """Dataclass for checkpointed fragment."""

    # The Parquet fragment
    fragment: pa.dataset.ParquetFileFragment
    # The row group index
    row_group_idx: int
    # The number of rows in the fragment
    num_rows: int
    # Whether all rows in the fragment are checkpointed
    fully_checkpointed: bool
    # The row IDs of the checkpointed rows in sorted order
    checkpointed_row_ids: Optional[
        pa.ListArray
    ]  # ListArray containing lists of row IDs per fragment
    # The number of checkpointed rows in the fragment
    checkpointed_row_count: int


def _create_empty_checkpointed_fragment_info(
    fragment: pa.dataset.ParquetFileFragment,
    row_group_idx: int,
) -> "CheckpointedFragmentInfo":
    """Create a CheckpointedFragmentInfo for a empty checkpointed fragment.

    Args:
        fragment: The Parquet fragment.
        row_group: Row group index.

    Returns:
        CheckpointedFragmentInfo indicating the fragment is fully checkpointed.
    """
    return CheckpointedFragmentInfo(
        fragment=fragment,
        row_group_idx=row_group_idx,
        num_rows=fragment.metadata.num_rows,  # Use actual fragment size
        fully_checkpointed=False,  # Empty checkpointed fragment
        checkpointed_row_ids=None,
        checkpointed_row_count=0,  # No rows checkpointed
    )


def get_checkpointed_fragment_info(
    fragment: pa.dataset.ParquetFileFragment,
    row_group_idx: int,
    checkpointed_ids: Block,
) -> CheckpointedFragmentInfo:
    """
    Get the checkpointed row IDs of a fragment from checkpointed_ids Block.
    Args:
        fragment: Parquet fragment to check.
        row_group_idx: Row group index.
        checkpointed_ids: Block containing checkpointed IDs.

    Returns:
        CheckpointedFragmentInfo: Checkpointed row IDs of the fragment.
    """
    accessor = ArrowBlockAccessor.for_block(checkpointed_ids)
    checkpointed_ids_table = accessor.to_arrow()
    if checkpointed_ids_table.num_rows == 0:
        # No checkpointed IDs, return empty checkpointed fragment info
        return _create_empty_checkpointed_fragment_info(fragment, row_group_idx)

    # Extract file path from fragment path
    fragment_path = fragment.path

    # Create the search key: /path/to/file/row_group=<row_group>
    search_key = f"{fragment_path}/row_group={row_group_idx}"

    # Use binary search to find the fragment in the checkpointed_ids table.
    checkpoint_fragment_col = checkpointed_ids_table[CHECKPOINTED_FRAGMENT_COLUMN_NAME]
    checkpoint_fragment_array = checkpoint_fragment_col.to_numpy()
    insert_idx = np.searchsorted(checkpoint_fragment_array, search_key, side="left")

    if (
        insert_idx >= len(checkpoint_fragment_array)
        or checkpoint_fragment_array[insert_idx] != search_key
    ):
        # Fragment not found, return empty checkpointed fragment info
        return _create_empty_checkpointed_fragment_info(fragment, row_group_idx)

    # Fragment found. Check if all the rows in the fragment are checkpointed.
    checkpointed_ids_fragment = checkpointed_ids_table.take([insert_idx])

    # Extract the checkpointed fragment row count
    checkpointed_row_count = checkpointed_ids_fragment[
        CHECKPOINTED_ROW_COUNT_COLUMN_NAME
    ][0].as_py()

    # Extract the fragment row count
    fragment_row_count = checkpointed_ids_fragment[NUM_ROWS_FIELD][0].as_py()
    assert fragment_row_count == fragment.metadata.row_group(row_group_idx).num_rows, (
        f"Fragment row count {fragment_row_count} is not equal to the fragment metadata "
        f"num_rows {fragment.metadata.row_group(row_group_idx).num_rows}"
    )

    if checkpointed_row_count == fragment_row_count:
        # All rows in the fragment are checkpointed
        fully_checkpointed = True

        # When all rows are checkpointed, the checkpointed_row_ids should be an empty list.
        checkpointed_row_ids_col = checkpointed_ids_fragment[
            CHECKPOINTED_ROW_IDS_COLUMN_NAME
        ]
        assert (
            len(checkpointed_row_ids_col) > 0 and len(checkpointed_row_ids_col[0]) == 0
        ), f"All rows are checkpointed, so checkpointed_row_ids {checkpointed_row_ids_col} should be an empty list"
        final_checkpointed_row_ids = checkpointed_row_ids_col
    else:
        # Some rows in the fragment are not checkpointed
        assert checkpointed_row_count < fragment_row_count, (
            f"Checkpointed row count {checkpointed_row_count} is greater than "
            f"fragment row count {fragment_row_count}"
        )
        fully_checkpointed = False
        final_checkpointed_row_ids = checkpointed_ids_fragment[
            CHECKPOINTED_ROW_IDS_COLUMN_NAME
        ].combine_chunks()

    return CheckpointedFragmentInfo(
        fragment=fragment,
        row_group_idx=row_group_idx,
        num_rows=fragment_row_count,
        fully_checkpointed=fully_checkpointed,
        checkpointed_row_ids=final_checkpointed_row_ids,
        checkpointed_row_count=checkpointed_row_count,
    )


def exclude_checkpointed_rows(
    table: pa.Table,
    checkpointed_fragment_info: CheckpointedFragmentInfo,
    current_row_offset: int,
    current_num_rows: int,
) -> pa.Table:
    """Exclude checkpointed rows from the table.
    Args:
        table: The table to exclude checkpointed rows from.
        checkpointed_fragment_info: The information of the checkpointed fragment.
        current_row_offset: The current row offset for the row IDs.
        current_num_rows: The current number of rows in the table.

    Returns:
        The table with checkpointed rows excluded.
    """
    # checkpointed_row_ids is a ListArray containing a boolean array.
    # The boolean array has True at checkpointed row positions, False at
    # non-checkpointed.
    checkpointed_row_ids = checkpointed_fragment_info.checkpointed_row_ids

    # If no rows are checkpointed, return the table as-is.
    if checkpointed_row_ids is None:
        assert (
            not checkpointed_fragment_info.fully_checkpointed
        ), "Checkpointed row ids is None, intended to be empty checkpointed fragment"
        return table

    # If checkpointed_row_ids is empty list, all rows are checkpointed.
    if len(checkpointed_row_ids) > 0 and len(checkpointed_row_ids[0]) == 0:
        assert (
            checkpointed_fragment_info.fully_checkpointed
        ), "Checkpointed row ids is empty list, intended to be fully checkpointed fragment"
        # Return empty table - all rows checkpointed
        return pa.table({})

    # Extract the first (and only) list from the ListArray
    checkpointed_row_ids_list = checkpointed_row_ids[0]

    # Convert the ListScalar to a regular boolean array
    checkpointed_row_ids_bool = checkpointed_row_ids_list.values

    # Create row indices for the current batch
    row_indices = np.arange(current_row_offset, current_row_offset + current_num_rows)

    # Check if the requested row range is completely inside the boolean array bounds.
    assert current_row_offset + current_num_rows <= len(checkpointed_row_ids_bool), (
        f"Current row offset {current_row_offset} + current num rows {current_num_rows} "
        f"is greater than the length of the boolean array {len(checkpointed_row_ids_bool)}"
    )

    # Extract the relevant portion of the boolean array.
    # Use take to get the boolean values for the current row range.
    # Set boundscheck=False to handle out-of-bounds gracefully.
    relevant_bools = pc.take(checkpointed_row_ids_bool, row_indices, boundscheck=False)

    # Invert the boolean array: True (checkpointed) becomes False (exclude).
    # False (not checkpointed) becomes True (keep).
    mask = pc.invert(relevant_bools)

    # Filter the table.
    return table.filter(mask)


def get_generated_id_column(
    path: str,
    row_group_idx: int,
    total_num_rows: int,
    current_row_offset: int,
    current_num_rows: int,
) -> pa.Array:
    """Helper function to get `generated_id_column`.

    Args:
        path: Full path to the file
        row_group_idx: Row group index
        total_num_rows: Total number of rows in the file row group
        current_row_offset: Current row offset for sequential IDs
        current_num_rows: Number of rows in the current batch

    Returns:
        PyArrow Array with `generated_id_column`
    """
    path_prefix = os.path.dirname(path)
    file_name = os.path.basename(path)
    row_ids = np.arange(current_row_offset, current_row_offset + current_num_rows)

    def create_array_for_field(field_name: str) -> pa.Array:
        if field_name == PATH_PREFIX_FIELD:
            array = pa.nulls(current_num_rows, type=pa.string())
            filled_array = pc.fill_null(array, path_prefix)
            return pc.dictionary_encode(filled_array)
        elif field_name == FILE_NAME_FIELD:
            array = pa.nulls(current_num_rows, type=pa.string())
            filled_array = pc.fill_null(array, file_name)
            return pc.dictionary_encode(filled_array)
        elif field_name == ROW_GROUP_FIELD:
            array = pa.nulls(current_num_rows, type=pa.int32())
            filled_array = pc.fill_null(array, row_group_idx)
            return pc.dictionary_encode(filled_array)
        elif field_name == NUM_ROWS_FIELD:
            array = pa.nulls(current_num_rows, type=pa.int32())
            filled_array = pc.fill_null(array, total_num_rows)
            return pc.dictionary_encode(filled_array)
        elif field_name == ROW_ID_FIELD:
            return pa.array(row_ids, type=pa.int32())
        else:
            raise ValueError(f"Unknown field name: {field_name}")

    arrays = [
        create_array_for_field(field_name)
        for field_name in GENERATED_ID_COLUMN_FIELD_NAMES
    ]
    return pa.StructArray.from_arrays(arrays, names=GENERATED_ID_COLUMN_FIELD_NAMES)


def normalize_id(id: Union[dict, int]) -> str:
    """Normalize an ID for use as a filename.

    Args:
        id: The ID (dict or int)

    Returns:
        A normalized string safe for use as a filename
    """
    if isinstance(id, int):
        return str(id)

    # Generate normalized ID using the format string.
    normalized_id = GENERATED_ID_STRING_FORMAT.format(**id)

    # Quote the normalized ID to make it safe for use as a filename.
    return urllib.parse.quote(normalized_id, safe="")


def get_struct_field_index(
    struct_array: Union[pa.Array, pa.ChunkedArray], field_name: str
) -> int:
    """
    Get the index of a field in a struct array, handling both regular Arrays and ChunkedArrays.

    Note:
    Given struct array in Arrow format is written as Parquet files and then read back
    as Arrow format, the fields order in struct is not guaranteed. So we need to look up
    explicitly with field name.

    Args:
        struct_array: The struct array to get the field index from
        field_name: The name of the field to find

    Returns:
        The index of the field in the struct

    Raises:
        ValueError: If the field is not found in the struct
    """
    if isinstance(struct_array, pa.ChunkedArray):
        # For ChunkedArrays, we need to get the type from the first chunk
        # since all chunks should have the same type
        struct_type = struct_array.chunks[0].type
    else:
        struct_type = struct_array.type

    field_index = struct_type.get_field_index(field_name)
    if field_index == -1:
        raise ValueError(f"Field '{field_name}' not found in struct type {struct_type}")

    return field_index
