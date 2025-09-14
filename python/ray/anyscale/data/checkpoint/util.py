from typing import Union, Iterable, Optional
import os
import urllib.parse
from enum import IntEnum
from dataclasses import dataclass

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset
import numpy as np
import logging
from ray.anyscale.data.checkpoint.interfaces import (
    CheckpointConfig,
)
from ray.data._internal.execution.interfaces.task_context import TaskContext
from ray.data.block import Block, BlockAccessor, DataBatch
from ray.data._internal.arrow_block import ArrowBlockAccessor


logger = logging.getLogger(__name__)


# Checkpoint keyword argument name
CHECKPOINTED_IDS_KWARG_NAME = "checkpointed_ids"


#
# Schema definition for `generated_id_column`
#

# Generated ID string format
GENERATED_ID_STRING_FORMAT = (
    "{path_prefix}/{file_name}/fragment={fragment}/num_rows={num_rows}/row_id={row_id}"
)

# Field names for `generated_id_column` schema
PATH_PREFIX_FIELD = "path_prefix"
FILE_NAME_FIELD = "file_name"
FRAGMENT_FIELD = "fragment"
NUM_FRAGMENTS_FIELD = "num_fragments"
NUM_ROWS_FIELD = "num_rows"
ROW_ID_FIELD = "row_id"

# Field types for `generated_id_column` schema
# Note: Dictionary encoding is used to reduce the size of the generated id column
# because of low cardinality of the fields.
GENERATED_ID_COLUMN_FIELDS = {
    # Path prefix for the file
    PATH_PREFIX_FIELD: pa.dictionary(pa.int32(), pa.string()),
    # File name
    FILE_NAME_FIELD: pa.dictionary(pa.int32(), pa.string()),
    # Fragment (chunk) index
    FRAGMENT_FIELD: pa.dictionary(pa.int32(), pa.int32()),
    # Total number of fragments in the file
    NUM_FRAGMENTS_FIELD: pa.dictionary(pa.int32(), pa.int32()),
    # Total number of rows in the file fragment
    NUM_ROWS_FIELD: pa.dictionary(pa.int32(), pa.int32()),
    # Row ID
    ROW_ID_FIELD: pa.int32(),
}

# Fields order
GENERATED_ID_COLUMN_FIELD_NAMES = [
    PATH_PREFIX_FIELD,
    FILE_NAME_FIELD,
    FRAGMENT_FIELD,
    NUM_FRAGMENTS_FIELD,
    NUM_ROWS_FIELD,
    ROW_ID_FIELD,
]


# Field indices for GENERATED_ID_COLUMN_TYPE struct
class GeneratedIdFieldIndex(IntEnum):
    PATH_PREFIX = 0
    FILE_NAME = 1
    FRAGMENT = 2
    NUM_FRAGMENTS = 3
    NUM_ROWS = 4
    ROW_ID = 5


# Explicit mapping from enum to field names
GENERATED_ID_FIELD_MAPPING = {
    GeneratedIdFieldIndex.PATH_PREFIX: PATH_PREFIX_FIELD,
    GeneratedIdFieldIndex.FILE_NAME: FILE_NAME_FIELD,
    GeneratedIdFieldIndex.FRAGMENT: FRAGMENT_FIELD,
    GeneratedIdFieldIndex.NUM_FRAGMENTS: NUM_FRAGMENTS_FIELD,
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
        pa.field(field_name, GENERATED_ID_COLUMN_FIELDS[field_name], nullable=False)
        for field_name in GENERATED_ID_COLUMN_FIELD_NAMES
    ]
)

#
# Generated ID column Checkpoint table schema and type definitions
#

# Checkpointed file column name
CHECKPOINTED_FILE_COLUMN_NAME = "checkpointed_file"

# Checkpointed file fragments column name
CHECKPOINTED_FILE_FRAGMENTS_COLUMN_NAME = "checkpointed_file_fragments"

# Schema for individual checkpointed fragment struct
CHECKPOINTED_FILE_FRAGMENT_ID_FIELD = "fragment_id"
CHECKPOINTED_FILE_FRAGMENT_NUM_ROWS_FIELD = "num_rows"
CHECKPOINTED_FILE_FRAGMENT_NUM_CHECKPOINTED_ROWS_FIELD = "num_checkpointed_rows"
CHECKPOINTED_FILE_FRAGMENT_CHECKPOINTED_ROW_IDS_FIELD = "checkpointed_row_ids"
CHECKPOINTED_FRAGMENT_TYPE = pa.struct(
    [
        # Fragment ID in this file
        pa.field(CHECKPOINTED_FILE_FRAGMENT_ID_FIELD, pa.int32(), nullable=False),
        # Total number of rows in this fragment
        pa.field(CHECKPOINTED_FILE_FRAGMENT_NUM_ROWS_FIELD, pa.int32(), nullable=False),
        # Number of checkpointed rows in this fragment
        pa.field(
            CHECKPOINTED_FILE_FRAGMENT_NUM_CHECKPOINTED_ROWS_FIELD,
            pa.int32(),
            nullable=False,
        ),
        # Boolean array of checkpointed rows (True = checkpointed)
        pa.field(
            CHECKPOINTED_FILE_FRAGMENT_CHECKPOINTED_ROW_IDS_FIELD,
            pa.large_list(pa.bool_()),
            nullable=True,
        ),
    ]
)

# Schema for checkpointed file fragments struct. This struct is passed in the
# file manifest to the readers.
CHECKPOINTED_FILE_FRAGMENTS_NUM_FRAGMENTS_FIELD = "num_fragments"
CHECKPOINTED_FILE_FRAGMENTS_FULLY_CHECKPOINTED_FIELD = "fully_checkpointed"
CHECKPOINTED_FILE_FRAGMENTS_FRAGMENTS_FIELD = "fragments"
CHECKPOINTED_FILE_FRAGMENTS_TYPE = pa.struct(
    [
        # Number of fragments for this file
        pa.field(
            CHECKPOINTED_FILE_FRAGMENTS_NUM_FRAGMENTS_FIELD,
            pa.int32(),
            nullable=False,
        ),
        # Whether all fragments in the file are checkpointed
        pa.field(
            CHECKPOINTED_FILE_FRAGMENTS_FULLY_CHECKPOINTED_FIELD,
            pa.bool_(),
            nullable=False,
        ),
        # List of checkpointed fragment structs
        pa.field(
            CHECKPOINTED_FILE_FRAGMENTS_FRAGMENTS_FIELD,
            pa.large_list(CHECKPOINTED_FRAGMENT_TYPE),
            nullable=True,
        ),
    ]
)

# Schema for the checkpointed generated id column table
CHECKPOINTED_GENERATED_ID_COLUMN_TABLE_SCHEMA = pa.schema(
    [
        pa.field(CHECKPOINTED_FILE_COLUMN_NAME, pa.string()),
        pa.field(
            CHECKPOINTED_FILE_FRAGMENTS_COLUMN_NAME,
            CHECKPOINTED_FILE_FRAGMENTS_TYPE,
        ),
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
        pa.Array
    ]  # PyArrow array of boolean values indicating checkpointed rows
    # The number of checkpointed rows in the fragment
    checkpointed_row_count: int


def _create_empty_checkpointed_fragment_info(
    fragment: pa.dataset.ParquetFileFragment,
    row_group_idx: int,
) -> "CheckpointedFragmentInfo":
    """Create a CheckpointedFragmentInfo for an empty checkpointed fragment.

    Args:
        fragment: The Parquet fragment.
        row_group_idx: Row group index.

    Returns:
        CheckpointedFragmentInfo indicating the fragment is fully checkpointed.
    """
    return CheckpointedFragmentInfo(
        fragment=fragment,
        row_group_idx=row_group_idx,
        num_rows=fragment.metadata.row_group(
            row_group_idx
        ).num_rows,  # Use actual fragment size
        fully_checkpointed=False,  # Empty checkpointed fragment
        checkpointed_row_ids=None,
        checkpointed_row_count=0,  # No rows checkpointed
    )


def get_checkpointed_fragment_info(
    fragment: pa.dataset.ParquetFileFragment,
    row_group_idx: int,
    checkpointed_file_fragments: pa.StructScalar,
) -> CheckpointedFragmentInfo:
    """Get the checkpointed row IDs of a fragment (parquet row group) from checkpointed_ids
    for this parquet file.

    Args:
        fragment: Parquet fragment for which to get checkpointed row IDs.
        row_group_idx: Row group index.
        checkpointed_file_fragments: PyArrow array containing checkpointed file fragments for
            this parquet file.

    Returns:
        CheckpointedFragmentInfo: Checkpointed row IDs of the fragment (parquet row group).
    """
    # Check if there are any checkpointed IDs for this file.
    if (
        checkpointed_file_fragments is None
        or checkpointed_file_fragments.is_valid is False
    ):
        return _create_empty_checkpointed_fragment_info(fragment, row_group_idx)

    fragments_field_idx = get_struct_field_index(
        checkpointed_file_fragments, CHECKPOINTED_FILE_FRAGMENTS_FRAGMENTS_FIELD
    )
    fragments = pc.struct_field(checkpointed_file_fragments, [fragments_field_idx])

    # Convert ListScalar to ListArray
    if fragments.is_valid is False or len(fragments) == 0:
        return _create_empty_checkpointed_fragment_info(fragment, row_group_idx)
    fragments_values = fragments.values  # StructArray

    # Extract fragment IDs as a sorted array
    fragment_id_field_idx = get_struct_field_index(
        fragments_values, CHECKPOINTED_FILE_FRAGMENT_ID_FIELD
    )
    fragment_ids = pc.struct_field(fragments_values, [fragment_id_field_idx])

    # Find matching fragment_id
    target_scalar = pa.scalar(row_group_idx, fragment_ids.type)
    wanted_mask = pc.equal(fragment_ids, target_scalar)

    if not pc.any(wanted_mask).as_py():
        return _create_empty_checkpointed_fragment_info(fragment, row_group_idx)

    # Get the first matching fragment by index
    # Convert mask to indices and take the first one
    indices = pc.indices_nonzero(wanted_mask)
    if len(indices) == 0:
        return _create_empty_checkpointed_fragment_info(fragment, row_group_idx)
    first_match_idx = indices[0].as_py()
    checkpointed_fragment = fragments_values[first_match_idx]
    checkpointed_row_ids_field_idx = get_struct_field_index(
        checkpointed_fragment, CHECKPOINTED_FILE_FRAGMENT_CHECKPOINTED_ROW_IDS_FIELD
    )
    checkpointed_row_ids = pc.struct_field(
        checkpointed_fragment, [checkpointed_row_ids_field_idx]
    )
    num_rows_field_idx = get_struct_field_index(
        checkpointed_fragment, CHECKPOINTED_FILE_FRAGMENT_NUM_ROWS_FIELD
    )
    num_rows = pc.struct_field(checkpointed_fragment, [num_rows_field_idx]).as_py()
    num_checkpointed_rows_field_idx = get_struct_field_index(
        checkpointed_fragment, CHECKPOINTED_FILE_FRAGMENT_NUM_CHECKPOINTED_ROWS_FIELD
    )
    num_checkpointed_rows = pc.struct_field(
        checkpointed_fragment, [num_checkpointed_rows_field_idx]
    ).as_py()

    # Get the actual number of rows in this specific row group
    actual_num_rows = fragment.metadata.row_group(row_group_idx).num_rows

    assert num_rows == actual_num_rows, (
        f"Number of rows in the row group {actual_num_rows} does not match "
        f"the number of rows in the checkpointed fragment {num_rows}"
    )

    # Check if this is a fully checkpointed fragment (empty list means all rows checkpointed)
    if len(checkpointed_row_ids) == 0:
        assert num_checkpointed_rows == num_rows, (
            f"Number of checkpointed rows {num_checkpointed_rows} does not match "
            f"the number of rows in the checkpointed fragment {num_rows}"
        )
        fully_checkpointed = True
        final_checkpointed_row_ids = pa.array([], type=pa.bool_())
    else:
        fully_checkpointed = False
        # checkpointed_row_ids is a ListScalar containing a list of booleans
        # Extract the values directly as a PyArrow boolean array without Python conversion
        final_checkpointed_row_ids = checkpointed_row_ids.values

    result = CheckpointedFragmentInfo(
        fragment=fragment,
        row_group_idx=row_group_idx,
        num_rows=num_rows,
        fully_checkpointed=fully_checkpointed,
        checkpointed_row_ids=final_checkpointed_row_ids,
        checkpointed_row_count=num_checkpointed_rows,
    )
    return result


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
    # checkpointed_row_ids is a PyArrow array of boolean values.
    # True at checkpointed row positions, False at non-checkpointed.
    checkpointed_row_ids = checkpointed_fragment_info.checkpointed_row_ids

    # If no rows are checkpointed, return the table as-is.
    if checkpointed_row_ids is None:
        assert (
            not checkpointed_fragment_info.fully_checkpointed
        ), "Checkpointed row ids is None, intended to be empty checkpointed fragment"
        return table

    # If checkpointed_row_ids is empty array, all rows are checkpointed.
    if len(checkpointed_row_ids) == 0:
        assert (
            checkpointed_fragment_info.fully_checkpointed
        ), "Checkpointed row ids is empty array, intended to be fully checkpointed fragment"
        # Return empty table - all rows checkpointed
        return table.slice(0, 0)

    # Create row indices for the current batch
    # Check if the requested row range is completely inside the boolean array bounds.
    assert current_row_offset + current_num_rows <= len(checkpointed_row_ids), (
        f"Current row offset {current_row_offset} + current num rows {current_num_rows} "
        f"is greater than the length of the PyArrow boolean array {len(checkpointed_row_ids)}"
    )

    # Extract the relevant portion of the boolean array.
    # Use take to get the boolean values for the current row range.
    # Set boundscheck=False to handle out-of-bounds gracefully.
    relevant_bools = checkpointed_row_ids.slice(current_row_offset, current_num_rows)

    # Invert the boolean array: True (checkpointed) becomes False (exclude).
    # False (not checkpointed) becomes True (keep).
    mask = pc.invert(relevant_bools)

    # Filter the table.
    return table.filter(mask)


def get_generated_id_column(
    path: str,
    row_group_idx: int,
    num_row_groups: int,
    total_num_rows: int,
    current_row_offset: int,
    current_num_rows: int,
) -> pa.Array:
    """Helper function to get `generated_id_column`.

    Args:
        path: Full path to the file
        row_group_idx: Row group index
        num_row_groups: Number of row groups in the file
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
        elif field_name == FRAGMENT_FIELD:
            array = pa.nulls(current_num_rows, type=pa.int32())
            filled_array = pc.fill_null(array, row_group_idx)
            return pc.dictionary_encode(filled_array)
        elif field_name == NUM_FRAGMENTS_FIELD:
            array = pa.nulls(current_num_rows, type=pa.int32())
            filled_array = pc.fill_null(array, num_row_groups)
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
    # Create proper pyarrow.Field objects for PyArrow 9 compatibility
    fields = [
        pa.field(field_name, field_type, nullable=False)
        for field_name, field_type in GENERATED_ID_COLUMN_FIELDS.items()
    ]
    return pa.StructArray.from_arrays(arrays, fields=fields)


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


def get_checkpoint_fragments(
    checkpointed_ids: Block,
    path: str,
    checkpointed_fragments_by_path: dict[str, int],
) -> Optional[pa.StructScalar]:
    """Filter checkpointed fragments based on the checkpointed IDs.

    Args:
        checkpointed_ids: A Block containing checkpointed IDs with schema CHECKPOINTED_GENERATED_ID_COLUMN_TABLE_SCHEMA
        path: The file path to get the checkpointed fragments for
        checkpointed_fragments_by_path: A dictionary mapping file path to the index of the file in the checkpointed IDs block

    Returns:
        A PyArrow StructScalar with schema CHECKPOINTED_FILE_FRAGMENTS_TYPE

    """
    if checkpointed_ids is None:
        # No checkpointed IDs
        return None

    accessor = ArrowBlockAccessor.for_block(checkpointed_ids)
    checkpointed_ids_table = accessor.to_arrow()
    if checkpointed_ids_table.num_rows == 0:
        # No checkpointed files
        return None

    if path not in checkpointed_fragments_by_path:
        # No checkpointed fragments for this path
        return None

    file_index = checkpointed_fragments_by_path[path]
    checkpointed_file_fragments_col = checkpointed_ids_table[
        CHECKPOINTED_FILE_FRAGMENTS_COLUMN_NAME
    ]
    checkpointed_file_fragments = checkpointed_file_fragments_col[file_index]
    return checkpointed_file_fragments


def index_checkpointed_fragments(
    checkpointed_ids: Block,
) -> dict[str, int]:
    """Index checkpointed fragments by file path based on the checkpointed IDs.

    Args:
        checkpointed_ids: A Block containing checkpointed IDs with CHECKPOINTED_GENERATED_ID_COLUMN_TABLE_SCHEMA

    Returns:
        A dictionary mapping file path to the index of the file in the checkpointed IDs block
    """
    if checkpointed_ids is None:
        return {}

    accessor = ArrowBlockAccessor.for_block(checkpointed_ids)
    checkpointed_ids_table = accessor.to_arrow()
    if checkpointed_ids_table.num_rows == 0:
        return {}

    file_path_col = checkpointed_ids_table[CHECKPOINTED_FILE_COLUMN_NAME]
    file_path_dict = {file_path.as_py(): i for i, file_path in enumerate(file_path_col)}
    return file_path_dict


def is_file_fragments_fully_checkpointed(
    checkpointed_file_fragments: pa.StructScalar,
) -> bool:
    """Check if the file fragments are fully checkpointed.

    Args:
        checkpointed_file_fragments: A PyArrow StructScalar with schema CHECKPOINTED_FILE_FRAGMENTS_TYPE

    Returns:
        True if the file fragments are fully checkpointed, False otherwise
    """
    fully_checkpointed_field_idx = get_struct_field_index(
        checkpointed_file_fragments,
        CHECKPOINTED_FILE_FRAGMENTS_FULLY_CHECKPOINTED_FIELD,
    )
    fully_checkpointed = pc.struct_field(
        checkpointed_file_fragments, [fully_checkpointed_field_idx]
    ).as_py()
    return fully_checkpointed


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
