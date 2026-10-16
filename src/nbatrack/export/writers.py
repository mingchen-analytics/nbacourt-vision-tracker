"""Validated table writers: Parquet (primary), CSV and JSON."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from nbatrack.export.schema import ENUMS, SCHEMAS, schema_metadata


class SchemaError(ValueError):
    pass


def _clean(v: Any) -> Any:
    """NaN / inf / numpy scalars -> python values with NULL for missing."""
    if v is None:
        return None
    if isinstance(v, np.ndarray):
        v = v.tolist()
    if isinstance(v, (list, tuple)):
        vals = [_clean(x) for x in v]
        return None if any(x is None for x in vals) else vals
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def build_table(name: str, rows: Iterable[dict]) -> pa.Table:
    """Validate rows against the schema (fields, nullability, enums) and build a table."""
    if name not in SCHEMAS:
        raise SchemaError(f"unknown table {name!r}")
    schema = SCHEMAS[name]
    names = schema.names
    enums = ENUMS.get(name, {})
    cols: dict[str, list] = {n: [] for n in names}
    for i, r in enumerate(rows):
        extra = set(r) - set(names)
        if extra:
            raise SchemaError(f"{name} row {i}: unknown fields {sorted(extra)}")
        for fld in schema:
            v = _clean(r.get(fld.name))
            if v is None and not fld.nullable:
                raise SchemaError(f"{name} row {i}: {fld.name} must not be null")
            if v is not None and fld.name in enums and v not in enums[fld.name]:
                raise SchemaError(f"{name} row {i}: {fld.name}={v!r} not in {sorted(enums[fld.name])}")
            cols[fld.name].append(v)
    arrays = [pa.array(cols[f.name], type=f.type) for f in schema]
    return pa.Table.from_arrays(arrays, schema=schema).replace_schema_metadata(schema_metadata(name))


def write_table(table: pa.Table, out_dir: str | Path, name: str,
                formats: tuple[str, ...] = ("parquet", "csv")) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    if "parquet" in formats:
        p = out_dir / f"{name}.parquet"
        pq.write_table(table, p, compression="zstd")
        paths.append(p)
    if "csv" in formats:
        # CSV cannot hold list columns: serialize them as JSON strings.
        t = table
        for i, f in enumerate(t.schema):
            if pa.types.is_list(f.type) or pa.types.is_fixed_size_list(f.type):
                col = [None if v is None else json.dumps(v) for v in t.column(i).to_pylist()]
                t = t.set_column(i, pa.field(f.name, pa.string()), pa.array(col, pa.string()))
        p = out_dir / f"{name}.csv"
        pacsv.write_csv(t, p)
        paths.append(p)
    if "json" in formats:
        p = out_dir / f"{name}.json"
        p.write_text(json.dumps(table.to_pylist(), indent=1), encoding="utf-8")
        paths.append(p)
    return paths


def write_rows(name: str, rows: Iterable[dict], out_dir: str | Path,
               formats: tuple[str, ...] = ("parquet", "csv")) -> list[Path]:
    return write_table(build_table(name, rows), out_dir, name, formats)
