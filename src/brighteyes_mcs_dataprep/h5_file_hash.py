"""Stable hash helpers for BrightEyes HDF5 source files."""

from __future__ import annotations

import hashlib
import json

import h5py
import numpy as np


CHANNEL_DATASET_CANDIDATES = (
    ("spad", ("/raw/spad", "/spad", "/data")),
    ("aux", ("/raw/aux", "/aux", "/data_channels_extra")),
    ("analog", ("/raw/analog", "/analog", "/data_analog")),
)


def _dataset_at(handle, path):
    key = str(path).strip("/")
    if key in handle and isinstance(handle[key], h5py.Dataset):
        return handle[key]
    return None


def _first_existing_channel_datasets(handle):
    datasets = []
    seen_names = set()
    for kind, candidates in CHANNEL_DATASET_CANDIDATES:
        for path in candidates:
            dataset = _dataset_at(handle, path)
            if dataset is None or dataset.name in seen_names:
                continue
            if dataset.ndim < 1 or dataset.shape[-1] <= 0:
                continue
            datasets.append((kind, dataset))
            seen_names.add(dataset.name)
            break
    return datasets


def channel_fingerprint(dataset):
    """Return the microscopy fingerprint: sum over all axes except channels."""

    channel_count = int(dataset.shape[-1])
    fingerprint = np.zeros(channel_count, dtype=np.float64)
    prefix_ndim = min(2, max(dataset.ndim - 1, 0))
    prefix_shape = tuple(int(size) for size in dataset.shape[:prefix_ndim])
    prefix_slice = (slice(None),) * max(dataset.ndim - prefix_ndim - 1, 0)

    for prefix in np.ndindex(prefix_shape or ()):
        block = np.asarray(dataset[prefix + prefix_slice + (slice(None),)], dtype=np.float64)
        if block.ndim == 1:
            fingerprint += block
        else:
            fingerprint += block.sum(axis=tuple(range(block.ndim - 1)))

    return fingerprint


def channel_fingerprint_file_hash(handle, algorithm="sha256"):
    """
    Hash a file identity vector built from all available channel fingerprints.

    The input vector concatenates the per-channel sums for SPAD, aux/extra, and
    analog datasets when present. Dataset path, shape, and dtype are included in
    the digest so files with identical count vectors but different channel
    layouts remain distinct.
    """

    hasher = hashlib.new(algorithm)
    source_paths = []
    channel_counts = []

    for kind, dataset in _first_existing_channel_datasets(handle):
        fingerprint = np.asarray(channel_fingerprint(dataset), dtype="<f8")
        source_paths.append(dataset.name)
        channel_counts.append(int(dataset.shape[-1]))
        header = {
            "kind": kind,
            "path": dataset.name,
            "shape": [int(size) for size in dataset.shape],
            "dtype": str(dataset.dtype),
            "channel_count": int(dataset.shape[-1]),
        }
        hasher.update(json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(fingerprint.tobytes(order="C"))
        hasher.update(b"\0")

    return {
        "algorithm": algorithm,
        "hash": hasher.hexdigest(),
        "source_paths": source_paths,
        "channel_counts": channel_counts,
    }


def channel_fingerprint_file_hash_attrs(handle, prefix="file"):
    """Return HDF5-ready attrs for :func:`channel_fingerprint_file_hash`."""

    payload = channel_fingerprint_file_hash(handle)
    return {
        f"{prefix}_{payload['algorithm']}": payload["hash"],
        f"{prefix}_hash_algorithm": payload["algorithm"],
        f"{prefix}_hash_source_paths_json": json.dumps(payload["source_paths"]),
        f"{prefix}_hash_channel_counts_json": json.dumps(payload["channel_counts"]),
    }
