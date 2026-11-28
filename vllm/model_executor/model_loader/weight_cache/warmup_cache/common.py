# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared compiled-cache bundling and installation helpers."""

import io
import tarfile
from pathlib import Path, PurePosixPath

from vllm.model_executor.model_loader.weight_cache.artifact_store import (
    atomic_write_bytes,
)

BUNDLE_LIMIT = 8 << 30


def _safe_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if not path.parts or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe cache path: {value!r}")
    return path


def _bundle(roots: dict[str, Path]) -> bytes | None:
    output = io.BytesIO()
    total = 0
    count = 0
    with tarfile.open(fileobj=output, mode="w:") as archive:
        for root_name, root in roots.items():
            if not root.is_dir():
                continue
            nested_roots = tuple(
                other
                for other in roots.values()
                if other != root and root in other.parents
            )
            for path in sorted(root.rglob("*")):
                if (
                    not path.is_file()
                    or path.is_symlink()
                    or path.name.endswith((".lock", ".tmp"))
                    or any(other in path.parents for other in nested_roots)
                ):
                    continue
                stat = path.stat()
                if total + stat.st_size > BUNDLE_LIMIT:
                    raise ValueError("compiled cache is too large")
                data = path.read_bytes()
                total += len(data)
                if total > BUNDLE_LIMIT:
                    raise ValueError("compiled cache is too large")
                relative = path.relative_to(root).as_posix()
                info = tarfile.TarInfo(f"{root_name}/{relative}")
                info.size, info.mode = len(data), 0o600
                info.mtime = stat.st_mtime_ns / 1e9
                archive.addfile(info, io.BytesIO(data))
                count += 1
    if not count:
        return None
    bundle = output.getvalue()
    if len(bundle) > BUNDLE_LIMIT:
        raise ValueError("compiled cache is too large")
    return bundle


def _install(bundle: bytes, roots: dict[str, Path]) -> None:
    verified: list[tuple[tarfile.TarInfo, Path]] = []
    with tarfile.open(fileobj=io.BytesIO(bundle), mode="r:") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                raise ValueError(f"unexpected compiled cache member: {member.name}")
            root_name, sep, rel = member.name.partition("/")
            if not sep or root_name not in roots or not rel:
                raise ValueError(f"unexpected compiled cache member: {member.name}")
            destination = roots[root_name] / _safe_path(rel)
            verified.append((member, destination))
        for member, destination in verified:
            extracted = archive.extractfile(member)
            assert extracted is not None
            atomic_write_bytes(
                destination,
                extracted.read(),
                mtime_ns=int(member.mtime * 1e9),
            )
