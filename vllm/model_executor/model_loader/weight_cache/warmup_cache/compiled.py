# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compiled runtime artifact warmup cache."""

import hashlib
import os
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_cache.artifact_store import (
    STORE_BACKENDS,
    ArtifactStore,
    artifact_key,
)
from vllm.model_executor.model_loader.weight_cache.transport import (
    Seedable,
    SeedTransport,
)
from vllm.model_executor.model_loader.weight_cache.warmup_cache.common import (
    BUNDLE_LIMIT,
    _bundle,
    _install,
)

logger = init_logger(__name__)


def _cache_roots() -> dict[str, Path]:
    return {
        name: Path(value).expanduser()
        for name, variable in (
            ("flashinfer-cubins", "FLASHINFER_CUBIN_DIR"),
            ("flashinfer-workspace", "FLASHINFER_WORKSPACE_BASE"),
        )
        if (value := os.environ.get(variable))
    }


class CompileArtifactWarmupCache(Seedable):
    """Compiled runtime cache seeded as part of daemon warmup."""

    kind = "compile_cache"

    def __init__(self, store_names: Iterable[str] = ()) -> None:
        self.roots = _cache_roots()
        self._store_names = tuple(store_names)
        self._store_sources: list[ArtifactStore] | None = None
        self._payload: dict[str, Any] | None = None

    @property
    def store_sources(self) -> list[ArtifactStore]:
        if self._store_sources is None:
            self._store_sources = [STORE_BACKENDS[name]() for name in self._store_names]
        return self._store_sources

    def adopt_before_load(self, transport: SeedTransport) -> None:
        if transport.source is None or not self.store_sources:
            return
        deadline = time.monotonic() + 300.0
        while time.monotonic() < deadline:
            response = transport.request({"kind": self.kind, "version": 1})
            if response is not None and response.get("status") == "ok":
                try:
                    payload = self.validate_payload(response.get("payload"))
                    self.install(payload)
                except Exception as error:
                    logger.warning(
                        "Cannot install remote compile cache; using local loading: %s",
                        error,
                    )
                    return
                logger.info("Adopted remote compile cache before model loading")
                return
            if response is not None and response.get("status") == "miss":
                return
            if response is not None and response.get("status") == "error":
                logger.warning(
                    "Source does not provide a compile cache; using local loading"
                )
                return
            time.sleep(1.0)
        logger.warning(
            "Timed out waiting for the source compile cache; using local loading"
        )

    def prepare(self) -> None:
        self._payload = None
        stores = self.store_sources
        if not stores:
            return
        try:
            bundle = _bundle(self.roots)
            if bundle is None:
                return
            digest = hashlib.sha256(bundle).hexdigest()
            key = artifact_key(self.kind, digest, "bundle.tar")
            for store in stores:
                store.put(key, bundle)
        except Exception as error:
            logger.warning(
                "Cannot publish compile cache; continuing without it: %s", error
            )
            return
        self._payload = {
            "version": 1,
            "key": key,
            "bundle_size": len(bundle),
            "sha256": digest,
        }

    def resolve(self, request: dict[str, Any]) -> dict[str, Any] | None:
        if request.get("version") != 1:
            return None
        return self._payload

    def validate_payload(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict) or payload.get("version") != 1:
            raise ValueError("invalid compile cache descriptor")
        size = payload.get("bundle_size")
        key, digest = payload.get("key"), payload.get("sha256")
        if isinstance(size, int) and not isinstance(size, bool) and size > BUNDLE_LIMIT:
            raise ValueError("compiled cache bundle is too large")
        if (
            not isinstance(size, int)
            or isinstance(size, bool)
            or size <= 0
            or not isinstance(digest, str)
            or key != artifact_key(self.kind, digest, "bundle.tar")
        ):
            raise ValueError("invalid compile cache descriptor")
        return {
            "version": 1,
            "key": key,
            "bundle_size": size,
            "sha256": digest,
        }

    def install(self, payload: dict[str, Any]) -> None:
        for store in self.store_sources:
            bundle = store.get(payload["key"])
            if bundle is None:
                continue
            if len(bundle) != payload["bundle_size"]:
                raise ValueError("compiled cache bundle size mismatch")
            if hashlib.sha256(bundle).hexdigest() != payload["sha256"]:
                raise ValueError("compiled cache bundle checksum mismatch")
            _install(bundle, self.roots)
            return
        raise RuntimeError("compiled cache bundle is missing from every shared store")
