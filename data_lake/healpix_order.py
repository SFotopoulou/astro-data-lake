"""Read the HEALPix order of a catalog, spectrum, or cutout layer.

The order is stored as ``healpix_order`` in ``catalog_info.json`` (and the
matching spectrum, cutout, crossmatch, and tile-index sidecars).

Readers of stored metadata use :func:`require_healpix_order` or
:func:`require_healpix_order_file`, which raise :class:`HealpixOrderError` when
the key is missing or invalid. There is no silent default. Validators use
:func:`read_healpix_order`, which returns ``None`` so they can report every
problem.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

HEALPIX_ORDER_KEY = "healpix_order"

__all__ = [
    "HEALPIX_ORDER_KEY",
    "HealpixOrderError",
    "read_healpix_order",
    "require_healpix_order",
    "require_healpix_order_file",
]


class HealpixOrderError(ValueError):
    """Stored metadata has no usable ``healpix_order``."""


def read_healpix_order(info: Mapping[str, Any] | None) -> int | None:
    """Return the stored HEALPix order, or ``None`` if missing or not an integer."""
    if not info:
        return None
    raw = info.get(HEALPIX_ORDER_KEY)
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def require_healpix_order(info: Mapping[str, Any] | None, source: str) -> int:
    """Return the stored HEALPix order or raise :class:`HealpixOrderError`.

    *source* names the metadata file in the error message.
    """
    value = read_healpix_order(info)
    if value is None:
        raise HealpixOrderError(
            f"{source}: missing or invalid 'healpix_order'. Reingest the layer, "
            "or add the key with the layer's HEALPix order."
        )
    return value


def require_healpix_order_file(path: Path | str) -> int:
    """Load a JSON sidecar and return its order, raising if the file or key is missing."""
    path = Path(path)
    if not path.is_file():
        raise HealpixOrderError(
            f"{path}: not found; cannot determine the HEALPix order. "
            "Pass the order explicitly (--healpix-order) or restore the metadata."
        )
    with open(path, encoding="utf-8") as fh:
        info = json.load(fh)
    return require_healpix_order(info, str(path))
