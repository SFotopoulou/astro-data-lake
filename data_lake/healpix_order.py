"""Read the HEALPix order of a catalog, spectrum, or cutout layer.

The order is stored as ``healpix_order`` in ``catalog_info.json`` (and the
matching spectrum, cutout, crossmatch, and tile-index sidecars).
"""

from __future__ import annotations

from typing import Any, Mapping

from data_lake.knobs import DEFAULT_HEALPIX_ORDER

HEALPIX_ORDER_KEY = "healpix_order"

__all__ = [
    "DEFAULT_HEALPIX_ORDER",
    "HEALPIX_ORDER_KEY",
    "read_healpix_order",
]


def read_healpix_order(
    info: Mapping[str, Any] | None,
    default: int | None = None,
) -> int | None:
    """Return the HEALPix order from a metadata mapping, or *default*.

    A value that is missing, ``None``, or not an integer returns *default*.
    """
    if not info:
        return default
    raw = info.get(HEALPIX_ORDER_KEY)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default
