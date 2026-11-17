"""Export sky regions and survey footprints as IVOA Multi-Order Coverage maps."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import numpy as np

from data_lake.discovery.region import Region, rescale_npix_nested
from data_lake.discovery.tile_index import survey_npix
from data_lake.lake_registry import registry_root
from data_lake.schema_registry import MODALITY_CATALOG

log = logging.getLogger(__name__)

MocFormat = Literal["fits", "json", "ascii"]

MOCS_DIRNAME = "mocs"

_MOC_FORMAT_SUFFIX: dict[MocFormat, str] = {
    "fits": ".moc.fits",
    "json": ".moc.json",
    "ascii": ".moc.txt",
}


@dataclass(frozen=True)
class MocExportResult:
    output: Path
    format: MocFormat
    moc_order: int
    n_cells: int
    max_order: int
    survey: str | None = None
    modality: str | None = None
    skipped: bool = False


@dataclass(frozen=True)
class MocBulkExportResult:
    """Summary of :func:`export_survey_mocs_from_registry`."""

    results: list[MocExportResult]
    errors: list[tuple[str, str, str]]  # survey, modality, message

    @property
    def n_ok(self) -> int:
        return sum(1 for r in self.results if not r.skipped)

    @property
    def n_skipped(self) -> int:
        return sum(1 for r in self.results if r.skipped)

    @property
    def n_error(self) -> int:
        return len(self.errors)


def moc_registry_dir(lake_root: Path | str) -> Path:
    """``<lake>/shared/registry/mocs/``."""
    return registry_root(lake_root) / MOCS_DIRNAME


def default_survey_moc_path(
    lake_root: Path | str,
    survey: str,
    modality: str = MODALITY_CATALOG,
    *,
    moc_order: int,
    fmt: MocFormat = "fits",
) -> Path:
    """Canonical path: ``shared/registry/mocs/<survey>.<modality>.Norder<N>.moc.<ext>``."""
    if fmt not in _MOC_FORMAT_SUFFIX:
        raise ValueError(f"unsupported MOC format {fmt!r}")
    order = int(moc_order)
    if order < 0:
        raise ValueError(f"moc_order must be >= 0; got {moc_order!r}")
    safe_survey = str(survey).strip()
    safe_mod = str(modality).strip()
    return (
        moc_registry_dir(lake_root)
        / f"{safe_survey}.{safe_mod}.Norder{order}{_MOC_FORMAT_SUFFIX[fmt]}"
    )


def _require_mocpy():
    try:
        from mocpy import MOC  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "MOC export requires the optional 'mocpy' dependency. "
            "Install with: uv sync --extra moc"
        ) from exc
    return MOC


def npix_set_to_moc(npix: Iterable[int], moc_order: int):
    """Build a compressed spatial MOC from NESTED HEALPix cells at *moc_order*."""
    if moc_order < 0:
        raise ValueError("moc_order must be >= 0")
    MOC = _require_mocpy()
    cells = sorted({int(p) for p in npix})
    if not cells:
        return MOC.from_string(f"{moc_order}/")
    ipix = np.array(cells, dtype=np.uint64)
    depth = np.uint8(moc_order)
    return MOC.from_healpix_cells(ipix, depth=depth, max_depth=moc_order)


def region_to_moc(region: Region, moc_order: int):
    """Resolve a :class:`Region` to a MOC at the requested HEALPix order."""
    npix = region.to_npix(moc_order)
    return npix_set_to_moc(npix, moc_order)


def survey_footprint_npix(
    lake_root: Path | str,
    survey: str,
    modality: str = MODALITY_CATALOG,
    *,
    region: Region | None = None,
    moc_order: int,
    allow_scan: bool = True,
) -> set[int]:
    """Populated survey tiles, optionally clipped to *region*, at *moc_order*."""
    npix_native, hats_order = survey_npix(
        lake_root, survey, modality, allow_scan=allow_scan,
    )
    if not npix_native:
        raise FileNotFoundError(
            f"No tiles found for {survey!r} modality {modality!r}"
        )
    if hats_order is None:
        raise ValueError(
            f"Cannot determine hats_order for {survey!r} {modality!r}; "
            "ingest metadata or refresh the tile index."
        )
    if region is not None:
        region_npix = region.to_npix(hats_order)
        npix_native &= region_npix
    if moc_order == hats_order:
        return npix_native
    return rescale_npix_nested(npix_native, hats_order, moc_order)


def write_moc(
    moc,
    output: Path | str,
    *,
    fmt: MocFormat = "fits",
    overwrite: bool = False,
    fits_keywords: dict[str, Any] | None = None,
) -> Path:
    """Serialize a MOC to disk (IVOA FITS by default)."""
    path = Path(output)
    if fmt not in ("fits", "json", "ascii"):
        raise ValueError(f"unsupported MOC format {fmt!r}")
    path.parent.mkdir(parents=True, exist_ok=True)
    # mocpy only accepts fits_keywords for FITS serialization.
    if fmt == "fits":
        moc.save(
            str(path),
            format=fmt,
            overwrite=overwrite,
            fits_keywords=fits_keywords,
        )
    else:
        moc.save(str(path), format=fmt, overwrite=overwrite)
    return path


def export_region_moc(
    region: Region,
    output: Path | str,
    *,
    moc_order: int,
    fmt: MocFormat = "fits",
    overwrite: bool = False,
    provenance: dict[str, Any] | None = None,
) -> MocExportResult:
    """Export any region selector as a MOC file."""
    moc = region_to_moc(region, moc_order)
    keywords = _fits_keywords(provenance, moc_order=moc_order)
    out = write_moc(moc, output, fmt=fmt, overwrite=overwrite, fits_keywords=keywords)
    n_cells = len(region.to_npix(moc_order))
    return MocExportResult(
        output=out,
        format=fmt,
        moc_order=moc_order,
        n_cells=n_cells,
        max_order=int(moc.max_order),
    )


def export_survey_moc(
    lake_root: Path | str,
    survey: str,
    output: Path | str | None = None,
    *,
    modality: str = MODALITY_CATALOG,
    moc_order: int,
    region: Region | None = None,
    fmt: MocFormat = "fits",
    overwrite: bool = False,
) -> MocExportResult:
    """Export populated survey tiles (optionally region-clipped) as a MOC.

    When *output* is ``None``, write to
    :func:`default_survey_moc_path` under ``shared/registry/mocs/``.
    """
    lake_root = Path(lake_root)
    out_path = (
        Path(output)
        if output is not None
        else default_survey_moc_path(
            lake_root, survey, modality, moc_order=moc_order, fmt=fmt
        )
    )
    if out_path.exists() and not overwrite:
        raise FileExistsError(
            f"MOC already exists: {out_path} (pass overwrite=True / --overwrite)"
        )
    npix = survey_footprint_npix(
        lake_root, survey, modality, region=region, moc_order=moc_order,
    )
    moc = npix_set_to_moc(npix, moc_order)
    provenance = {
        "survey": survey,
        "modality": modality,
        "source": "data_lake_tile_index",
    }
    keywords = _fits_keywords(provenance, moc_order=moc_order)
    out = write_moc(moc, out_path, fmt=fmt, overwrite=overwrite, fits_keywords=keywords)
    return MocExportResult(
        output=out,
        format=fmt,
        moc_order=moc_order,
        n_cells=len(npix),
        max_order=int(moc.max_order),
        survey=survey,
        modality=modality,
    )


def list_registry_surveys_for_moc(
    lake_root: Path | str,
    *,
    modality: str = MODALITY_CATALOG,
    kind: str = "ingested",
) -> list[str]:
    """Survey names from the lake registry for bulk MOC export."""
    from data_lake.lake_registry import (
        coalesce_registry_kinds,
        filter_lake_registry_table,
        filter_registry_by_kind,
        load_lake_registry,
    )

    table = coalesce_registry_kinds(load_lake_registry(lake_root))
    kind_key = None if str(kind).strip().lower() == "all" else kind
    table = filter_registry_by_kind(table, kind_key)
    table = filter_lake_registry_table(table, modality)
    if table.num_rows == 0 or "survey" not in table.schema.names:
        return []
    names = {str(s) for s in table.column("survey").to_pylist() if s}
    return sorted(names)


def export_survey_mocs_from_registry(
    lake_root: Path | str,
    *,
    moc_order: int,
    modality: str = MODALITY_CATALOG,
    kind: str = "ingested",
    fmt: MocFormat = "fits",
    overwrite: bool = False,
    continue_on_error: bool = True,
    surveys: Sequence[str] | None = None,
) -> MocBulkExportResult:
    """Write canonical MOCs for registry surveys (default: ingested × modality).

    Skips existing files when *overwrite* is false. Partner failures are
    recorded in ``errors``; with *continue_on_error* the rest still run.
    """
    lake_root = Path(lake_root)
    if surveys is None:
        targets = list_registry_surveys_for_moc(
            lake_root, modality=modality, kind=kind
        )
    else:
        targets = [str(s) for s in surveys]
    results: list[MocExportResult] = []
    errors: list[tuple[str, str, str]] = []
    for survey in targets:
        out_path = default_survey_moc_path(
            lake_root, survey, modality, moc_order=moc_order, fmt=fmt
        )
        if out_path.exists() and not overwrite:
            results.append(
                MocExportResult(
                    output=out_path,
                    format=fmt,
                    moc_order=moc_order,
                    n_cells=0,
                    max_order=moc_order,
                    survey=survey,
                    modality=modality,
                    skipped=True,
                )
            )
            log.info("Skip existing MOC %s", out_path)
            continue
        try:
            results.append(
                export_survey_moc(
                    lake_root,
                    survey,
                    out_path,
                    modality=modality,
                    moc_order=moc_order,
                    fmt=fmt,
                    overwrite=overwrite,
                )
            )
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            log.exception(
                "MOC export failed for %s (%s): %s", survey, modality, exc
            )
            errors.append((survey, modality, msg))
            if not continue_on_error:
                break
    return MocBulkExportResult(results=results, errors=errors)


def _fits_keywords(
    provenance: dict[str, Any] | None,
    *,
    moc_order: int,
) -> dict[str, Any]:
    kw: dict[str, Any] = {
        "ORIGIN": "data-lake",
        "MOCORD": moc_order,
    }
    if provenance:
        for key, value in provenance.items():
            if value is not None:
                kw[key.upper()[:8]] = str(value)[:68]
    return kw
