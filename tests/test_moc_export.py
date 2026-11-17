"""Tests for MOC export (requires optional mocpy)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from data_lake.discovery import tile_index as ti
from data_lake.discovery.moc_export import (
    export_region_moc,
    export_survey_moc,
    region_to_moc,
)
from data_lake.discovery.moc_export_cli import cli as export_moc_cli
from data_lake.discovery.region import Region
from data_lake.discovery.region_cli import cli as region_cli
from data_lake.ingest.fits_to_parquet import assign_healpix, healpix_dir


def _write_tiny_catalog(
    lake: Path,
    survey: str,
    *,
    ra: float,
    dec: float,
    norder: int = 5,
    source_id: int = 1,
) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    from data_lake.ingest.fits_to_parquet import LAKE_JOIN_ID_COLUMN

    npix = int(assign_healpix(np.array([ra]), np.array([dec]), norder)[0])
    tile_dir = lake / "catalogs" / survey / healpix_dir(norder, npix)
    tile_dir.mkdir(parents=True, exist_ok=True)
    hp_col = f"_healpix_norder{norder}"
    pq.write_table(
        pa.table(
            {
                LAKE_JOIN_ID_COLUMN: pa.array([source_id], type=pa.int64()),
                "ra": pa.array([ra], type=pa.float64()),
                "dec": pa.array([dec], type=pa.float64()),
                hp_col: pa.array([npix], type=pa.int64()),
            }
        ),
        tile_dir / f"Npix={npix}.parquet",
    )
    (lake / "catalogs" / survey / "catalog_info.json").write_text(
        json.dumps(
            {
                "hats_order": norder,
                "ra_column": "ra",
                "dec_column": "dec",
                "link_id_column": LAKE_JOIN_ID_COLUMN,
                "total_rows": 1,
            }
        )
    )
    ti.write_tile_index(lake, survey, "catalog")
    return npix


@pytest.fixture
def lake(tmp_path: Path) -> Path:
    lake = tmp_path / "lake"
    _write_tiny_catalog(lake, "ALLWISE", ra=120.0, dec=45.0, norder=5)
    return lake


class TestMocExport:
    def test_missing_mocpy_raises(self) -> None:
        region = Region.cone(120.0, 45.0, 60.0)
        try:
            import mocpy  # noqa: F401
        except ImportError:
            with pytest.raises(ImportError, match="mocpy"):
                region_to_moc(region, 5)
        else:
            pytest.skip("mocpy installed")

    def test_cone_fits_roundtrip(self, tmp_path: Path) -> None:
        pytest.importorskip("mocpy")
        region = Region.cone(120.0, 45.0, 600.0)
        out = tmp_path / "cone.moc.fits"
        export_region_moc(region, out, moc_order=5, overwrite=True)
        assert out.is_file()
        npix_in = region.to_npix(5)
        npix_back = Region.from_moc(path=out).to_npix(5)
        assert npix_back == npix_in

    def test_ascii_and_json_export(self, tmp_path: Path) -> None:
        """Non-FITS formats must not pass fits_keywords to mocpy.save."""
        pytest.importorskip("mocpy")
        region = Region.cone(120.0, 45.0, 600.0)
        ascii_out = tmp_path / "cone.moc.txt"
        json_out = tmp_path / "cone.moc.json"
        export_region_moc(
            region, ascii_out, moc_order=5, fmt="ascii", overwrite=True,
        )
        export_region_moc(
            region, json_out, moc_order=5, fmt="json", overwrite=True,
        )
        assert ascii_out.is_file() and ascii_out.stat().st_size > 0
        assert json_out.is_file() and json_out.stat().st_size > 0
        text = ascii_out.read_text().strip()
        assert "/" in text  # IVOA ASCII MOC: order/cells

    def test_survey_footprint(self, lake: Path, tmp_path: Path) -> None:
        pytest.importorskip("mocpy")
        out = tmp_path / "allwise.moc.fits"
        result = export_survey_moc(
            lake, "ALLWISE", out, moc_order=5, overwrite=True,
        )
        assert result.n_cells >= 1
        assert out.is_file()

    def test_finer_than_hats_uses_catalog_rows(
        self, lake: Path, tmp_path: Path
    ) -> None:
        """moc_order > HATS → source MOC, not subdivided empty tiles."""
        pytest.importorskip("mocpy")
        from mocpy import MOC

        # Tile-based upscale of one Norder=5 cell → 4^(8-5) = 64 cells at 8.
        tile_up = export_survey_moc(
            lake, "ALLWISE", tmp_path / "tile_eq.fits", moc_order=5, overwrite=True,
        )
        assert tile_up.n_cells == 1

        fine = export_survey_moc(
            lake, "ALLWISE", tmp_path / "fine.fits", moc_order=8, overwrite=True,
        )
        assert fine.n_cells == 1  # one source → one leaf cell, not 64
        moc = MOC.load(str(tmp_path / "fine.fits"), format="fits")
        assert int(moc.max_order) >= 8 or "8/" in moc.to_string("ascii")

    def test_region_cli_export_moc(self, lake: Path, tmp_path: Path) -> None:
        pytest.importorskip("mocpy")
        out = tmp_path / "field.moc.fits"
        runner = CliRunner()
        r = runner.invoke(
            region_cli,
            [
                str(lake), "--cone", "120.0", "45.0", "--radius-arcsec", "60",
                "--export-moc", str(out), "--moc-order", "5", "--overwrite",
            ],
        )
        assert r.exit_code == 0, r.output
        assert out.is_file()
        assert "Exported MOC" in r.output

    def test_export_moc_cli(self, lake: Path, tmp_path: Path) -> None:
        pytest.importorskip("mocpy")
        out = tmp_path / "wise.moc.fits"
        runner = CliRunner()
        r = runner.invoke(
            export_moc_cli,
            [
                str(lake), "--survey", "ALLWISE", "--moc-order", "5",
                "-o", str(out), "--overwrite",
            ],
        )
        assert r.exit_code == 0, r.output
        assert out.is_file()

    def test_export_moc_cli_default_path(self, lake: Path) -> None:
        pytest.importorskip("mocpy")
        from data_lake.discovery.moc_export import default_survey_moc_path

        runner = CliRunner()
        r = runner.invoke(
            export_moc_cli,
            [str(lake), "--survey", "ALLWISE", "--moc-order", "5", "--overwrite"],
        )
        assert r.exit_code == 0, r.output
        out = default_survey_moc_path(lake, "ALLWISE", "catalog", moc_order=5)
        assert out.is_file()
        assert "shared/registry/mocs" in str(out)
        assert out.name == "ALLWISE.catalog.Norder5.moc.fits"

    def test_export_moc_all_surveys(self, lake: Path) -> None:
        pytest.importorskip("mocpy")
        from data_lake.discovery.moc_export import default_survey_moc_path
        from data_lake.lake_registry import refresh_lake_registry

        # Second survey so bulk iterates more than one name.
        _write_tiny_catalog(lake, "DESI_DR1", ra=10.0, dec=0.0, norder=5, source_id=2)
        refresh_lake_registry(lake)

        runner = CliRunner()
        r = runner.invoke(
            export_moc_cli,
            [str(lake), "--all-surveys", "--moc-order", "5", "--overwrite"],
        )
        assert r.exit_code == 0, r.output
        assert default_survey_moc_path(
            lake, "ALLWISE", moc_order=5
        ).is_file()
        assert default_survey_moc_path(
            lake, "DESI_DR1", moc_order=5
        ).is_file()
        assert "ok=" in r.output
