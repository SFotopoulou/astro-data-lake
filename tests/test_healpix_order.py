"""Tests for the ``healpix_order`` readers: single key, no default, raise when missing."""

from __future__ import annotations

import json

import pytest

from data_lake.config import LakeConfig, LakeConfigInvalid
from data_lake.healpix_order import (
    HEALPIX_ORDER_KEY,
    HealpixOrderError,
    read_healpix_order,
    require_healpix_order,
    require_healpix_order_file,
)
from data_lake.knobs import DEFAULT_HEALPIX_ORDER


def test_key_is_read() -> None:
    assert read_healpix_order({HEALPIX_ORDER_KEY: 3}) == 3


def test_string_integer_is_accepted() -> None:
    assert read_healpix_order({HEALPIX_ORDER_KEY: "5"}) == 5


def test_legacy_key_is_ignored() -> None:
    assert read_healpix_order({"hats_order": 4}) is None


@pytest.mark.parametrize("info", [None, {}, {"ra_column": "ra"}])
def test_read_returns_none_when_missing(info) -> None:
    assert read_healpix_order(info) is None


@pytest.mark.parametrize("bad", [None, "", "abc", object()])
def test_read_returns_none_for_unusable_values(bad: object) -> None:
    assert read_healpix_order({HEALPIX_ORDER_KEY: bad}) is None


def test_require_returns_stored_value() -> None:
    assert require_healpix_order({HEALPIX_ORDER_KEY: 6}, "x") == 6


def test_require_raises_when_missing_and_names_source() -> None:
    with pytest.raises(HealpixOrderError, match="catalog_info.json"):
        require_healpix_order({"ra_column": "ra"}, "/lake/catalog_info.json")


def test_require_ignores_legacy_key() -> None:
    with pytest.raises(HealpixOrderError):
        require_healpix_order({"hats_order": 3}, "x")


def test_require_file_reads_sidecar(tmp_path) -> None:
    path = tmp_path / "catalog_info.json"
    path.write_text(json.dumps({HEALPIX_ORDER_KEY: 4}))
    assert require_healpix_order_file(path) == 4


def test_require_file_raises_when_file_missing(tmp_path) -> None:
    with pytest.raises(HealpixOrderError, match="not found"):
        require_healpix_order_file(tmp_path / "catalog_info.json")


def test_require_file_raises_when_key_missing(tmp_path) -> None:
    path = tmp_path / "catalog_info.json"
    path.write_text(json.dumps({"hats_order": 4}))
    with pytest.raises(HealpixOrderError, match="missing or invalid"):
        require_healpix_order_file(path)


def test_default_is_five() -> None:
    assert DEFAULT_HEALPIX_ORDER == 5


def test_tile_index_reads_new_key(tmp_path) -> None:
    from data_lake.discovery.tile_index import _read_healpix_order

    (tmp_path / "catalog_info.json").write_text(json.dumps({HEALPIX_ORDER_KEY: 4}))
    assert _read_healpix_order(tmp_path, "catalog") == 4


def test_tile_index_raises_on_legacy_key(tmp_path) -> None:
    from data_lake.discovery.tile_index import _read_healpix_order

    (tmp_path / "catalog_info.json").write_text(json.dumps({"hats_order": 3}))
    with pytest.raises(HealpixOrderError):
        _read_healpix_order(tmp_path, "catalog")


_LAKE_HEADER = '[lake]\nname = "t"\nroot = "."\n'


def test_config_partitioning_key_is_healpix_order(tmp_path) -> None:
    cfg_path = tmp_path / "lake_config.toml"
    cfg_path.write_text(_LAKE_HEADER + "[partitioning]\nhealpix_order = 6\n")
    assert LakeConfig.load(cfg_path).partitioning.healpix_order == 6


def test_config_legacy_partitioning_key_is_rejected(tmp_path) -> None:
    cfg_path = tmp_path / "lake_config.toml"
    cfg_path.write_text(_LAKE_HEADER + "[partitioning]\nhats_order = 6\n")
    with pytest.raises(LakeConfigInvalid, match="healpix_order"):
        LakeConfig.load(cfg_path)


def test_flag_helper_canonical_and_hidden_alias() -> None:
    import click
    from click.testing import CliRunner

    from data_lake.cli_utils import healpix_order_option

    @click.command()
    @healpix_order_option(default=5, type=int, help="HEALPix order.")
    def cmd(norder: int) -> None:
        click.echo(f"norder={norder}")

    runner = CliRunner()
    assert runner.invoke(cmd, []).output.strip() == "norder=5"
    assert runner.invoke(cmd, ["--healpix-order", "7"]).output.strip() == "norder=7"
    assert runner.invoke(cmd, ["--norder", "6"]).output.strip() == "norder=6"
    help_text = runner.invoke(cmd, ["--help"]).output
    assert "--healpix-order" in help_text
    assert "--norder" not in help_text


def test_region_cli_help_shows_healpix_order_only() -> None:
    from click.testing import CliRunner

    from data_lake.discovery.region_cli import cli

    help_text = CliRunner().invoke(cli, ["--help"]).output
    assert "--healpix-order" in help_text
    assert "--norder" not in help_text


def test_catalog_extract_norder_raises_without_key(tmp_path) -> None:
    from data_lake.export.catalog_extract import _lake_norder

    (tmp_path / "catalog_info.json").write_text(json.dumps({"ra_column": "ra"}))
    with pytest.raises(HealpixOrderError, match="catalog_info.json"):
        _lake_norder(tmp_path, None)
    assert _lake_norder(tmp_path, 3) == 3


def test_update_indices_norder_raises_without_key(tmp_path) -> None:
    from data_lake.ingest.update_catalog_indices import _resolve_catalog_norder

    (tmp_path / "catalog_info.json").write_text("{}")
    with pytest.raises(HealpixOrderError):
        _resolve_catalog_norder(tmp_path)
    assert _resolve_catalog_norder(tmp_path, override=4) == 4


def test_crossmatch_norder_falls_back_to_layout_only(tmp_path) -> None:
    from data_lake.io.crossmatch import _resolve_crossmatch_norder

    (tmp_path / "Norder=4").mkdir()
    assert _resolve_crossmatch_norder(tmp_path, None) == 4

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(HealpixOrderError):
        _resolve_crossmatch_norder(empty, None)


def test_survey_npix_raises_on_index_without_order(tmp_path) -> None:
    from data_lake.discovery.tile_index import survey_npix, tile_index_path

    path = tile_index_path(tmp_path, "ALLWISE", "catalog")
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"npix": [10, 20]}))
    with pytest.raises(HealpixOrderError):
        survey_npix(tmp_path, "ALLWISE", "catalog")


def test_link_validator_order_precedence(tmp_path) -> None:
    from data_lake.ingest.validate_catalog_spectra_link import _order_from_layer

    info = tmp_path / "catalog_info.json"
    info.write_text(json.dumps({HEALPIX_ORDER_KEY: 6}))
    assert _order_from_layer(info, 3) == 6  # stored value wins, as before

    info.write_text("{}")
    assert _order_from_layer(info, 3) == 3  # explicit flag when the key is absent
    with pytest.raises(HealpixOrderError):
        _order_from_layer(tmp_path / "missing.json", None)
