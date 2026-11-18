"""Tests for the ``healpix_order`` reader (single key, no legacy fallback)."""

from __future__ import annotations

import json

import pytest

from data_lake.config import LakeConfig, LakeConfigInvalid
from data_lake.healpix_order import (
    DEFAULT_HEALPIX_ORDER,
    HEALPIX_ORDER_KEY,
    read_healpix_order,
)


def test_key_is_read() -> None:
    assert read_healpix_order({HEALPIX_ORDER_KEY: 3}) == 3


def test_string_integer_is_accepted() -> None:
    assert read_healpix_order({HEALPIX_ORDER_KEY: "5"}) == 5


def test_legacy_key_is_ignored() -> None:
    assert read_healpix_order({"hats_order": 4}) is None
    assert read_healpix_order({"hats_order": 4}, default=9) == 9


def test_missing_key_returns_default() -> None:
    assert read_healpix_order({"ra_column": "ra"}, default=9) == 9
    assert read_healpix_order({}) is None


def test_none_info_returns_default() -> None:
    assert read_healpix_order(None, default=DEFAULT_HEALPIX_ORDER) == DEFAULT_HEALPIX_ORDER


@pytest.mark.parametrize("bad", [None, "", "abc", object()])
def test_unusable_values_return_default(bad: object) -> None:
    assert read_healpix_order({HEALPIX_ORDER_KEY: bad}, default=1) == 1


def test_default_is_five() -> None:
    assert DEFAULT_HEALPIX_ORDER == 5


def test_tile_index_reads_new_key(tmp_path) -> None:
    from data_lake.discovery.tile_index import _read_healpix_order

    (tmp_path / "catalog_info.json").write_text(json.dumps({HEALPIX_ORDER_KEY: 4}))
    assert _read_healpix_order(tmp_path, "catalog") == 4


def test_tile_index_ignores_legacy_key(tmp_path) -> None:
    from data_lake.discovery.tile_index import _read_healpix_order

    (tmp_path / "catalog_info.json").write_text(json.dumps({"hats_order": 3}))
    assert _read_healpix_order(tmp_path, "catalog") is None


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
