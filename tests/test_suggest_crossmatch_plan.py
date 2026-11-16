"""Unit tests for crossmatch plan suggest heuristics."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from click.testing import CliRunner

from data_lake.discovery.area_plan import (
    list_catalog_partners_from_registry,
    strip_crossmatch_plan_annotations,
    suggest_crossmatch_plan,
    suggest_partner_match,
)
from data_lake.lake_registry import registry_path


class TestSuggestPartnerMatch:
    def test_shared_targetid(self) -> None:
        partner, reason = suggest_partner_match(
            ["TARGETID", "RA", "DEC"],
            ["TARGETID", "Z"],
            partner_survey="DESI_DR1",
        )
        assert partner["match_mode"] == "column"
        assert partner["match_col_a"] == "TARGETID"
        assert partner["match_col_b"] == "TARGETID"
        assert "TARGETID" in reason

    def test_gaia_fk(self) -> None:
        partner, reason = suggest_partner_match(
            ["OBJECT_ID", "GAIA_ID", "right_ascension"],
            ["source_id", "ra", "dec", "designation"],
            partner_survey="GAIA_DR3_source",
            partner_manifest={"link_id_mode": "label:designation"},
        )
        assert partner["match_mode"] == "column"
        assert partner["match_col_a"] == "GAIA_ID"
        assert partner["match_col_b"] == "source_id"
        assert "GAIA_ID" in reason

    def test_sky_fallback(self) -> None:
        partner, reason = suggest_partner_match(
            ["OBJECT_ID", "FLUX"],
            ["w1mpro", "ra", "dec"],
            partner_survey="ALLWISE",
            default_sky_radius_arcsec=2.5,
        )
        assert partner == {
            "survey": "ALLWISE",
            "match_mode": "sky",
            "radius_arcsec": 2.5,
        }
        assert "sky" in reason.casefold()

    def test_never_joins_source_id_hash(self) -> None:
        partner, _reason = suggest_partner_match(
            ["_source_id", "RA"],
            ["_source_id", "ra"],
            partner_survey="OTHER",
        )
        assert partner["match_mode"] == "sky"
        assert partner.get("match_col_a") != "_source_id"


class TestStripAnnotations:
    def test_strips_suggestions(self) -> None:
        draft = {
            "base_catalog": "EUCLID_Q1_MER",
            "partners": [
                {
                    "survey": "GAIA_DR3_source",
                    "match_mode": "column",
                    "match_col_a": "GAIA_ID",
                    "match_col_b": "source_id",
                },
                {
                    "survey": "ALLWISE",
                    "match_mode": "sky",
                    "radius_arcsec": 1.0,
                },
            ],
            "reuse_existing": True,
            "_suggestions": {"ALLWISE": "sky: …"},
        }
        clean = strip_crossmatch_plan_annotations(draft)
        assert "_suggestions" not in clean
        assert clean["base_catalog"] == "EUCLID_Q1_MER"
        assert len(clean["partners"]) == 2


def _write_manifest(catalog_root: Path, survey: str, columns: list[str], **extra) -> None:
    catalog_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "manifest_version": "1",
        "survey": survey,
        "modality": "catalog",
        "columns": [{"name": c, "dtype": "string", "nullable": True, "role": "science"}
                    for c in columns],
        **extra,
    }
    (catalog_root / "schema_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


class TestSuggestCrossmatchPlanLake:
    def test_end_to_end_tmp_lake(self, tmp_path: Path) -> None:
        lake = tmp_path / "lake"
        _write_manifest(
            lake / "catalogs" / "EUCLID_Q1_MER",
            "EUCLID_Q1_MER",
            ["OBJECT_ID", "GAIA_ID", "right_ascension", "declination"],
            link_id_mode="column:OBJECT_ID",
        )
        _write_manifest(
            lake / "catalogs" / "GAIA_DR3_source",
            "GAIA_DR3_source",
            ["source_id", "designation", "ra", "dec"],
            link_id_mode="label:designation",
        )
        _write_manifest(
            lake / "catalogs" / "ALLWISE",
            "ALLWISE",
            ["designation", "ra", "dec", "w1mpro"],
            link_id_mode="label:designation",
        )
        plan = suggest_crossmatch_plan(
            lake,
            "EUCLID_Q1_MER",
            ["GAIA_DR3_source", "ALLWISE"],
        )
        assert plan["base_catalog"] == "EUCLID_Q1_MER"
        by_name = {p["survey"]: p for p in plan["partners"]}
        assert by_name["GAIA_DR3_source"]["match_mode"] == "column"
        assert by_name["GAIA_DR3_source"]["match_col_a"] == "GAIA_ID"
        assert by_name["GAIA_DR3_source"]["match_col_b"] == "source_id"
        assert by_name["ALLWISE"]["match_mode"] == "sky"
        assert "GAIA_DR3_source" in plan["_suggestions"]
        clean = strip_crossmatch_plan_annotations(plan)
        assert "_suggestions" not in clean


def _write_mini_registry(
    lake: Path,
    rows: list[dict],
) -> Path:
    """Write a minimal ``surveys.parquet`` for partner-discovery tests."""
    path = registry_path(lake)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)
    return path


class TestListCatalogPartnersFromRegistry:
    def test_excludes_base_sorts_by_rows_and_kind(self, tmp_path: Path) -> None:
        lake = tmp_path / "lake"
        _write_mini_registry(
            lake,
            [
                {
                    "survey": "BASE",
                    "modality": "catalog",
                    "kind": "ingested",
                    "total_rows": 10,
                },
                {
                    "survey": "SMALL",
                    "modality": "catalog",
                    "kind": "ingested",
                    "total_rows": 100,
                },
                {
                    "survey": "LARGE",
                    "modality": "catalog",
                    "kind": "ingested",
                    "total_rows": 1_000_000,
                },
                {
                    "survey": "PROD",
                    "modality": "catalog",
                    "kind": "product",
                    "total_rows": 50,
                },
                {
                    "survey": "SPECTRA_ONLY",
                    "modality": "spectra",
                    "kind": "ingested",
                    "total_rows": 5,
                },
            ],
        )
        partners = list_catalog_partners_from_registry(
            lake, base_catalog="BASE", kind="ingested"
        )
        assert partners == ["SMALL", "LARGE"]

        with_exclude = list_catalog_partners_from_registry(
            lake, base_catalog="BASE", kind="ingested", exclude=["LARGE"]
        )
        assert with_exclude == ["SMALL"]

        products = list_catalog_partners_from_registry(
            lake, base_catalog="BASE", kind="product"
        )
        assert products == ["PROD"]

        all_kinds = list_catalog_partners_from_registry(
            lake, base_catalog="BASE", kind="all"
        )
        assert all_kinds == ["PROD", "SMALL", "LARGE"]

    def test_suggest_from_registry(self, tmp_path: Path) -> None:
        lake = tmp_path / "lake"
        _write_manifest(
            lake / "catalogs" / "BASE",
            "BASE",
            ["OBJECT_ID", "RA", "DEC"],
            link_id_mode="column:OBJECT_ID",
        )
        _write_manifest(
            lake / "catalogs" / "SMALL",
            "SMALL",
            ["id", "ra", "dec"],
            link_id_mode="column:id",
        )
        _write_manifest(
            lake / "catalogs" / "LARGE",
            "LARGE",
            ["id", "ra", "dec"],
            link_id_mode="column:id",
        )
        _write_mini_registry(
            lake,
            [
                {
                    "survey": "BASE",
                    "modality": "catalog",
                    "kind": "ingested",
                    "total_rows": 10,
                },
                {
                    "survey": "SMALL",
                    "modality": "catalog",
                    "kind": "ingested",
                    "total_rows": 100,
                },
                {
                    "survey": "LARGE",
                    "modality": "catalog",
                    "kind": "ingested",
                    "total_rows": 9_000_000,
                },
            ],
        )
        plan = suggest_crossmatch_plan(
            lake,
            "BASE",
            partners_from_registry=True,
        )
        assert [p["survey"] for p in plan["partners"]] == ["SMALL", "LARGE"]
        assert plan["_suggestions"]["partner_order_reason"]

    def test_cli_partners_from_registry(self, tmp_path: Path) -> None:
        from data_lake.io.crossmatch_cli import cli

        lake = tmp_path / "lake"
        _write_manifest(
            lake / "catalogs" / "BASE",
            "BASE",
            ["OBJECT_ID", "RA", "DEC"],
        )
        _write_manifest(
            lake / "catalogs" / "SMALL",
            "SMALL",
            ["id", "ra", "dec"],
        )
        _write_mini_registry(
            lake,
            [
                {
                    "survey": "BASE",
                    "modality": "catalog",
                    "kind": "ingested",
                    "total_rows": 1,
                },
                {
                    "survey": "SMALL",
                    "modality": "catalog",
                    "kind": "ingested",
                    "total_rows": 2,
                },
            ],
        )
        out = tmp_path / "plan.json"
        result = CliRunner().invoke(
            cli,
            [
                str(lake),
                "--suggest-plan",
                "--base",
                "BASE",
                "--partners-from-registry",
                "-o",
                str(out),
            ],
        )
        assert result.exit_code == 0, result.output
        draft = json.loads(out.read_text(encoding="utf-8"))
        assert draft["base_catalog"] == "BASE"
        assert [p["survey"] for p in draft["partners"]] == ["SMALL"]
