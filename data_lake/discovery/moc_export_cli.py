"""``dl-export-moc`` – export a survey footprint (tiles or catalog rows) as a MOC."""

from __future__ import annotations

import logging
from pathlib import Path

from data_lake.discovery.moc_export import (
    MocFormat,
    MocSource,
    default_survey_moc_path,
    export_survey_moc,
    export_survey_mocs_from_registry,
)
from data_lake.discovery.region_cli import _region_from_args
from data_lake.schema_registry import MODALITY_CATALOG, MODALITY_CUTOUT, MODALITY_SPECTRA

log = logging.getLogger(__name__)

_ALL_MODALITIES = (MODALITY_CATALOG, MODALITY_SPECTRA, MODALITY_CUTOUT)


try:
    import click

    from data_lake.cli_utils import (
        config_option,
        configure_cli_logging,
        load_optional_config,
        logging_options,
        require_output_root,
        resolve_log_level,
        validate_quiet_verbose,
    )

    @click.command("dl-export-moc")
    @click.argument("output_root", type=click.Path(path_type=Path), required=False)
    @config_option
    @click.option(
        "--survey",
        default=None,
        help="Survey name (tile index / layer). Required unless --all-surveys.",
    )
    @click.option(
        "--all-surveys",
        is_flag=True,
        default=False,
        help=(
            "Export a MOC for every registry survey of --modality / --kind "
            "into shared/registry/mocs/ as "
            "<survey>.<modality>.Norder<N>.moc.* (-o not allowed)."
        ),
    )
    @click.option(
        "--kind",
        "registry_kind",
        type=click.Choice(["ingested", "product", "all"], case_sensitive=False),
        default="ingested",
        show_default=True,
        help="[all-surveys] Registry kind filter.",
    )
    @click.option(
        "--modality", default=MODALITY_CATALOG, show_default=True,
        type=click.Choice(_ALL_MODALITIES),
    )
    @click.option("--moc-order", type=int, required=True,
                  help="HEALPix order for the exported MOC (NESTED).")
    @click.option(
        "--from-tiles/--from-rows",
        "from_tiles",
        default=True,
        show_default=True,
        help=(
            "How to build the MOC. --from-tiles (default): populated HATS "
            "tiles (fast; needs --moc-order <= the survey HATS order). "
            "--from-rows: catalog RA/Dec streamed per tile (any --moc-order, "
            "including finer than HATS; catalog modality only)."
        ),
    )
    @click.option(
        "-o",
        "--output",
        "output_path",
        required=False,
        default=None,
        type=click.Path(path_type=Path),
        help=(
            "Output file (.fits, .json, or .txt). Default: "
            "shared/registry/mocs/<survey>.<modality>.Norder<N>.moc.<ext>"
        ),
    )
    @click.option(
        "--moc-format", "moc_format",
        type=click.Choice(["fits", "json", "ascii"]),
        default="fits", show_default=True,
    )
    @click.option("--from-area", default=None, help="Clip footprint to an area region.")
    @click.option("--cone", nargs=2, type=float, default=None,
                  help="Clip to cone: RA DEC (deg); needs --radius-arcsec.")
    @click.option("--radius-arcsec", type=float, default=None)
    @click.option("--bbox", nargs=4, type=float, default=None)
    @click.option("--npix", default=None, help="Clip to npix list (needs --norder).")
    @click.option("--norder", type=int, default=None)
    @click.option("--moc", "moc_path", type=click.Path(path_type=Path), default=None,
                  help="Clip to an input MOC file.")
    @click.option("--overwrite", is_flag=True, help="Replace an existing output file.")
    @click.option(
        "--continue-on-error/--fail-fast",
        default=True,
        show_default=True,
        help="[all-surveys] Continue after a per-survey failure (default).",
    )
    @logging_options
    def cli(
        output_root: Path | None,
        config_path: Path | None,
        survey: str | None,
        all_surveys: bool,
        registry_kind: str,
        modality: str,
        moc_order: int,
        from_tiles: bool,
        output_path: Path | None,
        moc_format: MocFormat,
        from_area: str | None,
        cone: tuple[float, float] | None,
        radius_arcsec: float | None,
        bbox: tuple[float, float, float, float] | None,
        npix: str | None,
        norder: int | None,
        moc_path: Path | None,
        overwrite: bool,
        continue_on_error: bool,
        log_file: Path | None,
        quiet: bool,
        verbose: bool,
    ) -> None:
        """Export a survey footprint as an IVOA MOC (optional region clip).

        The build method is explicit: --from-tiles (default) uses populated
        HATS tiles and needs --moc-order <= the survey HATS order;
        --from-rows uses catalog RA/Dec and works at any order.

        Default output (when ``-o`` is omitted)::

            <lake>/shared/registry/mocs/<survey>.<modality>.Norder<N>.moc.fits

        Examples::

            dl-export-moc --survey SDSS_DR17 --moc-order 5
            # → …/SDSS_DR17.catalog.Norder5.moc.fits
            dl-export-moc --survey SDSS_DR17 --moc-order 9 --from-rows
            dl-export-moc --all-surveys --moc-order 8
            dl-export-moc --survey DESI_DR1 --modality spectra \\
                --from-area MyCone --moc-order 8 -o desi_cone.moc.fits
        """
        validate_quiet_verbose(quiet, verbose)
        cfg = load_optional_config(config_path)
        configure_cli_logging(
            level=resolve_log_level(
                quiet=quiet, verbose=verbose,
                config_level=cfg.ingest.log_level if cfg else None,
            ),
            quiet=quiet,
            log_file=log_file,
        )
        lake = require_output_root(output_root, cfg)

        if all_surveys and survey:
            raise click.ClickException("Use only one of --survey or --all-surveys.")
        if not all_surveys and not survey:
            raise click.ClickException("Provide --survey or --all-surveys.")

        source: MocSource = "tiles" if from_tiles else "rows"
        if source == "rows" and modality != MODALITY_CATALOG:
            raise click.ClickException(
                f"--from-rows needs catalog RA/Dec rows; --modality {modality} "
                f"is not supported (use --modality {MODALITY_CATALOG} or "
                "--from-tiles)"
            )

        clip_selectors = [from_area, cone, bbox, npix, moc_path]
        n_clips = sum(x is not None for x in clip_selectors)
        if n_clips > 1:
            raise click.ClickException(
                "Use at most one region clip selector "
                "(--from-area | --cone | --bbox | --npix | --moc)"
            )

        if all_surveys:
            if output_path is not None:
                raise click.ClickException(
                    "--all-surveys writes canonical paths under "
                    "shared/registry/mocs/; do not pass -o / --output"
                )
            if n_clips:
                raise click.ClickException(
                    "--all-surveys exports full survey footprints; "
                    "region clips are not supported (use --survey + -o)"
                )
            try:
                bulk = export_survey_mocs_from_registry(
                    lake,
                    moc_order=moc_order,
                    modality=modality,
                    kind=registry_kind,
                    fmt=moc_format,
                    overwrite=overwrite,
                    continue_on_error=continue_on_error,
                    source=source,
                )
            except (ImportError, FileNotFoundError, ValueError) as exc:
                raise click.ClickException(str(exc)) from exc

            for r in bulk.results:
                if r.skipped:
                    click.echo(f"skip {r.survey}: exists → {r.output}")
                else:
                    click.echo(
                        f"ok {r.survey}: {r.n_cells:,} cells "
                        f"(order={r.moc_order}) → {r.output}"
                    )
            for survey_b, mod, msg in bulk.errors:
                click.echo(f"error {survey_b} ({mod}): {msg}", err=True)
            click.echo(
                f"Done. ok={bulk.n_ok} skip={bulk.n_skipped} "
                f"error={bulk.n_error} modality={modality} kind={registry_kind}"
            )
            if bulk.n_error:
                raise click.ClickException(
                    f"{bulk.n_error} MOC export(s) failed; see messages above"
                )
            return

        region = None
        if n_clips:
            try:
                region, _ = _region_from_args(
                    from_area=from_area, npix=npix, norder=norder, cone=cone,
                    radius_arcsec=radius_arcsec, bbox=bbox, moc=moc_path,
                    lake_root=lake,
                )
            except (ValueError, FileNotFoundError) as exc:
                raise click.ClickException(str(exc)) from exc
            if output_path is None:
                click.echo(
                    "Note: clipped MOC writing to canonical path; "
                    "prefer -o for a distinct filename.",
                    err=True,
                )

        resolved = (
            output_path
            if output_path is not None
            else default_survey_moc_path(
                lake,
                survey,  # type: ignore[arg-type]
                modality,
                moc_order=moc_order,
                fmt=moc_format,
            )
        )
        try:
            result = export_survey_moc(
                lake, survey, resolved,  # type: ignore[arg-type]
                modality=modality,
                moc_order=moc_order,
                region=region,
                fmt=moc_format,
                overwrite=overwrite,
                source=source,
            )
        except (ImportError, ValueError, FileNotFoundError, OSError, FileExistsError) as exc:
            raise click.ClickException(str(exc)) from exc

        click.echo(
            f"Wrote MOC ({result.format}, order={result.moc_order}, "
            f"{result.n_cells:,} input cells, max_order={result.max_order}) "
            f"→ {result.output}"
        )

except ImportError:
    cli = None  # type: ignore[misc, assignment]
