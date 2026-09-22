# Space asset sources

Retrieved on 2026-09-22. Files in `source/` are immutable source material;
runtime-ready derivatives must be written outside this directory and retain
the attribution below.

## Hipparcos Main Catalogue

- Catalogue: `I/239/hip_main`, 118,218 source records
- Publisher: European Space Agency (1997)
- Catalogue page: <https://vizier.cds.unistra.fr/viz-bin/cat/I/239>
- Download mirror:
  <https://vizier.cfa.harvard.edu/ftp/cats/I/239/version_cd/cats/hip_main.dat.gz>
- VizieR licence: CC-BY-NC-3.0 IGO
- Local files:
  - `source/hip_main.dat.gz`
  - `source/hipparcos_ReadMe.txt`
- SHA-256:
  - `023555ca4c3781372c1dcfc9dc7f2d468f6c0f867ce3eb421f60f722d2a6e6af`
  - `7a7d54903d16460a76c240f07c41b019d57becfd27b5dfa2908e828f52f90c27`

The stable presets use Hipparcos in place of BSC5. Of the 118,218 source
records, 117,955 contain usable ICRS position and visual magnitude fields.
Stars through V=6.5 remain individual HDR point sources; 109,085 fainter
records are baked into a fixed celestial texture.

## Bright Star Catalogue, 5th Revised Edition (BSC5)

This source is retained for provenance and fallback comparisons but is no
longer selected by the stable presets.

- Catalogue: `V/50/catalog`, 9,110 rows
- Authors: Hoffleit, D. and Warren, W. H., Jr. (1991)
- Source: <https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=V/50>
- Download query: <https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source=V%2F50%2Fcatalog&-out.all&-out.max=unlimited>
- Usage terms: <https://cds.unistra.fr/vizier-org/licences_vizier.html>
- Requested acknowledgement: "This research has made use of the VizieR
  catalogue access tool, CDS, Strasbourg, France (DOI:
  10.26093/cds/vizier)."
- Local files:
  - `source/bsc5_catalog.tsv`
  - `source/bsc5_ReadMe.txt`
- SHA-256:
  - `981d6b77b379847ec9f00d7ba2ffab2feb2a04e4589717c08479b9d61a45bedd`
  - `44fd9c73e2eecad0beb47bdfa3f01c60fd43f93d6964198e31fcd48732de5b33`

The import path should consume J2000 right ascension/declination, visual
magnitude and B-V colour. Proper motion is present but is not required for the
initial fixed-epoch renderer.

## ESO Milky Way panorama

- Image id: `eso0932a`, 6000 x 3000 JPEG
- Credit: **ESO/S. Brunier**
- Source page: <https://www.eso.org/public/images/eso0932a/>
- Download: <https://cdn.eso.org/images/large/eso0932a.jpg>
- ESO usage policy: <https://www.eso.org/public/copyright/>
- Local file: `source/eso0932a_large.jpg`
- SHA-256:
  `60400c92c54b7c1bd12299c69e83b16e5b6256e7dabacc478c021758ecd28179`

The downloaded JPEG is display-referred source material, not a linear-HDR
runtime texture. Preprocessing must linearize it, suppress resolvable point
stars to avoid duplication with BSC5, downsample it, and preserve the full
credit in distributed documentation or an in-application credits view.

## Generated runtime assets

Install the optional preprocessing dependency with `pip install -e .[assets]`,
then run `python tools/build_space_assets.py` from the repository root. The
runtime renderer itself does not require Pillow. The tool produces:

- `runtime/bsc5_stars.npz`: 9,096 valid-position/photometry records from the
  9,110-row source catalogue; the default magnitude limit selects 8,404.
- `runtime/hipparcos_stars.npz`: 117,955 valid Hipparcos records; runtime uses
  the 8,870 records through V=6.5 as individually projected HDR points.
- `runtime/hipparcos_faint_4096x2048.npz`: RGBE-encoded linear-HDR celestial
  texture containing the remaining 109,085 stars. It costs one lookup per
  output pixel rather than 109,085 per-star projection/splat operations.
- `runtime/hipparcos_faint_4096x2048_preview.jpg`: exposure-adjusted inspection
  image; runtime does not load it.
- `runtime/milky_way_2048x1024.npz`: sRGB-encoded compact source for runtime
  linearization and GPU upload. Point-like high-frequency energy is clamped
  and median-filtered so BSC5 remains the authoritative resolvable-star layer.
- `runtime/milky_way_2048x1024_preview.jpg`: inspection only; runtime code does
  not load it.

The generated files are deterministic derivatives of the sources above and
retain their attribution and usage requirements.
