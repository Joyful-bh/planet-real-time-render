"""Diagnostic views for isolating atmosphere pipeline stages.

The values are part of the preview/debug contract only.  They do not alter
the physical atmosphere model or the contents of any LUT.
"""

from __future__ import annotations

from enum import IntEnum


class AtmosphereDiagnosticView(IntEnum):
    """Select the final diagnostic composition shown by the preview."""

    COMPOSITE = 0
    SKY_VIEW = 1
    CAMERA_TRANSMITTANCE = 2
    TRANSMITTANCE_LUT = 3
    MULTI_SCATTERING_LUT = 4
    AERIAL_SCATTERING = 5
    AERIAL_TRANSMITTANCE = 6
    SURFACE_MASK = 7

    @property
    def label(self) -> str:
        return {
            self.COMPOSITE: "Composite",
            self.SKY_VIEW: "Sky-view radiance",
            self.CAMERA_TRANSMITTANCE: "Camera transmittance",
            self.TRANSMITTANCE_LUT: "Transmittance LUT",
            self.MULTI_SCATTERING_LUT: "Multi-scattering LUT",
            self.AERIAL_SCATTERING: "Aerial scattering",
            self.AERIAL_TRANSMITTANCE: "Aerial transmittance",
            self.SURFACE_MASK: "Surface mask",
        }[self]

    @property
    def is_radiance(self) -> bool:
        """Whether this view needs exposure and display tone mapping."""

        return self in {
            self.COMPOSITE,
            self.SKY_VIEW,
            self.MULTI_SCATTERING_LUT,
            self.AERIAL_SCATTERING,
        }

    @property
    def uses_bloom(self) -> bool:
        """Bloom is a camera effect and belongs only to final composition."""

        return self is self.COMPOSITE


__all__ = ["AtmosphereDiagnosticView"]
