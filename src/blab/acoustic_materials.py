"""Physical acoustic-loss parameter helpers shared by the UI and solvers."""

from __future__ import annotations

import math
from typing import Any, Mapping

REGION_BULK_LOSS_FACTOR_KEY = "bulk_loss_factor"
FEM_BULK_LOSS_FACTOR_OPTIONS = (0.0, 0.002, 0.005, 0.01, 0.02, 0.05)
WALL_IMPEDANCE_KEY = "wall_impedance"
WALL_IMPEDANCE_MODEL_MIKI = "miki"
TRANSFER_IMPEDANCE_KEY = "transfer_impedance"
TRANSFER_IMPEDANCE_MODEL_RESISTANCE_MASS = "resistance_mass"
TRANSFER_IMPEDANCE_RESISTANCE_KEY = "resistance_pa_s_per_m"
TRANSFER_IMPEDANCE_MASS_KEY = "mass_kg_per_m2"

DEFAULT_WALL_LINING_THICKNESS_M = 0.03
DEFAULT_WALL_LINING_FLOW_RESISTIVITY_PA_S_PER_M2 = 5_000.0


def region_bulk_loss_factor(loss_model: Mapping[str, Any] | None) -> float:
    """Return and validate a bounded-region bulk loss factor."""

    raw = 0.0 if not loss_model else loss_model.get(REGION_BULK_LOSS_FACTOR_KEY, 0.0)
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("Region FEM bulk loss factor must be a finite number between 0 and 1.") from exc
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("Region FEM bulk loss factor must be a finite number between 0 and 1.")
    return value


def wall_impedance_parameters(parameters: Mapping[str, Any] | None) -> dict[str, float | str] | None:
    """Return a normalized Miki wall-lining description, or ``None``."""

    if not parameters or WALL_IMPEDANCE_KEY not in parameters:
        return None
    raw = parameters[WALL_IMPEDANCE_KEY]
    if not isinstance(raw, Mapping):
        raise ValueError("Wall impedance parameters must be an object.")
    model = str(raw.get("model", WALL_IMPEDANCE_MODEL_MIKI)).strip().lower()
    if model != WALL_IMPEDANCE_MODEL_MIKI:
        raise ValueError(f"Unsupported wall impedance model {model!r}; expected 'miki'.")
    try:
        thickness_m = float(raw.get("thickness_m", DEFAULT_WALL_LINING_THICKNESS_M))
        flow_resistivity = float(
            raw.get(
                "flow_resistivity_pa_s_per_m2",
                DEFAULT_WALL_LINING_FLOW_RESISTIVITY_PA_S_PER_M2,
            )
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Wall lining thickness and airflow resistivity must be finite positive numbers.") from exc
    if not math.isfinite(thickness_m) or thickness_m <= 0.0:
        raise ValueError("Wall lining thickness must be a finite positive number.")
    if not math.isfinite(flow_resistivity) or flow_resistivity <= 0.0:
        raise ValueError("Wall lining airflow resistivity must be a finite positive number.")
    return {
        "model": WALL_IMPEDANCE_MODEL_MIKI,
        "thickness_m": thickness_m,
        "flow_resistivity_pa_s_per_m2": flow_resistivity,
    }


def miki_wall_impedance_parameters(
    *,
    thickness_m: float = DEFAULT_WALL_LINING_THICKNESS_M,
    flow_resistivity_pa_s_per_m2: float = DEFAULT_WALL_LINING_FLOW_RESISTIVITY_PA_S_PER_M2,
) -> dict[str, dict[str, float | str]]:
    """Build normalized boundary parameters for a rigid-backed Miki layer."""

    normalized = wall_impedance_parameters(
        {
            WALL_IMPEDANCE_KEY: {
                "model": WALL_IMPEDANCE_MODEL_MIKI,
                "thickness_m": thickness_m,
                "flow_resistivity_pa_s_per_m2": flow_resistivity_pa_s_per_m2,
            }
        }
    )
    assert normalized is not None
    return {WALL_IMPEDANCE_KEY: normalized}


def transfer_impedance_parameters(parameters: Mapping[str, Any] | None) -> dict[str, float | str] | None:
    """Return a normalized interface transfer-impedance layer, or ``None``.

    The layer is a thin two-sided resistive/inertive sheet across an FEM-BEM
    interface: ``p_bounded - p_unbounded = (R + i omega M) v_n`` in
    ``exp(+i omega t)``, with ``v_n`` along the bounded region's outward
    normal. ``R`` (Pa s/m) and ``M`` (kg/m^2) are referred to the interface
    face area. Every key is required; unknown keys are refused, matching the
    engine's parser.
    """

    if not parameters or TRANSFER_IMPEDANCE_KEY not in parameters:
        return None
    raw = parameters[TRANSFER_IMPEDANCE_KEY]
    if not isinstance(raw, Mapping):
        raise ValueError("Transfer impedance parameters must be an object.")
    expected = {"model", TRANSFER_IMPEDANCE_RESISTANCE_KEY, TRANSFER_IMPEDANCE_MASS_KEY}
    if set(raw) != expected:
        raise ValueError(
            "Transfer impedance parameters must have exactly the keys "
            + ", ".join(sorted(expected))
            + f"; found {', '.join(sorted(map(str, raw))) or 'none'}."
        )
    if raw["model"] != TRANSFER_IMPEDANCE_MODEL_RESISTANCE_MASS:
        raise ValueError(
            f"Unsupported transfer impedance model {raw['model']!r}; "
            f"expected {TRANSFER_IMPEDANCE_MODEL_RESISTANCE_MASS!r}."
        )
    values = {}
    for key in (TRANSFER_IMPEDANCE_RESISTANCE_KEY, TRANSFER_IMPEDANCE_MASS_KEY):
        value = raw[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Transfer impedance {key} must be a finite non-negative number.")
        value = float(value)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"Transfer impedance {key} must be a finite non-negative number.")
        values[key] = value
    return {"model": TRANSFER_IMPEDANCE_MODEL_RESISTANCE_MASS, **values}


def resistance_mass_transfer_impedance_parameters(
    *,
    resistance_pa_s_per_m: float,
    mass_kg_per_m2: float = 0.0,
) -> dict[str, dict[str, float | str]]:
    """Build normalized boundary parameters for a resistance-mass interface layer."""

    normalized = transfer_impedance_parameters(
        {
            TRANSFER_IMPEDANCE_KEY: {
                "model": TRANSFER_IMPEDANCE_MODEL_RESISTANCE_MASS,
                TRANSFER_IMPEDANCE_RESISTANCE_KEY: resistance_pa_s_per_m,
                TRANSFER_IMPEDANCE_MASS_KEY: mass_kg_per_m2,
            }
        }
    )
    assert normalized is not None
    return {TRANSFER_IMPEDANCE_KEY: normalized}
