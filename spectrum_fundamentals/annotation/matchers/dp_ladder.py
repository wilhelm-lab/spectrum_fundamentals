"""Dynamic-programming matcher over the b/y ion ladder."""

# Per ion series at a fixed charge, a shortest-path DP walks the theoretical fragment ladder in
# increasing m/z and picks, per slot, the observed peak minimising an additive cost: an emission term
# (ppm / s)^2 - w_I * intensity, a transition term w_L * (gap_ppm / s)^2 penalising mismatch between
# the observed and theoretical gap to the previous assignment (drift-robust -- a constant calibration
# offset cancels in the gap), and a flat skip penalty. Monotonicity keeps assigned m/z increasing.
#
# Unlike `nearest` (per-slot closest m/z) and `global_ransac` (one global calibration line), this uses
# the ladder's sequential structure, so it can reject in-window noise that breaks the gap pattern and
# bridge missing fragments. The cost is purely geometric: it never sees Prosit's predicted intensities,
# since it runs before the spectral angle is computed. Like `global_ransac` it degrades to `nearest`
# on short ladders and when `min_match_fraction` is not met.

import logging
import numbers
from typing import Any

import numpy as np
import pandas as pd

from .nearest import nearest_resolver

logger = logging.getLogger(__name__)

# Columns every candidate row must carry (shared with the other resolvers).
_REQUIRED_COLUMNS = ("ion_type", "no", "charge", "exp_mass", "theoretical_mass", "intensity")
# A fragment slot is identified by this triple; a ladder by (ion_type, charge).
_SLOT_COLUMNS = ["ion_type", "no", "charge"]

# ppm scale used when it cannot be derived from a ppm matching tolerance
# (tolerance unset or given in Da). ~Orbitrap MS2 full window.
_DEFAULT_PPM_SCALE = 20.0
# Infinitesimal reward per assignment so that, on an exact cost tie, matching a
# peak beats skipping it -- makes clean spectra reproduce ``nearest`` exactly.
_MATCH_REWARD_EPS = 1e-9
# Emission residual (ppm) substituted for a non-finite one so the candidate's
# cost is huge-but-finite -- it gets skipped instead of poisoning the DP.
_NONFINITE_EMIT_PPM = 1e6

# Sentinels for the DP. ``_START`` is the "no real peak assigned yet" anchor;
# ``_SKIP`` marks a slot left unmatched.
_START = object()
_SKIP = object()


def dp_ladder_resolver(
    candidates: list[dict[str, Any]] | None,
    peaks_masses: np.ndarray | None = None,
    peaks_intensity: np.ndarray | None = None,
    unmod_sequence: str | None = None,
    *,
    ppm_scale: float | None = None,
    skip_penalty: float = 1.0,
    ladder_weight: float = 2.0,
    intensity_weight: float = 0.0,
    unique_peak: bool = True,
    min_match_fraction: float = 0.0,
    mass_tolerance: float | None = None,
    unit_mass_tolerance: str | None = None,
    **kwargs: Any,
) -> tuple[pd.DataFrame, int]:
    """Resolve candidates by DP alignment to the b/y ion ladder.

    :param candidates: rows from ``match_peaks``; may be ``None`` or empty
    :param peaks_masses: unused; part of the resolver contract
    :param peaks_intensity: unused; part of the resolver contract
    :param unmod_sequence: unused; part of the resolver contract
    :param ppm_scale: ppm scale ``s`` of the emission and transition penalties; a peak ``s`` ppm off
        theoretical costs 1. Defaults to ``mass_tolerance`` when that is in ppm, else 20 ppm.
    :param skip_penalty: flat cost of leaving a fragment unmatched. At the default scale a
        ladder-consistent in-window peak always beats skipping, so only inconsistent peaks are dropped.
    :param ladder_weight: weight ``w_L`` of the gap-consistency term; ``0`` reduces the DP to per-slot
        ppm picking, larger values are more robust to global mass drift
    :param intensity_weight: weight ``w_I`` rewarding intense *observed* peaks; ``0`` keeps the cost
        purely geometric
    :param unique_peak: let each observed peak back at most one fragment across ladders, resolving
        cross-series collisions by ascending ppm error (within a ladder monotonicity already does)
    :param min_match_fraction: in ``[0, 1]``; below this share of matched slots the whole spectrum is
        deferred to ``nearest``. ``0`` disables the floor.
    :param mass_tolerance: tolerance ``ppm_scale`` is derived from when not given explicitly
    :param unit_mass_tolerance: unit of ``mass_tolerance`` (``"ppm"`` or ``"da"``)
    :raises ValueError: if a hyperparameter is out of range or a candidate row misses a column
    :return: ``(matched_peaks_df, n_dropped)``, where ``n_dropped`` counts input rows that did not
        survive. The frame carries the contract columns plus a ``ppm_residual`` diagnostic.
    """
    _validate_hyperparameters(skip_penalty, ladder_weight, intensity_weight, min_match_fraction)
    scale = _resolve_ppm_scale(ppm_scale, mass_tolerance, unit_mass_tolerance)

    if not candidates:
        return pd.DataFrame(), 0
    n_input = len(candidates)

    df = pd.DataFrame(candidates)
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"dp_ladder: candidate rows missing required columns {missing}")

    # Drop rows that can't yield a finite ppm residual (mirrors global_ransac's
    # data hygiene): non-finite masses or a non-positive theoretical mass.
    exp = df["exp_mass"].to_numpy(dtype=float)
    theo = df["theoretical_mass"].to_numpy(dtype=float)
    valid = np.isfinite(exp) & np.isfinite(theo) & (theo > 0)
    n_invalid = int((~valid).sum())
    if n_invalid:
        logger.warning(
            "dp_ladder: dropping %d/%d candidate rows with non-finite or non-positive mass",
            n_invalid,
            n_input,
        )
        df = df.loc[valid].reset_index(drop=True)

    if len(df) == 0:
        return pd.DataFrame(columns=df.columns), n_input

    df["ppm_residual"] = (
        1e6 * (df["exp_mass"].to_numpy() - df["theoretical_mass"].to_numpy()) / df["theoretical_mass"].to_numpy()
    )

    n_slots = df[_SLOT_COLUMNS].drop_duplicates().shape[0]

    # Run the DP independently on each ladder (one ion series at one charge).
    chosen_idx = _run_dp_assignment(df, scale, skip_penalty, ladder_weight, intensity_weight)

    if unique_peak:
        chosen_idx = _greedy_unique_peaks(df, chosen_idx)

    # Safety floor: a DP that explains too few slots is more likely mis-tuned
    # than informative; fall back to nearest rather than return a sparse set.
    matched_slots = df.loc[chosen_idx, _SLOT_COLUMNS].drop_duplicates().shape[0]
    if min_match_fraction > 0 and matched_slots < min_match_fraction * n_slots:
        logger.info(
            "dp_ladder: matched only %d/%d slots (< %.2f), falling back to nearest",
            matched_slots,
            n_slots,
            min_match_fraction,
        )
        return _fallback_to_nearest(df, n_input)

    chosen = df.loc[chosen_idx].reset_index(drop=True)
    return chosen, n_input - len(chosen)


def _run_dp_assignment(
    df: pd.DataFrame,
    scale: float,
    skip_penalty: float,
    ladder_weight: float,
    intensity_weight: float,
    emit_col: str = "ppm_residual",
) -> list[int]:
    """Run the per-ladder DP across every ``(ion_type, charge)`` series in ``df``.

    ``emit_col`` selects the ppm column used for the emission penalty: the raw
    ``"ppm_residual"`` for the plain ladder DP, or a drift-corrected
    deviation-from-line column for the calibrated variant (``dp_calibrated``).
    Returns the chosen df indices (at most one observed peak per fragment slot).
    """
    chosen_idx: list[int] = []
    for _, ladder_df in df.groupby(["ion_type", "charge"], sort=False):
        chosen_idx.extend(_resolve_ladder(ladder_df, scale, skip_penalty, ladder_weight, intensity_weight, emit_col))
    return chosen_idx


def _resolve_ladder(
    ladder_df: pd.DataFrame,
    scale: float,
    skip_penalty: float,
    ladder_weight: float,
    intensity_weight: float,
    emit_col: str = "ppm_residual",
) -> list[int]:
    """Run the alignment DP over one ion series, returning chosen df indices.

    ``ladder_df`` holds every candidate row for a single ``(ion_type, charge)``
    pair; it may contain several candidate peaks per fragment slot. Slots are
    ordered by theoretical m/z (equivalently fragment number). ``emit_col`` is
    the column whose value drives the per-peak emission penalty (see
    :func:`_run_dp_assignment`).
    """
    # Build ordered slots, each carrying its candidate peaks.
    slots: list[dict[str, Any]] = []
    for theo, slot_df in sorted(ladder_df.groupby("theoretical_mass", sort=False), key=lambda kv: kv[0]):
        cands = []
        for idx, row in slot_df.iterrows():
            # A non-finite intensity would poison the additive cost (note that
            # ``0.0 * nan == nan``, so even intensity_weight=0 is not safe);
            # treat it as zero reward. The untouched value still reaches output.
            intensity = float(row["intensity"])
            if not np.isfinite(intensity):
                intensity = 0.0
            # The emission residual is finite for both callers (filtered inputs);
            # guard defensively so a stray non-finite value forces a skip rather
            # than poisoning every cost it touches.
            eppm = float(row[emit_col])
            if not np.isfinite(eppm):
                eppm = _NONFINITE_EMIT_PPM
            cands.append(
                {
                    "idx": int(idx),
                    "mz": float(row["exp_mass"]),
                    "intensity": intensity,
                    "eppm": eppm,
                }
            )
        slots.append({"theo": float(theo), "cands": cands})

    n = len(slots)
    if n == 0:
        return []

    # Forward DP. ``prev_states`` maps an anchor key -> (cost, prev_key, option)
    # where the anchor is the last *real* peak assigned so far. ``layers[k]`` is
    # the state table after processing slot k, used for backtracking.
    layers: list[dict] = []
    prev_states: dict = {_START: (0.0, None, None)}

    for k in range(n):
        theo_k = slots[k]["theo"]
        new_states: dict = {}
        for akey, (acost, _, _) in prev_states.items():
            # Option 1: skip slot k -- anchor (last real peak) is unchanged.
            _relax(new_states, akey, acost + skip_penalty, akey, _SKIP)

            # Option 2: assign one of slot k's candidate peaks.
            if akey is _START:
                a_mz = None
                a_theo = None
            else:
                aj, ali = akey
                a_mz = slots[aj]["cands"][ali]["mz"]
                a_theo = slots[aj]["theo"]
            for li, c in enumerate(slots[k]["cands"]):
                if a_mz is not None:
                    d_obs = c["mz"] - a_mz
                    if d_obs <= 0:  # hard monotonicity: peaks must increase in m/z
                        continue
                    d_theo = theo_k - a_theo
                    gap_ppm = 1e6 * (d_obs - d_theo) / theo_k
                    trans = ladder_weight * (gap_ppm / scale) ** 2
                else:
                    trans = 0.0  # first real peak in the ladder: emission only
                emit = (c["eppm"] / scale) ** 2 - intensity_weight * c["intensity"] - _MATCH_REWARD_EPS
                _relax(new_states, (k, li), acost + emit + trans, akey, li)
        layers.append(new_states)
        prev_states = new_states

    # Pick the lowest-cost end state and backtrack the per-slot decisions.
    best_key = min(prev_states, key=lambda key: prev_states[key][0])
    chosen: list[int] = []
    key = best_key
    for k in range(n - 1, -1, -1):
        _, prev_key, option = layers[k][key]
        if option is not _SKIP and option is not None:
            chosen.append(slots[k]["cands"][option]["idx"])
        key = prev_key
    return chosen


def _relax(states: dict, key: Any, cost: float, prev_key: Any, option: Any) -> None:
    """Keep the minimum-cost route to ``key`` (stable on exact ties)."""
    current = states.get(key)
    if current is None or cost < current[0]:
        states[key] = (cost, prev_key, option)


def _greedy_unique_peaks(df: pd.DataFrame, chosen_idx: list[int], sort_col: str = "ppm_residual") -> list[int]:
    """Enforce one fragment per observed peak across ladders.

    Walks the chosen rows in ascending ``|sort_col|`` and keeps a row only while
    its ``exp_mass`` is still unclaimed. Resolves cross-series collisions (the
    same peak picked by, say, a b ladder and a y ladder); within a ladder the
    DP's monotonicity already guarantees distinct peaks. ``sort_col`` is the
    quantity the DP minimised (raw ppm for ``dp_ladder``, deviation-from-line
    for ``dp_calibrated``), so the better-fitting fragment wins the shared peak.
    """
    if not chosen_idx:
        return chosen_idx
    order = df.loc[chosen_idx, sort_col].abs().sort_values(kind="stable").index
    used_peaks: set[float] = set()
    kept: list[int] = []
    for idx in order:
        peak = float(df.at[idx, "exp_mass"])
        if peak in used_peaks:
            continue
        used_peaks.add(peak)
        kept.append(int(idx))
    return kept


def _fallback_to_nearest(df: pd.DataFrame, n_input: int) -> tuple[pd.DataFrame, int]:
    """Delegate to ``nearest_resolver`` with the diagnostic column stripped.

    Uses the already-filtered candidate set so the fallback inherits the same
    data hygiene; ``n_input`` is the original row count so ``n_dropped``
    reflects "input rows not in output", including non-finite drops.
    """
    cleaned = df.drop(columns=["ppm_residual"], errors="ignore")
    chosen, _ = nearest_resolver(candidates=cleaned.to_dict("records"))
    chosen = chosen.reset_index(drop=True)
    return chosen, n_input - len(chosen)


def _resolve_ppm_scale(
    ppm_scale: float | None,
    mass_tolerance: float | None,
    unit_mass_tolerance: str | None,
) -> float:
    """Resolve the ppm scale ``s`` for the emission/transition penalties.

    An explicit ``ppm_scale`` always wins (and is validated). Otherwise derive
    it from the matching tolerance: the tolerance itself when given in ppm
    (so an in-window peak has emission <= 1), else a fixed fallback.

    :raises ValueError: if an explicit ``ppm_scale`` is not a positive finite number.
    """
    if ppm_scale is not None:
        if not (isinstance(ppm_scale, numbers.Real) and np.isfinite(float(ppm_scale)) and ppm_scale > 0):
            raise ValueError(f"ppm_scale must be a positive finite number, got {ppm_scale!r}")
        return float(ppm_scale)

    if (
        unit_mass_tolerance is not None
        and str(unit_mass_tolerance).lower() == "ppm"
        and isinstance(mass_tolerance, numbers.Real)
        and np.isfinite(float(mass_tolerance))
        and mass_tolerance > 0
    ):
        return float(mass_tolerance)

    return _DEFAULT_PPM_SCALE


def _validate_hyperparameters(
    skip_penalty: float,
    ladder_weight: float,
    intensity_weight: float,
    min_match_fraction: float,
) -> None:
    if not (isinstance(skip_penalty, numbers.Real) and np.isfinite(float(skip_penalty)) and skip_penalty > 0):
        raise ValueError(f"skip_penalty must be a positive finite number, got {skip_penalty!r}")
    if not (isinstance(ladder_weight, numbers.Real) and np.isfinite(float(ladder_weight)) and ladder_weight >= 0):
        raise ValueError(f"ladder_weight must be a non-negative finite number, got {ladder_weight!r}")
    if not (
        isinstance(intensity_weight, numbers.Real) and np.isfinite(float(intensity_weight)) and intensity_weight >= 0
    ):
        raise ValueError(f"intensity_weight must be a non-negative finite number, got {intensity_weight!r}")
    if not (
        isinstance(min_match_fraction, numbers.Real)
        and np.isfinite(float(min_match_fraction))
        and 0.0 <= min_match_fraction <= 1.0
    ):
        raise ValueError(f"min_match_fraction must be a number in [0, 1], got {min_match_fraction!r}")
