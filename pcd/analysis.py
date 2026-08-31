"""What to ask the solver for, and how to read the answer back.

Two analyses are supported, and a single solver run can produce both:

    tran  -> waveform.csv    time, voltage, current   (design against a target waveform)
    ac    -> ac.csv          frequency response       (design the match itself)

AC matters for a matching network because matching is a frequency-domain
problem: the quantity you actually want is the impedance looking into the
network, and how far it sits from the source impedance.

Like :mod:`pcd.spice`, this module is shared vocabulary — it imports nothing
else from ``pcd`` and knows nothing about cases being simulated or scored.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .component_models import ComponentObservation, observed_components
from .signals import harmonic_phasors, periodic_window, time_average

#: Reference impedance for reflection coefficient / VSWR, in ohms.
DEFAULT_Z0 = 50.0

AC_FILE = "ac.csv"
WAVEFORM_FILE = "waveform.csv"


@dataclass(frozen=True)
class AcSweep:
    """A frequency sweep, mirroring ngspice's `.ac` arguments."""

    sweep: str = "dec"  # dec | oct | lin
    points: int = 20  # per decade/octave, or total for lin
    start_hz: float = 1e6
    stop_hz: float = 1e8

    @classmethod
    def from_config(cls, cfg: dict[str, Any], params: dict[str, Any] | None = None) -> AcSweep:
        if "frequency_Hz" in cfg:
            frequency = float(_resolve_analysis_value(cfg["frequency_Hz"], params))
            return cls(sweep="lin", points=1, start_hz=frequency, stop_hz=frequency)
        return cls(
            sweep=str(cfg.get("sweep", "dec")),
            points=int(cfg.get("points", 20)),
            start_hz=float(cfg.get("start_Hz", 1e6)),
            stop_hz=float(cfg.get("stop_Hz", 1e8)),
        )

    def command(self) -> str:
        return f"ac {self.sweep} {self.points} {self.start_hz:g} {self.stop_hz:g}"


def _resolve_analysis_value(value: Any, params: dict[str, Any] | None) -> Any:
    """Resolve a bare or ``$`` parameter reference without owning case semantics."""

    if isinstance(value, str) and params is not None:
        key = value[1:] if value.startswith("$") else value
        if key in params:
            return params[key]
    return value


def ac_sweep(solver_cfg: dict[str, Any], params: dict[str, Any] | None = None) -> AcSweep | None:
    """The AC request, resolved for this scenario, or ``None``.

    ``ac.frequency_Hz`` is deliberately a one-point analysis.  It lets a CSV
    scenario table carry independent measured ``R+jX`` points without asking a
    one-frequency equivalent circuit to invent behavior between those points.
    """

    cfg = solver_cfg.get("ac")
    return AcSweep.from_config(cfg or {}, params) if "ac" in solver_cfg else None


def transient_requested(solver_cfg: dict[str, Any]) -> bool:
    """Whether this run should produce a time-domain waveform.

    Historically every case received a default transient analysis.  That
    remains the default for cases that declare neither analysis, while an
    explicit AC-only case no longer pays for an unrelated transient run.
    """

    return "tran" in solver_cfg or "ac" not in solver_cfg


#: Column name for the source-node voltage, always recorded so the real power
#: leaving the source can be computed without extra configuration.
SOURCE_VOLTAGE = "source_voltage_V"

#: The zero-volt source inserted in series with the load by
#: ``measurement.load_current: auto``, and the column its current lands in.
LOAD_AMMETER = "Vload_meter"
LOAD_CURRENT = "load_current_A"
AC_LOAD_VOLTAGE = "load_voltage_V"


def source_node(case: Any) -> str:
    """The node the source drives, i.e. its positive terminal."""

    data = case.data
    sources = data.get("sources") or ([data["source"]] if data.get("source") else [])
    return str(sources[0].get("p", "src")) if sources else "src"


def load_current(case: Any) -> str | None:
    """The waveform column holding the current into the load, if any.

    ``load_current: auto`` asks the platform to insert its own ammeter, which
    is the only way to get an exact figure out of a built-in circuit builder --
    they place matching elements, not measurement hardware.  Any other value
    names a column the case arranged for itself.
    """

    declared = (case.data.get("measurement", {}) or {}).get("load_current")
    if not declared:
        return None
    return LOAD_CURRENT if str(declared).lower() == "auto" else str(declared)


def _declared_probe_pairs(case: Any) -> list[tuple[str, str]]:
    """Return ``(vector, column)`` pairs from explicit and component probes."""

    meas = case.data.get("measurement", {}) or {}
    raw = meas.get("probes") or []
    if isinstance(raw, Mapping):
        pairs = [(str(vector), str(name)) for name, vector in raw.items()]
    elif isinstance(raw, list):
        pairs = [(str(vector), str(vector)) for vector in raw]
    else:
        raise ValueError("measurement.probes must be a list or a name-to-vector mapping")

    for component in observed_components(case):
        pairs.extend(
            [
                (component.voltage_vector, component.voltage_column),
                (component.current_vector, component.current_column),
            ]
        )
    names: dict[str, str] = {}
    for vector, name in pairs:
        if name in names and names[name] != vector:
            raise ValueError(f"probe column {name!r} names both {names[name]!r} and {vector!r}")
        names[name] = vector
    return [(vector, name) for name, vector in names.items()]


def probe_plan(case: Any) -> tuple[list[str], list[str]]:
    """What to record, and what to call it when reading it back.

    Returns ``(ngspice vectors, column names)`` as one pair so the netlist
    writer and the result parser cannot drift out of step.  The source-node
    voltage is always appended: it costs one vector, and without it no
    power-flow question can be answered at all.
    """

    pairs = _declared_probe_pairs(case)
    vectors = [vector for vector, _name in pairs]
    columns = [name for _vector, name in pairs]
    if load_current(case) == LOAD_CURRENT:
        vectors.append(f"i({LOAD_AMMETER})")
        columns.append(LOAD_CURRENT)
    return [*vectors, f"v({source_node(case)})"], [*columns, SOURCE_VOLTAGE]


def ac_probe_plan(case: Any) -> tuple[list[str], list[str]]:
    """Named extra vectors written after the standard AC source/load fields."""

    pairs = _declared_probe_pairs(case)
    vectors = [vector for vector, _name in pairs]
    columns = [name for _vector, name in pairs]
    if load_current(case) == LOAD_CURRENT:
        vectors.append(f"i({LOAD_AMMETER})")
        columns.append(LOAD_CURRENT)
    return vectors, columns


def control_lines(
    solver_cfg: dict[str, Any],
    output_vector: str,
    source: str,
    drive_node: str,
    probes: list[str],
    params: dict[str, Any] | None = None,
    ac_probes: list[str] | None = None,
) -> list[str]:
    """The `.control` block: run each requested analysis and write its file."""

    voltage_vector = output_vector if output_vector.strip().lower().startswith("v(") else f"v({output_vector})"
    vectors = [voltage_vector, f"i({source})", *probes]
    sweep = ac_sweep(solver_cfg, params)
    # The declared output is part of the minimum AC result, even when no
    # optional probes are requested.  This keeps source-to-output gain and
    # phase available for later interpretation without asking users to add a
    # figure-specific probe.
    ac_extras = [voltage_vector, *(ac_probes or [])]
    saved = list(vectors) if transient_requested(solver_cfg) else []
    if sweep:
        saved += [f"v({drive_node})", f"i({source})", *ac_extras]
    saved = list(dict.fromkeys(saved))
    lines = [
        f".save {' '.join(saved)}",
        ".control",
        # Impedance divides voltage by current.  The default wrdata precision
        # loses the small in-phase current of high-Q RF loads and can turn a
        # formatting error into several ohms of apparent loss.
        "set numdgt=15",
    ]
    if transient_requested(solver_cfg):
        tran = solver_cfg.get("tran", {}) or {}
        lines += [
            f"tran {tran.get('step_s', 1e-9)} {tran.get('stop_s', 1e-6)}",
            f"wrdata {WAVEFORM_FILE} time {' '.join(vectors)}",
        ]
    if sweep:
        # The drive-node voltage and the source current together give the
        # impedance the source sees, which is the point of running AC here.
        ac_vectors = [f"v({drive_node})", f"i({source})", *ac_extras]
        lines += [sweep.command(), f"wrdata {AC_FILE} {' '.join(ac_vectors)}"]
    lines += ["quit", ".endc", ".end"]
    return lines


# -----------------------------------------------------------------------------
# Reading results back
# -----------------------------------------------------------------------------


def read_ac(path: str | Path, extra_columns: list[str] | tuple[str, ...] | None = None) -> pd.DataFrame:
    """Parse an ngspice AC `wrdata` file.

    ngspice writes each complex vector as ``(scale, real, imag)``, so a file
    holding v and i has six columns.  The result keeps real and imaginary parts
    in separate float columns: a complex dtype would not survive a CSV round
    trip, and mixing complex with real columns silently upcasts a whole row.
    """

    arr = np.loadtxt(path)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] < 6:
        raise ValueError(f"AC output needs 6 columns (scale, re, im per vector): {path}")
    data = {
        "frequency_Hz": arr[:, 0],
        "voltage_re": arr[:, 1],
        "voltage_im": arr[:, 2],
        "current_re": arr[:, 4],
        "current_im": arr[:, 5],
    }
    for index, name in enumerate(extra_columns or (), start=2):
        base = 3 * index
        if base + 2 >= arr.shape[1]:
            raise ValueError(f"AC output is missing vector {name!r}: {path}")
        data[f"{name}_re"] = arr[:, base + 1]
        data[f"{name}_im"] = arr[:, base + 2]
    return pd.DataFrame(data)


def input_impedance(ac: pd.DataFrame, z0: float = DEFAULT_Z0) -> pd.DataFrame:
    """Impedance seen by the source, plus reflection coefficient and VSWR.

    ngspice reports ``i(Vsrc)`` flowing *into* the source's positive terminal,
    so the current delivered to the network is its negative.
    """

    voltage = ac["voltage_re"].to_numpy() + 1j * ac["voltage_im"].to_numpy()
    delivered = -(ac["current_re"].to_numpy() + 1j * ac["current_im"].to_numpy())
    with np.errstate(divide="ignore", invalid="ignore"):
        z = voltage / delivered
        # Algebraically (Z - Z0)/(Z + Z0), but written in V and I so an open
        # circuit stays finite: there I = 0 gives gamma = 1 (total reflection)
        # instead of inf/inf = nan, which would poison the objective silently.
        gamma = (voltage - z0 * delivered) / (voltage + z0 * delivered)
    magnitude = np.abs(gamma)
    # Total reflection gives infinite VSWR.  np.where evaluates both branches,
    # so the division is guarded rather than selected after the fact.
    with np.errstate(divide="ignore", invalid="ignore"):
        vswr = np.where(magnitude < 1.0, (1 + magnitude) / (1 - magnitude), np.inf)
    return ac.assign(
        resistance_ohm=np.real(z),
        reactance_ohm=np.imag(z),
        magnitude_ohm=np.abs(z),
        # Gamma is kept in rectangular form as well: it is what a Smith chart
        # plots, and recomputing it from Z would reintroduce the open-circuit
        # nan that writing it in V and I above avoids.
        reflection_re=np.real(gamma),
        reflection_im=np.imag(gamma),
        reflection_magnitude=magnitude,
        reflection_db=20.0 * np.log10(np.maximum(magnitude, 1e-30)),
        vswr=vswr,
    )


def _ac_phasor(row: pd.Series, column: str) -> complex | None:
    real, imag = f"{column}_re", f"{column}_im"
    if real not in row or imag not in row:
        return None
    return complex(float(row[real]), float(row[imag]))


def ac_power_flow(row: pd.Series, load_current_column: str | None = None) -> dict[str, float]:
    """Electrical port power from peak AC phasors.

    ``load_real_power_W`` is accepted power at the named load reference plane;
    it is not a plasma-species or heating-path allocation.
    """

    source_voltage = complex(float(row["voltage_re"]), float(row["voltage_im"]))
    source_current = -complex(float(row["current_re"]), float(row["current_im"]))
    source_power = 0.5 * float(np.real(source_voltage * np.conj(source_current)))
    source_current_rms = float(abs(source_current)) / np.sqrt(2.0)
    source_voltage_rms = float(abs(source_voltage)) / np.sqrt(2.0)
    out = {
        "source_real_power_W": source_power,
        "source_current_rms_A": source_current_rms,
        "source_apparent_power_VA": source_voltage_rms * source_current_rms,
    }
    load_voltage = _ac_phasor(row, AC_LOAD_VOLTAGE)
    load_current_phasor = _ac_phasor(row, load_current_column) if load_current_column else None
    if load_voltage is None or load_current_phasor is None:
        return out

    load_power = 0.5 * float(np.real(load_voltage * np.conj(load_current_phasor)))
    network_loss = source_power - load_power
    out.update(
        {
            "load_real_power_W": load_power,
            "network_loss_W": network_loss,
            "transfer_efficiency": load_power / source_power if abs(source_power) > 1e-30 else 0.0,
        }
    )
    return out


def ac_component_metrics(row: pd.Series, components: tuple[ComponentObservation, ...]) -> dict[str, float]:
    """Peak/RMS terminal stress and effective series-resistance loss."""

    out: dict[str, float] = {}
    losses: list[float] = []
    for component in components:
        voltage = _ac_phasor(row, component.voltage_column)
        current = _ac_phasor(row, component.current_column)
        if voltage is None or current is None:
            raise ValueError(f"AC response is missing probes for observed component {component.reference}")
        prefix = f"component_{component.metric_id}"
        voltage_peak, current_peak = float(abs(voltage)), float(abs(current))
        voltage_rms, current_rms = voltage_peak / np.sqrt(2.0), current_peak / np.sqrt(2.0)
        out.update(
            {
                f"{prefix}_voltage_peak_V": voltage_peak,
                f"{prefix}_voltage_rms_V": voltage_rms,
                f"{prefix}_current_peak_A": current_peak,
                f"{prefix}_current_rms_A": current_rms,
            }
        )
        if component.series_resistance_ohm is not None:
            loss = current_rms**2 * component.series_resistance_ohm
            out[f"{prefix}_loss_W"] = loss
            losses.append(loss)
    if losses:
        out["modeled_component_loss_W"] = float(sum(losses))
    return out


def component_loss_balance(metrics: Mapping[str, Any]) -> dict[str, float]:
    """Compare summed explicit ESR/DCR loss with source-to-load network loss."""

    if "modeled_component_loss_W" not in metrics or "network_loss_W" not in metrics:
        return {}
    residual = float(metrics["network_loss_W"]) - float(metrics["modeled_component_loss_W"])
    source = abs(float(metrics.get("source_real_power_W", 0.0)))
    return {
        "component_loss_balance_residual_W": residual,
        "component_loss_balance_fraction_of_source": abs(residual) / max(source, 1e-30),
    }


def at_frequency(ac: pd.DataFrame, frequency_hz: float) -> pd.Series:
    """Read a requested point from a simulated AC sweep without extrapolation.

    Only solver-produced phasors are interpolated here. Independent measured
    load-table rows are never interpolated into new electrical conditions.
    """

    if ac.empty:
        raise ValueError("frequency response is empty")
    target = float(frequency_hz)
    ordered = ac.sort_values("frequency_Hz").reset_index(drop=True)
    frequencies = ordered["frequency_Hz"].to_numpy(float)
    if target < frequencies[0] or target > frequencies[-1]:
        raise ValueError(
            f"frequency {target:g} Hz is outside the simulated sweep [{frequencies[0]:g}, {frequencies[-1]:g}] Hz"
        )
    exact = np.flatnonzero(np.isclose(frequencies, target, rtol=1e-12, atol=0.0))
    if len(exact):
        return ordered.iloc[int(exact[0])]

    upper = int(np.searchsorted(frequencies, target, side="right"))
    lower = upper - 1
    fraction = (target - frequencies[lower]) / (frequencies[upper] - frequencies[lower])
    values: dict[str, Any] = {"frequency_Hz": target}
    for column in ordered.columns:
        if column == "frequency_Hz":
            continue
        lo, hi = ordered.iloc[lower][column], ordered.iloc[upper][column]
        if pd.api.types.is_numeric_dtype(ordered[column]):
            values[column] = float(lo) + fraction * (float(hi) - float(lo))
        else:
            values[column] = lo if fraction <= 0.5 else hi
    return pd.Series(values)


# -----------------------------------------------------------------------------
# Where the power goes
# -----------------------------------------------------------------------------


def mean_over_time(values: np.ndarray, time_s: np.ndarray) -> float:
    """Time average of an unevenly sampled signal.

    ngspice chooses its own timesteps, and over one transient they can span
    three orders of magnitude.  An unweighted ``mean`` then counts a 0.1 ps
    sample as heavily as a 0.2 ns one, which is not a time average at all.

    It matters most exactly where power questions are hardest: driving a nearly
    reactive load, the real power is a small residue of a large ``v*i`` product,
    and on a measured CCP case the unweighted mean came out 22% high.
    """

    return time_average(values, time_s)


def harmonic_spectrum(
    time_s: np.ndarray, values: np.ndarray, fundamental_hz: float, count: int = 3
) -> np.ndarray | None:
    """Complex amplitude at exact harmonics of the fundamental.

    A time-weighted least-squares fit replaces FFT bin selection.  It works on
    adaptive solver timesteps directly and does not require the requested tone
    to coincide with a discrete frequency bin.
    """

    if len(time_s) < 4 or fundamental_hz <= 0:
        return None
    fitted = harmonic_phasors(time_s, values, fundamental_hz, range(1, count + 1))
    if len(fitted) != count:
        return None
    return np.asarray([fitted[h] for h in range(1, count + 1)], dtype=complex)


def power_flow(waveform: pd.DataFrame, load_current: str | None = None) -> dict[str, float]:
    """Real power delivered by the source and reaching the electrode.

    A simulator run includes the source term; imported port data may omit it.
    The load term needs the current actually entering the load, which only a probe can supply -- the standard
    ``current_A`` channel is the *source* current, and in a matching network
    those differ by everything the match itself draws.

    Ask the platform for the probe, and an efficiency comes with it::

        measurement: {load_current: auto}
    """

    if waveform.empty:
        return {}
    time_s = waveform["time_s"].to_numpy(float)
    out: dict[str, float] = {}
    if "source_voltage_V" in waveform and "current_A" in waveform:
        v_src = waveform["source_voltage_V"].to_numpy(float)
        i_src = waveform["current_A"].to_numpy(float)
        # ngspice reports current into the source's + terminal, so delivered
        # power is the negative of the product.
        source_voltage_rms = float(np.sqrt(max(mean_over_time(v_src**2, time_s), 0.0)))
        source_current_rms = float(np.sqrt(max(mean_over_time(i_src**2, time_s), 0.0)))
        out.update(
            {
                "source_power_W": mean_over_time(-v_src * i_src, time_s),
                "source_current_rms_A": source_current_rms,
                "source_apparent_power_VA": source_voltage_rms * source_current_rms,
            }
        )
    if load_current and load_current in waveform:
        v_load = waveform["voltage_V"].to_numpy(float)
        i_load = waveform[load_current].to_numpy(float)
        p_load = mean_over_time(v_load * i_load, time_s)
        out["load_power_W"] = p_load
        if "source_power_W" in out:
            p_source = out["source_power_W"]
            out["network_loss_W"] = p_source - p_load
            out["transfer_efficiency"] = p_load / p_source if abs(p_source) > 1e-30 else 0.0
    return out


def _final_periodic_cycles(waveform: pd.DataFrame, fundamental_hz: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Trim a waveform to final whole cycles and retain convergence evidence."""

    if waveform.empty:
        return waveform, {}
    time_s = waveform["time_s"].to_numpy(float)
    voltage = waveform["voltage_V"].to_numpy(float)
    window = periodic_window(time_s, voltage, fundamental_hz, measure_cycles=3, consecutive=2, tolerance=1e-3)
    if window is None or len(waveform) < 4:
        return waveform, {"periodic_settled": False, "periodic_residual": None, "measurement_cycles": 0}

    grid = np.linspace(window.start_s, window.end_s, max(len(waveform), 16))
    measured = pd.DataFrame(
        {
            "time_s": grid,
            **{
                column: np.interp(grid, time_s, waveform[column].to_numpy(float))
                for column in waveform.columns
                if column != "time_s"
            },
        }
    )
    return measured, {
        "periodic_settled": bool(window.settled),
        "periodic_residual": window.residual,
        "measurement_cycles": window.cycles,
    }


def transient_component_metrics(
    waveform: pd.DataFrame,
    fundamental_hz: float,
    components: tuple[ComponentObservation, ...],
) -> dict[str, float]:
    """Component stress/loss over the same final-cycle window as RF-port power."""

    if waveform.empty or not components:
        return {}
    measured, _evidence = _final_periodic_cycles(waveform, fundamental_hz)
    time_s = measured["time_s"].to_numpy(float)
    out: dict[str, float] = {}
    losses: list[float] = []
    for component in components:
        if component.voltage_column not in measured or component.current_column not in measured:
            raise ValueError(f"waveform is missing probes for observed component {component.reference}")
        voltage = measured[component.voltage_column].to_numpy(float)
        current = measured[component.current_column].to_numpy(float)
        voltage_rms = float(np.sqrt(max(mean_over_time(voltage**2, time_s), 0.0)))
        current_rms = float(np.sqrt(max(mean_over_time(current**2, time_s), 0.0)))
        prefix = f"component_{component.metric_id}"
        out.update(
            {
                f"{prefix}_voltage_peak_V": float(np.nanmax(np.abs(voltage))),
                f"{prefix}_voltage_rms_V": voltage_rms,
                f"{prefix}_current_peak_A": float(np.nanmax(np.abs(current))),
                f"{prefix}_current_rms_A": current_rms,
            }
        )
        if component.series_resistance_ohm is not None:
            loss = current_rms**2 * component.series_resistance_ohm
            out[f"{prefix}_loss_W"] = loss
            losses.append(loss)
    if losses:
        out["modeled_component_loss_W"] = float(sum(losses))
    return out


def _harmonic_amplitudes(spectrum: np.ndarray | None) -> dict[str, float]:
    if spectrum is None:
        return {}
    return {f"h{index}": float(abs(value)) for index, value in enumerate(spectrum, start=1)}


def rf_port_metrics(
    waveform: pd.DataFrame,
    fundamental_hz: float,
    load_current_column: str | None,
) -> dict[str, Any]:
    """Measure the electrical RF-load port without inferring plasma physics.

    ``load_real_power_W`` is accepted power at this electrical port. Voltage
    is the canonical differential load-port voltage written to
    ``voltage_V``.  Current must be a probe physically placed in series with
    the load; silently substituting source current would make shunt matching
    networks look more efficient than they are.
    """

    if waveform.empty:
        raise ValueError("waveform is empty; RF-port metrics are unavailable")
    if not load_current_column or load_current_column not in waveform:
        raise ValueError("RF-port metrics require measurement.load_current (use 'auto' for built-in loads)")
    if fundamental_hz <= 0:
        raise ValueError("a positive source fundamental frequency is required for RF-port metrics")

    measured, evidence = _final_periodic_cycles(waveform, fundamental_hz)
    time_s = measured["time_s"].to_numpy(float)
    voltage = measured["voltage_V"].to_numpy(float)
    current = measured[load_current_column].to_numpy(float)
    flow = power_flow(measured, load_current_column)
    voltage_h = harmonic_spectrum(time_s, voltage, fundamental_hz, 3)
    current_h = harmonic_spectrum(time_s, current, fundamental_hz, 3)
    z1: complex | None = None
    if voltage_h is not None and current_h is not None and abs(current_h[0]) > 1e-30:
        z1 = complex(voltage_h[0] / current_h[0])

    result: dict[str, Any] = {
        "fundamental_Hz": float(fundamental_hz),
        "load_v_peak_V": float(np.nanmax(np.abs(voltage))),
        "load_v_rms_V": float(np.sqrt(max(mean_over_time(voltage**2, time_s), 0.0))),
        "load_v_dc_V": float(mean_over_time(voltage, time_s)),
        "load_i_rms_A": float(np.sqrt(max(mean_over_time(current**2, time_s), 0.0))),
        "load_real_power_W": float(flow["load_power_W"]),
        "voltage_harmonic_amplitude_V": _harmonic_amplitudes(voltage_h),
        "current_harmonic_amplitude_A": _harmonic_amplitudes(current_h),
        **evidence,
    }
    if z1 is not None:
        result.update(
            {
                "load_fundamental_resistance_ohm": float(z1.real),
                "load_fundamental_reactance_ohm": float(z1.imag),
            }
        )
    if "source_power_W" in flow:
        result.update(
            {
                "source_real_power_W": float(flow["source_power_W"]),
                "source_current_rms_A": float(flow["source_current_rms_A"]),
                "source_apparent_power_VA": float(flow["source_apparent_power_VA"]),
                "network_loss_W": float(flow["network_loss_W"]),
                "transfer_efficiency": float(flow["transfer_efficiency"]),
            }
        )
    return result
