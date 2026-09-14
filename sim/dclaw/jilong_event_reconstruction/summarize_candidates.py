"""Write the Phase-2B candidate figure and report; does not run D-Claw."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

CASE = Path(__file__).resolve().parent
PROJECT = CASE.parents[2]
CSV = CASE / "CANDIDATE_RUNS.csv"
FIGURE = PROJECT / "outputs" / "phase2b" / "event_reconstruction_candidates.png"
REPORT = CASE / "RESULTS_REPORT.md"


def number(value: str | None) -> float | None:
    if value in (None, "", "None"):
        return None
    return float(value)


def main() -> None:
    with CSV.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError("no candidate rows")
    for row in rows:
        row["runtime_s"] = number(row["runtime_s"])
        row["arrival_time_s"] = number(row["arrival_time_s"])
        row["port_peak_discharge_m3s"] = number(row["port_peak_discharge_m3s"])
        row["port_peak_depth_m"] = number(row["port_peak_depth_m"])
        row["port_peak_speed_ms"] = number(row["port_peak_speed_ms"])
        row["source_volume_m3"] = number(row["source_volume_m3"])
        row["stable"] = row["stable"].lower() == "true"
        row["reached_port"] = row["reached_port"].lower() == "true"

    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    labels = [r["run_id"] for r in rows]
    x = list(range(len(rows)))
    arrivals = [r["arrival_time_s"] if r["arrival_time_s"] is not None else 1980.0 for r in rows]
    peaks = [r["port_peak_discharge_m3s"] or 0.0 for r in rows]
    colors = ["#2b8cbe" if r["stable"] else "#d7301f" for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    axes[0].scatter(x, arrivals, s=78, c=colors, zorder=3)
    axes[0].axhline(1800, color="0.25", ls="--", lw=1, label="soft target: 1800 s")
    axes[0].set(xticks=x, xticklabels=labels, ylim=(0, 2100), ylabel="port arrival time (s)", title="Arrival diagnostic")
    axes[0].legend(fontsize=8)
    for i, row in enumerate(rows):
        if row["arrival_time_s"] is None:
            axes[0].annotate("not reached", (i, 1980), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=7)
    axes[1].scatter(x, peaks, s=78, c=colors, zorder=3)
    axes[1].axhspan(3e4, 5e4, color="#74c476", alpha=.25, label="soft target: 3–5 × 10⁴ m³/s")
    axes[1].set(xticks=x, xticklabels=labels, ylim=(0, 5.5e4), ylabel="port peak discharge (m³/s)", title="Cross-section flux diagnostic")
    axes[1].legend(fontsize=8)
    fig.suptitle("Phase 2B: moving-initial-mass source candidates\nblue = stable; red = unstable", fontsize=11)
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)

    table = []
    for r in rows:
        table.append(
            f"| {r['run_id']} | {float(r['source_depth_m']):g} | {float(r['initial_speed_ms']):g} | "
            f"{r['source_volume_m3']:,.0f} | {str(r['reached_port']).lower()} | "
            f"{r['arrival_time_s'] if r['arrival_time_s'] is not None else 'not reached'} | "
            f"{r['port_peak_discharge_m3s']:,.1f} | {r['port_peak_depth_m']:.3f} | "
            f"{r['port_peak_speed_ms']:.3f} | {r['runtime_s']:.1f} | {str(r['stable']).lower()} |"
        )
    report = f"""# Phase 2B — first event-constrained downstream reconstruction

## Scope and source representation

This is a first, deliberately small source-only reconstruction exercise for propagation **after entry to the main river**.  It uses the fixed Phase-2B entry, 64 m grid, 1800 s duration, DEM construction, D-Claw material parameters, Manning coefficient, friction angle, viscosity, solid-fraction defaults, and disabled entrainment/segregation from `jilong_baseline`.  The original baseline was not modified.

The sole source mechanism is D-Claw-supported `qinit_dclaw_data`: a fixed elliptical initial mass with nonzero downstream `u,v` state.  This represents already-moving material at the model entry.  `source_depth_m` and `initial_speed_ms` are **inverse-model parameters**, not observations and not claims about entry depth or velocity.  `SOURCE_MODEL.md` records the mechanism and fixed assumptions.

Arrival is the first 60 s output with interpolated depth at the official port point above 0.01 m.  Discharge is positive downstream flux integrated across a fixed 384 m, six-strip (64 m each) numerical cross-section centered at that point.  It is not inferred from the point gauge.

## Candidate runs

| run_id | source depth (m) | initial speed (m/s) | initial grid volume (m³) | reached port | arrival (s) | port peak Q (m³/s) | port peak depth (m) | port peak speed (m/s) | runtime (s) | stable |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|
{chr(10).join(table)}

The first two candidates are stable but do not reach the port by 1800 s.  The third broad adaptive increase remains short of the port and produces a domain maximum speed of about 95.8 m/s, so it is excluded as numerically/physically unsuitable despite completing without NaNs.

## Selection and comparison with soft event constraints

No event-scale candidate was found.  The most informative stable case is `run_02` (`h=10 m`, `V0=14 m/s`), because it is the stronger of the stable tests; it still does not reach Jilong Port, so it has no port arrival time and zero diagnostic port discharge.

- **Arrival mismatch:** no stable candidate reached the port by the 1800 s soft downstream consistency time.
- **Peak-discharge mismatch:** all candidates have 0 m³/s at the fixed port section versus the soft 3–5 × 10⁴ m³/s estimate.
- **Stability:** `run_01` and `run_02` have finite fields and domain maximum speeds below 36 m/s. `run_03` is excluded because its maximum is about 95.8 m/s; it is not an acceptable way to force transport.

The sole diagnostic figure is [event_reconstruction_candidates.png](../../../outputs/phase2b/event_reconstruction_candidates.png).

## Interpretation and stop decision

Within this three-run, two-parameter adaptive search, simple moving-initial-mass source adjustment is **not sufficient** to create an event-consistent downstream reconstruction.  Increasing both source thickness and momentum from 6 m/8 m s⁻¹ to 25 m/28 m s⁻¹ changed transport only from sub-kilometre progression to approximately one kilometre, before unacceptable high velocities appeared.  Continuing to force this same source representation would not be scientifically justified.

Further source-representation work is needed before material-parameter calibration.  In particular, these results do **not** justify changing friction angle, viscosity, solid fraction, or entrainment to compensate.  No hydrograph has been inferred or created in this phase.
"""
    REPORT.write_text(report)
    print(f"wrote {REPORT} and {FIGURE}")


if __name__ == "__main__":
    main()
