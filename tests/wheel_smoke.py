"""Exercise an installed PCD wheel from outside the source checkout."""

from __future__ import annotations

import csv
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit


def pcd(*args: str | Path) -> str:
    result = subprocess.run(
        [sys.executable, "-I", "-m", "pcd", *map(str, args)],
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode:
        raise SystemExit(f"pcd {args[0]} exited with {result.returncode}\n{result.stdout}{result.stderr}")
    return result.stdout


def main() -> None:
    share = Path(sys.prefix) / "share" / "pcd"
    examples = share / "examples"
    manual = share / "docs" / "manual-ja.md"
    assert (share / "docs" / "assets" / "rc-circuit.svg").is_file()
    for target in re.findall(r"\]\(([^)\s]+)\)", manual.read_text(encoding="utf-8")):
        url = urlsplit(target)
        if not url.scheme and not url.netloc and url.path:
            assert (manual.parent / unquote(url.path)).exists(), f"missing wheel manual link: {target}"

    pcd("solver-diagnose", "--json")
    circuit = examples / "circuit_external.yaml"
    pcd("validate-case", circuit, "--plan")
    pcd("sim-netlist", circuit, "--out", "package-smoke.cir")
    pcd("visualize-netlist", "package-smoke.cir", "--out", "package-smoke.png")
    assert Path("package-smoke.png").is_file()
    simulation = json.loads(pcd("sim-run", circuit, "--run-root", "simulation", "--json"))
    analysis = json.loads(pcd("analyze", simulation["run_dir"], "--out", "redraw", "--json"))
    assert analysis["status"] == "complete"
    assert analysis["provenance"]["source_integrity"] == "verified"
    # The installed example is a 50 ohm series resistor driving a 50 ohm load.
    assert math.isclose(analysis["analyses"]["ac"]["resistance_ohm"], 100.0, rel_tol=1e-9)

    Path("matched.cir").write_text("Rload port 0 50\n", encoding="utf-8")
    Path("network.yaml").write_text(
        "schema: pcd.network.v1\n"
        "dut: {file: matched.cir}\n"
        "ports: [{name: P1, p: port, reference_impedance_ohm: 50}]\n"
        "ac: {sweep: lin, points: 1, start_Hz: 1e6, stop_Hz: 1e6}\n",
        encoding="utf-8",
    )
    network = json.loads(pcd("network-run", "network.yaml", "--out", "network", "--json"))
    assert network["status"] == "complete"
    with (Path("network") / network["artifacts"]["scattering"]).open(encoding="utf-8", newline="") as handle:
        (reflection,) = csv.DictReader(handle)
    assert abs(complex(float(reflection["s_re"]), float(reflection["s_im"]))) < 1e-12

    design = json.loads(
        pcd("run", examples / "circuit_design.yaml", "--output", "study", "--require-acceptance", "--json")
    )
    assert design["best"]["status"] == "meets_declared_acceptance"
    generation = Path(design["run_root"]) / design["artifacts"]["generation"]
    pcd("result-summary", generation, "--out", "candidates.csv")
    with Path("candidates.csv").open(encoding="utf-8", newline="") as handle:
        selected = [row["candidate_id"] for row in csv.DictReader(handle) if row["selected"] == "True"]
    assert selected == [design["best"]["candidate"]["candidate_id"]]
    print("Installed wheel examples, real simulation, scattering, saved generation, and manual links passed.")


if __name__ == "__main__":
    main()
