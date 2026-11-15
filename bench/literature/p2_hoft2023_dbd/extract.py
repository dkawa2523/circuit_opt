"""Digitize the electrical plots in Figs. 6 and 8(a) of Höft et al. (2023)."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from hashlib import sha256
from importlib import import_module
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Panel:
    condition: str
    pdf_page: int
    pixels: tuple[int, int]
    image_sha256: str
    time_axis_x: tuple[float, float]  # 0 and 100 µs
    voltage_y_zero: float
    voltage_v_per_pixel: float
    voltage_crop_bottom_y: int
    current_y_zero: float
    current_ua_per_pixel: float
    current_crop_y: tuple[int, int]


PANELS = (
    Panel(
        "diffuse_12p6_kVpp",
        5,
        (1733, 1327),
        "9e6661587429cac241f5a70fb201501bb201901eac7b7083c1f7df544801fb62",
        (243.5, 1464.5),
        129.5,
        6000 / 110,
        260,
        405,
        50 / 110,
        (310, 500),
    ),
    Panel(
        "hybrid_14_kVpp",
        7,
        (1500, 2244),
        "b2cb79267a94ac75f1100eb226e2cc6414fff01aae79e89bccfe711384fa4ec4",
        (258.5, 1251.5),
        126.5,
        9000 / 114,
        226,
        274,
        600 / 156.5,
        (252, 400),
    ),
)

CSV_COLUMNS = ("condition", "time_us", "voltage_V", "current_uA", "ink_lower_uA", "ink_upper_uA")


def digitize_panel(raw_image: bytes, panel: Panel) -> list[tuple[str, ...]]:
    """Return the trace center and red ink span in each unobscured pixel column."""
    if sha256(raw_image).hexdigest() != panel.image_sha256:
        raise ValueError(f"embedded image has changed for {panel.condition}")
    rgb = np.asarray(Image.open(BytesIO(raw_image)).convert("RGB"), dtype=np.int16)
    r, g, b = rgb.transpose(2, 0, 1)
    red_ink = (r > 120) & (r - g > 75) & (r - b > 70)
    black_ink = np.max(rgb, axis=2) < 75
    left, right = panel.time_axis_x
    x0 = round(left) + 5
    x1 = round(right) - 16  # the red right-axis ticks protrude into the plot
    y0, y1 = panel.current_crop_y
    rows: list[tuple[str, ...]] = []
    for x in range(x0, x1 + 1):
        phase = (x - left) / (right - left)
        if abs(10 * phase - round(10 * phase)) * (right - left) / 10 < 3:
            continue  # vertical optical markers occlude both traces
        center_y = panel.voltage_y_zero - 6500 * np.sin(2 * np.pi * phase) / panel.voltage_v_per_pixel
        lower_y = max(5, round(center_y) - 20)
        upper_y = min(panel.voltage_crop_bottom_y, round(center_y) + 21)
        v_ink = lower_y + np.flatnonzero(black_ink[lower_y:upper_y, x])
        i_ink = y0 + np.flatnonzero(red_ink[y0:y1, x])
        if not len(v_ink) or not len(i_ink):
            continue
        voltage = (panel.voltage_y_zero - np.median(v_ink)) * panel.voltage_v_per_pixel
        current, lower, upper = (
            (panel.current_y_zero - y) * panel.current_ua_per_pixel
            for y in ((i_ink[0] + i_ink[-1]) / 2, i_ink[-1] + 1, i_ink[0] - 1)
        )
        rows.append(
            (
                panel.condition,
                f"{100 * phase:.6f}",
                f"{voltage:.4f}",
                *(f"{value:.6f}" for value in (current, lower, upper)),
            )
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        pymupdf = import_module("pymupdf")
    except ModuleNotFoundError as error:
        if error.name != "pymupdf":
            raise
        parser.error("PDF extraction requires PyMuPDF. Run with `uv run --with pymupdf`.")

    rows = []
    with pymupdf.open(args.pdf) as pdf:
        for panel in PANELS:
            (xref,) = [image[0] for image in pdf[panel.pdf_page].get_images() if image[2:4] == panel.pixels]
            rows.extend(digitize_panel(pdf.extract_image(xref)["image"], panel))
    with args.output.open("x", encoding="ascii", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(CSV_COLUMNS)
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
