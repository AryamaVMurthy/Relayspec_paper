"""Draw the two introduction figures from saved autoregressive comparisons."""

from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


root = Path(__file__).resolve().parents[2]
out = Path(__file__).parent


def saved_ar_rows(filename):
    source = (root / filename).read_text()
    section = source.split("## Saved AR comparison", 1)[1].split("\n## ", 1)[0]
    rows = {}
    for line in section.splitlines():
        if not line.startswith("| "):
            continue
        cells = [cell.strip() for cell in line.strip("| ").split("|")]
        if len(cells) != 5 or not cells[2].replace(".", "", 1).isdigit():
            continue
        dataset, method, ar_tps, method_tps, ratio = cells
        rows[(dataset, method)] = {
            "ar_tps": float(ar_tps),
            "method_tps": float(method_tps),
            "speedup_vs_ar": float(ratio.removesuffix("×")),
        }
    return rows


phase1 = saved_ar_rows("new_phase1.md")
phase2 = saved_ar_rows("new_phase2.md")


def cohort(rows, datasets, adapted_method):
    result = []
    for dataset in datasets:
        reference = rows[(dataset, "DFlash native")]
        adapted = rows[(dataset, adapted_method)]
        assert reference["ar_tps"] == adapted["ar_tps"], dataset
        for item in (reference, adapted):
            assert abs(item["method_tps"] / item["ar_tps"] - item["speedup_vs_ar"]) < 0.002, (dataset, item)
        result.append({"dataset": dataset, "autoregressive_tps": reference["ar_tps"],
                       "reference": reference["speedup_vs_ar"],
                       "relayspec": adapted["speedup_vs_ar"]})
    return result


lora = cohort(phase1, ["GSM8K", "KiCad", "NanoCoder"], "5W AUF")
transfer = cohort(phase2, ["Math", "GSM8K", "Code", "Chat"], "5W MSE25")
(out / "speedups_data.json").write_text(json.dumps({
    "metric": "arithmetic mean per-request tokens/s divided by matching autoregressive mean tokens/s",
    "lora": lora,
    "transfer": transfer,
    "sources": ["new_phase1.md, Saved AR comparison", "new_phase2.md, Saved AR comparison"],
}, indent=2) + "\n")

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42})
gray, blue = "#b8c7d2", "#266a94"


def draw(data, reference_name, output_name, upper):
    fig, ax = plt.subplots(figsize=(3.2, 2.15))
    x = np.arange(len(data))
    reference = [row["reference"] for row in data]
    adapted = [row["relayspec"] for row in data]
    ax.bar(x - 0.19, reference, 0.36, color=gray, label=reference_name)
    ax.bar(x + 0.19, adapted, 0.36, color=blue, label="RelaySpec")
    for series, (offsets, values) in enumerate(((-0.19, reference), (0.19, adapted))):
        for index, (xpos, value) in enumerate(zip(x + offsets, values)):
            shown = Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            stagger = 0.28 if series == 1 and abs(reference[index] - adapted[index]) < 0.5 else 0
            ax.text(xpos, value + upper * 0.018 + stagger, f"{shown}×", ha="center", fontsize=7)
    ax.set_xticks(x, [row["dataset"] for row in data])
    ax.set_ylim(0, upper)
    ax.set_ylabel("Speedup vs AR (×)")
    ax.axhline(1, color="#575757", linewidth=0.9, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, alpha=0.12)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=2,
              fontsize=6.5, frameon=False, handlelength=1.0, columnspacing=0.8)
    fig.tight_layout(pad=0.2)
    fig.savefig(out / f"{output_name}.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out / f"{output_name}.png", dpi=180, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


draw(lora, "Unchanged DFlash", "relayspec_lora_ar", 10.1)
draw(transfer, "Native 8B drafter", "relayspec_transfer_ar", 7.1)
