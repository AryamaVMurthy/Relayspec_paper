"""Fit-time companion for the supervision-scaling figure."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


root = Path(__file__).resolve().parents[2]
out = Path(__file__).parent


def markdown_rows(block):
    return [[cell.strip() for cell in line.strip("|").split("|")]
            for line in block.splitlines() if line.startswith("|")]


phase1 = (root / "phase1_results.md").read_text()
block1 = phase1[phase1.index("### scaling"):phase1.index("### serving")]
rows1 = markdown_rows(block1)
head1 = rows1[0]
index1 = {name: head1.index(name) for name in
          ["domain", "architecture", "examples", "anchors", "training_seconds", "loss"]}
large_lora = {}
anchors = {}
for row in rows1[2:]:
    if len(row) != len(head1) or row[index1["loss"]] != "AUF":
        continue
    domain = row[index1["domain"]]
    architecture = "maps" if row[index1["architecture"]].startswith("maps") else "body"
    examples = int(row[index1["examples"]])
    anchor_count = int(row[index1["anchors"]])
    seconds = float(row[index1["training_seconds"]])
    if architecture == "maps" and anchor_count == 32:
        large_lora[(domain, examples)] = seconds
    if examples == 4096:
        anchors[(domain, architecture, anchor_count)] = seconds

small = (root / "small_scaling.md").read_text()


def table_after(marker):
    start = small.index("|", small.index(marker) + len(marker))
    end = small.index("\n\n", start)
    return markdown_rows(small[start:end])


small_lora = {}
for row in table_after("**LoRA target adapters, AUF**")[2:]:
    small_lora[(row[0], int(row[1].replace(",", "")))] = float(row[2])
small_transfer = {}
for row in table_after("**T1 cross-size transfer, MSE**")[2:]:
    small_transfer[int(row[1].replace(",", ""))] = float(row[2])

phase2 = (root / "phase2_results.md").read_text()
start2 = phase2.index("## MSE 50% coverage results")
end2 = phase2.index("### Completed training", start2)
rows2 = markdown_rows(phase2[start2:end2])
head2 = rows2[0]
index2 = {name: head2.index(name) for name in ["Fit", "Fit min"]}
large_transfer = {}
for row in rows2[2:]:
    if len(row) != len(head2):
        continue
    fit = row[index2["Fit"]]
    if not fit.startswith("T1_mse50_n") or not fit.endswith("_e3_lr0.001"):
        continue
    examples = int(fit.split("_n")[1].split("_")[0])
    large_transfer[examples] = 60 * float(row[index2["Fit min"]])

counts_lora = [16, 128, 512, 1024, 2048, 4096, 8192, 12288, 16384]
counts_transfer = [16, 128, 512, 1024, 2048, 4096, 8192, 16384]
anchor_counts = [8, 24, 32, 64, 128]
domains = [("math", "GSM8K", "#c2703a"), ("kicad", "KiCad", "#266a94"),
           ("nanocoder", "NanoCoder", "#5b8c5a")]
for domain, _, _ in domains:
    assert all((domain, count) in small_lora or (domain, count) in large_lora
               for count in counts_lora)
    assert all((domain, architecture, count) in anchors
               for architecture in ("maps", "body") for count in anchor_counts)
assert all(count in small_transfer or count in large_transfer for count in counts_transfer)


def lora_time(domain, count):
    if (domain, count) in small_lora:
        return small_lora[(domain, count)]
    return large_lora[(domain, count)]


def transfer_time(count):
    if count in small_transfer:
        return small_transfer[count]
    return large_transfer[count]


payload = {
    "sources": [
        "phase1_results.md: scaling table",
        "phase2_results.md: MSE 50% coverage results",
        "small_scaling.md: completed small-data extension",
    ],
    "unit": "optimizer-loop fit seconds on two L40S GPUs",
    "lora_examples": [
        {"domain": domain, "examples": count, "fit_seconds": lora_time(domain, count)}
        for domain, _, _ in domains for count in counts_lora
    ],
    "lora_anchors": [
        {"domain": domain, "architecture": architecture, "anchors": count,
         "fit_seconds": anchors[(domain, architecture, count)]}
        for domain, _, _ in domains for architecture in ("maps", "body")
        for count in anchor_counts
    ],
    "transfer_examples": [
        {"examples": count, "fit_seconds": transfer_time(count)}
        for count in counts_transfer
    ],
}
with open(out / "scaling_fit_times_data.json", "w") as handle:
    json.dump(payload, handle, indent=2)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "pdf.fonttype": 42})
fig, (left, middle, right) = plt.subplots(1, 3, figsize=(7.6, 2.35))
ticks_lora = range(len(counts_lora))
for domain, label, color in domains:
    values = [lora_time(domain, count) for count in counts_lora]
    left.plot(ticks_lora, values, "-o", color=color, ms=3.2, lw=1.5, label=label)
left.set(xticks=ticks_lora,
         xticklabels=[f"{count // 1024}k" if count >= 1024 else str(count)
                     for count in counts_lora],
         xlabel="Example presentations", ylabel="Fit time (s)")
left.set_title("(a) LoRA-fine-tuned", loc="left", fontsize=8, weight="bold")
left.legend(fontsize=6.4, frameon=False, loc="upper left", handlelength=1.5)
left.tick_params(axis="x", labelrotation=35)

for domain, label, color in domains:
    map_values = [anchors[(domain, "maps", count)] for count in anchor_counts]
    body_values = [anchors[(domain, "body", count)] for count in anchor_counts]
    middle.plot(anchor_counts, map_values, "-o", color=color, ms=3.2, lw=1.5)
    middle.plot(anchor_counts, body_values, "--s", color=color, ms=2.8, lw=1.0, alpha=.7)
middle.set(xlabel="Anchors per example (4,096 ex.)", ylabel="Fit time (s)")
middle.set_title("(b) LoRA-fine-tuned", loc="left", fontsize=8, weight="bold")
middle.text(0.02, 0.96, "Solid: maps\nDashed: drafter LoRA", transform=middle.transAxes,
            va="top", fontsize=6.3)

values = [transfer_time(count) for count in counts_transfer]
right.plot(range(len(counts_transfer)), values, "-o", color="#266a94", ms=3.2, lw=1.5)
right.set(xticks=range(len(counts_transfer)),
          xticklabels=[f"{count // 1024}k" if count >= 1024 else str(count)
                      for count in counts_transfer],
          xlabel="Training examples", ylabel="Fit time (s)")
right.set_title("(c) Cross-transfer", loc="left", fontsize=8, weight="bold")
right.tick_params(axis="x", labelrotation=35)

for axis in (left, middle, right):
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_axisbelow(True)
    axis.yaxis.grid(True, alpha=.14)
fig.tight_layout(w_pad=1.1)
fig.savefig(out / "relayspec_scaling_fit_times.pdf", bbox_inches="tight", pad_inches=.02)
fig.savefig(out / "relayspec_scaling_fit_times.png", dpi=220, bbox_inches="tight", pad_inches=.02)
