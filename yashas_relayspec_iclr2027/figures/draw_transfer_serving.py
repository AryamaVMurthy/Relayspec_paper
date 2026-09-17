"""Concurrent Qwen3-8B <- Qwen3-4B cross-transfer serving figure."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


root = Path(__file__).resolve().parents[2]
out = Path(__file__).parent
text = (root / "phase2_results.md").read_text()
start = text.index("## Optimized vLLM concurrency")
end = text.index("### Optimized serving latency", start)
block = text[start:end]
rows = [[cell.strip() for cell in line.strip("|").split("|")]
        for line in block.splitlines() if line.startswith("|")]
header = rows[0]
index = {name: header.index(name) for name in
         ["Pair", "Task", "Clients", "Native TPS", "Mapped TPS"]}

values = {}
for row in rows[2:]:
    if len(row) != len(header) or row[index["Pair"]] != "T1":
        continue
    task = row[index["Task"]]
    clients = int(row[index["Clients"]])
    values[(task, clients)] = {
        "native": float(row[index["Native TPS"]]),
        "mapped": float(row[index["Mapped TPS"]]),
    }

# Client-side request measurements use the request latency after client-slot
# acquisition and the complete returned-token count.  This is intentionally
# distinct from server aggregate TPS, which divides by workload wall time.
latency_start = text.index("### Optimized serving latency")
latency_end = text.index("### Optimized serving output and memory accounting", latency_start)
latency_rows = [[cell.strip() for cell in line.strip("|").split("|")]
                for line in text[latency_start:latency_end].splitlines()
                if line.startswith("|")]
latency_header = latency_rows[0]
latency_index = {name: latency_header.index(name) for name in
                 ["Pair", "Task", "Clients", "Arm", "Mean latency"]}
latency = {}
for row in latency_rows[2:]:
    if len(row) != len(latency_header) or row[latency_index["Pair"]] != "T1":
        continue
    latency[(row[latency_index["Task"]], int(row[latency_index["Clients"]]),
             row[latency_index["Arm"]])] = float(row[latency_index["Mean latency"]])

output_start = text.index("### Optimized serving output and memory accounting")
output_end = text.index("### Optimized acceptance survival", output_start)
output_rows = [[cell.strip() for cell in line.strip("|").split("|")]
               for line in text[output_start:output_end].splitlines()
               if line.startswith("|")]
output_header = output_rows[0]
output_index = {name: output_header.index(name) for name in
                ["Pair", "Task", "Clients", "Arm", "Output tokens"]}
for row in output_rows[2:]:
    if len(row) != len(output_header) or row[output_index["Pair"]] != "T1":
        continue
    task = row[output_index["Task"]]
    clients = int(row[output_index["Clients"]])
    arm = row[output_index["Arm"]]
    # This pooled client-side rate is total tokens divided by the sum of the
    # 128 client-side request latencies, not a server-wall-time rate.
    values[(task, clients)][f"{arm}_client_tps"] = (
        float(row[output_index["Output tokens"]]) /
        (128 * latency[(task, clients, arm)])
    )

clients = [1, 8, 16, 32]
tasks = [("math", "Math", "#1f77b4"), ("gsm", "GSM8K", "#ff7f0e"),
         ("code", "Code", "#2ca02c"), ("chat", "Chat", "#9467bd")]
for task, _, _ in tasks:
    assert all((task, client) in values for client in clients), task

json.dump(
    {
        "setting": "Qwen3-8B <- Qwen3-4B cross-transfer, vLLM, one L40S, 128 prompts per workload",
        "source": "phase2_results.md: Optimized vLLM concurrency, T1",
        "aggregate_definition": "returned tokens divided by workload wall time including drain",
        "client_definition": "returned tokens divided by the sum of 128 client-side request latencies",
        "rows": [
            {"task": task, "clients": client, **values[(task, client)]}
            for task, _, _ in tasks for client in clients
        ],
    },
    open(out / "transfer_serving_data.json", "w"), indent=2,
)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42})
fig, (left, right) = plt.subplots(1, 2, figsize=(7.6, 2.6))
for task, label, color in tasks:
    native = [values[(task, client)]["native"] for client in clients]
    mapped = [values[(task, client)]["mapped"] for client in clients]
    left.plot(clients, mapped, "-o", color=color, ms=3.8, lw=1.6, label=label)
    left.plot(clients, native, "--", color=color, lw=1.3)
    native_client = [values[(task, client)]["native_client_tps"] for client in clients]
    mapped_client = [values[(task, client)]["mapped_client_tps"] for client in clients]
    right.plot(clients, mapped_client, "-o", color=color, ms=3.8, lw=1.6, label=label)
    right.plot(clients, native_client, "--", color=color, lw=1.3)

left.set(xlabel="Concurrent clients", ylabel="Aggregate tokens/s", xscale="symlog", xlim=(0.8, 36))
left.set_xticks(clients)
left.set_xticklabels(clients)
left.set_title("(a) Server throughput", loc="left", fontsize=10, weight="bold")
right.set(xlabel="Concurrent clients", ylabel="Client-side per-request tokens/s", xscale="symlog", xlim=(0.8, 36))
right.set_xticks(clients)
right.set_xticklabels(clients)
right.set_title("(b) Client-side per-request throughput", loc="left", fontsize=10, weight="bold")
for axis in (left, right):
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_axisbelow(True)
    axis.yaxis.grid(True, alpha=0.14)
left.legend(title="Workload", fontsize=7, title_fontsize=7, frameon=False, ncol=2)
fig.text(0.34, 0.01, "Solid: RelaySpec    Dashed: native drafter", ha="center", fontsize=7.5)
fig.tight_layout(w_pad=1.8, rect=(0, 0.05, 1, 1))
fig.savefig(out / "relayspec_transfer_serving.pdf", bbox_inches="tight", pad_inches=0.025)
fig.savefig(out / "relayspec_transfer_serving.png", dpi=220, bbox_inches="tight", pad_inches=0.025)
