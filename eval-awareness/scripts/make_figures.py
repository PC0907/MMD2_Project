"""Generate the four paper figures from results CSVs. Run after stages 3-5."""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def main(model, results_dir):
    safe = model.replace("/", "__")
    figdir = Path(results_dir) / "figures"; figdir.mkdir(parents=True, exist_ok=True)

    # Fig 1: layerwise AUROC
    p = Path(results_dir) / "probes" / safe / "layer_auroc.csv"
    if p.exists():
        df = pd.read_csv(p)
        plt.figure()
        plt.plot(df.layer, df.auroc_kfold, label="k-fold")
        plt.plot(df.layer, df.auroc_lofo, label="leave-one-family-out")
        plt.axhline(0.5, ls=":", c="gray"); plt.xlabel("layer"); plt.ylabel("AUROC")
        plt.title(f"Eval-vs-deploy probe AUROC — {model}"); plt.legend()
        plt.savefig(figdir / "fig1_layer_auroc.png", dpi=150, bbox_inches="tight")

    # Fig 4: steering dose-response
    p = Path(results_dir) / "steering" / safe / "dose_response.csv"
    if p.exists():
        df = pd.read_csv(p)
        fig, ax1 = plt.subplots()
        for name, g in df.groupby("direction"):
            ax1.plot(g.alpha, g.safe_rate, marker="o", label=f"{name}: safe rate")
        ax1.set_xlabel("steering coefficient (alpha)"); ax1.set_ylabel("safe rate")
        ax1.legend(loc="lower right"); ax1.set_title(f"Steering dose-response — {model}")
        plt.savefig(figdir / "fig4_steering.png", dpi=150, bbox_inches="tight")
    print("figures ->", figdir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--results-dir", default="results")
    main(*vars(ap.parse_args()).values())
