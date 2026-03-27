"""
Statistical Analysis for LLM Cognitive Bias Study
==================================================
Run after the experiment finishes to generate publication-ready analysis.

Usage:
    pip install scipy matplotlib seaborn
    python analyze_results.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Try imports — guide user if missing
try:
    from scipy import stats
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")
    import seaborn as sns
except ImportError:
    print("Install dependencies: pip install scipy matplotlib seaborn")
    exit(1)

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

# Premium Modern Styling
sns.set_context("talk")
sns.set_style("white")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

PALETTE = {
    "Gemini 2.5 Flash": "#3b82f6",  # Blue 500
    "Llama 3.1 8B": "#10b981",      # Emerald 500
    "Llama 3.3 70B": "#0f766e",     # Emerald 700
    "GPT-4o-mini": "#8b5cf6",       # Violet 500
    "Command R": "#f43f5e",         # Rose 500
    "Command R+": "#be123c",        # Rose 700
    "Human": "#64748b"              # Slate 500
}

def load_data() -> pd.DataFrame:
    path = RESULTS_DIR / "raw_responses.csv"
    if not path.exists():
        print(f"❌ {path} not found. Run run_experiments.py first.")
        exit(1)
    df = pd.read_csv(path)
    df = df[df["parse_success"] == True].copy()
    
    # Beautiful display names
    name_map = {
        "gemini-2.5-flash": "Gemini 2.5 Flash",
        "llama-3.1-8b-instant": "Llama 3.1 8B",
        "llama-3.3-70b-versatile": "Llama 3.3 70B",
        "gpt-4o-mini": "GPT-4o-mini",
        "command-r-08-2024": "Command R",
        "command-r-plus-08-2024": "Command R+",
        "ollama:llama3.1": "Ollama Llama 3.1"
    }
    df["model"] = df["model"].map(lambda x: name_map.get(x, x))
    
    return df


# ═══════════════════════════════════════════════════════════════════
#  1. Anchoring Analysis
# ═══════════════════════════════════════════════════════════════════

def analyze_anchoring(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("📊 ANCHORING EFFECT")
    print("=" * 60)

    data = df[df["bias"] == "anchoring"].copy()
    data["parsed_value"] = data["parsed_value"].astype(float)

    for model in data["model"].unique():
        m = data[data["model"] == model]
        high = m[m["condition"] == "high_anchor"]["parsed_value"]
        low = m[m["condition"] == "low_anchor"]["parsed_value"]

        t_stat, p_val = stats.ttest_ind(high, low)
        cohens_d = (high.mean() - low.mean()) / np.sqrt((high.std()**2 + low.std()**2) / 2)

        print(f"\n  {model}:")
        print(f"    High anchor: M={high.mean():.1f}, SD={high.std():.1f}")
        print(f"    Low anchor:  M={low.mean():.1f}, SD={low.std():.1f}")
        print(f"    t={t_stat:.3f}, p={p_val:.4f}, Cohen's d={cohens_d:.3f}")
        print(f"    Human baseline d=1.24 (Tversky & Kahneman, 1974)")

    # Plot - Modern Violin + Strip (Raincloud aesthetic) with Quartiles
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.yaxis.grid(True, color="#e5e7eb", linestyle="--", linewidth=1, zorder=0)
    
    sns.violinplot(data=data, x="model", y="parsed_value", hue="condition",
                   palette=["#f87171", "#38bdf8"], inner="quartile", linewidth=1, alpha=0.3, ax=ax, zorder=1)
    sns.stripplot(data=data, x="model", y="parsed_value", hue="condition",
                  palette=["#dc2626", "#0284c7"], dodge=True, alpha=0.6, size=4, jitter=0.25, ax=ax, zorder=2)
    
    ax.set_title("Anchoring Effect: Response Distribution by Model", fontweight="bold", pad=20)
    ax.set_ylabel("Estimated Percentage (%)", fontweight="medium", color="#4b5563")
    ax.set_xlabel("")
    ax.tick_params(axis='x', labelrotation=15, colors="#374151")
    ax.tick_params(axis='y', colors="#374151")
    
    ax.axhline(y=28, color="#9ca3af", linestyle=":", linewidth=2, label="Actual (~28%)", zorder=3)
    
    # Clean up duplicate legend from stripplot
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:3], ["High Anchor", "Low Anchor", "Actual (~28%)"], title="Condition", frameon=False, loc="upper left", bbox_to_anchor=(1, 1))
    
    sns.despine(trim=True, offset=10)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "anchoring.png", dpi=400, bbox_inches='tight')
    print(f"  → Figure saved: {FIGURES_DIR / 'anchoring.png'}")


# ═══════════════════════════════════════════════════════════════════
#  2. Framing Effect Analysis
# ═══════════════════════════════════════════════════════════════════

def analyze_framing(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("📊 FRAMING EFFECT")
    print("=" * 60)

    data = df[df["bias"] == "framing"].copy()

    for model in data["model"].unique():
        m = data[data["model"] == model]
        gain = m[m["condition"] == "gain_frame"]
        loss = m[m["condition"] == "loss_frame"]

        gain_A = (gain["parsed_value"] == "A").sum() / len(gain) * 100
        loss_A = (loss["parsed_value"] == "A").sum() / len(loss) * 100

        # Chi-square test
        table = pd.crosstab(m["condition"], m["parsed_value"])
        if table.shape == (2, 2):
            chi2, p_val, dof, _ = stats.chi2_contingency(table)
            cramers_v = np.sqrt(chi2 / len(m))
        else:
            chi2, p_val, cramers_v = float("nan"), float("nan"), float("nan")

        print(f"\n  {model}:")
        print(f"    Gain frame → A (safe): {gain_A:.1f}%")
        print(f"    Loss frame → A (safe): {loss_A:.1f}%")
        print(f"    χ²={chi2:.3f}, p={p_val:.4f}, Cramér's V={cramers_v:.3f}")
        print(f"    Human baseline: Gain→A=72%, Loss→A=22%")

    # Plot
    plot_data = []
    for _, row in data.iterrows():
        plot_data.append({
            "model": row["model"],
            "condition": row["condition"],
            "chose_safe": 1 if row["parsed_value"] == "A" else 0,
        })
    pdf = pd.DataFrame(plot_data)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.yaxis.grid(True, color="#e5e7eb", linestyle="--", linewidth=1, zorder=0)

    # Plot raw data so Seaborn computes and draws 95% Confidence Interval error bars automatically!
    bars = sns.barplot(data=pdf, x="model", y="chose_safe", hue="condition", errorbar=("ci", 95), capsize=0.1, err_kws={'linewidth': 1.5},
                palette=["#38bdf8", "#f87171"], ax=ax, zorder=2, alpha=0.9, edgecolor="white", linewidth=1.5)
    
    # Accurate Label percentages
    for container in ax.containers:
        if hasattr(container, "datavalues"):
            ax.bar_label(container, fmt='%.2f', padding=3, color="#4b5563", fontsize=10)

    ax.set_title("Framing Effect: Proportion Choosing the 'Safe' Program", fontweight="bold", pad=20)
    ax.set_ylabel("Proportion Choosing Program A (0 to 1.0)", fontweight="medium", color="#4b5563")
    ax.set_xlabel("")
    ax.tick_params(axis='x', labelrotation=15, colors="#374151")
    ax.tick_params(axis='y', colors="#374151")
    ax.set_ylim(0, 1.1)

    # Human baselines
    ax.axhline(y=0.72, color="#0284c7", linestyle=":", linewidth=2, label="Human (Gain) = 72%", zorder=3)
    ax.axhline(y=0.22, color="#dc2626", linestyle=":", linewidth=2, label="Human (Loss) = 22%", zorder=3)
    
    ax.legend(title="Linguistic Frame", frameon=False, loc="upper right", bbox_to_anchor=(1.25, 1))
    
    sns.despine(trim=True, offset=10)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "framing.png", dpi=400, bbox_inches='tight')
    print(f"  → Figure saved: {FIGURES_DIR / 'framing.png'}")


# ═══════════════════════════════════════════════════════════════════
#  3. Sunk Cost Analysis
# ═══════════════════════════════════════════════════════════════════

def analyze_sunk_cost(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("📊 SUNK COST FALLACY")
    print("=" * 60)

    data = df[df["bias"] == "sunk_cost"].copy()

    for model in data["model"].unique():
        m = data[data["model"] == model]
        high = m[m["condition"] == "high_sunk_cost"]
        none = m[m["condition"] == "no_sunk_cost"]

        high_mich = (high["parsed_value"] == "Michigan").sum() / len(high) * 100
        none_mich = (none["parsed_value"] == "Michigan").sum() / len(none) * 100

        print(f"\n  {model}:")
        print(f"    With sunk cost   → Michigan: {high_mich:.1f}%")
        print(f"    Without sunk cost → Michigan: {none_mich:.1f}%")
        print(f"    Human baseline: With=54%, Without=15%")


# ═══════════════════════════════════════════════════════════════════
#  4. Decoy Effect Analysis
# ═══════════════════════════════════════════════════════════════════

def analyze_decoy(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("📊 DECOY EFFECT")
    print("=" * 60)

    data = df[df["bias"] == "decoy"].copy()

    for model in data["model"].unique():
        m = data[data["model"] == model]
        no_d = m[m["condition"] == "no_decoy"]
        with_d = m[m["condition"] == "with_decoy"]

        no_d_A = (no_d["parsed_value"] == "A").sum() / len(no_d) * 100
        with_d_A = (with_d["parsed_value"] == "A").sum() / len(with_d) * 100

        print(f"\n  {model}:")
        print(f"    No decoy   → A: {no_d_A:.1f}%")
        print(f"    With decoy → A: {with_d_A:.1f}%")
        print(f"    Human baseline: No decoy→A=50%, With decoy→A=67%")


# ═══════════════════════════════════════════════════════════════════
#  5. Base Rate Neglect Analysis
# ═══════════════════════════════════════════════════════════════════

def analyze_base_rate(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("📊 BASE RATE NEGLECT")
    print("=" * 60)

    data = df[df["bias"] == "base_rate"].copy()
    data["parsed_value"] = data["parsed_value"].astype(float)

    for model in data["model"].unique():
        m = data[data["model"] == model]
        high = m[m["condition"] == "high_base_rate"]["parsed_value"]
        low = m[m["condition"] == "low_base_rate"]["parsed_value"]

        t_stat, p_val = stats.ttest_ind(high, low)

        print(f"\n  {model}:")
        print(f"    30 eng / 70 law → P(engineer): M={high.mean():.1f}")
        print(f"    70 eng / 30 law → P(engineer): M={low.mean():.1f}")
        print(f"    t={t_stat:.3f}, p={p_val:.4f}")
        print(f"    Human baseline: ~same estimate regardless (base rate neglect)")


# ═══════════════════════════════════════════════════════════════════
#  Master Summary Figure
# ═══════════════════════════════════════════════════════════════════

def plot_effect_size_comparison(df: pd.DataFrame):
    """Bar chart comparing LLM effect sizes to human baselines."""
    print("\n" + "=" * 60)
    print("📊 EFFECT SIZE COMPARISON (LLM vs. Human)")
    print("=" * 60)

    records = []

    # Anchoring
    for model in df[df["bias"] == "anchoring"]["model"].unique():
        m = df[(df["bias"] == "anchoring") & (df["model"] == model)]
        high = m[m["condition"] == "high_anchor"]["parsed_value"].astype(float)
        low = m[m["condition"] == "low_anchor"]["parsed_value"].astype(float)
        d = (high.mean() - low.mean()) / np.sqrt((high.std()**2 + low.std()**2) / 2)
        records.append({"bias": "Anchoring", "model": model, "effect_size": abs(d)})
    records.append({"bias": "Anchoring", "model": "Human", "effect_size": 1.24})

    # Base Rate
    for model in df[df["bias"] == "base_rate"]["model"].unique():
        m = df[(df["bias"] == "base_rate") & (df["model"] == model)]
        high = m[m["condition"] == "high_base_rate"]["parsed_value"].astype(float)
        low = m[m["condition"] == "low_base_rate"]["parsed_value"].astype(float)
        if high.std() > 0 and low.std() > 0:
            d = (high.mean() - low.mean()) / np.sqrt((high.std()**2 + low.std()**2) / 2)
        else:
            d = 0
        records.append({"bias": "Base Rate", "model": model, "effect_size": abs(d)})
    records.append({"bias": "Base Rate", "model": "Human", "effect_size": 0.1})

    edf = pd.DataFrame(records)
    
    # Modern Horizontal Facet Grid
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    
    # Sort models nicely
    model_order = ["Human", "Gemini 2.5 Flash", "Llama 3.1 8B", "Command R", 
                   "Llama 3.3 70B", "Command R+", "GPT-4o-mini"]
    
    # Clean missing/NaN models from plot invisibly
    available_models = [m for m in model_order if m in edf["model"].values]

    # Anchoring Plot
    anch_data = edf[edf["bias"] == "Anchoring"]
    bars1 = sns.barplot(data=anch_data, y="model", x="effect_size", order=available_models,
                palette=PALETTE, ax=ax1, alpha=0.9, edgecolor="white", linewidth=1)
    for container in ax1.containers:
        ax1.bar_label(container, fmt='%.2fd', padding=5, color="#1f2937", fontsize=10, fontweight="bold")
        
    ax1.set_title("Anchoring Effect Size", fontweight="bold")
    ax1.set_xlabel("Absolute Cohen's d")
    ax1.set_ylabel("")
    ax1.xaxis.grid(True, color="#e5e7eb", linestyle="--", linewidth=1, zorder=0)

    # Base Rate Plot
    base_data = edf[edf["bias"] == "Base Rate"]
    bars2 = sns.barplot(data=base_data, y="model", x="effect_size", order=available_models,
                palette=PALETTE, ax=ax2, alpha=0.9, edgecolor="white", linewidth=1)
    for container in ax2.containers:
        ax2.bar_label(container, fmt='%.2fd', padding=5, color="#1f2937", fontsize=10, fontweight="bold")

    ax2.set_title("Base Rate Neglect Effect Size", fontweight="bold")
    ax2.set_xlabel("Absolute Cohen's d")
    ax2.set_ylabel("")
    ax2.xaxis.grid(True, color="#e5e7eb", linestyle="--", linewidth=1, zorder=0)

    sns.despine(trim=True)
    fig.suptitle("Effect Sizes: LLMs vs. Humans (Cognitive Bias Magnitude)", fontsize=18, fontweight="black", y=1.05)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "effect_size_comparison.png", dpi=400, bbox_inches='tight')
    print(f"  → Figure saved: {FIGURES_DIR / 'effect_size_comparison.png'}")


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    df = load_data()

    analyze_anchoring(df)
    analyze_framing(df)
    analyze_sunk_cost(df)
    analyze_decoy(df)
    analyze_base_rate(df)
    plot_effect_size_comparison(df)

    print("\n" + "=" * 60)
    print("✅ All analyses complete. Figures saved in:", FIGURES_DIR)
    print("=" * 60)
