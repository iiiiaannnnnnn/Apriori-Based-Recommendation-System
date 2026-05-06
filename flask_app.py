import io
import base64
import textwrap
import json
import sys
import os
from pathlib import Path


PROJECT_DIR = Path(__file__).parent
LOCAL_VENV_PYTHON = PROJECT_DIR / ".venv" / "Scripts" / "python.exe"


def _running_in_local_venv() -> bool:
    try:
        return Path(sys.executable).resolve() == LOCAL_VENV_PYTHON.resolve()
    except OSError:
        return False


if LOCAL_VENV_PYTHON.exists() and not _running_in_local_venv():
    # Re-launch with the project virtualenv instead of mixing global Python
    # with packages installed for a different interpreter version.
    os.execv(str(LOCAL_VENV_PYTHON), [str(LOCAL_VENV_PYTHON), __file__, *sys.argv[1:]])

try:
    from google import genai as _genai_sdk
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from flask import Flask, render_template, request, jsonify
from mlxtend.frequent_patterns import apriori, association_rules
from data_loader import load_transaction_data

app = Flask(__name__)

DATA_DIR = Path(__file__).parent / "data"


@app.after_request
def add_no_cache_headers(response):
    if response.mimetype == "text/html":
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


# ─────────────────────────────────────────────
#  Gemini key rotator
# ─────────────────────────────────────────────

class GeminiRotator:
    """Rotate through multiple Gemini API keys to spread token usage."""
    def __init__(self):
        self.keys:  list = []
        self.idx:   int  = 0
        self.model: str  = "gemini-1.5-flash"

    def set_keys(self, keys: list, model: str = ""):
        self.keys = [k.strip() for k in keys if k.strip()]
        self.idx  = 0
        if model:
            self.model = model.strip()

    def available(self) -> bool:
        return GEMINI_AVAILABLE and bool(self.keys)

    def call(self, prompt: str, model: str = "") -> str:
        if not self.available():
            raise RuntimeError(
                "Gemini unavailable — install google-genai and add API keys."
            )
        import time
        use_model = model or self.model
        last_err = None
        for attempt in range(len(self.keys)):
            key = self.keys[self.idx % len(self.keys)]
            self.idx = (self.idx + 1) % len(self.keys)
            try:
                client = _genai_sdk.Client(api_key=key)
                response = client.models.generate_content(
                    model=use_model,
                    contents=prompt,
                )
                return response.text.strip()
            except Exception as exc:
                last_err = exc
                # back off briefly on rate-limit before trying next key
                if "429" in str(exc) and attempt < len(self.keys) - 1:
                    time.sleep(2)
        raise RuntimeError(f"All Gemini keys failed. Last: {last_err}")


_rotator = GeminiRotator()


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def wrap(text, width=16):
    return "\n".join(textwrap.wrap(str(text), width=width)) or str(text)


# ─────────────────────────────────────────────
#  Data loading
# ─────────────────────────────────────────────

def load_data(data_dir: Path):
    return load_transaction_data(data_dir)


# ─────────────────────────────────────────────
#  Chart generators  (return base64 PNG string)
# ─────────────────────────────────────────────

def chart_itemsets(fi_df):
    chart_df = fi_df.head(12).sort_values("support", ascending=True)
    colors   = ["#bfd7b5" if c == 1 else "#2e7d32" for c in chart_df["item_count"]]

    fig, ax = plt.subplots(figsize=(11, 6), dpi=110)
    bars = ax.barh(chart_df["itemset_label"], chart_df["support"],
                   color=colors, edgecolor="#1b4332")
    for bar, sup in zip(bars, chart_df["support"]):
        ax.text(bar.get_width() + 0.0005,
                bar.get_y() + bar.get_height() / 2,
                f"{sup:.4f}", va="center", fontsize=9)
    ax.set_title("1. Frequent Itemsets by Support", fontsize=14, pad=10)
    ax.set_xlabel("Support")
    ax.set_ylabel("Itemset")
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    fig.tight_layout()
    return fig_to_b64(fig)


def scatter_sample_rules(rules_df, max_points=260, bins=12):
    """Use a representative spread of rules instead of only the top lift slice."""
    if rules_df is None or rules_df.empty or len(rules_df) <= max_points:
        return rules_df.copy() if rules_df is not None else rules_df

    plot = rules_df.copy().reset_index(drop=True)
    bin_count = min(bins, max(2, len(plot)))
    plot["_support_bin"] = pd.qcut(
        plot["support"].rank(method="first"),
        q=bin_count,
        labels=False,
        duplicates="drop",
    )
    plot["_confidence_bin"] = pd.qcut(
        plot["confidence"].rank(method="first"),
        q=bin_count,
        labels=False,
        duplicates="drop",
    )

    representatives = (
        plot.sort_values(["_support_bin", "_confidence_bin", "lift"],
                         ascending=[True, True, False])
            .groupby(["_support_bin", "_confidence_bin"], dropna=False)
            .head(2)
    )

    if len(representatives) < max_points:
        remaining = plot.drop(index=representatives.index, errors="ignore")
        needed = max_points - len(representatives)
        if not remaining.empty:
            remaining = remaining.sort_values(["support", "confidence", "lift"])
            step = max(len(remaining) / needed, 1)
            fill_indexes = [remaining.index[min(int(i * step), len(remaining) - 1)]
                            for i in range(min(needed, len(remaining)))]
            representatives = pd.concat([representatives, remaining.loc[fill_indexes]])

    return (
        representatives.drop(columns=["_support_bin", "_confidence_bin"], errors="ignore")
        .drop_duplicates()
        .sort_values(["support", "confidence", "lift"], ascending=[True, True, False])
        .head(max_points)
    )


def chart_scatter(rules_df):
    plot = scatter_sample_rules(rules_df)

    fig, ax = plt.subplots(figsize=(11, 6), dpi=110)
    sc = ax.scatter(plot["support"], plot["confidence"],
                    s=48, c=plot["lift"], cmap="YlGnBu",
                    alpha=0.72, edgecolors="#173f5f", linewidths=0.6)
    label_indexes = (
        plot.index if len(plot) <= 35
        else plot.assign(_score=plot["lift"] * 2 + plot["confidence"] + plot["support"])
                 .nlargest(24, "_score").index
    )
    for label_no, idx in enumerate(plot.index, start=1):
        if idx not in label_indexes:
            continue
        row = plot.loc[idx]
        ax.annotate(str(label_no), (row["support"], row["confidence"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=7,
                    weight="bold", color="#0b2f4a")
    ax.set_title("2. Support vs Confidence Scatter Plot", fontsize=14, pad=10)
    ax.set_xlabel("Support")
    ax.set_ylabel("Confidence")
    ax.grid(linestyle="--", alpha=0.25)
    corr = plot["support"].corr(plot["confidence"]) if len(plot) > 1 else None
    if pd.notna(corr):
        ax.text(0.01, 0.98, f"Numbered rules shown - Pearson r = {corr:.2f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=8,
                color="#607d9c")
    fig.colorbar(sc, ax=ax).set_label("Lift")
    fig.tight_layout()
    return fig_to_b64(fig)


def chart_network(graph_df):
    G = nx.DiGraph()
    for _, row in graph_df.iterrows():
        G.add_edge(row["Item A"], row["Item B"],
                   lift=float(row["lift"]),
                   confidence=float(row["confidence"]),
                   support=float(row["support"]))

    pos = nx.spring_layout(G, seed=42, k=2.8, iterations=120)
    pos = {n: (x * 1.5, y * 1.5) for n, (x, y) in pos.items()}

    in_deg  = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    node_sizes  = [3000 + (in_deg.get(n, 0) + out_deg.get(n, 0)) * 450 for n in G.nodes()]
    node_colors = ["#2e7d32" if out_deg.get(n, 0) > 0 else "#a8d5a2" for n in G.nodes()]
    edge_widths = [1.2 + min(G[u][v]["lift"] * 0.55, 4.5) for u, v in G.edges()]
    edge_colors = [G[u][v]["confidence"] for u, v in G.edges()]
    edge_cmap   = colormaps["YlOrRd"]

    fig, ax = plt.subplots(figsize=(14, 9), dpi=110)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes,
                           node_color=node_colors, edgecolors="#0a2e18",
                           linewidths=1.5, alpha=0.92)
    wrapped_labels = {n: wrap(n, 14) for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=wrapped_labels, ax=ax,
                            font_size=7.5, font_weight="bold", font_color="#ffffff")
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True,
                           width=edge_widths, arrowsize=18,
                           edge_color=edge_colors, edge_cmap=edge_cmap,
                           edge_vmin=min(edge_colors) if edge_colors else 0,
                           edge_vmax=max(edge_colors) if edge_colors else 1,
                           alpha=0.82, connectionstyle="arc3,rad=0.18",
                           min_source_margin=24, min_target_margin=24)
    edge_labels = {(u, v): f"L={G[u][v]['lift']:.1f}\nC={G[u][v]['confidence']:.2f}"
                   for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax,
                                 font_size=6.5, rotate=False, label_pos=0.38,
                                 bbox={"boxstyle": "round,pad=0.12",
                                       "fc": "#fffde7", "ec": "#cccc88", "alpha": 0.88})
    sm = ScalarMappable(norm=Normalize(vmin=min(edge_colors) if edge_colors else 0,
                                       vmax=max(edge_colors) if edge_colors else 1),
                        cmap=edge_cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.02).set_label("Confidence", fontsize=9)
    ax.set_title("3. Network Graph of Top Association Rules", fontsize=13, pad=10)
    ax.axis("off")
    ax.margins(0.2)
    fig.tight_layout(pad=1.5)
    return fig_to_b64(fig)


def chart_heatmap(rules_df):
    src = rules_df[
        (~rules_df["Item A"].str.contains(",", regex=False)) &
        (~rules_df["Item B"].str.contains(",", regex=False))
    ].copy()
    if src.empty:
        return None

    top_items = pd.Index(
        pd.concat([src["Item A"], src["Item B"]])
        .value_counts().head(10).index
    )
    src = src[src["Item A"].isin(top_items) & src["Item B"].isin(top_items)]
    hm = src.pivot_table(index="Item A", columns="Item B",
                          values="lift", aggfunc="max", fill_value=0)
    if hm.empty:
        return None

    fig, ax = plt.subplots(figsize=(11, 6), dpi=110)
    im = ax.imshow(hm.values, cmap="YlOrRd", aspect="auto")
    ax.set_title("4. Association Strength Heatmap (Lift)", fontsize=14, pad=10)
    ax.set_xticks(range(len(hm.columns)))
    ax.set_yticks(range(len(hm.index)))
    ax.set_xticklabels([wrap(c, 14) for c in hm.columns],
                       rotation=35, ha="right", fontsize=8)
    ax.set_yticklabels([wrap(c, 16) for c in hm.index], fontsize=8)
    for ri in range(len(hm.index)):
        for ci in range(len(hm.columns)):
            v = hm.iat[ri, ci]
            if v > 0:
                ax.text(ci, ri, f"{v:.2f}", ha="center", va="center",
                        fontsize=7, color="#3b1f00")
    fig.colorbar(im, ax=ax).set_label("Lift")
    fig.tight_layout()
    return fig_to_b64(fig)


def chart_bubble(rules_df):
    plot   = rules_df.head(80).copy()
    sizes  = (plot["lift"] ** 2) * 48

    fig, ax = plt.subplots(figsize=(12, 6), dpi=110)
    sc = ax.scatter(plot["support"], plot["confidence"],
                    s=sizes, c=plot["lift"], cmap="RdYlGn",
                    alpha=0.75, edgecolors="#444444", linewidths=0.5)
    for _, row in plot.nlargest(7, "lift").iterrows():
        ax.annotate(f"{row['Item A']} → {row['Item B']}",
                    (row["support"], row["confidence"]),
                    textcoords="offset points", xytext=(18, 8), fontsize=8,
                    bbox={"boxstyle": "round,pad=0.25", "fc": "white",
                          "ec": "#bbbbbb", "alpha": 0.9})
    ax.set_title(
        "5. Bubble Plot  —  X: Support  |  Y: Confidence  |  Bubble Size ∝ Lift²",
        fontsize=13, pad=10)
    ax.set_xlabel("Support")
    ax.set_ylabel("Confidence")
    ax.grid(linestyle="--", alpha=0.2)
    if not plot.empty:
        x_min, x_max = plot["support"].min(), plot["support"].max()
        x_span = max(x_max - x_min, 0.001)
        ax.set_xlim(x_min - x_span * 0.04, x_max + x_span * 0.45)
        y_min, y_max = plot["confidence"].min(), plot["confidence"].max()
        y_span = max(y_max - y_min, 0.001)
        ax.set_ylim(max(0, y_min - y_span * 0.12),
                    min(1, y_max + y_span * 0.22))
    fig.colorbar(sc, ax=ax).set_label("Lift")
    fig.tight_layout()
    return fig_to_b64(fig)


def chart_parallel(rules_df):
    plot    = rules_df.head(50).copy().reset_index(drop=True)
    metrics = ["support", "confidence", "lift"]
    actual_min, actual_max = {}, {}

    norm_df = plot[metrics].copy()
    for col in metrics:
        cmin, cmax = norm_df[col].min(), norm_df[col].max()
        actual_min[col] = cmin
        actual_max[col] = cmax
        norm_df[col] = ((norm_df[col] - cmin) / (cmax - cmin)
                        if cmax > cmin else 0.5)

    lift_cmap = colormaps["RdYlGn"]
    lift_norm = Normalize(vmin=plot["lift"].min(), vmax=plot["lift"].max())

    fig, ax = plt.subplots(figsize=(11, 6), dpi=110)
    for i in range(len(norm_df)):
        color = lift_cmap(lift_norm(plot["lift"].iloc[i]))
        ax.plot(range(len(metrics)),
                [norm_df[m].iloc[i] for m in metrics],
                color=color, alpha=0.5, linewidth=1.3)

    for ci, col in enumerate(metrics):
        ax.axvline(x=ci, color="#aaaaaa", linewidth=0.8, zorder=0)
        ax.text(ci, -0.08, f"min\n{actual_min[col]:.4f}",
                ha="center", fontsize=7.5, color="#555555", va="top")
        ax.text(ci, 1.06, f"max\n{actual_max[col]:.4f}",
                ha="center", fontsize=7.5, color="#555555", va="bottom")

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(["Support", "Confidence", "Lift"],
                       fontsize=12, fontweight="bold")
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["Min", "25%", "50%", "75%", "Max"], fontsize=9)
    ax.set_ylim(-0.18, 1.22)
    ax.set_xlim(-0.3, len(metrics) - 0.7)
    ax.set_title("6. Parallel Coordinates (colored by Lift)", fontsize=13, pad=10)
    ax.set_ylabel("Normalized Value")
    ax.grid(axis="x", linestyle="--", alpha=0.25)

    sm = ScalarMappable(norm=lift_norm, cmap=lift_cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.8).set_label("Lift")
    fig.tight_layout()
    return fig_to_b64(fig)


# ─────────────────────────────────────────────
#  Recommendation engine
# ─────────────────────────────────────────────

def build_recommendations(rules_df, ll_df, fi_df, price_summary):
    """Generate structured business recommendations from analysis results."""
    recs = []
    prio = lambda lft: "High" if lft >= 2.0 else ("Medium" if lft >= 1.5 else "Low")

    # 1. Cross-sell opportunities (top 5 by lift, single-category only)
    cross = rules_df[
        ~rules_df["Item A"].str.contains(",", regex=False) &
        ~rules_df["Item B"].str.contains(",", regex=False)
    ].nlargest(5, "lift")
    for _, r in cross.iterrows():
        recs.append({
            "category": "Cross-Sell",
            "priority": prio(r["lift"]),
            "action":   f"Promote '{r['Item B']}' alongside '{r['Item A']}'",
            "why":      (f"Customers buying {r['Item A']} are {r['lift']:.1f}\u00d7 "
                         f"more likely to buy {r['Item B']} "
                         f"({r['confidence']*100:.0f}% confidence)"),
            "metrics":  {"lift": round(float(r["lift"]), 3),
                         "confidence": round(float(r["confidence"]), 3),
                         "support": round(float(r["support"]), 4)},
        })

    # 2. Loss leader pricing strategies
    if not ll_df.empty:
        for _, r in ll_df.nlargest(3, "loss_leader_score").iterrows():
            recs.append({
                "category": "Loss Leader",
                "priority": "High",
                "action":   f"Discount '{r['Item A']}' to pull buyers toward '{r['Item B']}'",
                "why":      (f"Avg price R${r['average_price']:.2f} \u00b7 "
                             f"{int(r['total_units_sold'])} units sold \u00b7 "
                             f"Score {r['loss_leader_score']:.1f}"),
                "metrics":  {"lift": round(float(r["lift"]), 3),
                             "confidence": round(float(r["confidence"]), 3),
                             "support": round(float(r["support"]), 4)},
            })

    # 3. Bundle deals (high confidence + support)
    bundles = rules_df[
        (rules_df["confidence"] >= 0.30) &
        ~rules_df["Item A"].str.contains(",", regex=False) &
        ~rules_df["Item B"].str.contains(",", regex=False)
    ].nlargest(3, "support")
    for _, r in bundles.iterrows():
        recs.append({
            "category": "Bundle",
            "priority": prio(r["lift"]),
            "action":   f"Bundle deal: '{r['Item A']}' + '{r['Item B']}'",
            "why":      (f"Co-purchased in {r['support']*100:.2f}% of orders \u00b7 "
                         f"{r['confidence']*100:.0f}% of '{r['Item A']}' buyers also buy '{r['Item B']}'"),
            "metrics":  {"lift": round(float(r["lift"]), 3),
                         "confidence": round(float(r["confidence"]), 3),
                         "support": round(float(r["support"]), 4)},
        })

    # 4. Inventory priority (top single-item frequent categories)
    for _, r in fi_df[fi_df["item_count"] == 1].nlargest(3, "support").iterrows():
        recs.append({
            "category": "Inventory",
            "priority": "Low",
            "action":   f"Maintain consistent stock of '{r['itemset_label']}'",
            "why":      f"Appears in {r['support']*100:.2f}% of all orders \u2014 consistently high demand",
            "metrics":  {"support": round(float(r["support"]), 4)},
        })

    # 5. Multi-category bundle opportunities (3-itemsets)
    for _, r in fi_df[fi_df["item_count"] >= 3].nlargest(2, "support").iterrows():
        recs.append({
            "category": "Multi-Bundle",
            "priority": "Medium",
            "action":   f"Create {r['item_count']}-item bundle: {r['itemset_label']}",
            "why":      f"Bought together in {r['support']*100:.3f}% of orders",
            "metrics":  {"support": round(float(r["support"]), 4)},
        })

    return recs


# ─────────────────────────────────────────────
#  Gemini chunked interpretation
# ─────────────────────────────────────────────

def _fmt_rules(rules_list: list, n: int = 10) -> str:
    """Compact rule text to minimise Gemini token usage."""
    lines = []
    for r in rules_list[:n]:
        lines.append(
            f"  {r['Item A']} \u2192 {r['Item B']}  "
            f"sup={r['support']:.3f} conf={r['confidence']:.2f} lift={r['lift']:.2f}"
        )
    return "\n".join(lines) or "  (no rules)"


def interpret_chunks(analysis_data: dict) -> dict:
    """
    Send 4 focused prompts to Gemini (reduced from 7 to save API quota).
    Each chunk covers related visualizations. Returns dict: {key: interpretation_string}.
    """
    from collections import Counter
    s     = analysis_data["summary"]
    rules = analysis_data["rules"]
    ll    = analysis_data["loss_leaders"]
    isets = analysis_data.get("itemsets_data", [])

    size_dist = Counter(it["item_count"] for it in isets)
    size_text = ", ".join(
        f"{cnt}\u00d7{sz}-item" for sz, cnt in sorted(size_dist.items())
    )
    top_isets = "\n".join(
        f"  {it['itemset_label']} (sup={it['support']:.3f})"
        for it in isets[:5]  # Reduced from 8 to 5
    ) or "  (none)"

    by_lift = sorted(rules, key=lambda r: r["lift"], reverse=True)
    ctx = (
        f"Olist e-commerce: {s['basket_orders']} orders, "
        f"{s['unique_categories']} categories, {s['association_rules']} rules. "
        f"Reply in 2-3 concise sentences."
    )

    # Combine similar visualizations to reduce API calls from 7 to 4
    chunks = [
        {"key": "itemsets",
         "prompt": (f"{ctx}\n\nFrequent Itemsets: {size_text}\nTop 5:\n{top_isets}\n"
                    f"Key patterns?")},
        {"key": "scatter",
         "prompt": (f"{ctx}\n\nScatter Plot (support vs confidence):\n"
                    f"{_fmt_rules(rules, 4)}\n"  # Reduced from 8 to 4
                    f"Pattern insights?")},
        {"key": "network",
         "prompt": (f"{ctx}\n\nNetwork & Heatmap (category associations):\n"
                    f"{_fmt_rules(by_lift, 4)}\n"  # Reduced from 6 to 4, combines network+heatmap
                    f"Hub categories and strongest links?")},
        {"key": "overall",
         "prompt": (f"{ctx}\n\nLoss Leaders:\n"
                    f"{_fmt_rules(ll, 3) if ll else '(none)'}\n"  # Reduced from 5 to 3
                    f"Cutoff: R${s['low_price_cutoff']}, {s['loss_leaders']} candidates.\n"
                    f"Top actionable recommendation?")},
    ]

    results = {}
    # Fill in missing keys with combined results
    results["bubble"] = ""     # Will use scatter interpretation
    results["parallel"] = ""   # Will use scatter interpretation
    results["heatmap"] = ""    # Will use network interpretation
    
    for chunk in chunks:
        try:
            results[chunk["key"]] = _rotator.call(chunk["prompt"])
        except Exception as exc:
            results[chunk["key"]] = f"[Unavailable: {exc}]"
    
    # Copy scatter interpretation to bubble and parallel
    if results.get("scatter"):
        results["bubble"] = results["scatter"]
        results["parallel"] = results["scatter"]
    # Copy network interpretation to heatmap
    if results.get("network"):
        results["heatmap"] = results["network"]
    
    return results


# ─────────────────────────────────────────────
#  Core analysis
# ─────────────────────────────────────────────

def run_analysis(params):
    data_dir        = Path(params.get("data_folder", str(DATA_DIR)))
    min_support     = float(params["min_support"])
    min_confidence  = float(params["min_confidence"])
    min_lift        = float(params["min_lift"])
    max_len         = int(params.get("max_len", 3)) or None  # None = unlimited
    price_pct       = float(params["low_price_percentile"])
    graph_limit     = int(params["graph_limit"])

    df = load_data(data_dir)

    price_summary = (
        df.groupby("item_name")
        .agg(average_price=("price", "mean"),
             total_units_sold=("quantity", "sum"),
             total_revenue=("price", "sum"))
        .reset_index()
    )
    low_price_cutoff = price_summary["average_price"].quantile(price_pct / 100)

    basket_source = df[["order_id", "item_name"]].drop_duplicates()
    counts = basket_source.groupby("order_id")["item_name"].nunique()
    basket_source = basket_source[
        basket_source["order_id"].isin(counts[counts >= 2].index)
    ]

    if basket_source.empty:
        raise ValueError("No multi-category orders found. Apriori needs orders with at least two categories.")

    basket = pd.crosstab(basket_source["order_id"],
                          basket_source["item_name"]).astype(bool)

    frequent_itemsets = apriori(basket, min_support=min_support, use_colnames=True, max_len=max_len)
    if frequent_itemsets.empty:
        raise ValueError("No frequent itemsets found. Try lowering min_support.")

    rules = association_rules(
        frequent_itemsets,
        num_itemsets=basket.shape[0],
        metric="confidence",
        min_threshold=min_confidence,
    )
    if rules.empty:
        raise ValueError("No association rules found. Try lowering min_confidence.")

    rules = rules[rules["lift"] >= min_lift].copy()
    if rules.empty:
        raise ValueError("No rules passed the lift threshold. Try lowering min_lift.")

    istr = lambda s: ", ".join(sorted(list(s)))

    # frequent itemsets display
    fi_disp = frequent_itemsets.copy()
    fi_disp["itemset_label"] = fi_disp["itemsets"].apply(istr)
    fi_disp["item_count"]    = fi_disp["itemsets"].apply(len)
    fi_disp["support"]       = fi_disp["support"].round(4)
    fi_disp = fi_disp[["itemset_label", "item_count", "support"]] \
                .sort_values(["support", "item_count"], ascending=[False, False])

    rules["Item A"] = rules["antecedents"].apply(istr)
    rules["Item B"] = rules["consequents"].apply(istr)
    rules_disp = rules[["Item A", "Item B", "support", "confidence", "lift"]] \
                    .sort_values("lift", ascending=False).copy()
    for col in ["support", "confidence", "lift"]:
        rules_disp[col] = rules_disp[col].round(4)

    # loss leaders
    single_rules = rules_disp[~rules_disp["Item A"].str.contains(",", regex=False)].copy()
    lc = single_rules.merge(price_summary, left_on="Item A",
                             right_on="item_name", how="left")
    lc = lc[lc["average_price"] <= low_price_cutoff].copy()

    if not lc.empty:
        lc["loss_leader_score"] = (
            lc["confidence"] * lc["lift"] * lc["total_units_sold"] /
            (lc["average_price"] + 1)
        )
        lc = lc.sort_values("loss_leader_score", ascending=False)
        ll_disp = lc[["Item A", "Item B", "average_price", "total_units_sold",
                       "support", "confidence", "lift", "loss_leader_score"]].copy()
        ll_disp["average_price"]     = ll_disp["average_price"].round(2)
        ll_disp["loss_leader_score"] = ll_disp["loss_leader_score"].round(2)
    else:
        ll_disp = pd.DataFrame(
            columns=["Item A", "Item B", "average_price", "total_units_sold",
                     "support", "confidence", "lift", "loss_leader_score"])

    graph_df = rules_disp[
        (~rules_disp["Item A"].str.contains(",", regex=False)) &
        (~rules_disp["Item B"].str.contains(",", regex=False))
    ].sort_values(["lift", "confidence", "support"], ascending=False) \
     .head(graph_limit).copy()

    summary = {
        "total_records":     int(len(df)),
        "unique_orders":     int(df["order_id"].nunique()),
        "unique_categories": int(df["item_name"].nunique()),
        "basket_orders":     int(basket.shape[0]),
        "frequent_itemsets": int(len(frequent_itemsets)),
        "association_rules": int(len(rules_disp)),
        "low_price_cutoff":  round(float(low_price_cutoff), 2),
        "loss_leaders":      int(len(ll_disp)),
        "graph_rules":       int(len(graph_df)),
        "top_rule": (
            f"If customers buy '{rules_disp.iloc[0]['Item A']}', "
            f"they are likely to also buy '{rules_disp.iloc[0]['Item B']}' — "
            f"Support: {rules_disp.iloc[0]['support']}, "
            f"Confidence: {rules_disp.iloc[0]['confidence']}, "
            f"Lift: {rules_disp.iloc[0]['lift']}"
        ) if not rules_disp.empty else ""
    }

    charts = {
        "itemset":  chart_itemsets(fi_disp),
        "scatter":  chart_scatter(rules_disp),
        "network":  chart_network(graph_df),
        "heatmap":  chart_heatmap(rules_disp),
        "bubble":   chart_bubble(rules_disp),
        "parallel": chart_parallel(rules_disp),
    }

    # itemset size breakdown (e.g. {"1": 45, "2": 12, "3": 2})
    size_counts = fi_disp.groupby("item_count").size().reset_index(name="count")
    summary["itemset_sizes"] = {
        str(int(r["item_count"])): int(r["count"])
        for _, r in size_counts.iterrows()
    }

    recommendations = build_recommendations(rules_disp, ll_disp, fi_disp, price_summary)

    return {
        "summary":         summary,
        "rules":           rules_disp.to_dict(orient="records"),
        "loss_leaders":    ll_disp.to_dict(orient="records"),
        "charts":          charts,
        "itemsets_data":   fi_disp.to_dict(orient="records"),
        "recommendations": recommendations,
    }


# ─────────────────────────────────────────────
#  Flask routes
# ─────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", default_data_folder=str(DATA_DIR))


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        params = request.get_json(force=True)
        result = run_analysis(params)
        return jsonify({"ok": True, **result})
    except (FileNotFoundError, ValueError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"ok": False, "error": f"Unexpected error: {exc}"}), 500


@app.route("/set-keys", methods=["POST"])
def set_keys():
    data = request.get_json(force=True)
    keys = data.get("keys", [])
    model = data.get("model", "").strip()
    if isinstance(keys, str):
        keys = [k.strip() for k in keys.splitlines() if k.strip()]
    _rotator.set_keys(keys, model)
    return jsonify({
        "ok":             True,
        "count":          len(_rotator.keys),
        "model":          _rotator.model,
        "gemini_package": GEMINI_AVAILABLE,
    })


@app.route("/interpret", methods=["POST"])
def interpret():
    if not _rotator.available():
        msg = ("Gemini not ready \u2014 "
               + ("add API keys in the sidebar." if GEMINI_AVAILABLE
                  else "run: pip install google-genai, then add keys."))
        return jsonify({"ok": False, "error": msg}), 400
    try:
        data   = request.get_json(force=True)
        result = interpret_chunks(data)
        return jsonify({"ok": True, "interpretations": result})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/browse")
def browse():
    """Server-side directory browser (local use only)."""
    path = request.args.get("path", "").strip()
    if not path:
        path = str(Path.home())
    try:
        p = Path(path).resolve()
        if not p.exists():
            return jsonify({"ok": False, "error": "Path does not exist"}), 400
        if not p.is_dir():
            p = p.parent
        items = []
        for item in sorted(p.iterdir(),
                            key=lambda x: (not x.is_dir(), x.name.lower())):
            items.append({
                "name":   item.name,
                "path":   str(item),
                "is_dir": item.is_dir(),
            })
        return jsonify({
            "ok":      True,
            "current": str(p),
            "parent":  str(p.parent) if str(p.parent) != str(p) else None,
            "items":   items,
        })
    except PermissionError:
        return jsonify({"ok": False, "error": "Permission denied"}), 403
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


if __name__ == "__main__":
    print("Starting Apriori Loss Leader Discovery \u2014 Flask App")
    print("Open http://127.0.0.1:5000 in your browser")
    app.run(debug=False, port=5000)
