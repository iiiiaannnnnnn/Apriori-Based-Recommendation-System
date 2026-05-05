import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import textwrap

import pandas as pd
import networkx as nx

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import colormaps
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure

from mlxtend.frequent_patterns import apriori, association_rules


class AprioriLossLeaderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Apriori Loss Leader Discovery Tool")
        self.root.geometry("1300x780")

        self.rules_df = None
        self.loss_leader_df = None
        self.frequent_itemsets_df = None

        self.data_folder = tk.StringVar(value=str(Path.cwd() / "data"))

        self.min_support = tk.StringVar(value="0.003")
        self.min_confidence = tk.StringVar(value="0.30")
        self.min_lift = tk.StringVar(value="1.20")
        self.low_price_percentile = tk.StringVar(value="25")
        self.graph_limit = tk.StringVar(value="20")

        self.build_ui()

    def build_ui(self):
        title = tk.Label(
            self.root,
            text="Apriori-Based Hidden Product Connection and Loss Leader Discovery Tool",
            font=("Arial", 16, "bold")
        )
        title.pack(pady=10)

        control_frame = tk.LabelFrame(self.root, text="Settings", padx=10, pady=10)
        control_frame.pack(fill="x", padx=15, pady=5)

        tk.Label(control_frame, text="Data Folder:").grid(row=0, column=0, sticky="w")
        tk.Entry(control_frame, textvariable=self.data_folder, width=80).grid(row=0, column=1, padx=5, sticky="w")

        tk.Button(
            control_frame,
            text="Browse",
            command=self.browse_folder
        ).grid(row=0, column=2, padx=5)

        tk.Label(control_frame, text="Min Support:").grid(row=1, column=0, sticky="w", pady=5)
        tk.Entry(control_frame, textvariable=self.min_support, width=10).grid(row=1, column=1, sticky="w")

        tk.Label(control_frame, text="Min Confidence:").grid(row=1, column=1, sticky="w", padx=(100, 0))
        tk.Entry(control_frame, textvariable=self.min_confidence, width=10).grid(row=1, column=1, sticky="w", padx=(210, 0))

        tk.Label(control_frame, text="Min Lift:").grid(row=1, column=1, sticky="w", padx=(330, 0))
        tk.Entry(control_frame, textvariable=self.min_lift, width=10).grid(row=1, column=1, sticky="w", padx=(390, 0))

        tk.Label(control_frame, text="Low Price Percentile:").grid(row=1, column=1, sticky="w", padx=(510, 0))
        tk.Entry(control_frame, textvariable=self.low_price_percentile, width=10).grid(row=1, column=1, sticky="w", padx=(650, 0))

        tk.Label(control_frame, text="Graph Rules:").grid(row=1, column=1, sticky="w", padx=(770, 0))
        tk.Entry(control_frame, textvariable=self.graph_limit, width=10).grid(row=1, column=1, sticky="w", padx=(850, 0))

        tk.Button(
            control_frame,
            text="Run Apriori Analysis",
            command=self.run_analysis,
            bg="#2e7d32",
            fg="white",
            width=20
        ).grid(row=1, column=2, padx=10)

        tk.Button(
            control_frame,
            text="Export Rules",
            command=self.export_rules,
            width=15
        ).grid(row=1, column=3, padx=5)

        tk.Button(
            control_frame,
            text="Export Loss Leaders",
            command=self.export_loss_leaders,
            width=18
        ).grid(row=1, column=4, padx=5)

        summary_frame = tk.LabelFrame(self.root, text="Analysis Summary", padx=10, pady=10)
        summary_frame.pack(fill="x", padx=15, pady=5)

        self.summary_text = tk.Text(summary_frame, height=7, wrap="word")
        self.summary_text.pack(fill="x")

        self.summary_text.insert(
            "end",
            "Click 'Run Apriori Analysis' to start.\n\n"
            "This tool treats each order as a basket and each product category as an item.\n"
            "It discovers rules such as Item A → Item B and identifies low-priced Item A candidates as possible loss leaders.\n"
            "The Network Graph tab visualizes the hidden item associations."
        )

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=15, pady=10)

        self.rules_tab = ttk.Frame(self.notebook)
        self.loss_tab = ttk.Frame(self.notebook)
        self.itemset_tab = ttk.Frame(self.notebook)
        self.scatter_tab = ttk.Frame(self.notebook)
        self.heatmap_tab = ttk.Frame(self.notebook)
        self.bubble_tab = ttk.Frame(self.notebook)
        self.parallel_tab = ttk.Frame(self.notebook)
        self.graph_tab = ttk.Frame(self.notebook)
        self.guide_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.rules_tab, text="Association Rules")
        self.notebook.add(self.loss_tab, text="Possible Loss Leaders")
        self.notebook.add(self.itemset_tab, text="1. Frequent Itemsets")
        self.notebook.add(self.scatter_tab, text="2. Scatter Plot")
        self.notebook.add(self.heatmap_tab, text="4. Association Heatmap")
        self.notebook.add(self.bubble_tab, text="5. Bubble Plot")
        self.notebook.add(self.parallel_tab, text="6. Parallel Coordinates")
        self.notebook.add(self.graph_tab, text="3. Network Graph")
        self.notebook.add(self.guide_tab, text="Guide / Interpretation")

        self.create_guide_tab()

    def format_itemset_label(self, value, width=18):
        if isinstance(value, (set, frozenset, list, tuple)):
            label = ", ".join(sorted(str(item) for item in value))
        else:
            label = str(value)

        return "\n".join(textwrap.wrap(label, width=width)) or label

    def browse_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.data_folder.set(folder)

    def load_data(self):
        data_dir = Path(self.data_folder.get())

        order_items_path = data_dir / "olist_order_items_dataset.csv"
        products_path = data_dir / "olist_products_dataset.csv"
        translation_path = data_dir / "product_category_name_translation.csv"

        missing_files = []

        if not order_items_path.exists():
            missing_files.append("olist_order_items_dataset.csv")

        if not products_path.exists():
            missing_files.append("olist_products_dataset.csv")

        if not translation_path.exists():
            missing_files.append("product_category_name_translation.csv")

        if missing_files:
            messagebox.showerror(
                "Missing Files",
                "The following files are missing:\n\n" + "\n".join(missing_files)
            )
            return None

        order_items = pd.read_csv(order_items_path)
        products = pd.read_csv(products_path)
        translation = pd.read_csv(translation_path)

        df = order_items.merge(products, on="product_id", how="left")
        df = df.merge(translation, on="product_category_name", how="left")

        df["item_name"] = df["product_category_name_english"].fillna(df["product_category_name"])
        df = df.dropna(subset=["order_id", "item_name"])

        return df

    def run_analysis(self):
        try:
            min_support = float(self.min_support.get())
            min_confidence = float(self.min_confidence.get())
            min_lift = float(self.min_lift.get())
            price_percentile = float(self.low_price_percentile.get())
            graph_limit = int(self.graph_limit.get())
        except ValueError:
            messagebox.showerror(
                "Invalid Input",
                "Please enter valid numeric values for support, confidence, lift, price percentile, and graph rules."
            )
            return

        if min_support <= 0:
            messagebox.showerror("Invalid Input", "Minimum support must be greater than 0.")
            return

        if not 0 < min_confidence <= 1:
            messagebox.showerror("Invalid Input", "Minimum confidence must be between 0 and 1.")
            return

        if min_lift < 1:
            messagebox.showerror("Invalid Input", "Minimum lift should be 1 or higher.")
            return

        if not 1 <= price_percentile <= 100:
            messagebox.showerror("Invalid Input", "Low price percentile must be between 1 and 100.")
            return

        if graph_limit <= 0:
            messagebox.showerror("Invalid Input", "Graph rules must be greater than 0.")
            return

        self.summary_text.delete("1.0", "end")
        self.summary_text.insert("end", "Loading data and running Apriori analysis...\n")
        self.root.update_idletasks()

        df = self.load_data()

        if df is None:
            return

        price_summary = (
            df.groupby("item_name")
            .agg(
                average_price=("price", "mean"),
                total_units_sold=("order_item_id", "count"),
                total_revenue=("price", "sum")
            )
            .reset_index()
        )

        low_price_cutoff = price_summary["average_price"].quantile(price_percentile / 100)

        basket_source = df[["order_id", "item_name"]].drop_duplicates()

        order_category_count = basket_source.groupby("order_id")["item_name"].nunique()
        multi_item_orders = order_category_count[order_category_count >= 2].index

        basket_source = basket_source[basket_source["order_id"].isin(multi_item_orders)]

        if basket_source.empty:
            messagebox.showwarning(
                "No Basket Data",
                "No multi-category orders were found. Apriori needs orders with at least two categories."
            )
            return

        basket = pd.crosstab(
            basket_source["order_id"],
            basket_source["item_name"]
        )

        basket = basket.astype(bool)

        frequent_itemsets = apriori(
            basket,
            min_support=min_support,
            use_colnames=True
        )

        if frequent_itemsets.empty:
            messagebox.showwarning(
                "No Frequent Itemsets",
                "No frequent itemsets found. Try lowering the minimum support."
            )
            return

        rules = association_rules(
            frequent_itemsets,
            metric="confidence",
            min_threshold=min_confidence
        )

        if rules.empty:
            messagebox.showwarning(
                "No Rules",
                "No association rules found. Try lowering the minimum confidence."
            )
            return

        rules = rules[rules["lift"] >= min_lift].copy()

        if rules.empty:
            messagebox.showwarning(
                "No Rules After Lift Filter",
                "No rules passed the lift threshold. Try lowering the minimum lift."
            )
            return

        def itemset_to_string(itemset):
            return ", ".join(sorted(list(itemset)))

        frequent_itemsets_display = frequent_itemsets.copy()
        frequent_itemsets_display["itemset_label"] = frequent_itemsets_display["itemsets"].apply(itemset_to_string)
        frequent_itemsets_display["item_count"] = frequent_itemsets_display["itemsets"].apply(len)
        frequent_itemsets_display["support"] = frequent_itemsets_display["support"].round(4)
        self.frequent_itemsets_df = frequent_itemsets_display[
            ["itemset_label", "item_count", "support"]
        ].sort_values(by=["support", "item_count"], ascending=[False, False])

        rules["Item A"] = rules["antecedents"].apply(itemset_to_string)
        rules["Item B"] = rules["consequents"].apply(itemset_to_string)

        rules_display = rules[
            ["Item A", "Item B", "support", "confidence", "lift"]
        ].sort_values(by="lift", ascending=False)

        rules_display["support"] = rules_display["support"].round(4)
        rules_display["confidence"] = rules_display["confidence"].round(4)
        rules_display["lift"] = rules_display["lift"].round(4)

        self.rules_df = rules_display.copy()

        single_item_rules = rules_display[
            ~rules_display["Item A"].str.contains(",", regex=False)
        ].copy()

        loss_candidates = single_item_rules.merge(
            price_summary,
            left_on="Item A",
            right_on="item_name",
            how="left"
        )

        loss_candidates = loss_candidates[
            loss_candidates["average_price"] <= low_price_cutoff
        ].copy()

        if not loss_candidates.empty:
            loss_candidates["loss_leader_score"] = (
                loss_candidates["confidence"]
                * loss_candidates["lift"]
                * loss_candidates["total_units_sold"]
                / (loss_candidates["average_price"] + 1)
            )

            loss_candidates = loss_candidates.sort_values(
                by="loss_leader_score",
                ascending=False
            )

            self.loss_leader_df = loss_candidates[
                [
                    "Item A",
                    "Item B",
                    "average_price",
                    "total_units_sold",
                    "support",
                    "confidence",
                    "lift",
                    "loss_leader_score"
                ]
            ].copy()

            self.loss_leader_df["average_price"] = self.loss_leader_df["average_price"].round(2)
            self.loss_leader_df["loss_leader_score"] = self.loss_leader_df["loss_leader_score"].round(2)
        else:
            self.loss_leader_df = pd.DataFrame()

        self.display_dataframe(self.rules_tab, self.rules_df)
        self.display_dataframe(self.loss_tab, self.loss_leader_df)
        self.display_itemset_chart(self.itemset_tab, self.frequent_itemsets_df)
        self.display_rule_scatter(self.scatter_tab, self.rules_df)
        self.display_association_heatmap(self.heatmap_tab, self.rules_df)
        self.display_bubble_plot(self.bubble_tab, self.rules_df)
        self.display_parallel_coords(self.parallel_tab, self.rules_df)

        graph_df = self.rules_df[
            (~self.rules_df["Item A"].str.contains(",", regex=False)) &
            (~self.rules_df["Item B"].str.contains(",", regex=False))
        ].sort_values(by=["lift", "confidence", "support"], ascending=False).head(graph_limit).copy()

        self.display_network_graph(self.graph_tab, graph_df)

        self.summary_text.delete("1.0", "end")
        self.summary_text.insert("end", "Analysis Completed Successfully!\n\n")
        self.summary_text.insert("end", f"Total order-item records: {len(df):,}\n")
        self.summary_text.insert("end", f"Unique orders: {df['order_id'].nunique():,}\n")
        self.summary_text.insert("end", f"Unique product categories: {df['item_name'].nunique():,}\n")
        self.summary_text.insert("end", f"Orders used for Apriori: {basket.shape[0]:,}\n")
        self.summary_text.insert("end", f"Frequent itemsets found: {len(frequent_itemsets):,}\n")
        self.summary_text.insert("end", f"Association rules found: {len(self.rules_df):,}\n")
        self.summary_text.insert("end", f"Low-price threshold: {low_price_cutoff:.2f}\n")
        self.summary_text.insert("end", f"Possible loss-leader candidates: {len(self.loss_leader_df):,}\n")
        self.summary_text.insert("end", f"Rules shown in network graph: {len(graph_df):,}\n\n")

        if not self.rules_df.empty:
            top_rule = self.rules_df.iloc[0]
            self.summary_text.insert(
                "end",
                f"Strongest rule: If customers buy '{top_rule['Item A']}', "
                f"they are also likely to buy '{top_rule['Item B']}'.\n"
            )
            self.summary_text.insert(
                "end",
                f"Support: {top_rule['support']} | "
                f"Confidence: {top_rule['confidence']} | "
                f"Lift: {top_rule['lift']}\n"
            )

    def display_dataframe(self, parent, df):
        for widget in parent.winfo_children():
            widget.destroy()

        if df is None or df.empty:
            label = tk.Label(
                parent,
                text="No results to display.",
                font=("Arial", 12)
            )
            label.pack(pady=20)
            return

        frame = tk.Frame(parent)
        frame.pack(fill="both", expand=True)

        tree = ttk.Treeview(frame, columns=list(df.columns), show="headings")

        vertical_scroll = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        horizontal_scroll = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)

        tree.configure(
            yscrollcommand=vertical_scroll.set,
            xscrollcommand=horizontal_scroll.set
        )

        tree.grid(row=0, column=0, sticky="nsew")
        vertical_scroll.grid(row=0, column=1, sticky="ns")
        horizontal_scroll.grid(row=1, column=0, sticky="ew")

        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        for col in df.columns:
            tree.heading(col, text=col)
            tree.column(col, width=170, anchor="center")

        for _, row in df.iterrows():
            tree.insert("", "end", values=list(row))

    def display_itemset_chart(self, parent, df):
        for widget in parent.winfo_children():
            widget.destroy()

        if df is None or df.empty:
            label = tk.Label(parent, text="No frequent itemsets to visualize.", font=("Arial", 12))
            label.pack(pady=20)
            return

        chart_df = df.head(12).copy().sort_values(by="support", ascending=True)

        fig = Figure(figsize=(11.5, 6.5), dpi=100)
        ax = fig.add_subplot(111)

        labels = [self.format_itemset_label(label, width=22) for label in chart_df["itemset_label"]]
        colors = ["#bfd7b5" if count == 1 else "#2e7d32" for count in chart_df["item_count"]]

        bars = ax.barh(labels, chart_df["support"], color=colors, edgecolor="#1b4332")
        ax.set_title("Top Frequent Itemsets by Support", fontsize=14, pad=12)
        ax.set_xlabel("Support")
        ax.set_ylabel("Itemset")
        ax.grid(axis="x", linestyle="--", alpha=0.25)

        for bar, support in zip(bars, chart_df["support"]):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2, f"{support:.4f}", va="center", fontsize=9)

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def display_rule_scatter(self, parent, df):
        for widget in parent.winfo_children():
            widget.destroy()

        if df is None or df.empty:
            label = tk.Label(parent, text="No association rules to visualize.", font=("Arial", 12))
            label.pack(pady=20)
            return

        plot_df = df.head(200).copy()
        bubble_sizes = plot_df["lift"].clip(lower=1).mul(140)

        fig = Figure(figsize=(11.5, 6.5), dpi=100)
        ax = fig.add_subplot(111)

        scatter = ax.scatter(
            plot_df["support"],
            plot_df["confidence"],
            s=bubble_sizes,
            c=plot_df["lift"],
            cmap="YlGnBu",
            alpha=0.7,
            edgecolors="#173f5f",
            linewidths=0.6
        )

        top_points = plot_df.nlargest(8, ["lift", "confidence"])
        for _, row in top_points.iterrows():
            ax.annotate(
                f"{row['Item A']} -> {row['Item B']}",
                (row["support"], row["confidence"]),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=8,
                bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "#cccccc", "alpha": 0.85}
            )

        ax.set_title("Support vs Confidence of Association Rules", fontsize=14, pad=12)
        ax.set_xlabel("Support")
        ax.set_ylabel("Confidence")
        ax.grid(linestyle="--", alpha=0.25)
        colorbar = fig.colorbar(scatter, ax=ax)
        colorbar.set_label("Lift")
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def display_association_heatmap(self, parent, df):
        for widget in parent.winfo_children():
            widget.destroy()

        if df is None or df.empty:
            label = tk.Label(parent, text="No association heatmap available.", font=("Arial", 12))
            label.pack(pady=20)
            return

        heatmap_source = df[
            (~df["Item A"].str.contains(",", regex=False)) &
            (~df["Item B"].str.contains(",", regex=False))
        ].copy()

        if heatmap_source.empty:
            label = tk.Label(parent, text="Heatmap needs one-to-one category rules.", font=("Arial", 12))
            label.pack(pady=20)
            return

        top_items = pd.Index(
            pd.concat([heatmap_source["Item A"], heatmap_source["Item B"]])
            .value_counts()
            .head(10)
            .index
        )

        heatmap_source = heatmap_source[
            heatmap_source["Item A"].isin(top_items) & heatmap_source["Item B"].isin(top_items)
        ]

        heatmap_df = heatmap_source.pivot_table(
            index="Item A",
            columns="Item B",
            values="lift",
            aggfunc="max",
            fill_value=0
        )

        if heatmap_df.empty:
            label = tk.Label(parent, text="Not enough rule overlap for a readable heatmap.", font=("Arial", 12))
            label.pack(pady=20)
            return

        fig = Figure(figsize=(11.5, 6.5), dpi=100)
        ax = fig.add_subplot(111)

        image = ax.imshow(heatmap_df.values, cmap="YlOrRd", aspect="auto")
        ax.set_title("Association Strength Heatmap (Lift)", fontsize=14, pad=12)
        ax.set_xticks(range(len(heatmap_df.columns)))
        ax.set_yticks(range(len(heatmap_df.index)))
        ax.set_xticklabels([self.format_itemset_label(label, width=14) for label in heatmap_df.columns], rotation=35, ha="right", fontsize=8)
        ax.set_yticklabels([self.format_itemset_label(label, width=16) for label in heatmap_df.index], fontsize=8)

        for row_index in range(len(heatmap_df.index)):
            for column_index in range(len(heatmap_df.columns)):
                value = heatmap_df.iat[row_index, column_index]
                if value > 0:
                    ax.text(column_index, row_index, f"{value:.2f}", ha="center", va="center", fontsize=7, color="#3b1f00")

        colorbar = fig.colorbar(image, ax=ax)
        colorbar.set_label("Lift")
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def display_bubble_plot(self, parent, df):
        for widget in parent.winfo_children():
            widget.destroy()

        if df is None or df.empty:
            tk.Label(parent, text="No data to display.", font=("Arial", 12)).pack(pady=20)
            return

        plot_df = df.head(80).copy()
        bubble_sizes = (plot_df["lift"] ** 2) * 90

        fig = Figure(figsize=(11.5, 6.5), dpi=100)
        ax = fig.add_subplot(111)

        scatter = ax.scatter(
            plot_df["support"],
            plot_df["confidence"],
            s=bubble_sizes,
            c=plot_df["lift"],
            cmap="RdYlGn",
            alpha=0.75,
            edgecolors="#444444",
            linewidths=0.5
        )

        top = plot_df.nlargest(7, "lift")
        for _, row in top.iterrows():
            ax.annotate(
                f"{row['Item A']} \u2192 {row['Item B']}",
                (row["support"], row["confidence"]),
                textcoords="offset points",
                xytext=(8, 8),
                fontsize=8,
                bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#bbbbbb", "alpha": 0.9}
            )

        ax.set_title(
            "Bubble Plot  \u2014  X: Support  |  Y: Confidence  |  Bubble Size \u221d Lift\u00b2",
            fontsize=13, pad=12
        )
        ax.set_xlabel("Support")
        ax.set_ylabel("Confidence")
        ax.grid(linestyle="--", alpha=0.2)
        cb = fig.colorbar(scatter, ax=ax)
        cb.set_label("Lift")
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def display_parallel_coords(self, parent, df):
        for widget in parent.winfo_children():
            widget.destroy()

        if df is None or df.empty:
            tk.Label(parent, text="No data to display.", font=("Arial", 12)).pack(pady=20)
            return

        plot_df = df.head(50).copy().reset_index(drop=True)
        metrics = ["support", "confidence", "lift"]

        norm_df = plot_df[metrics].copy()
        actual_min = {}
        actual_max = {}
        for col in metrics:
            col_min = norm_df[col].min()
            col_max = norm_df[col].max()
            actual_min[col] = col_min
            actual_max[col] = col_max
            norm_df[col] = (norm_df[col] - col_min) / (col_max - col_min) if col_max > col_min else 0.5

        fig = Figure(figsize=(11.5, 6.5), dpi=100)
        ax = fig.add_subplot(111)

        lift_cmap = colormaps["RdYlGn"]
        lift_norm = Normalize(vmin=plot_df["lift"].min(), vmax=plot_df["lift"].max())

        for i in range(len(norm_df)):
            lift_val = plot_df["lift"].iloc[i]
            color = lift_cmap(lift_norm(lift_val))
            values = [norm_df[m].iloc[i] for m in metrics]
            ax.plot(range(len(metrics)), values, color=color, alpha=0.5, linewidth=1.3)

        for col_idx, col in enumerate(metrics):
            ax.axvline(x=col_idx, color="#aaaaaa", linewidth=0.8, zorder=0)
            ax.text(col_idx, -0.08, f"min\n{actual_min[col]:.4f}", ha="center", fontsize=7.5,
                    color="#555555", va="top")
            ax.text(col_idx, 1.06, f"max\n{actual_max[col]:.4f}", ha="center", fontsize=7.5,
                    color="#555555", va="bottom")

        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(["Support", "Confidence", "Lift"], fontsize=12, fontweight="bold")
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["Min", "25%", "50%", "75%", "Max"], fontsize=9)
        ax.set_ylim(-0.15, 1.2)
        ax.set_xlim(-0.3, len(metrics) - 0.7)
        ax.set_title("Parallel Coordinates Plot of Association Rules (colored by Lift)", fontsize=13, pad=12)
        ax.set_ylabel("Normalized Value")
        ax.grid(axis="x", linestyle="--", alpha=0.25)

        scalar_mappable = ScalarMappable(norm=lift_norm, cmap=lift_cmap)
        scalar_mappable.set_array([])
        cb = fig.colorbar(scalar_mappable, ax=ax, shrink=0.8)
        cb.set_label("Lift")
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def display_network_graph(self, parent, df):
        for widget in parent.winfo_children():
            widget.destroy()

        if df is None or df.empty:
            tk.Label(
                parent,
                text="No network graph available. Try lowering support, confidence, or lift.",
                font=("Arial", 12)
            ).pack(pady=20)
            return

        main_frame = tk.Frame(parent)
        main_frame.pack(fill="both", expand=True)

        tk.Label(
            main_frame,
            text=(
                "Network Graph  \u2014  "
                "Node size = connection count.  "
                "Dark green node = originates rules.  "
                "Arrow = Item A \u2192 Item B.  "
                "Thicker edge = higher lift.  "
                "Edge color = confidence (yellow \u2192 red = low \u2192 high)."
            ),
            font=("Arial", 9),
            wraplength=1200,
            justify="left",
            fg="#333333"
        ).pack(padx=12, pady=(6, 2), anchor="w")

        graph_frame = tk.Frame(main_frame)
        graph_frame.pack(fill="both", expand=True)

        G = nx.DiGraph()
        for _, row in df.iterrows():
            G.add_edge(
                row["Item A"], row["Item B"],
                lift=float(row["lift"]),
                confidence=float(row["confidence"]),
                support=float(row["support"])
            )

        fig = Figure(figsize=(14, 9), dpi=96)
        ax = fig.add_subplot(111)

        pos = nx.spring_layout(G, seed=42, k=2.8, iterations=120)
        pos = {node: (x * 1.5, y * 1.5) for node, (x, y) in pos.items()}

        in_deg = dict(G.in_degree())
        out_deg = dict(G.out_degree())
        node_sizes = [3000 + (in_deg.get(n, 0) + out_deg.get(n, 0)) * 450 for n in G.nodes()]
        node_colors = ["#2e7d32" if out_deg.get(n, 0) > 0 else "#a8d5a2" for n in G.nodes()]

        edge_widths = [1.2 + min(G[u][v]["lift"] * 0.55, 4.5) for u, v in G.edges()]
        edge_colors = [G[u][v]["confidence"] for u, v in G.edges()]
        edge_cmap = colormaps["YlOrRd"]

        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_size=node_sizes,
            node_color=node_colors,
            edgecolors="#0a2e18",
            linewidths=1.5,
            alpha=0.92
        )

        wrapped_labels = {
            node: "\n".join(textwrap.wrap(node, width=14)) or node
            for node in G.nodes()
        }
        nx.draw_networkx_labels(
            G, pos, labels=wrapped_labels, ax=ax,
            font_size=7.5, font_weight="bold", font_color="#ffffff"
        )

        nx.draw_networkx_edges(
            G, pos, ax=ax,
            arrows=True,
            width=edge_widths,
            arrowsize=18,
            edge_color=edge_colors,
            edge_cmap=edge_cmap,
            edge_vmin=min(edge_colors) if edge_colors else 0,
            edge_vmax=max(edge_colors) if edge_colors else 1,
            alpha=0.82,
            connectionstyle="arc3,rad=0.18",
            min_source_margin=24,
            min_target_margin=24
        )

        edge_labels = {
            (u, v): f"L={G[u][v]['lift']:.1f}\nC={G[u][v]['confidence']:.2f}"
            for u, v in G.edges()
        }
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=edge_labels, ax=ax,
            font_size=6.5,
            bbox={"boxstyle": "round,pad=0.12", "fc": "#fffde7", "ec": "#cccc88", "alpha": 0.88},
            rotate=False,
            label_pos=0.38
        )

        if edge_colors:
            sm = ScalarMappable(
                norm=Normalize(vmin=min(edge_colors), vmax=max(edge_colors)),
                cmap=edge_cmap
            )
            sm.set_array([])
            cb = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)
            cb.set_label("Confidence", fontsize=9)

        ax.set_title("3. Network Graph of Top Association Rules", fontsize=13, pad=10)
        ax.axis("off")
        ax.margins(0.2)
        fig.tight_layout(pad=1.5)

        canvas = FigureCanvasTkAgg(fig, master=graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def export_rules(self):
        if self.rules_df is None or self.rules_df.empty:
            messagebox.showwarning("No Data", "No association rules to export.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv")],
            initialfile="apriori_association_rules.csv"
        )

        if file_path:
            self.rules_df.to_csv(file_path, index=False)
            messagebox.showinfo("Export Successful", "Association rules exported successfully.")

    def export_loss_leaders(self):
        if self.loss_leader_df is None or self.loss_leader_df.empty:
            messagebox.showwarning("No Data", "No loss-leader candidates to export.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv")],
            initialfile="possible_loss_leader_candidates.csv"
        )

        if file_path:
            self.loss_leader_df.to_csv(file_path, index=False)
            messagebox.showinfo("Export Successful", "Loss-leader candidates exported successfully.")

    def create_guide_tab(self):
        guide = tk.Text(self.guide_tab, wrap="word", font=("Arial", 11))
        guide.pack(fill="both", expand=True, padx=10, pady=10)

        guide.insert("end", "HOW TO INTERPRET THE OUTPUT\n\n")

        guide.insert("end", "1. Item A → Item B\n")
        guide.insert(
            "end",
            "This means that when customers buy Item A, they are also likely to buy Item B.\n\n"
        )

        guide.insert("end", "2. Support\n")
        guide.insert(
            "end",
            "Support shows how often Item A and Item B appear together in all transactions.\n\n"
        )

        guide.insert("end", "3. Confidence\n")
        guide.insert(
            "end",
            "Confidence shows how often Item B is bought when Item A is bought.\n\n"
        )

        guide.insert("end", "4. Lift\n")
        guide.insert(
            "end",
            "Lift shows the strength of the relationship. A lift greater than 1 means Item A increases the chance of buying Item B.\n\n"
        )

        guide.insert("end", "5. Possible Loss Leader\n")
        guide.insert(
            "end",
            "A possible loss leader is a low-priced Item A that has a strong relationship with Item B. "
            "This means the shop may promote or lower the price of Item A to attract buyers and encourage them to buy Item B.\n\n"
        )

        guide.insert("end", "6. Network Graph\n")
        guide.insert(
            "end",
            "The network graph visually presents the association rules. Each node represents an item or product category. "
            "Each arrow represents a rule from Item A to Item B. The edge label shows the lift and confidence values. "
            "This makes it easier to see hidden product connections compared with looking only at a table.\n\n"
        )

        guide.insert("end", "SUGGESTED RECITATION EXPLANATION\n\n")
        guide.insert(
            "end",
            "This system uses Apriori association rule mining to discover hidden connections among product categories. "
            "The system treats each customer order as a basket. Then, it finds product categories that are frequently bought together. "
            "If a low-priced product category appears as Item A in a strong rule, it may be considered a possible loss-leader candidate. "
            "The network graph helps visualize these hidden item relationships. This supports shop owners in deciding which items can be promoted "
            "or discounted to encourage the purchase of related products."
        )

        guide.config(state="disabled")


if __name__ == "__main__":
    root = tk.Tk()
    app = AprioriLossLeaderApp(root)
    root.mainloop()