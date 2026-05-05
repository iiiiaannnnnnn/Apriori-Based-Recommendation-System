# Apriori-Based Recommendation System

Interactive market basket analysis web application for e-commerce data using the Apriori algorithm. Features AI-powered insights, loss leader detection, and comprehensive visualizations.

## Features

- 🛒 **Market Basket Analysis** — Apriori algorithm for finding frequent itemsets and association rules
- 🤖 **AI Insights** — Google Gemini integration for intelligent pattern interpretation
- 💸 **Loss Leader Detection** — Automatic identification of low-margin, high-association products
- 📊 **6 Interactive Visualizations**:
  - Frequent Itemsets Bar Chart
  - Support-Confidence Scatter Plot
  - Network Graph (force-directed)
  - Heatmap (category associations)
  - Bubble Plot (multi-metric)
  - Parallel Coordinates
- 💾 **Smart Caching** — AI outputs cached in localStorage for presentations
- ⚙️ **Professional UI** — Settings modal, tab groups, folder browser

## Dataset

**Olist Brazilian E-Commerce** (Kaggle)
- ~100k orders
- ~74 product categories
- Structured tabular data

## Tech Stack

**Backend:** Python 3.x, Flask 3.1.3, mlxtend, pandas, numpy  
**AI:** Google Generative AI SDK (Gemini 1.5 Flash)  
**Visualization:** Plotly.js  
**Frontend:** Vanilla HTML/CSS/JavaScript  

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/iiiiaannnnnnn/Apriori-Based-Recommendation-System.git
   cd Apriori-Based-Recommendation-System
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv .venv
   .venv\Scripts\activate    # Windows
   source .venv/bin/activate # macOS/Linux
   ```

   Python 3.12 is recommended on Windows for the pinned pandas and matplotlib wheels.

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare a dataset**
   - Olist still works out of the box with `olist_order_items_dataset.csv`, `olist_products_dataset.csv`, and `product_category_name_translation.csv`.
   - Other CSV datasets are also supported. Put them in a folder and choose that folder in the UI.
   - A generic single CSV should contain an order column such as `order_id`, `invoice`, or `transaction_id`, plus an item column such as `item_name`, `product`, `category`, or `description`.
   - Optional columns such as `price`, `unit_price`, `amount`, `quantity`, or `qty` are used for loss-leader scoring when present.
   - If a dataset has transaction lines with product IDs and a separate product catalog CSV, the app attempts to merge them by `product_id`, `sku`, `item_id`, or `stockcode`.

## Usage

1. **Start the server**
   ```bash
   python flask_app.py
   ```

   On Windows, you can also run `run_flask_app.bat` to force the project venv.

2. **Open in browser**
   ```
   http://127.0.0.1:5000
   ```

3. **Configure API keys** (for AI features)
   - Click ⚙ Settings in header
   - Add Google Gemini API keys
   - Select model (gemini-1.5-flash recommended)

4. **Run analysis**
   - Set data folder path (or use default `data/`)
   - Adjust parameters (Min Support, Confidence, Lift, Max Itemset Size)
   - Click "Run Apriori Analysis"

## Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| **Min Support** | Minimum fraction of orders containing itemset | 0.003 | 0.001–1.0 |
| **Min Confidence** | P(B\|A): likelihood of B given A | 0.30 | 0.01–1.0 |
| **Min Lift** | Association strength vs random | 1.20 | ≥ 1.0 |
| **Max Itemset Size** | Maximum items per combo (2=pairs, 3=triples) | 3 | 2–10 |
| **Low Price Percentile** | Threshold for loss leader candidates | 25 | 1–100 |
| **Max Graph Rules** | Top N rules in network graph | 20 | 1+ |

## Results Tabs

- 📋 **Association Rules** — All discovered rules with metrics
- 💸 **Loss Leaders** — Low-price products with strong associations
- 📦 **Frequent Itemsets** — Full table grouped by size (1-item, 2-item, etc.)
- 💡 **Recommendations** — AI-generated business insights

## Team Workflow

**For teammates to contribute:**

1. **Clone the repo**
   ```bash
   git clone https://github.com/iiiiaannnnnnn/Apriori-Based-Recommendation-System.git
   cd Apriori-Based-Recommendation-System
   ```

2. **Pull latest changes** (before starting work)
   ```bash
   git pull origin main
   ```

3. **Make changes and commit**
   ```bash
   git add .
   git commit -m "Describe your changes"
   ```

4. **Push to GitHub**
   ```bash
   git push origin main
   ```

## Project Structure

```
olist_apriori_ui/
├── flask_app.py              # Flask backend + analysis logic
├── templates/
│   └── index.html            # Main UI
├── static/
│   └── style.css             # Stylesheet
├── data/                     # CSV files (not included in repo)
│   ├── olist_order_items_dataset.csv
│   ├── olist_products_dataset.csv
│   └── product_category_name_translation.csv
├── requirements.txt          # Python dependencies
├── run_flask_app.bat         # Windows launcher for the Flask app
├── SYSTEM_GUIDE.txt         # Presentation guide
└── README.md
```

## License

MIT License (or specify your license)

## Contributors

- []

## Acknowledgments

- Dataset: [Olist Brazilian E-Commerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
- Apriori implementation: [mlxtend](https://rasbt.github.io/mlxtend/)
- AI: Google Gemini API
