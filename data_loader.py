from pathlib import Path

import pandas as pd


ORDER_COLUMN_CANDIDATES = [
    "order_id", "order id", "orderid", "invoice", "invoice_no", "invoiceno",
    "invoice number", "transaction_id", "transaction id", "transaction",
    "basket_id", "basket id", "receipt_id", "receipt id", "bill_no",
    "bill number", "ticket_id", "sale_id",
]

ITEM_COLUMN_CANDIDATES = [
    "item_name", "item name", "product_category_name_english",
    "product_category_name", "category", "product_name", "product name",
    "product", "item", "description", "sku_name", "article", "department",
    "stockcode", "stock_code",
]

PRODUCT_ID_CANDIDATES = [
    "product_id", "product id", "productid", "sku", "sku_id", "item_id",
    "item id", "stockcode", "stock_code", "barcode", "upc",
]

PRICE_COLUMN_CANDIDATES = [
    "price", "unit_price", "unit price", "unitprice", "item_price",
    "item price", "sales", "sale_price", "amount", "line_total",
    "total_price", "revenue",
]

QUANTITY_COLUMN_CANDIDATES = [
    "quantity", "qty", "units", "unit_count", "count",
]


def _normal_name(name):
    return str(name).strip().lower().replace("-", "_")


def _column_lookup(df):
    return {_normal_name(col): col for col in df.columns}


def _find_column(df, candidates):
    lookup = _column_lookup(df)
    for candidate in candidates:
        key = _normal_name(candidate)
        if key in lookup:
            return lookup[key]
    return None


def _read_csv(path):
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1")


def _standardize_transactions(df, source_name, allow_product_id_as_item=False):
    order_col = _find_column(df, ORDER_COLUMN_CANDIDATES)
    item_col = _find_column(df, ITEM_COLUMN_CANDIDATES)
    if not item_col and allow_product_id_as_item:
        item_col = _find_column(df, PRODUCT_ID_CANDIDATES)
    price_col = _find_column(df, PRICE_COLUMN_CANDIDATES)
    quantity_col = _find_column(df, QUANTITY_COLUMN_CANDIDATES)

    if not order_col or not item_col:
        return None

    out = pd.DataFrame({
        "order_id": df[order_col],
        "item_name": df[item_col],
    })
    out["price"] = pd.to_numeric(df[price_col], errors="coerce") if price_col else 1.0
    out["quantity"] = (
        pd.to_numeric(df[quantity_col], errors="coerce").fillna(1)
        if quantity_col else 1
    )
    out["order_item_id"] = [f"{source_name}:{i}" for i in range(len(out))]
    return _clean_standardized(out)


def _clean_standardized(df):
    df = df.copy()
    df["order_id"] = df["order_id"].astype(str).str.strip()
    df["item_name"] = df["item_name"].astype(str).str.strip()
    df = df.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(1.0)
    df["quantity"] = pd.to_numeric(df.get("quantity", 1), errors="coerce").fillna(1)
    df = df.dropna(subset=["order_id", "item_name"])
    return df[df["item_name"].astype(str).str.len() > 0]


def _load_olist(data_dir):
    oi_path = data_dir / "olist_order_items_dataset.csv"
    p_path = data_dir / "olist_products_dataset.csv"
    t_path = data_dir / "product_category_name_translation.csv"
    if not (oi_path.exists() and p_path.exists() and t_path.exists()):
        return None

    order_items = _read_csv(oi_path)
    products = _read_csv(p_path)
    translation = _read_csv(t_path)

    df = order_items.merge(products, on="product_id", how="left")
    df = df.merge(translation, on="product_category_name", how="left")
    df["item_name"] = df["product_category_name_english"].fillna(df["product_category_name"])
    df["quantity"] = 1
    return _clean_standardized(df)


def _load_merged_product_dataset(csv_paths):
    frames = {path: _read_csv(path) for path in csv_paths}
    for tx_path, tx_df in frames.items():
        order_col = _find_column(tx_df, ORDER_COLUMN_CANDIDATES)
        tx_product_col = _find_column(tx_df, PRODUCT_ID_CANDIDATES)
        if not order_col or not tx_product_col:
            continue

        for product_path, product_df in frames.items():
            if product_path == tx_path:
                continue
            product_col = _find_column(product_df, PRODUCT_ID_CANDIDATES)
            item_col = _find_column(product_df, ITEM_COLUMN_CANDIDATES)
            if not product_col or not item_col:
                continue

            merged = tx_df.merge(
                product_df[[product_col, item_col]].drop_duplicates(product_col),
                left_on=tx_product_col,
                right_on=product_col,
                how="left",
            )
            standardized = _standardize_transactions(merged, tx_path.name)
            if standardized is not None and not standardized.empty:
                return standardized
    return None


def _load_transaction_format(path):
    """Load CSV where each row contains comma-separated items (transaction format)."""
    try:
        transactions = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    items = [item.strip() for item in line.split(',') if item.strip()]
                    if items:
                        for item in items:
                            transactions.append({
                                'order_id': f"transaction_{line_num}",
                                'item_name': item,
                                'price': 1.0,
                                'quantity': 1.0,
                                'order_item_id': f"{path.name}:{line_num}:{item}"
                            })
        return pd.DataFrame(transactions)
    except UnicodeDecodeError:
        transactions = []
        with open(path, 'r', encoding='latin1') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    items = [item.strip() for item in line.split(',') if item.strip()]
                    if items:
                        for item in items:
                            transactions.append({
                                'order_id': f"transaction_{line_num}",
                                'item_name': item,
                                'price': 1.0,
                                'quantity': 1.0,
                                'order_item_id': f"{path.name}:{line_num}:{item}"
                            })
        return pd.DataFrame(transactions)


def load_transaction_data(data_dir):
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data folder does not exist: {data_dir}")

    olist_df = _load_olist(data_dir)
    if olist_df is not None and not olist_df.empty:
        return olist_df

    csv_paths = sorted(data_dir.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in data folder: {data_dir}")

    merged = _load_merged_product_dataset(csv_paths)
    if merged is not None and not merged.empty:
        return merged

    # Try traditional format first
    single_file_matches = []
    for path in csv_paths:
        df = _read_csv(path)
        standardized = _standardize_transactions(
            df,
            path.name,
            allow_product_id_as_item=True,
        )
        if standardized is not None and not standardized.empty:
            single_file_matches.append(standardized)

    if single_file_matches:
        return pd.concat(single_file_matches, ignore_index=True)

    # Try transaction format (each line = one transaction with comma-separated items)
    transaction_matches = []
    for path in csv_paths:
        try:
            transaction_df = _load_transaction_format(path)
            if not transaction_df.empty:
                transaction_matches.append(transaction_df)
        except Exception:
            continue

    if transaction_matches:
        return pd.concat(transaction_matches, ignore_index=True)

    # Clean, helpful error message
    error_msg = (
        "Unable to load data. The system supports these formats:\n"
        "1. Traditional format: CSV with order_id, item_name columns (price/quantity optional)\n"
        "2. Transaction format: CSV where each line contains comma-separated items\n"
        "3. Olist format: Standard Olist dataset files\n\n"
        f"Files found: {[p.name for p in csv_paths]}"
    )
    raise ValueError(error_msg)
