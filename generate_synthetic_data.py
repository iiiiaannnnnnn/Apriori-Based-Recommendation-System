from __future__ import annotations

import csv
import random
from pathlib import Path


def weighted_pick(weights: dict[str, float]) -> str:
    items = list(weights.keys())
    probs = list(weights.values())
    return random.choices(items, weights=probs, k=1)[0]


def build_synthetic_olist_data(out_dir: Path, order_count: int = 1400, seed: int = 42) -> None:
    random.seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    category_to_en = {
        "beleza_saude": "beauty_health",
        "cama_mesa_banho": "bed_bath_table",
        "utilidades_domesticas": "housewares",
        "esporte_lazer": "sports_leisure",
        "informatica_acessorios": "computers_accessories",
        "telefonia": "telephony",
        "automotivo": "automotive",
        "ferramentas_jardim": "garden_tools",
        "pet_shop": "pet_shop",
        "bebes": "baby",
        "alimentos_bebidas": "food_drink",
        "papelaria": "stationery",
        "moveis_decoracao": "furniture_decor",
        "relogios_presentes": "watches_gifts",
        "brinquedos": "toys",
        "livros_interesse_geral": "books_general_interest",
        "fashion_bolsas_e_acessorios": "fashion_bags_accessories",
        "instrumentos_musicais": "musical_instruments",
        "saude_fitness": "health_fitness",
        "eletrodomesticos": "home_appliances",
    }

    cat_price_band = {
        "beleza_saude": (18, 95),
        "cama_mesa_banho": (25, 180),
        "utilidades_domesticas": (15, 130),
        "esporte_lazer": (35, 260),
        "informatica_acessorios": (45, 420),
        "telefonia": (60, 520),
        "automotivo": (28, 340),
        "ferramentas_jardim": (30, 230),
        "pet_shop": (12, 110),
        "bebes": (20, 190),
        "alimentos_bebidas": (8, 70),
        "papelaria": (5, 55),
        "moveis_decoracao": (60, 650),
        "relogios_presentes": (80, 700),
        "brinquedos": (18, 170),
        "livros_interesse_geral": (12, 90),
        "fashion_bolsas_e_acessorios": (22, 210),
        "instrumentos_musicais": (70, 800),
        "saude_fitness": (35, 280),
        "eletrodomesticos": (90, 900),
    }

    core_patterns: list[list[str]] = [
        ["beleza_saude", "cama_mesa_banho"],
        ["esporte_lazer", "saude_fitness"],
        ["informatica_acessorios", "telefonia"],
        ["pet_shop", "alimentos_bebidas"],
        ["bebes", "brinquedos"],
        ["automotivo", "ferramentas_jardim"],
        ["papelaria", "livros_interesse_geral"],
        ["moveis_decoracao", "utilidades_domesticas"],
        ["fashion_bolsas_e_acessorios", "relogios_presentes"],
    ]
    triple_patterns: list[list[str]] = [
        ["esporte_lazer", "saude_fitness", "alimentos_bebidas"],
        ["bebes", "brinquedos", "alimentos_bebidas"],
        ["informatica_acessorios", "telefonia", "papelaria"],
        ["moveis_decoracao", "cama_mesa_banho", "utilidades_domesticas"],
    ]

    categories = list(category_to_en.keys())
    base_weights = {c: 1.0 for c in categories}
    for c in ["cama_mesa_banho", "beleza_saude", "esporte_lazer", "utilidades_domesticas"]:
        base_weights[c] = 1.7
    for c in ["instrumentos_musicais", "eletrodomesticos", "relogios_presentes"]:
        base_weights[c] = 0.5

    products: list[dict[str, str]] = []
    product_ids: dict[str, list[str]] = {}
    prod_seq = 1
    for cat in categories:
        n = 20 if base_weights[cat] > 1 else 12
        product_ids[cat] = []
        lo, hi = cat_price_band[cat]
        for _ in range(n):
            pid = f"syn_prod_{prod_seq:05d}"
            prod_seq += 1
            product_ids[cat].append(pid)
            products.append(
                {
                    "product_id": pid,
                    "product_category_name": cat,
                    "base_price": f"{random.uniform(lo, hi):.2f}",
                }
            )

    order_items_rows: list[dict[str, str]] = []
    for i in range(1, order_count + 1):
        order_id = f"syn_order_{i:06d}"
        line_no = 1

        draw = random.random()
        if draw < 0.46:
            cats = random.choice(core_patterns).copy()
            if random.random() < 0.35:
                cats.append(weighted_pick(base_weights))
        elif draw < 0.68:
            cats = random.choice(triple_patterns).copy()
        else:
            size = random.choices([2, 3, 4, 5], weights=[0.45, 0.3, 0.18, 0.07], k=1)[0]
            cats = []
            while len(cats) < size:
                c = weighted_pick(base_weights)
                if c not in cats:
                    cats.append(c)

        if random.random() < 0.08:
            rare_bundle = ["instrumentos_musicais", "eletrodomesticos", "relogios_presentes"]
            cats = list(dict.fromkeys(cats + random.sample(rare_bundle, k=2)))

        for cat in cats:
            pid = random.choice(product_ids[cat])
            lo, hi = cat_price_band[cat]
            price = round(random.uniform(lo, hi), 2)
            if cat in {"alimentos_bebidas", "papelaria", "pet_shop"} and random.random() < 0.35:
                price = round(price * random.uniform(0.6, 0.85), 2)
            order_items_rows.append(
                {
                    "order_id": order_id,
                    "order_item_id": str(line_no),
                    "product_id": pid,
                    "price": f"{price:.2f}",
                }
            )
            line_no += 1

    with (out_dir / "olist_products_dataset.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["product_id", "product_category_name", "base_price"])
        writer.writeheader()
        writer.writerows(products)

    with (out_dir / "olist_order_items_dataset.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["order_id", "order_item_id", "product_id", "price"])
        writer.writeheader()
        writer.writerows(order_items_rows)

    with (out_dir / "product_category_name_translation.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["product_category_name", "product_category_name_english"]
        )
        writer.writeheader()
        for pt, en in category_to_en.items():
            writer.writerow(
                {
                    "product_category_name": pt,
                    "product_category_name_english": en,
                }
            )

    print(f"Wrote synthetic dataset to: {out_dir}")
    print(f"Orders: {order_count}")
    print(f"Order line items: {len(order_items_rows)}")
    print(f"Products: {len(products)}")
    print("Files:")
    print("- olist_order_items_dataset.csv")
    print("- olist_products_dataset.csv")
    print("- product_category_name_translation.csv")


if __name__ == "__main__":
    root = Path(__file__).parent
    target = root / "data" / "synthetic_olist_v1"
    build_synthetic_olist_data(target)
