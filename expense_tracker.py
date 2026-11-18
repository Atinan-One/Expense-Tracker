"""Personal Expense Tracker & Spending Forecaster.

What it does with your expenses CSV (columns: date, description, amount, category):
  1. Cleans the data                         (Pandas)
  2. Auto-fills missing categories from the
     description, e.g. "Uber ride" -> Transport   (ML: text classification)
  3. Summarises spending by month and category (Pandas)
  4. Forecasts next month's spending           (ML: linear regression)
  5. Warns about overspending and checks your budget, then saves a report chart

Usage:
    python expense_tracker.py                       # runs on built-in sample data
    python expense_tracker.py --file my_expenses.csv --budget 80000
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

DISCRETIONARY = ["Shopping", "Entertainment", "Food"]  # easiest places to save money

SAMPLE_ITEMS = {  # category -> (descriptions, typical amount in PKR)
    "Food": (["Foodpanda dinner", "KFC lunch", "Grocery store", "Tea and snacks", "Restaurant dinner", "Bakery"], 1100),
    "Transport": (["Uber ride", "Petrol", "Careem trip", "Bus fare", "Rickshaw", "Car service"], 800),
    "Shopping": (["Clothes outlet", "Daraz order", "New shoes", "Mobile accessory", "Gift for friend"], 3200),
    "Health": (["Pharmacy", "Doctor visit", "Lab test", "Gym fee"], 1800),
    "Entertainment": (["Netflix", "Cinema ticket", "Game purchase", "Spotify", "Picnic"], 1500),
}
MONTHLY_BILLS = {"Electricity bill": 9000, "Internet bill": 3500, "Mobile package": 1200, "Gas bill": 2500}


# ------------------------------------------------------------- sample data (NumPy)
def make_sample(path, seed=3):
    rng = np.random.default_rng(seed)
    rows = []
    for day in pd.date_range("2025-10-01", "2026-09-30"):
        months_in = (day.year - 2025) * 12 + day.month - 10
        for _ in range(rng.poisson(1.3)):
            cat = rng.choice(list(SAMPLE_ITEMS), p=[0.40, 0.25, 0.12, 0.10, 0.13])
            names, typical = SAMPLE_ITEMS[cat]
            growth = 1 + 0.025 * months_in if cat == "Food" else 1   # food creeps up over time
            desc = rng.choice(names)
            desc = f"{desc} {rng.choice(['', '', 'paid', 'cash', 'card', 'online', 'today'])}".strip()
            if rng.random() < 0.15:                      # people write messy notes
                desc = rng.choice(["misc", "payment", "other", "shopping trip", "bill", desc.split()[0].lower()])
            rows.append((day, desc, round(rng.lognormal(np.log(typical * growth), 0.5), -1), cat))
        if day.day == 5:
            for name, amount in MONTHLY_BILLS.items():
                rows.append((day, name, round(amount * rng.uniform(0.9, 1.15), -1), "Bills"))
    df = pd.DataFrame(rows, columns=["date", "description", "amount", "category"])
    df.loc[rng.random(len(df)) < 0.15, "category"] = np.nan   # 15% uncategorised
    df.to_csv(path, index=False)


# ------------------------------------------------------------------ cleaning (Pandas)
def load_and_clean(path):
    df = pd.read_csv(path)
    n0 = len(df)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["description"] = df["description"].astype(str).str.strip()
    df["category"] = df["category"].astype("string").str.strip().str.title()
    df = df.dropna(subset=["date", "amount"]).query("amount > 0").drop_duplicates()
    print(f"Loaded {n0} rows, kept {len(df)} valid expenses.")
    return df.reset_index(drop=True)


# ------------------------------------------------- ML 1: auto-categorise descriptions
def fill_categories(df):
    known, unknown = df[df.category.notna()], df[df.category.isna()]
    if unknown.empty or known.category.nunique() < 2:
        return df.assign(auto_categorised=False)
    model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), LogisticRegression(max_iter=1000))
    Xtr, Xte, ytr, yte = train_test_split(known.description, known.category, test_size=0.2, random_state=0)
    acc = model.fit(Xtr, ytr).score(Xte, yte)
    print(f"Category model accuracy on held-out expenses: {acc:.0%}")
    model.fit(known.description, known.category)          # refit on everything we know
    df = df.copy()
    df["auto_categorised"] = df.category.isna()
    df.loc[unknown.index, "category"] = model.predict(unknown.description)
    print(f"Auto-categorised {len(unknown)} expenses.")
    return df


# ------------------------------------------------------- summary + ML 2: forecasting
def monthly_table(df):
    df = df.assign(month=df.date.dt.to_period("M"))
    return df.pivot_table(index="month", columns="category", values="amount", aggfunc="sum", fill_value=0)


def forecast_next(table):
    t = np.arange(len(table)).reshape(-1, 1)
    preds = {}
    for cat in table.columns:
        lr = LinearRegression().fit(t, table[cat].values)
        preds[cat] = max(0.0, float(lr.predict([[len(table)]])[0]))   # spending can't be negative
    return pd.Series(preds)


def overspending(table):
    """Categories where last month was 20%+ above the average of the 3 months before."""
    if len(table) < 4:
        return {}
    last, base = table.iloc[-1], table.iloc[-4:-1].mean()
    return {c: (last[c] / base[c] - 1) for c in table.columns if base[c] > 0 and last[c] > 1.2 * base[c]}


def make_chart(table, forecast, path):
    nxt = table.index[-1] + 1
    full = pd.concat([table, pd.DataFrame([forecast], index=[nxt])]).fillna(0)
    full.index = [f"{m} (forecast)" if m == nxt else str(m) for m in full.index]   # plain text labels
    ax = full.plot(kind="bar", stacked=True, figsize=(11, 5), colormap="Set2", width=0.8)
    ax.axvline(len(table) - 0.5, color="grey", linestyle="--")
    ax.set_title("Monthly spending by category"); ax.set_xlabel(""); ax.set_ylabel("PKR")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.2)
    ax.legend(loc="upper left", ncol=3, fontsize=8)
    plt.xticks(rotation=45); plt.savefig(path, dpi=120, bbox_inches="tight"); plt.close()


def main(args):
    path = Path(args.file)
    if not path.exists():
        print(f"No file found, creating sample data at {path}")
        path.parent.mkdir(exist_ok=True)
        make_sample(path)
    df = fill_categories(load_and_clean(path))
    table = monthly_table(df)
    forecast = forecast_next(table)
    nxt = table.index[-1] + 1

    if len(table) < 6:
        print(f"\nNote: only {len(table)} month(s) of data. Forecasts need about 6+ months to be reliable.")
    print(f"\n=== Spending report ({table.index[0]} to {table.index[-1]}) ===")
    print(f"Average monthly spending: {table.sum(axis=1).mean():,.0f}")
    print("\nLast month by category:")
    print(table.iloc[-1].sort_values(ascending=False).round(0).to_string())
    print(f"\nForecast for {nxt}: {forecast.sum():,.0f}")
    print(forecast.sort_values(ascending=False).round(0).to_string())

    flags = overspending(table)
    if flags:
        print("\nWatch out: spending jumped last month in:")
        for cat, pct in sorted(flags.items(), key=lambda x: -x[1]):
            print(f"  - {cat}: +{pct:.0%} vs the previous 3-month average")

    if args.budget:
        gap = forecast.sum() - args.budget
        if gap <= 0:
            print(f"\nBudget check: forecast is {-gap:,.0f} UNDER your {args.budget:,.0f} budget. Nice.")
        else:
            flex = forecast[[c for c in DISCRETIONARY if c in forecast]].sum()
            print(f"\nBudget check: forecast is {gap:,.0f} OVER your {args.budget:,.0f} budget.")
            if flex > 0 and gap / flex <= 0.5:
                print(f"  Tip: cut {', '.join(DISCRETIONARY)} by about {gap / flex:.0%} to get back on track.")
            else:
                print("  Cutting flexible spending alone is not realistic; review your fixed bills or raise the budget.")

    out = Path("reports"); out.mkdir(exist_ok=True)
    df.to_csv(out / "categorized_expenses.csv", index=False)
    table.round(0).to_csv(out / "monthly_summary.csv")
    make_chart(table, forecast, out / "spending_report.png")
    print("\nSaved reports/categorized_expenses.csv, monthly_summary.csv and spending_report.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Personal expense tracker and forecaster")
    ap.add_argument("--file", default="data/expenses.csv", help="CSV with date, description, amount, category")
    ap.add_argument("--budget", type=float, help="your monthly budget (optional)")
    main(ap.parse_args())
