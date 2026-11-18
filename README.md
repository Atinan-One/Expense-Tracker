# Personal Expense Tracker & Spending Forecaster

A small Python tool that turns a plain list of expenses into a clear picture of
where your money goes, predicts next month's spending, and warns you before you
go over budget. It works on your own data, so it is useful in daily life.

**Stack:** Python · NumPy · Pandas · scikit-learn · Matplotlib

## What it does
1. **Cleans your data** (Pandas): fixes dates and amounts, removes duplicates, bad rows and negative amounts.
2. **Fills in missing categories** (machine learning): learns from your categorised expenses so "Uber ride" becomes *Transport* automatically (TF-IDF + logistic regression).
3. **Summarises spending** by month and category (Pandas).
4. **Forecasts next month** per category (linear regression).
5. **Gives advice:** flags categories that jumped last month and checks your budget, suggesting how much to cut from flexible spending (Food, Shopping, Entertainment).
6. **Saves a report** chart and CSV files in `reports/`.

## Run it
```bash
pip install -r requirements.txt
python expense_tracker.py                              # demo with sample data
python expense_tracker.py --file my_expenses.csv --budget 80000   # your own data
```
Your CSV needs these columns (category may be blank):
```
date,description,amount,category
2026-09-02,Uber ride,650,Transport
2026-09-03,Grocery store,3000,
```

## Example output
```
Category model accuracy on held-out expenses: 94%
Auto-categorised 84 expenses.
Average monthly spending: 85,389
Forecast for 2026-10: 87,968
Budget check: forecast is 7,968 OVER your 80,000 budget.
  Tip: cut Shopping, Entertainment, Food by about 15% to get back on track.
```

## Files
```
expense-tracker/
├── expense_tracker.py   # the whole tool
├── data/expenses.csv    # sample data (created on first run)
├── reports/             # chart + CSV results
└── requirements.txt
```

## Limitations and next steps
- The demo data is **simulated**; accuracy on your own data will differ.
- The forecast is a simple trend line and needs about 6+ months of data.
- Next: a web dashboard (Flask), receipt scanning, and income tracking.
