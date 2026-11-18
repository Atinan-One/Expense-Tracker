# Personal Expense Tracker & Spending Forecaster

A Python-based tool that helps users track personal expenses, understand spending patterns, predict next month's expenses, and monitor their budgets.

## Technologies Used
- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

## Features
1. **Data Cleaning:** Cleans expense records, fixes dates and amounts, and removes duplicate or invalid entries.
2. **Automatic Expense Categorization:** Uses Machine Learning to automatically categorize expenses.
3. **Spending Analysis:** Summarizes monthly expenses by category.
4. **Expense Prediction:** Uses Linear Regression to forecast next month's spending.
5. **Budget Monitoring:** Identifies overspending and provides budget suggestions.
6. **Reports:** Generates charts and CSV reports.

## How to Run

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python expense_tracker.py
```

## Sample CSV Format

```csv
date,description,amount,category
2026-09-02,Uber ride,650,Transport
2026-09-03,Grocery store,3000,Food
```

## Project Structure

```text
expense-tracker/
├── expense_tracker.py
├── data/
├── reports/
└── requirements.txt
```

## Future Improvements
- Web-based dashboard
- Receipt scanning
- Income tracking
- Improved expense prediction

## Author
Atinan Sohail