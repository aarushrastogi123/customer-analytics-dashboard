# data/

This directory stores local data files. **None of these files are committed to Git.**

## Dataset: UCI Online Retail II

The dashboard uses the **UCI Online Retail II** dataset.

| Field | Details |
|-------|---------|
| **Source** | https://archive.ics.uci.edu/dataset/502/online+retail+ii |
| **Licence** | CC BY 4.0 (free to redistribute with attribution) |
| **Format** | Excel (.xlsx), two sheets: 2009-2010 and 2010-2011 |
| **Size** | ~6 MB zipped, ~45 MB unzipped |
| **Rows** | ~1 million (we use the 2010-2011 sheet, ~541 k rows) |

## Automatic Download

When you run `streamlit run app.py`, the app will automatically:
1. Download `online_retail_II.zip` from the UCI repository
2. Extract `online_retail_II.xlsx` into this directory
3. Create `customer_analytics.db` (SQLite database of cleaned data)

## Manual Download

If the automatic download fails (e.g. network restriction):
1. Visit https://archive.ics.uci.edu/dataset/502/online+retail+ii
2. Download the zip file
3. Extract and place `online_retail_II.xlsx` in this `data/` folder
4. Re-run `streamlit run app.py`

## Files in this directory (not committed)

| File | Description |
|------|------------|
| `online_retail_II.xlsx` | Raw dataset |
| `customer_analytics.db` | SQLite database (auto-generated) |
