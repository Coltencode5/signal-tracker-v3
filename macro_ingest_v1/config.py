# Basic configuration for Macro Ingest Module
MODULE_VERSION = "v1.0"
DEFAULT_OUTPUT_FILE = "MacroInputs.xlsx"

# Countries we'll start with
INITIAL_COUNTRIES = ["Turkey", "Argentina"]

# Dataset categories
DATASET_CATEGORIES = {
    "A": "Sovereign Risk & Credit Fragility",
    "B": "Capital Flows & Monetary Expansion", 
    "C": "FX Reserves & External Fragility",
    "D": "Inflation, GDP, and Trade"
}

# Excel sheet names for each dataset
SHEET_MAPPINGS = {
    "A1": "CDS",
    "A2": "RolloverRatings", 
    "B1": "CapitalFlows",
    "B2": "MoneySupplyCB",
    "C1": "FXReserves",
    "D1": "Inflation",
    "D2": "TradeBalance"
}