import pandas as pd
import os
from datetime import datetime

# === CONSTANTS ===
EXCEL_PATH = "signals.xlsx"
IMPACT_MAPPING_SHEET = "ImpactMapping_v1"
SIGNAL_FEED_SHEET = "Primary"
OUTPUT_SHEET = "ImpactResults_v1"
MIN_REGION_CONFIDENCE = 0.5
SCHEMA_VERSION = "v1.0"

# === MAPPING COLUMNS ===
MAPPING_COLUMNS = {
    'EVENT_TYPE': 'EventType',
    'SYNONYMS': 'Synonyms',
    'PRIMARY_SECTORS': 'PrimarySectors',
    'EXAMPLE_TICKERS': 'ExampleTickers',
    'EXPECTED_IMPACT': 'ExpectedImpact'
}

# === OUTPUT COLUMNS ===
OUTPUT_COLUMNS = {
    'MAPPED_PRIMARY_SECTORS': 'MappedPrimarySectors',
    'MAPPED_EXAMPLE_TICKERS': 'MappedExampleTickers',
    'MAPPED_EXPECTED_IMPACT': 'MappedExpectedImpact',
    'MAPPING_FOUND': 'MappingFound'
}

def load_impact_mapping():
    """Load the impact mapping table from Excel."""
    try:
        impact_mapping_df = pd.read_excel(EXCEL_PATH, sheet_name=IMPACT_MAPPING_SHEET, engine="openpyxl")
        impact_mapping_df.dropna(how="all", inplace=True)
        print("✅ Impact Mapping Table Loaded:")
        print(impact_mapping_df.head())
        return impact_mapping_df
    except Exception as e:
        print(f"❌ Error loading impact mapping: {e}")
        return pd.DataFrame()

def load_signal_feed():
    """Load signals from the signal tracker."""
    try:
        signal_df = pd.read_excel(EXCEL_PATH, sheet_name=SIGNAL_FEED_SHEET, engine="openpyxl")
        signal_df.dropna(how="all", inplace=True)
        print("✅ Signal Feed Loaded:")
        print(signal_df.head())
        return signal_df
    except Exception as e:
        print(f"❌ Error loading signal feed: {e}")
        return pd.DataFrame()

def get_event_synonyms(map_row):
    """Extract and normalize event synonyms from mapping row."""
    primary_event = str(map_row.get(MAPPING_COLUMNS['EVENT_TYPE'], '')).strip().lower()
    synonyms_text = str(map_row.get(MAPPING_COLUMNS['SYNONYMS'], '')).lower()
    synonyms = [s.strip() for s in synonyms_text.split(',') if s.strip()]
    return [primary_event] + synonyms

def map_signal_to_impact(signal_row, impact_mapping_df):
    """Map a single signal to its impact based on event type and region."""
    event_type = str(signal_row.get('classified_event', '')).strip().lower()
    region = str(signal_row.get('region', '')).strip()
    region_confidence = float(signal_row.get('region_confidence', 0.0))
    
    # Skip low-confidence signals
    if region_confidence < MIN_REGION_CONFIDENCE:
        return {
            'mapped_sectors': 'Low confidence - skipped',
            'mapped_tickers': 'Low confidence - skipped',
            'mapped_impact': 'Low confidence - skipped',
            'mapping_found': False,
            'low_confidence': True
        }
    
    # Split region into individual countries
    countries = [c.strip() for c in region.split(",") if c.strip()]
    
    # Search for mapping
    for country in countries:
        for _, map_row in impact_mapping_df.iterrows():
            possible_events = get_event_synonyms(map_row)
            
            if event_type in possible_events:
                mapped_sectors = f"{map_row.get(MAPPING_COLUMNS['PRIMARY_SECTORS'], '')} (Country: {country})"
                
                return {
                    'mapped_sectors': mapped_sectors,
                    'mapped_tickers': map_row.get(MAPPING_COLUMNS['EXAMPLE_TICKERS'], ''),
                    'mapped_impact': map_row.get(MAPPING_COLUMNS['EXPECTED_IMPACT'], ''),
                    'mapping_found': True,
                    'low_confidence': False
                }
    
    # No mapping found
    return {
        'mapped_sectors': 'No mapping found',
        'mapped_tickers': 'No mapping found',
        'mapped_impact': 'No mapping found',
        'mapping_found': False,
        'low_confidence': False
    }

def process_signals(signal_df, impact_mapping_df):
    """Process all signals and map them to impacts."""
    output_df = signal_df.copy()
    low_confidence_log = []
    
    # Initialize output columns
    for col in OUTPUT_COLUMNS.values():
        output_df[col] = ''
    output_df[OUTPUT_COLUMNS['MAPPING_FOUND']] = False
    
    # Process each signal
    for idx, row in output_df.iterrows():
        mapping_result = map_signal_to_impact(row, impact_mapping_df)
        
        if mapping_result['low_confidence']:
            low_confidence_log.append({
                "timestamp": row.get('timestamp'),
                "classified_event": row.get('classified_event'),
                "region": row.get('region'),
                "region_confidence": row.get('region_confidence')
            })
            continue
        
        # Update output dataframe
        output_df.at[idx, OUTPUT_COLUMNS['MAPPED_PRIMARY_SECTORS']] = mapping_result['mapped_sectors']
        output_df.at[idx, OUTPUT_COLUMNS['MAPPED_EXAMPLE_TICKERS']] = mapping_result['mapped_tickers']
        output_df.at[idx, OUTPUT_COLUMNS['MAPPED_EXPECTED_IMPACT']] = mapping_result['mapped_impact']
        output_df.at[idx, OUTPUT_COLUMNS['MAPPING_FOUND']] = mapping_result['mapping_found']
    
    print("✅ Event-to-Impact Mapping with region splitting and confidence filtering complete:")
    print(output_df.head())
    
    # Display low-confidence log
    if low_confidence_log:
        low_conf_df = pd.DataFrame(low_confidence_log)
        print("⚠️ Low-confidence signals logged:")
        print(low_conf_df)
    else:
        print("✅ No low-confidence signals detected.")
    
    return output_df

def export_results(output_df):
    """Export results to Excel sheet."""
    try:
        with pd.ExcelWriter(EXCEL_PATH, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
            output_df.to_excel(writer, sheet_name=OUTPUT_SHEET, index=False)
        print(f"✅ Mapped results written to '{OUTPUT_SHEET}' tab in Excel.")
    except Exception as e:
        print(f"❌ Error exporting results: {e}")

def main():
    """Main execution function."""
    print("🔄 Starting Industry Impact Engine...")
    
    # Load data
    impact_mapping_df = load_impact_mapping()
    if impact_mapping_df.empty:
        print("❌ Failed to load impact mapping. Exiting.")
        return
    
    signal_df = load_signal_feed()
    if signal_df.empty:
        print("❌ Failed to load signal feed. Exiting.")
        return
    
    # Process signals
    output_df = process_signals(signal_df, impact_mapping_df)
    
    # Export results
    export_results(output_df)
    
    print("✅ Industry Impact Engine completed successfully.")

if __name__ == "__main__":
    main()