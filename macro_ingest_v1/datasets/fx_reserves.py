"""
FX Reserves datasets - Central Bank foreign exchange reserves
"""

import pandas as pd
import numpy as np
import logging
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path
import refinitiv.data as rd

logger = logging.getLogger(__name__)

class FXReservesDataset:
    """Dataset C1: FX Reserves with derived metrics"""
    
    def __init__(self, country: str, start_date: str, end_date: str):
        self.country = country
        self.start_date = start_date
        self.end_date = end_date
        self.dataset_code = "C1"
        self.raw_data = None
        self.processed_data = None
        
        # ECI mapping by country for FX Reserves data
        self.eci_mapping = {
            "Turkey": [
                "TRGFXR=ECI",   # Primary: Turkey, Reserves, Central Bank's gross foreign exchange reserves
                "TRGFXR=ECIX",  # Fallback 1: Alternative ECI series
                "aTRFXRES"      # Fallback 2: Alternative series
            ],
            "Argentina": [
                "ARGFXR=ECI",   # Primary: Argentina, Reserves, Central Bank's gross foreign exchange reserves
                "ARGFXR=ECIX",  # Fallback 1: Alternative ECI series
                "aARFXRES"      # Fallback 2: Alternative series
            ]
        }
        self.working_eci = None
        self.data_source = None  # "historical" or "snapshot"
        self.fallback_used = False
        self.unit_info = None  # Track units from the API response
        
        # Local history file path
        self.history_dir = Path("data_history")
        self.history_dir.mkdir(exist_ok=True)
        self.history_file = self.history_dir / f"fx_reserves_history_{country.lower()}.json"
        
    def _load_local_history(self) -> pd.DataFrame:
        """Load existing local history for this country"""
        try:
            if self.history_file.exists():
                with open(self.history_file, 'r') as f:
                    history_data = json.load(f)
                
                # Convert to DataFrame
                df = pd.DataFrame(history_data)
                if not df.empty:
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.sort_values('date').drop_duplicates(subset=['date'], keep='last')
                    logger.info(f"📚 Loaded {len(df)} historical records from local storage")
                    return df
        except Exception as e:
            logger.warning(f"⚠️ Could not load local history: {e}")
        
        return pd.DataFrame()
    
    def _save_local_history(self, df: pd.DataFrame):
        """Save current data to local history"""
        try:
            # Convert DataFrame to JSON-serializable format
            history_data = []
            for _, row in df.iterrows():
                # Only save FX_Reserves_USD records to history (not derived metrics)
                if row['metric'] == 'FX_Reserves_USD':
                    history_data.append({
                        'date': row['date'].strftime('%Y-%m-%d'),
                        'value_usd': float(row['value_usd']),
                        'eci': row.get('eci', self.working_eci),
                        'metric': row['metric']
                    })
            
            with open(self.history_file, 'w') as f:
                json.dump(history_data, f, indent=2)
            
            logger.info(f" Saved {len(history_data)} records to local history")
            
        except Exception as e:
            logger.error(f"❌ Failed to save local history: {e}")
    
    def _try_eci_historical(self, eci: str) -> Optional[pd.DataFrame]:
        """Try to fetch historical data for a specific ECI code"""
        try:
            logger.info(f"🔄 Trying historical data for ECI: {eci}")
            
            # Parse dates
            start_dt = datetime.strptime(self.start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(self.end_date, "%Y-%m-%d")
            
            # Create historical pricing definition - using the same API pattern as inflation
            fx_reserves_def = rd.content.historical_pricing.summaries.Definition(
                universe=eci,
                start=start_dt.strftime("%Y-%m-%d"),
                end=end_dt.strftime("%Y-%m-%d"),
                interval="P1M"  # Monthly data
            )
            
            # Fetch the data
            response = fx_reserves_def.get_data()
            
            if response.data and response.data.df is not None:
                df = response.data.df
                if not df.empty:
                    # Debug: print the actual columns we got
                    logger.info(f"🔍 Response columns: {list(df.columns)}")
                    logger.info(f"🔍 Response shape: {df.shape}")
                    
                    # Check if we have the right field (either VALUE or INDEX_VALUE)
                    if 'VALUE' in df.columns:
                        value_col = 'VALUE'
                    elif 'INDEX_VALUE' in df.columns:
                        value_col = 'INDEX_VALUE'
                    else:
                        logger.warning(f"⚠️ No VALUE or INDEX_VALUE field found for ECI: {eci}")
                        logger.warning(f"Available columns: {list(df.columns)}")
                        return None
                    
                    # Filter to just the value field and date - handle date column properly
                    if 'Date' in df.columns:
                        date_col = 'Date'
                    elif 'date' in df.columns:
                        date_col = 'date'
                    else:
                        # If no date column, use index
                        date_col = None
                    
                    if date_col:
                        fx_reserves_data = df[[date_col, value_col]].copy()
                    else:
                        # Use index as date
                        fx_reserves_data = df[[value_col]].copy()
                        fx_reserves_data.index.name = 'Date'
                    
                    fx_reserves_data = fx_reserves_data.dropna()
                    
                    if not fx_reserves_data.empty:
                        logger.info(f"✅ Historical data successful for ECI: {eci} ({len(fx_reserves_data)} records)")
                        return fx_reserves_data
            
            logger.warning(f"⚠️ Historical data empty for ECI: {eci}")
            return None
            
        except Exception as e:
            logger.warning(f"⚠️ Historical data failed for ECI: {eci}: {e}")
            return None
    
    def _try_eci_snapshot(self, eci: str) -> Optional[pd.DataFrame]:
        """Try to fetch snapshot data for a specific ECI code"""
        try:
            logger.info(f"🔄 Trying snapshot data for ECI: {eci}")
            
            # Create pricing definition for current snapshot
            pricing_def = rd.content.pricing.Definition(
                universe=eci,
                fields=["VALUE", "INDEX_VALUE"]
            )
            
            # Fetch the data
            response = pricing_def.get_data()
            
            if response.data and response.data.df is not None:
                df = response.data.df
                if not df.empty:
                    # Determine which field to use
                    if 'VALUE' in df.columns:
                        value_col = 'VALUE'
                        value = df['VALUE'].iloc[0]
                    elif 'INDEX_VALUE' in df.columns:
                        value_col = 'INDEX_VALUE'
                        value = df['INDEX_VALUE'].iloc[0]
                    else:
                        logger.warning(f"⚠️ No VALUE or INDEX_VALUE field found for ECI: {eci}")
                        return None
                    
                    # Create a single-row DataFrame with today's date
                    today = datetime.now().strftime("%Y-%m-%d")
                    snapshot_data = pd.DataFrame({
                        'Date': [today],
                        value_col: [value]
                    }).set_index('Date')
                    
                    logger.info(f"✅ Snapshot data successful for ECI: {eci}")
                    return snapshot_data
            
            logger.warning(f"⚠️ Snapshot data empty for ECI: {eci}")
            return None
            
        except Exception as e:
            logger.warning(f"⚠️ Snapshot data failed for ECI: {eci}: {e}")
            return None
    
    def _detect_units(self, df: pd.DataFrame) -> str:
        """Detect and log the units from the data"""
        try:
            if df.empty:
                return "unknown"
            
            # Get the value column name
            value_col = 'VALUE' if 'VALUE' in df.columns else 'INDEX_VALUE'
            
            # Sample a few values to estimate units
            sample_values = df[value_col].dropna().head(10)
            if sample_values.empty:
                return "unknown"
            
            max_value = sample_values.max()
            min_value = sample_values.min()
            
            # Estimate units based on typical FX reserves values
            if max_value > 1000000000:  # > 1 billion
                unit_info = "USD (billions)"
                self.unit_info = "billions"
            elif max_value > 1000000:  # > 1 million
                unit_info = "USD (millions)"
                self.unit_info = "millions"
            else:
                unit_info = "USD"
                self.unit_info = "base"
            
            logger.info(f"🔍 Detected units: {unit_info} (max value: {max_value:,.0f})")
            return unit_info
            
        except Exception as e:
            logger.warning(f"⚠️ Could not detect units: {e}")
            return "unknown"
    
    def fetch_data(self) -> bool:
        """Fetch FX reserves data from Refinitiv Workspace using ECI fallback strategy"""
        try:
            logger.info(f" Starting FX reserves data fetch for {self.country}")
            
            # Open Refinitiv session - same pattern as working datasets
            rd.open_session()
            logger.info("✅ Refinitiv session opened")
            
            # Get ECI mapping for this country
            country_eci_codes = self.eci_mapping.get(self.country, [])
            if not country_eci_codes:
                logger.error(f"❌ No ECI mapping found for country: {self.country}")
                return False
            
            # Try each ECI code in order until one works
            for i, eci in enumerate(country_eci_codes):
                logger.info(f"🔄 Testing ECI: {eci}")
                
                # First try historical data
                historical_data = self._try_eci_historical(eci)
                if historical_data is not None:
                    self.raw_data = historical_data
                    self.working_eci = eci
                    self.data_source = "historical"
                    self.fallback_used = (i > 0)  # True if not the primary ECI
                    
                    # Detect units from the data
                    self._detect_units(historical_data)
                    
                    logger.info(f"✅ Successfully fetched historical FX reserves data for {self.country} using ECI: {eci}")
                    return True
                
                # If historical fails, try snapshot
                snapshot_data = self._try_eci_snapshot(eci)
                if snapshot_data is not None:
                    self.raw_data = snapshot_data
                    self.working_eci = eci
                    self.data_source = "snapshot"
                    self.fallback_used = (i > 0)  # True if not the primary ECI
                    
                    # Detect units from the data
                    self._detect_units(snapshot_data)
                    
                    logger.info(f"✅ Successfully fetched snapshot FX reserves data for {self.country} using ECI: {eci}")
                    return True
            
            # If we get here, no ECI worked
            logger.error(f"❌ All ECI codes failed for {self.country}")
            return False
                
        except Exception as e:
            logger.error(f"❌ Error fetching FX reserves data for {self.country}: {e}")
            return False
    
    def process_data(self) -> bool:
        """Process raw FX reserves data and merge with local history"""
        try:
            if self.raw_data is None or self.raw_data.empty:
                return False
            
            # Load existing local history
            local_history = self._load_local_history()
            
            # Process new data
            new_records = []
            
            # Get the value column name
            value_col = 'VALUE' if 'VALUE' in self.raw_data.columns else 'INDEX_VALUE'
            
            for date, row in self.raw_data.iterrows():
                # Convert date index to string if it's a datetime
                if isinstance(date, datetime):
                    date_str = date.strftime("%Y-%m-%d")
                else:
                    date_str = str(date)
                
                # Get the FX reserves value
                fx_value = float(row[value_col])
                
                # Create unit notes
                unit_notes = ""
                if self.unit_info == "billions":
                    unit_notes = " - values in USD billions"
                elif self.unit_info == "millions":
                    unit_notes = " - values in USD millions"
                
                new_records.append({
                    "country": self.country,
                    "metric": "FX_Reserves_USD",
                    "date": date_str,
                    "value_usd": fx_value,
                    "source": "Refinitiv",
                    "eci": self.working_eci,
                    "notes": f"FX Reserves for {self.country}" + 
                            unit_notes +
                            (f" - {self.data_source}_only" if self.data_source == "snapshot" else "") +
                            (f" - fallback_eci" if self.fallback_used else ""),
                    "frequency": "monthly"
                })
            
            new_df = pd.DataFrame(new_records)
            new_df['date'] = pd.to_datetime(new_df['date'])
            
            # Merge with local history
            if not local_history.empty:
                # Combine and remove duplicates (keep newest)
                combined_df = pd.concat([local_history, new_df], ignore_index=True)
                combined_df = combined_df.drop_duplicates(subset=['date'], keep='last')
                combined_df = combined_df.sort_values('date')
                
                logger.info(f" Merged {len(new_df)} new records with {len(local_history)} existing records")
                logger.info(f"📈 Total records available: {len(combined_df)}")
                
                self.processed_data = combined_df
            else:
                self.processed_data = new_df
                logger.info(f"📊 First run: {len(new_df)} new records")
            
            # Save updated history
            self._save_local_history(self.processed_data)
            
            logger.info(f"✅ Processed FX reserves data for {self.country}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error processing FX reserves data for {self.country}: {e}")
            return False
    
    def calculate_derived_metrics(self) -> bool:
        """Calculate month-over-month change and placeholder coverage ratio"""
        try:
            if self.processed_data is None or self.processed_data.empty:
                return False
            
            df = self.processed_data.copy()
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # Filter to just FX_Reserves_USD records for calculations
            fx_df = df[df['metric'] == 'FX_Reserves_USD'].copy()
            
            if fx_df.empty:
                logger.warning("⚠️ No FX_Reserves_USD records found for derived metrics")
                return False
            
            # Calculate month-over-month percentage change
            fx_df['change_1m_pct'] = fx_df['value_usd'].pct_change(1) * 100
            
            # Add derived metrics as new records
            derived_records = []
            
            for _, row in fx_df.iterrows():
                if pd.notna(row['change_1m_pct']):
                    derived_records.append({
                        "country": self.country,
                        "metric": "change_1m_pct",
                        "date": row['date'].strftime("%Y-%m-%d"),
                        "value_usd": float(row['change_1m_pct']),
                        "source": "Refinitiv",
                        "eci": self.working_eci,
                        "notes": "Month-over-month % change in FX reserves",
                        "frequency": "monthly"
                    })
                
                # Add placeholder coverage ratio record
                derived_records.append({
                    "country": self.country,
                    "metric": "coverage_ratio",
                    "date": row['date'].strftime("%Y-%m-%d"),
                    "value_usd": np.nan,  # Placeholder - will be computed when imports series is added
                    "source": "Refinitiv",
                    "eci": self.working_eci,
                    "notes": "needs_imports_series",
                    "frequency": "monthly"
                })
            
            # Combine original and derived records
            combined_df = pd.concat([
                self.processed_data,
                pd.DataFrame(derived_records)
            ], ignore_index=True)
            
            self.processed_data = combined_df
            logger.info(f"✅ Calculated derived metrics for {self.country} FX reserves data")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error calculating derived metrics for {self.country} FX reserves: {e}")
            return False
    
    def run_full_pipeline(self) -> bool:
        """Run the complete data pipeline"""
        try:
            logger.info(f" Starting FX reserves pipeline for {self.country}")
            
            # Step 1: Fetch data
            if not self.fetch_data():
                return False
            
            # Step 2: Process data
            if not self.process_data():
                return False
            
            # Step 3: Calculate derived metrics
            if not self.calculate_derived_metrics():
                return False
            
            # Log final status
            logger.info(f"✅ FX reserves pipeline completed for {self.country}")
            logger.info(f"📊 Used ECI: {self.working_eci}")
            logger.info(f" Data source: {self.data_source}")
            logger.info(f"🔄 Fallback used: {self.fallback_used}")
            logger.info(f" Units: {self.unit_info}")
            logger.info(f" Records: {len(self.processed_data)}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ FX reserves pipeline failed for {self.country}: {e}")
            return False
    
    def get_data(self):
        """Get the processed data"""
        return self.processed_data