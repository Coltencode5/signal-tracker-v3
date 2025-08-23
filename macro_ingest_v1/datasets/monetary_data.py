"""
Monetary datasets - M2 money supply and derived metrics
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

class MonetaryDataset:
    """Dataset B2: M2 money supply with derived metrics"""
    
    def __init__(self, country: str, start_date: str, end_date: str):
        self.country = country
        self.start_date = start_date
        self.end_date = end_date
        self.dataset_code = "B2"
        self.raw_data = None
        self.processed_data = None
        
        # RIC mapping by country for M2 money supply data
        self.ric_mapping = {
            "Turkey": [
                "aTRM2",        # Primary: Turkey M2, Monthly, current prices, NSA
                "TRM2=ECI"      # Fallback: Alternative ECI series if available
            ],
            "Argentina": [
                "aARM2",        # Primary: Argentina M2, Monthly, current prices, NSA
                "ARM2=ECI"      # Fallback: Alternative ECI series if available
            ]
        }
        self.working_ric = None
        self.data_source = None  # "historical" or "snapshot"
        self.fallback_used = False
        
        # Local history file path
        self.history_dir = Path("data_history")
        self.history_dir.mkdir(exist_ok=True)
        self.history_file = self.history_dir / f"monetary_history_{country.lower()}.json"
        
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
                # Only save M2_LEVEL_TRY records to history (not derived metrics)
                if row['metric'] == 'M2_LEVEL_TRY':
                    history_data.append({
                        'date': row['date'].strftime('%Y-%m-%d'),
                        'value': float(row['value']),
                        'ric': row.get('ric', self.working_ric),
                        'metric': row['metric']
                    })
            
            with open(self.history_file, 'w') as f:
                json.dump(history_data, f, indent=2)
            
            logger.info(f" Saved {len(history_data)} records to local history")
            
        except Exception as e:
            logger.error(f"❌ Failed to save local history: {e}")
    
    def _try_ric_historical(self, ric: str) -> Optional[pd.DataFrame]:
        """Try to fetch historical data for a specific RIC"""
        try:
            logger.info(f"🔄 Trying historical data for RIC: {ric}")
            
            # Parse dates - fetch last 10 years as specified in requirements
            end_dt = datetime.strptime(self.end_date, "%Y-%m-%d")
            start_dt = end_dt - timedelta(days=3650)  # 10 years
            
            # Create historical pricing definition - using the same API pattern as working datasets
            monetary_def = rd.content.historical_pricing.summaries.Definition(
                universe=ric,
                start=start_dt.strftime("%Y-%m-%d"),
                end=end_dt.strftime("%Y-%m-%d"),
                interval="P1M"  # Monthly data
            )
            
            # Fetch the data
            response = monetary_def.get_data()
            
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
                        logger.warning(f"⚠️ No VALUE or INDEX_VALUE field found for RIC: {ric}")
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
                        monetary_data = df[[date_col, value_col]].copy()
                    else:
                        # Use index as date
                        monetary_data = df[[value_col]].copy()
                        monetary_data.index.name = 'Date'
                    
                    monetary_data = monetary_data.dropna()
                    
                    if not monetary_data.empty:
                        logger.info(f"✅ Historical data successful for RIC: {ric} ({len(monetary_data)} records)")
                        return monetary_data
            
            logger.warning(f"⚠️ Historical data empty for RIC: {ric}")
            return None
            
        except Exception as e:
            logger.warning(f"⚠️ Historical data failed for RIC: {ric}: {e}")
            return None
    
    def _try_ric_snapshot(self, ric: str) -> Optional[pd.DataFrame]:
        """Try to fetch snapshot data for a specific RIC"""
        try:
            logger.info(f"🔄 Trying snapshot data for RIC: {ric}")
            
            # Create pricing definition for current snapshot
            pricing_def = rd.content.pricing.Definition(
                universe=ric,
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
                        logger.warning(f"⚠️ No VALUE or INDEX_VALUE field found for RIC: {ric}")
                        return None
                    
                    # Create a single-row DataFrame with today's date
                    today = datetime.now().strftime("%Y-%m-%d")
                    snapshot_data = pd.DataFrame({
                        'Date': [today],
                        value_col: [value]
                    }).set_index('Date')
                    
                    logger.info(f"✅ Snapshot data successful for RIC: {ric}")
                    return snapshot_data
            
            logger.warning(f"⚠️ Snapshot data empty for RIC: {ric}")
            return None
            
        except Exception as e:
            logger.warning(f"⚠️ Snapshot data failed for RIC: {ric}: {e}")
            return None
    
    def fetch_data(self) -> bool:
        """Fetch monetary data from Refinitiv Workspace using RIC fallback strategy"""
        try:
            logger.info(f" Starting monetary data fetch for {self.country}")
            
            # Open Refinitiv session - same pattern as working datasets
            rd.open_session()
            logger.info("✅ Refinitiv session opened")
            
            # Get RIC mapping for this country
            country_rics = self.ric_mapping.get(self.country, [])
            if not country_rics:
                logger.error(f"❌ No RIC mapping found for country: {self.country}")
                return False
            
            # Try each RIC in order until one works
            for i, ric in enumerate(country_rics):
                logger.info(f"🔄 Testing RIC: {ric}")
                
                # First try historical data
                historical_data = self._try_ric_historical(ric)
                if historical_data is not None:
                    self.raw_data = historical_data
                    self.working_ric = ric
                    self.data_source = "historical"
                    self.fallback_used = (i > 0)  # True if not the primary RIC
                    logger.info(f"✅ Successfully fetched historical monetary data for {self.country} using RIC: {ric}")
                    return True
                
                # If historical fails, try snapshot
                snapshot_data = self._try_ric_snapshot(ric)
                if snapshot_data is not None:
                    self.raw_data = snapshot_data
                    self.working_ric = ric
                    self.data_source = "snapshot"
                    self.fallback_used = (i > 0)  # True if not the primary RIC
                    logger.info(f"✅ Successfully fetched snapshot monetary data for {self.country} using RIC: {ric}")
                    return True
            
            # If we get here, no RIC worked
            logger.error(f"❌ All RICs failed for {self.country}")
            return False
                
        except Exception as e:
            logger.error(f"❌ Error fetching monetary data for {self.country}: {e}")
            return False
    
    def process_data(self) -> bool:
        """Process raw monetary data and merge with local history"""
        try:
            if self.raw_data is None or self.raw_data.empty:
                logger.error("❌ No raw data available to process")
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
                
                # Get the M2 value
                m2_value = float(row[value_col])
                
                new_records.append({
                    "country": "TUR",  # As specified in requirements
                    "metric": "M2_LEVEL_TRY",
                    "date": date_str,
                    "value": m2_value,
                    "source": "Refinitiv",
                    "ric": self.working_ric,
                    "notes": f"M2 money supply for {self.country} (TRY)" + 
                            (f" - {self.data_source}_only" if self.data_source == "snapshot" else "") +
                            (f" - fallback_ric" if self.fallback_used else "")
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
            
            logger.info(f"✅ Processed monetary data for {self.country}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error processing monetary data for {self.country}: {e}")
            return False
    
    def calculate_derived_metrics(self) -> bool:
        """Calculate year-over-year percentage change"""
        try:
            if self.processed_data is None or self.processed_data.empty:
                return False
            
            df = self.processed_data.copy()
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # Filter to just M2_LEVEL_TRY records for calculations
            m2_df = df[df['metric'] == 'M2_LEVEL_TRY'].copy()
            
            if m2_df.empty:
                logger.warning("⚠️ No M2_LEVEL_TRY records found for derived metrics")
                return False
            
            # Calculate year-over-year percentage change (12-month lag)
            m2_df['M2_YOY_PCT'] = m2_df['value'].pct_change(12) * 100
            
            # Add derived metrics as new records
            derived_records = []
            
            for _, row in m2_df.iterrows():
                if pd.notna(row['M2_YOY_PCT']):
                    derived_records.append({
                        "country": "TUR",  # As specified in requirements
                        "metric": "M2_YOY_PCT",
                        "date": row['date'].strftime("%Y-%m-%d"),
                        "value": float(row['M2_YOY_PCT']),
                        "source": "Refinitiv",
                        "ric": self.working_ric,
                        "notes": "M2 year-over-year percentage change (%)"
                    })
            
            # Combine original and derived records
            combined_df = pd.concat([
                self.processed_data,
                pd.DataFrame(derived_records)
            ], ignore_index=True)
            
            self.processed_data = combined_df
            logger.info(f"✅ Calculated derived metrics for {self.country} monetary data")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error calculating derived metrics for {self.country} monetary data: {e}")
            return False
    
    def run_full_pipeline(self) -> bool:
        """Run the complete data pipeline"""
        try:
            logger.info(f" Starting monetary pipeline for {self.country}")
            
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
            logger.info(f"✅ Monetary pipeline completed for {self.country}")
            logger.info(f"📊 Used RIC: {self.working_ric}")
            logger.info(f" Data source: {self.data_source}")
            logger.info(f"🔄 Fallback used: {self.fallback_used}")
            logger.info(f" Records: {len(self.processed_data)}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Monetary pipeline failed for {self.country}: {e}")
            return False
    
    def get_data(self):
        """Get the processed data"""
        return self.processed_data