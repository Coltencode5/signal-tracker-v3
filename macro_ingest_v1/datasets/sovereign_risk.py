"""
Sovereign Risk datasets - CDS spreads and credit ratings
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

class SovereignCDSDataset:
    """Dataset A1: Sovereign CDS 5Y spreads with derived metrics"""
    
    def __init__(self, country: str, start_date: str, end_date: str):
        self.country = country
        self.start_date = start_date
        self.end_date = end_date
        self.dataset_code = "A1"
        self.raw_data = None
        self.processed_data = None
        
        # RIC mapping by country for 5Y USD sovereign CDS
        self.ric_mapping = {
            "Turkey": [
                "TRGV5YUSAC=R",   # Primary: Turkey, 5Y, USD, Senior Unsecured, Refinitiv EOD
                "TRGV5YUSAC=MG",  # Fallback 1
                "TRGV5YUSAC=MP",  # Fallback 2
                "TRGV5YUSAC=MT",  # Fallback 3
                "TRGV5YUSAC=RR"   # Fallback 4
            ],
            "Argentina": [
                "ARGV5YUSAC=R",   # Primary: Argentina, 5Y, USD, Senior Unsecured, Refinitiv EOD
                "ARGV5YUSAC=MG",  # Fallback 1
                "ARGV5YUSAC=MP",  # Fallback 2
                "ARGV5YUSAC=MT",  # Fallback 3
                "ARGV5YUSAC=RR"   # Fallback 4
            ]
        }
        self.working_ric = None
        self.data_source = None  # "historical" or "snapshot"
        
        # Local history file path
        self.history_dir = Path("macro_ingest_v1/data_history")
        self.history_dir.mkdir(exist_ok=True)
        self.history_file = self.history_dir / f"cds_history_{country.lower()}.json"
        
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
                history_data.append({
                    'date': row['date'].strftime('%Y-%m-%d'),
                    'mid_spread_bp': float(row['mid_spread_bp']),
                    'ric': row['ric']
                })
            
            with open(self.history_file, 'w') as f:
                json.dump(history_data, f, indent=2)
            
            logger.info(f" Saved {len(history_data)} records to local history")
            
        except Exception as e:
            logger.error(f"❌ Failed to save local history: {e}")
    
    def _try_ric_historical(self, ric: str) -> Optional[pd.DataFrame]:
        """Try to fetch historical data for a specific RIC (90 days only)"""
        try:
            logger.info(f"🔄 Trying historical data for RIC: {ric}")
            
            # Fetch only last 90 days
            end_date = datetime.now()
            start_date = end_date - timedelta(days=90)
            
            # Create historical pricing definition
            cds_def = rd.content.historical_pricing.summaries.Definition(
                universe=ric,
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                interval="P1D"  # Daily data
            )
            
            # Fetch the data
            response = cds_def.get_data()
            
            if response.data and response.data.df is not None:
                df = response.data.df
                if not df.empty and 'MID_SPREAD' in df.columns:
                    # Filter to just MID_SPREAD field
                    mid_spread_data = df[['MID_SPREAD']].copy()
                    mid_spread_data = mid_spread_data.dropna()
                    
                    if not mid_spread_data.empty:
                        logger.info(f"✅ Historical data successful for RIC: {ric} ({len(mid_spread_data)} records)")
                        return mid_spread_data
            
            logger.warning(f"⚠️ Historical data empty or missing MID_SPREAD for RIC: {ric}")
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
                fields=["MID_SPREAD"]
            )
            
            # Fetch the data
            response = pricing_def.get_data()
            
            if response.data and response.data.df is not None:
                df = response.data.df
                if not df.empty and 'MID_SPREAD' in df.columns:
                    # Create a single-row DataFrame with today's date
                    today = datetime.now().strftime("%Y-%m-%d")
                    snapshot_data = pd.DataFrame({
                        'MID_SPREAD': [df['MID_SPREAD'].iloc[0]],
                        'Date': [today]
                    }).set_index('Date')
                    
                    logger.info(f"✅ Snapshot data successful for RIC: {ric}")
                    return snapshot_data
            
            logger.warning(f"⚠️ Snapshot data empty or missing MID_SPREAD for RIC: {ric}")
            return None
            
        except Exception as e:
            logger.warning(f"⚠️ Snapshot data failed for RIC: {ric}: {e}")
            return None
    
    def fetch_data(self) -> bool:
        """Fetch CDS data from Refinitiv Workspace using RIC fallback strategy"""
        try:
            logger.info(f"🚀 Starting CDS data fetch for {self.country}")
            
            # Open Refinitiv session
            rd.open_session()
            logger.info("✅ Refinitiv session opened")
            
            # Get RIC mapping for this country
            country_rics = self.ric_mapping.get(self.country, [])
            if not country_rics:
                logger.error(f"❌ No RIC mapping found for country: {self.country}")
                return False
            
            # Try each RIC in order until one works
            for ric in country_rics:
                logger.info(f"🔄 Testing RIC: {ric}")
                
                # First try historical data
                historical_data = self._try_ric_historical(ric)
                if historical_data is not None:
                    self.raw_data = historical_data
                    self.working_ric = ric
                    self.data_source = "historical"
                    logger.info(f"✅ Successfully fetched historical CDS data for {self.country} using RIC: {ric}")
                    return True
                
                # If historical fails, try snapshot
                snapshot_data = self._try_ric_snapshot(ric)
                if snapshot_data is not None:
                    self.raw_data = snapshot_data
                    self.working_ric = ric
                    self.data_source = "snapshot"
                    logger.info(f"✅ Successfully fetched snapshot CDS data for {self.country} using RIC: {ric}")
                    return True
            
            # If we get here, no RIC worked
            logger.error(f"❌ All RICs failed for {self.country}")
            return False
                
        except Exception as e:
            logger.error(f"❌ Error fetching CDS data for {self.country}: {e}")
            return False
    
    def process_data(self) -> bool:
        """Process raw CDS data and merge with local history"""
        try:
            if self.raw_data is None or self.raw_data.empty:
                return False
            
            # Load existing local history
            local_history = self._load_local_history()
            
            # Process new data
            new_records = []
            for date, row in self.raw_data.iterrows():
                # Convert date index to string if it's a datetime
                if isinstance(date, datetime):
                    date_str = date.strftime("%Y-%m-%d")
                else:
                    date_str = str(date)
                
                new_records.append({
                    "country": self.country,
                    "ric": self.working_ric,
                    "date": date_str,
                    "mid_spread_bp": float(row['MID_SPREAD']),
                    "chg_7d_bps": np.nan,  # Will be calculated in derived metrics
                    "z_90d": np.nan,        # Will be calculated in derived metrics
                    "z_6m": np.nan,         # Will be calculated in derived metrics
                    "source": "Refinitiv",
                    "notes": f"CDS 5Y spread for {self.country}" + 
                            (f" - {self.data_source}_only" if self.data_source == "snapshot" else "")
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
            
            logger.info(f"✅ Processed CDS data for {self.country}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error processing CDS data for {self.country}: {e}")
            return False
    
    def calculate_derived_metrics(self) -> bool:
        """Calculate 7-day changes and z-scores"""
        try:
            if self.processed_data is None or self.processed_data.empty:
                return False
            
            df = self.processed_data.copy()
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # Calculate 7-day change (7 trading days)
            df['chg_7d_bps'] = df['mid_spread_bp'].diff(7)
            
            # Calculate 90-day z-score (using available data)
            # Start calculating from row 30 onwards (minimum data needed)
            df['z_90d'] = np.nan
            if len(df) >= 30:
                for i in range(29, len(df)):  # Start from index 29 (30th row)
                    window_size = min(90, i + 1)  # Use available data up to current row
                    if window_size >= 30:  # Need at least 30 days for meaningful z-score
                        window_data = df.iloc[i-window_size+1:i+1]['mid_spread_bp']
                        if len(window_data) >= 30:
                            mean_val = window_data.mean()
                            std_val = window_data.std()
                            if std_val > 0:  # Avoid division by zero
                                df.loc[df.index[i], 'z_90d'] = (df.iloc[i]['mid_spread_bp'] - mean_val) / std_val
            
            # Calculate 6-month z-score (126 trading days) - conditional
            z_6m_computed = False
            df['z_6m'] = np.nan
            
            if len(df) >= 126:
                for i in range(125, len(df)):  # Start from index 125 (126th row)
                    window_data = df.iloc[i-125:i+1]['mid_spread_bp']
                    if len(window_data) == 126:
                        mean_val = window_data.mean()
                        std_val = window_data.std()
                        if std_val > 0:  # Avoid division by zero
                            df.loc[df.index[i], 'z_6m'] = (df.iloc[i]['mid_spread_bp'] - mean_val) / std_val
                
                z_6m_computed = True
                logger.info(f"✅ z_6m computed using {len(df)} days of history")
            else:
                # Add note for insufficient history
                if len(df) > 0:
                    df.loc[df.index[-1], 'notes'] = df.loc[df.index[-1], 'notes'] + " - insufficient_history_6m"
                logger.info(f"⚠️ z_6m skipped - only {len(df)} days available (need 126+)")
            
            # Update the processed data with calculated metrics
            self.processed_data = df.copy()
            
            # Count non-NaN values for logging
            z_90d_count = df['z_90d'].notna().sum()
            z_6m_count = df['z_6m'].notna().sum()
            
            logger.info(f"✅ Calculated derived metrics for {self.country} CDS data")
            logger.info(f" z_90d: {z_90d_count}/{len(df)} records calculated")
            logger.info(f" z_6m: {z_6m_count}/{len(df)} records calculated")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error calculating derived metrics for {self.country} CDS: {e}")
            return False
    
    def run_full_pipeline(self) -> bool:
        """Run the complete data pipeline"""
        try:
            logger.info(f"🚀 Starting CDS pipeline for {self.country}")
            
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
            logger.info(f"✅ CDS pipeline completed for {self.country}")
            logger.info(f"📊 Used RIC: {self.working_ric}")
            logger.info(f" Data source: {self.data_source}")
            logger.info(f" Records: {len(self.processed_data)}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ CDS pipeline failed for {self.country}: {e}")
            return False
    
    def get_data(self):
        """Get the processed data"""
        return self.processed_data