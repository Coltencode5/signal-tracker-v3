"""
Inflation datasets - CPI YoY with derived metrics
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

class InflationDataset:
    """Dataset D1: CPI YoY with derived metrics"""
    
    def __init__(self, country: str, start_date: str, end_date: str):
        self.country = country
        self.start_date = start_date
        self.end_date = end_date
        self.dataset_code = "D1"
        self.raw_data = None
        self.processed_data = None
        
        # RIC mapping by country for CPI data
        self.ric_mapping = {
            "Turkey": [
                "TRCPIY=ECI",   # Primary: TurkStat CPI, change Y/Y, monthly, % y/y
                "TRCPI=ECI"     # Fallback: All Items CPI index, monthly; compute YoY in code
            ],
            "Argentina": [
                "ARCPIY=ECI",   # Primary: Argentina CPI, change Y/Y, monthly, % y/y
                "ARCPI=ECI"     # Fallback: All Items CPI index, monthly; compute YoY in code
            ]
        }
        self.working_ric = None
        self.data_source = None  # "historical" or "snapshot"
        self.fallback_used = False
        
        # Local history file path
        self.history_dir = Path("data_history")
        self.history_dir.mkdir(exist_ok=True)
        self.history_file = self.history_dir / f"inflation_history_{country.lower()}.json"
        
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
                # Only save CPI_YoY records to history (not derived metrics)
                if row['metric'] == 'CPI_YoY':
                    history_data.append({
                        'date': row['date'].strftime('%Y-%m-%d'),
                        'value_yoy_pct': float(row['value_yoy_pct']),
                        'ric': row.get('ric', self.working_ric),  # Use ric field if available
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
            
            # Parse dates
            start_dt = datetime.strptime(self.start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(self.end_date, "%Y-%m-%d")
            
            # Create historical pricing definition - EXACTLY like sovereign_risk.py
            inflation_def = rd.content.historical_pricing.summaries.Definition(
                universe=ric,
                start=start_dt.strftime("%Y-%m-%d"),
                end=end_dt.strftime("%Y-%m-%d"),
                interval="P1M"  # Monthly data
            )
            
            # Fetch the data
            response = inflation_def.get_data()
            
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
                        inflation_data = df[[date_col, value_col]].copy()
                    else:
                        # Use index as date
                        inflation_data = df[[value_col]].copy()
                        inflation_data.index.name = 'Date'
                    
                    inflation_data = inflation_data.dropna()
                    
                    if not inflation_data.empty:
                        logger.info(f"✅ Historical data successful for RIC: {ric} ({len(inflation_data)} records)")
                        return inflation_data
            
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
    
    def _compute_yoy_from_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute YoY percentage change from CPI index data"""
        try:
            if df.empty or len(df) < 2:
                return df
            
            # Sort by date to ensure proper calculation
            df = df.sort_values('Date')
            
            # Get the value column name
            value_col = 'VALUE' if 'VALUE' in df.columns else 'INDEX_VALUE'
            
            # Calculate YoY change (assuming monthly data)
            df['value_yoy_pct'] = df[value_col].pct_change(12) * 100
            
            # Remove rows where we can't calculate YoY (first 12 months)
            df = df.dropna(subset=['value_yoy_pct'])
            
            logger.info(f"✅ Computed YoY from index data: {len(df)} records")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error computing YoY from index: {e}")
            return df
    
    def fetch_data(self) -> bool:
        """Fetch inflation data from Refinitiv Workspace using RIC fallback strategy"""
        try:
            logger.info(f" Starting inflation data fetch for {self.country}")
            
            # Open Refinitiv session - EXACTLY like sovereign_risk.py
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
                    logger.info(f"✅ Successfully fetched historical inflation data for {self.country} using RIC: {ric}")
                    return True
                
                # If historical fails, try snapshot
                snapshot_data = self._try_ric_snapshot(ric)
                if snapshot_data is not None:
                    self.raw_data = snapshot_data
                    self.working_ric = ric
                    self.data_source = "snapshot"
                    self.fallback_used = (i > 0)  # True if not the primary RIC
                    logger.info(f"✅ Successfully fetched snapshot inflation data for {self.country} using RIC: {ric}")
                    return True
            
            # If we get here, no RIC worked
            logger.error(f"❌ All RICs failed for {self.country}")
            return False
                
        except Exception as e:
            logger.error(f"❌ Error fetching inflation data for {self.country}: {e}")
            return False
    
    def process_data(self) -> bool:
        """Process raw inflation data and merge with local history"""
        try:
            if self.raw_data is None or self.raw_data.empty:
                return False
            
            # Load existing local history
            local_history = self._load_local_history()
            
            # Process new data
            new_records = []
            
            # Determine if we need to compute YoY from index
            value_col = 'VALUE' if 'VALUE' in self.raw_data.columns else 'INDEX_VALUE'
            is_index_data = value_col == 'INDEX_VALUE'
            
            # If it's index data, compute YoY
            if is_index_data:
                self.raw_data = self._compute_yoy_from_index(self.raw_data)
                if self.raw_data.empty:
                    logger.error("❌ Failed to compute YoY from index data")
                    return False
            
            for date, row in self.raw_data.iterrows():
                # Convert date index to string if it's a datetime
                if isinstance(date, datetime):
                    date_str = date.strftime("%Y-%m-%d")
                else:
                    date_str = str(date)
                
                # Get the YoY value
                if 'value_yoy_pct' in self.raw_data.columns:
                    yoy_value = float(row['value_yoy_pct'])
                else:
                    # If we have direct YoY data
                    yoy_value = float(row[value_col])
                
                new_records.append({
                    "country": self.country,
                    "metric": "CPI_YoY",
                    "date": date_str,
                    "value_yoy_pct": yoy_value,
                    "source": "Refinitiv",
                    "ric": self.working_ric,  # Add RIC code to track which one worked
                    "notes": f"CPI YoY for {self.country} (%)" + 
                            (f" - computed_from_index" if is_index_data else "") +
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
            
            logger.info(f"✅ Processed inflation data for {self.country}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error processing inflation data for {self.country}: {e}")
            return False
    
    def calculate_derived_metrics(self) -> bool:
        """Calculate 3-month average and acceleration flag"""
        try:
            if self.processed_data is None or self.processed_data.empty:
                return False
            
            df = self.processed_data.copy()
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # Filter to just CPI_YoY records for calculations
            cpi_df = df[df['metric'] == 'CPI_YoY'].copy()
            
            if cpi_df.empty:
                logger.warning("⚠️ No CPI_YoY records found for derived metrics")
                return False
            
            # Calculate 3-month average
            cpi_df['3m_avg_cpi'] = cpi_df['value_yoy_pct'].rolling(window=3).mean()
            
            # Calculate acceleration flag (is inflation speeding up?)
            cpi_df['acceleration_flag'] = cpi_df['value_yoy_pct'] > cpi_df['3m_avg_cpi']
            
            # Add derived metrics as new records
            derived_records = []
            
            for _, row in cpi_df.iterrows():
                if pd.notna(row['3m_avg_cpi']):
                    derived_records.append({
                        "country": self.country,
                        "metric": "3m_avg_cpi",
                        "date": row['date'].strftime("%Y-%m-%d"),
                        "value_yoy_pct": float(row['3m_avg_cpi']),
                        "source": "Refinitiv",
                        "ric": self.working_ric,
                        "notes": "3-month average CPI YoY (%)",
                        "dataset_code": self.dataset_code,
                        "frequency": "monthly",
                        "last_updated": datetime.now().isoformat()
                    })
                
                if pd.notna(row['acceleration_flag']):
                    derived_records.append({
                        "country": self.country,
                        "metric": "acceleration_flag",
                        "date": row['date'].strftime("%Y-%m-%d"),
                        "value_yoy_pct": float(row['acceleration_flag']),
                        "source": "Refinitiv",
                        "ric": self.working_ric,
                        "notes": "CPI acceleration flag (1=accelerating, 0=decelerating)",
                        "dataset_code": self.dataset_code,
                        "frequency": "monthly",
                        "last_updated": datetime.now().isoformat()
                    })
            
            # Combine original and derived records
            combined_df = pd.concat([
                self.processed_data,
                pd.DataFrame(derived_records)
            ], ignore_index=True)
            
            self.processed_data = combined_df
            logger.info(f"✅ Calculated derived metrics for {self.country} CPI data")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error calculating derived metrics for {self.country} CPI: {e}")
            return False
    
    def run_full_pipeline(self) -> bool:
        """Run the complete data pipeline"""
        try:
            logger.info(f" Starting inflation pipeline for {self.country}")
            
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
            logger.info(f"✅ Inflation pipeline completed for {self.country}")
            logger.info(f"📊 Used RIC: {self.working_ric}")
            logger.info(f" Data source: {self.data_source}")
            logger.info(f"🔄 Fallback used: {self.fallback_used}")
            logger.info(f" Records: {len(self.processed_data)}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Inflation pipeline failed for {self.country}: {e}")
            return False
    
    def get_data(self):
        """Get the processed data"""
        return self.processed_data