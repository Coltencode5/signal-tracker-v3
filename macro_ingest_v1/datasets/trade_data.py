"""
Trade datasets - Trade balance and derived metrics
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

class TradeBalanceDataset:
    """Dataset D2: Trade balance with derived metrics"""
    
    def __init__(self, country: str, start_date: str, end_date: str, 
                 init_backfill_years: int = 5, excel_window_years: int = 3):
        self.country = country
        self.start_date = start_date
        self.end_date = end_date
        self.dataset_code = "D2"
        self.raw_data = None
        self.processed_data = None
        
        # Configuration
        self.init_backfill_years = init_backfill_years
        self.excel_window_years = excel_window_years
        
        # Country code mapping
        self.country_codes = {
            "Turkey": "TUR",
            "Argentina": "ARG"
        }
        
        # ECI series mapping by country
        self.eci_mapping = {
            "Turkey": {
                "trade_balance": "TRTRD=ECI",      # Primary: Monthly trade balance, USD, NSA
                "exports": "TREXP=ECI",            # Fallback: Exports, USD
                "imports": "TRIMP=ECI",            # Fallback: Imports, USD
                "gdp": "TRGDPQ=ECI"               # GDP denominator, USD (corrected)
            },
            "Argentina": {
                "trade_balance": "ARTRD=ECI",      # Primary: Monthly trade balance, USD, NSA
                "exports": "AREXP=ECI",            # Fallback: Exports, USD
                "imports": "ARIMP=ECI",            # Fallback: Imports, USD
                "gdp": "ARGDPA=ECI"                # GDP denominator, USD
            }
        }
        
        # Local history file path
        self.history_dir = Path("data_history")
        self.history_dir.mkdir(exist_ok=True)
        self.history_file = self.history_dir / f"trade_history_{country.lower()}.json"
        
        # Data tracking
        self.working_series = None
        self.data_source = None
        self.fallback_used = False
        self.gdp_available = False
        self.latest_gdp = None
        
        logger.info(f"📊 Initialized Trade Balance Dataset for {country}")
    
    def _load_local_history(self) -> pd.DataFrame:
        """Load existing local history from JSON file"""
        try:
            if self.history_file.exists():
                with open(self.history_file, 'r') as f:
                    data = json.load(f)
                
                # Convert to DataFrame
                df = pd.DataFrame(data)
                if not df.empty:
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.sort_values('date')
                    df = df.drop_duplicates(subset=['date'], keep='last')
                    logger.info(f"📂 Loaded {len(df)} existing records from local history")
                    return df
            else:
                logger.info("📂 No existing local history found")
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.warning(f"⚠️ Error loading local history: {e}")
            return pd.DataFrame()
    
    def _save_local_history(self, df: pd.DataFrame) -> None:
        """Save processed data to local JSON history file"""
        try:
            if df is not None and not df.empty:
                # Convert to records format for JSON
                records = df.to_dict('records')
                
                # Convert datetime to string for JSON serialization
                for record in records:
                    if isinstance(record['date'], datetime):
                        record['date'] = record['date'].strftime("%Y-%m-%d")
                
                with open(self.history_file, 'w') as f:
                    json.dump(records, f, indent=2)
                
                logger.info(f" Saved {len(records)} records to local history")
            else:
                logger.warning("⚠️ No data to save to local history")
                
        except Exception as e:
            logger.error(f"❌ Error saving local history: {e}")
    
    def _fetch_gdp_data(self, country: str) -> Optional[float]:
        """Fetch latest available GDP data for percentage calculations"""
        try:
            gdp_series = self.eci_mapping[country]["gdp"]
            logger.info(f"🔄 Fetching GDP data using series: {gdp_series}")
            
            # Create historical pricing definition for GDP
            gdp_def = rd.content.historical_pricing.summaries.Definition(
                universe=gdp_series,
                start=(datetime.now() - timedelta(days=365*5)).strftime("%Y-%m-%d"),  # Last 5 years
                end=datetime.now().strftime("%Y-%m-%d"),
                interval="P1Y"  # Annual data
            )
            
            # Fetch the data
            response = gdp_def.get_data()
            
            if response.data and response.data.df is not None:
                df = response.data.df
                if not df.empty:
                    # Get the latest GDP value
                    if 'VALUE' in df.columns:
                        latest_gdp = float(df['VALUE'].iloc[-1])
                        logger.info(f"✅ GDP data successful: {latest_gdp:,.0f} USD")
                        return latest_gdp
                    elif 'INDEX_VALUE' in df.columns:
                        latest_gdp = float(df['INDEX_VALUE'].iloc[-1])
                        logger.info(f"✅ GDP data successful: {latest_gdp:,.0f} USD")
                        return latest_gdp
            
            logger.warning(f"⚠️ GDP data empty for series: {gdp_series}")
            return None
            
        except Exception as e:
            logger.warning(f"⚠️ GDP data failed for series: {gdp_series}: {e}")
            return None
    
    def _try_eci_series(self, series: str, is_fallback: bool = False) -> Optional[pd.DataFrame]:
        """Try to fetch data for a specific ECI series"""
        try:
            logger.info(f" Trying {'fallback' if is_fallback else 'primary'} ECI series: {series}")
            
            # Determine fetch window
            if self._load_local_history().empty:
                # First run: backfill N years
                end_dt = datetime.strptime(self.end_date, "%Y-%m-%d")
                start_dt = end_dt - timedelta(days=365 * self.init_backfill_years)
                logger.info(f"📅 Initial backfill: fetching {self.init_backfill_years} years")
            else:
                # Subsequent run: fetch from last cached date - 13 months
                last_cached = self._load_local_history()['date'].max()
                start_dt = last_cached - timedelta(days=13*30)  # 13 months back
                end_dt = datetime.strptime(self.end_date, "%Y-%m-%d")
                logger.info(f" Incremental update: fetching from {start_dt.strftime('%Y-%m-%d')}")
            
            # Create historical pricing definition
            trade_def = rd.content.historical_pricing.summaries.Definition(
                universe=series,
                start=start_dt.strftime("%Y-%m-%d"),
                end=end_dt.strftime("%Y-%m-%d"),
                interval="P1M"  # Monthly data
            )
            
            # Fetch the data
            response = trade_def.get_data()
            
            if response.data and response.data.df is not None:
                df = response.data.df
                if not df.empty:
                    # Check if we have the right field
                    if 'VALUE' in df.columns:
                        value_col = 'VALUE'
                    elif 'INDEX_VALUE' in df.columns:
                        value_col = 'INDEX_VALUE'
                    else:
                        logger.warning(f"⚠️ No VALUE or INDEX_VALUE field found for series: {series}")
                        return None
                    
                    # Handle date column
                    if 'Date' in df.columns:
                        date_col = 'Date'
                    elif 'date' in df.columns:
                        date_col = 'date'
                    else:
                        date_col = None
                    
                    if date_col:
                        trade_data = df[[date_col, value_col]].copy()
                    else:
                        # Use index as date
                        trade_data = df[[value_col]].copy()
                        trade_data.index.name = 'Date'
                    
                    trade_data = trade_data.dropna()
                    
                    if not trade_data.empty:
                        logger.info(f"✅ ECI series successful: {series} ({len(trade_data)} records)")
                        return trade_data
            
            logger.warning(f"⚠️ ECI series empty: {series}")
            return None
            
        except Exception as e:
            logger.warning(f"⚠️ ECI series failed: {series}: {e}")
            return None
    
    def _calculate_fallback_trade_balance(self, country: str) -> Optional[pd.DataFrame]:
        """Calculate trade balance from exports - imports if primary series fails"""
        try:
            country_series = self.eci_mapping[country]
            
            if "exports" not in country_series or "imports" not in country_series:
                logger.warning("⚠️ Fallback series not configured for this country")
                return None
            
            logger.info("🔄 Attempting fallback calculation: exports - imports")
            
            # Fetch exports data
            exports_data = self._try_eci_series(country_series["exports"], is_fallback=True)
            if exports_data is None:
                logger.warning("⚠️ Exports data unavailable for fallback")
                return None
            
            # Fetch imports data
            imports_data = self._try_eci_series(country_series["imports"], is_fallback=True)
            if imports_data is None:
                logger.warning("⚠️ Imports data unavailable for fallback")
                return None
            
            # Align dates and calculate trade balance
            exports_data = exports_data.reset_index()
            imports_data = imports_data.reset_index()
            
            # Merge on date
            merged = pd.merge(exports_data, imports_data, on='Date', how='inner', suffixes=('_exp', '_imp'))
            
            if merged.empty:
                logger.warning("⚠️ No overlapping dates for fallback calculation")
                return None
            
            # Calculate trade balance
            merged['trade_balance'] = merged['VALUE_exp'] - merged['VALUE_imp']
            
            # Return in expected format
            result = merged[['Date', 'trade_balance']].copy()
            result.columns = ['Date', 'VALUE']
            result = result.set_index('Date')
            
            logger.info(f"✅ Fallback calculation successful: {len(result)} records")
            return result
            
        except Exception as e:
            logger.warning(f"⚠️ Fallback calculation failed: {e}")
            return None
    
    def fetch_data(self) -> bool:
        """Fetch trade balance data from Refinitiv using ECI series"""
        try:
            logger.info(f"🚀 Starting trade balance data fetch for {self.country}")
            
            # Open Refinitiv session
            rd.open_session()
            logger.info("✅ Refinitiv session opened")
            
            # Get ECI mapping for this country
            country_series = self.eci_mapping.get(self.country, {})
            if not country_series:
                logger.error(f"❌ No ECI mapping found for country: {self.country}")
                return False
            
            # Try primary trade balance series first
            if "trade_balance" in country_series:
                primary_data = self._try_eci_series(country_series["trade_balance"])
                if primary_data is not None:
                    self.raw_data = primary_data
                    self.working_series = country_series["trade_balance"]
                    self.data_source = "primary"
                    self.fallback_used = False
                    logger.info(f"✅ Successfully fetched primary trade balance data for {self.country}")
                    return True
            
            # If primary fails, try fallback calculation
            fallback_data = self._calculate_fallback_trade_balance(self.country)
            if fallback_data is not None:
                self.raw_data = fallback_data
                self.working_series = f"fallback_{country_series.get('exports', 'unknown')}_{country_series.get('imports', 'unknown')}"
                self.data_source = "fallback"
                self.fallback_used = True
                logger.info(f"✅ Successfully fetched fallback trade balance data for {self.country}")
                return True
            
            # If we get here, no method worked
            logger.error(f"❌ All trade balance methods failed for {self.country}")
            return False
                
        except Exception as e:
            logger.error(f"❌ Error fetching trade balance data for {self.country}: {e}")
            return False
    
    def process_data(self) -> bool:
        """Process raw trade balance data and merge with local history"""
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
                
                # Get the trade balance value
                trade_balance = float(row[value_col])
                
                new_records.append({
                    "country": self.country_codes.get(self.country, self.country.upper()[:3]),
                    "ric_or_series": self.working_series,
                    "date": date_str,
                    "trade_balance_usd": trade_balance,
                    "source": "Refinitiv",
                    "notes": f"Trade balance for {self.country} (USD)" + 
                            (f" - {self.data_source}_series" if self.data_source == "fallback" else "") +
                            (f" - fallback_calculation" if self.fallback_used else "")
                })
            
            new_df = pd.DataFrame(new_records)
            new_df['date'] = pd.to_datetime(new_df['date'])
            
            # Merge with local history
            if not local_history.empty:
                # Combine and remove duplicates (keep newest)
                combined_df = pd.concat([local_history, new_df], ignore_index=True)
                combined_df = combined_df.drop_duplicates(subset=['date'], keep='last')
                combined_df = combined_df.sort_values('date')
                
                # Keep only last 10 years
                cutoff_date = datetime.now() - timedelta(days=365*10)
                combined_df = combined_df[combined_df['date'] >= cutoff_date]
                
                logger.info(f" Merged {len(new_df)} new records with {len(local_history)} existing records")
                logger.info(f"📈 Total records available: {len(combined_df)}")
                
                self.processed_data = combined_df
            else:
                self.processed_data = new_df
                logger.info(f"📊 First run: {len(new_df)} new records")
            
            # Save updated history
            self._save_local_history(self.processed_data)
            
            logger.info(f"✅ Processed trade balance data for {self.country}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error processing trade balance data for {self.country}: {e}")
            return False
    
    def calculate_derived_metrics(self) -> bool:
        """Calculate derived trade balance metrics"""
        try:
            if self.processed_data is None or self.processed_data.empty:
                return False
            
            df = self.processed_data.copy()
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # Filter to just trade balance records (primary or fallback)
            # Look for records that are either the primary series or fallback calculations
            trade_df = df[
                (df['ric_or_series'].str.contains('TRTRD=ECI|ARTRD=ECI')) |  # Primary series
                (df['ric_or_series'].str.contains('fallback')) |              # Fallback calculations
                (df['ric_or_series'].str.contains('trade_balance'))           # Any trade balance records
            ].copy()
            
            if trade_df.empty:
                logger.warning("⚠️ No trade balance records found for derived metrics")
                return False
            
            # Calculate rolling 12-month sum
            trade_df['tb_12m_sum_usd'] = trade_df['trade_balance_usd'].rolling(window=12, min_periods=1).sum()
            
            # Calculate year-over-year change
            trade_df['change_1y_usd'] = trade_df['trade_balance_usd'].diff(12)
            
            # Try to fetch GDP for percentage calculations
            self.latest_gdp = self._fetch_gdp_data(self.country)
            if self.latest_gdp is not None:
                self.gdp_available = True
                # Calculate deficit as percentage of GDP
                trade_df['deficit_pct_gdp'] = np.where(
                    trade_df['tb_12m_sum_usd'] < 0,
                    abs(trade_df['tb_12m_sum_usd']) / self.latest_gdp * 100,
                    np.nan
                )
                logger.info(f"✅ GDP calculations applied using latest GDP: {self.latest_gdp:,.0f} USD")
            else:
                self.gdp_available = False
                trade_df['deficit_pct_gdp'] = np.nan
                logger.info("⚠️ GDP not available for percentage calculations")
            
            # Add derived metrics as new records
            derived_records = []
            
            for _, row in trade_df.iterrows():
                if pd.notna(row['tb_12m_sum_usd']):
                    derived_records.append({
                        "country": self.country_codes.get(self.country, self.country.upper()[:3]),
                        "ric_or_series": "tb_12m_sum_usd",
                        "date": row['date'].strftime("%Y-%m-%d"),
                        "trade_balance_usd": float(row['tb_12m_sum_usd']),
                        "source": "Refinitiv",
                        "notes": "12-month rolling sum of trade balance"
                    })
                
                if pd.notna(row['change_1y_usd']):
                    derived_records.append({
                        "country": self.country_codes.get(self.country, self.country.upper()[:3]),
                        "ric_or_series": "change_1y_usd",
                        "date": row['date'].strftime("%Y-%m-%d"),
                        "trade_balance_usd": float(row['change_1y_usd']),  # ✅ CORRECT
                        "source": "Refinitiv",
                        "notes": "Year-over-year change in trade balance"
                    })
                
                if pd.notna(row['deficit_pct_gdp']):
                    derived_records.append({
                        "country": self.country_codes.get(self.country, self.country.upper()[:3]),
                        "ric_or_series": "deficit_pct_gdp",
                        "date": row['date'].strftime("%Y-%m-%d"),
                        "trade_balance_usd": float(row['deficit_pct_gdp']),
                        "source": "Refinitiv",
                        "notes": f"Deficit as % of GDP (GDP: {self.latest_gdp:,.0f} USD)"
                    })
            
            # Combine original and derived records
            combined_df = pd.concat([
                self.processed_data,
                pd.DataFrame(derived_records)
            ], ignore_index=True)
            
            self.processed_data = combined_df
            logger.info(f"✅ Calculated derived metrics for {self.country} trade data")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error calculating derived metrics for {self.country} trade data: {e}")
            return False
    
    def run_full_pipeline(self) -> bool:
        """Run the complete data pipeline"""
        try:
            logger.info(f"🚀 Starting trade balance pipeline for {self.country}")
            
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
            logger.info(f"✅ Trade balance pipeline completed for {self.country}")
            logger.info(f" Used series: {self.working_series}")
            logger.info(f" Data source: {self.data_source}")
            logger.info(f"🔄 Fallback used: {self.fallback_used}")
            logger.info(f"📊 GDP available: {self.gdp_available}")
            logger.info(f" Records: {len(self.processed_data)}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Trade balance pipeline failed for {self.country}: {e}")
            return False
    
    def get_data(self):
        """Get the processed data"""
        return self.processed_data