"""
Capital flows datasets - Portfolio investment flows and derived metrics
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

class CapitalFlowsDataset:
    """Dataset B1: Portfolio capital flows with derived metrics"""
    
    def __init__(self, country: str, start_date: str, end_date: str, 
                 init_backfill_years: int = 5, excel_window_years: int = 3):
        self.country = country
        self.start_date = start_date
        self.end_date = end_date
        self.dataset_code = "B1"
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
                "portfolio_net": "aTRPIBALT",      # Primary: Portfolio Investment Balance, Total, USD
                "gdp": "TRGDPQ=ECI"               # GDP denominator, USD (annual)
            },
            "Argentina": {
                "portfolio_net": "aARPIBALT",      # Primary: Portfolio Investment Balance, Total, USD
                "gdp": "ARGDPA=ECI"               # GDP denominator, USD (annual)
            }
        }
        
        # Local history file path
        self.history_dir = Path("data_history")
        self.history_dir.mkdir(exist_ok=True)
        self.history_file = self.history_dir / f"capital_flows_history_{country.lower()}.json"
        
        # Data tracking
        self.working_series = None
        self.data_source = None
        self.gdp_available = False
        self.latest_gdp = None
        
        logger.info(f"📊 Initialized Capital Flows Dataset for {country}")
    
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
    
    def _try_eci_series(self, series: str) -> Optional[pd.DataFrame]:
        """Try to fetch data for a specific ECI series"""
        try:
            logger.info(f"�� Trying ECI series: {series}")
            
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
                logger.info(f"�� Incremental update: fetching from {start_dt.strftime('%Y-%m-%d')}")
            
            # Create historical pricing definition
            capital_flows_def = rd.content.historical_pricing.summaries.Definition(
                universe=series,
                start=start_dt.strftime("%Y-%m-%d"),
                end=end_dt.strftime("%Y-%m-%d"),
                interval="P1M"  # Monthly data
            )
            
            # Fetch the data
            response = capital_flows_def.get_data()
            
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
                        capital_flows_data = df[[date_col, value_col]].copy()
                    else:
                        # Use index as date
                        capital_flows_data = df[[value_col]].copy()
                        capital_flows_data.index.name = 'Date'
                    
                    capital_flows_data = capital_flows_data.dropna()
                    
                    if not capital_flows_data.empty:
                        logger.info(f"✅ ECI series successful: {series} ({len(capital_flows_data)} records)")
                        return capital_flows_data
            
            logger.warning(f"⚠️ ECI series empty: {series}")
            return None
            
        except Exception as e:
            logger.warning(f"⚠️ ECI series failed: {series}: {e}")
            return None
    
    def fetch_data(self) -> bool:
        """Fetch capital flows data from Refinitiv using ECI series"""
        try:
            logger.info(f"🚀 Starting capital flows data fetch for {self.country}")
            
            # Open Refinitiv session
            rd.open_session()
            logger.info("✅ Refinitiv session opened")
            
            # Get ECI mapping for this country
            country_series = self.eci_mapping.get(self.country, {})
            if not country_series:
                logger.error(f"❌ No ECI mapping found for country: {self.country}")
                return False
            
            # Try portfolio net series
            if "portfolio_net" in country_series:
                portfolio_data = self._try_eci_series(country_series["portfolio_net"])
                if portfolio_data is not None:
                    self.raw_data = portfolio_data
                    self.working_series = country_series["portfolio_net"]
                    self.data_source = "primary"
                    logger.info(f"✅ Successfully fetched capital flows data for {self.country}")
                    return True
            
            # If we get here, no method worked
            logger.error(f"❌ All capital flows methods failed for {self.country}")
            return False
                
        except Exception as e:
            logger.error(f"❌ Error fetching capital flows data for {self.country}: {e}")
            return False
    
    def process_data(self) -> bool:
        """Process raw capital flows data and merge with local history"""
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
                
                # Get the portfolio net value
                portfolio_net = float(row[value_col])
                
                new_records.append({
                    "country": self.country_codes.get(self.country, self.country.upper()[:3]),
                    "series_id": self.working_series,
                    "date": date_str,
                    "portfolio_net_usd": portfolio_net,
                    "source": "Refinitiv",
                    "notes": f"Portfolio investment balance for {self.country} (USD)"
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
                
                logger.info(f"�� Merged {len(new_df)} new records with {len(local_history)} existing records")
                logger.info(f"📈 Total records available: {len(combined_df)}")
                
                self.processed_data = combined_df
            else:
                self.processed_data = new_df
                logger.info(f"📊 First run: {len(new_df)} new records")
            
            # Save updated history
            self._save_local_history(self.processed_data)
            
            logger.info(f"✅ Processed capital flows data for {self.country}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error processing capital flows data for {self.country}: {e}")
            return False
    
    def calculate_derived_metrics(self) -> bool:
        """Calculate derived capital flows metrics"""
        try:
            if self.processed_data is None or self.processed_data.empty:
                return False
            
            df = self.processed_data.copy()
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # Filter to just portfolio_net_usd records for calculations
            portfolio_df = df[df['series_id'].str.contains('PIBALT')].copy()
            
            if portfolio_df.empty:
                logger.warning("⚠️ No portfolio investment records found for derived metrics")
                return False
            
            # Calculate rolling 12-month sum
            portfolio_df['cf_12m_sum_usd'] = portfolio_df['portfolio_net_usd'].rolling(window=12, min_periods=1).sum()
            
            # Calculate year-over-year change
            portfolio_df['cf_yoy_usd'] = portfolio_df['portfolio_net_usd'].diff(12)
            
            # Try to fetch GDP for percentage calculations
            self.latest_gdp = self._fetch_gdp_data(self.country)
            if self.latest_gdp is not None:
                self.gdp_available = True
                # Calculate as percentage of GDP
                portfolio_df['cf_pct_gdp'] = portfolio_df['cf_12m_sum_usd'] / self.latest_gdp * 100
                logger.info(f"✅ GDP calculations applied using latest GDP: {self.latest_gdp:,.0f} USD")
            else:
                self.gdp_available = False
                portfolio_df['cf_pct_gdp'] = np.nan
                logger.info("⚠️ GDP not available for percentage calculations")
            
            # Add derived metrics as new records
            derived_records = []
            
            for _, row in portfolio_df.iterrows():
                if pd.notna(row['cf_12m_sum_usd']):
                    derived_records.append({
                        "country": self.country_codes.get(self.country, self.country.upper()[:3]),
                        "series_id": "cf_12m_sum_usd",
                        "date": row['date'].strftime("%Y-%m-%d"),
                        "portfolio_net_usd": float(row['cf_12m_sum_usd']),
                        "source": "Refinitiv",
                        "notes": "12-month rolling sum of portfolio investment balance"
                    })
                
                if pd.notna(row['cf_yoy_usd']):
                    derived_records.append({
                        "country": self.country_codes.get(self.country, self.country.upper()[:3]),
                        "series_id": "cf_yoy_usd",
                        "date": row['date'].strftime("%Y-%m-%d"),
                        "portfolio_net_usd": float(row['cf_yoy_usd']),
                        "source": "Refinitiv",
                        "notes": "Year-over-year change in portfolio investment balance"
                    })
                
                if pd.notna(row['cf_pct_gdp']):
                    derived_records.append({
                        "country": self.country_codes.get(self.country, self.country.upper()[:3]),
                        "series_id": "cf_pct_gdp",
                        "date": row['date'].strftime("%Y-%m-%d"),
                        "portfolio_net_usd": float(row['cf_pct_gdp']),
                        "source": "Refinitiv",
                        "notes": f"Portfolio flows as % of GDP (GDP: {self.latest_gdp:,.0f} USD)"
                    })
            
            # Combine original and derived records
            combined_df = pd.concat([
                self.processed_data,
                pd.DataFrame(derived_records)
            ], ignore_index=True)
            
            self.processed_data = combined_df
            logger.info(f"✅ Calculated derived metrics for {self.country} capital flows data")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error calculating derived metrics for {self.country} capital flows data: {e}")
            return False
    
    def run_full_pipeline(self) -> bool:
        """Run the complete data pipeline"""
        try:
            logger.info(f"🚀 Starting capital flows pipeline for {self.country}")
            
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
            logger.info(f"✅ Capital flows pipeline completed for {self.country}")
            logger.info(f"�� Used series: {self.working_series}")
            logger.info(f"�� Data source: {self.data_source}")
            logger.info(f"📊 GDP available: {self.gdp_available}")
            logger.info(f"�� Records: {len(self.processed_data)}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Capital flows pipeline failed for {self.country}: {e}")
            return False
    
    def get_data(self):
        """Get the processed data"""
        return self.processed_data