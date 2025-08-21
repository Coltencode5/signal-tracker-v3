"""
Trade Balance datasets
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class TradeBalanceDataset:
    """Dataset D2: Trade balance with derived metrics"""
    
    def __init__(self, country: str, start_date: str, end_date: str):
        self.country = country
        self.start_date = start_date
        self.end_date = end_date
        self.dataset_code = "D2"
        self.raw_data = None
        self.processed_data = None
        
    def fetch_data(self) -> bool:
        """Create dummy trade balance data"""
        try:
            logger.info(f" Creating dummy trade data for {self.country}")
            
            # Create monthly data
            dates = pd.date_range(start=self.start_date, end=self.end_date, freq='M')
            
            np.random.seed(42)  # For reproducible results
            
            # Trade balance (in millions USD) - can be negative (deficit)
            trade_balance = np.random.normal(-2000, 1000, len(dates))
            
            # GDP for ratio calculations (in millions USD)
            gdp = np.random.normal(200000, 10000, len(dates))
            gdp = np.abs(gdp)
            
            self.raw_data = pd.DataFrame({
                'date': dates,
                'trade_balance': trade_balance,
                'gdp': gdp
            })
            
            logger.info(f"✅ Created dummy trade data for {self.country} ({len(dates)} months)")
            return True
                
        except Exception as e:
            logger.error(f"❌ Error creating dummy trade data for {self.country}: {e}")
            return False
    
    def process_data(self) -> bool:
        """Process raw data into standardized format"""
        try:
            if self.raw_data is None or self.raw_data.empty:
                return False
            
            processed_records = []
            
            for _, row in self.raw_data.iterrows():
                # Trade balance
                processed_records.append({
                    "country": self.country,
                    "metric": "trade_balance",
                    "value": float(row['trade_balance']),
                    "date": row['date'].strftime("%Y-%m-%d"),
                    "source": "Refinitiv",
                    "notes": f"Trade balance for {self.country} (millions USD)",
                    "dataset_code": self.dataset_code,
                    "frequency": "monthly",
                    "last_updated": datetime.now().isoformat()
                })
                
                # GDP
                processed_records.append({
                    "country": self.country,
                    "metric": "gdp",
                    "value": float(row['gdp']),
                    "date": row['date'].strftime("%Y-%m-%d"),
                    "source": "Refinitiv",
                    "notes": f"GDP for {self.country} (millions USD)",
                    "dataset_code": self.dataset_code,
                    "frequency": "monthly",
                    "last_updated": datetime.now().isoformat()
                })
            
            self.processed_data = pd.DataFrame(processed_records)
            logger.info(f"✅ Processed trade data for {self.country}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error processing trade data for {self.country}: {e}")
            return False
    
    def calculate_derived_metrics(self) -> bool:
        """Calculate year-over-year change and deficit percentage of GDP"""
        try:
            if self.processed_data is None or self.processed_data.empty:
                return False
            
            # Get the raw data for calculations
            raw_df = self.raw_data.copy()
            
            # Calculate 1-year change
            raw_df['1y_change'] = raw_df['trade_balance'].pct_change(12) * 100
            
            # Calculate deficit as percentage of GDP
            raw_df['deficit_pct_gdp'] = (raw_df['trade_balance'] / raw_df['gdp']) * 100
            
            # Add derived metrics as new records
            derived_records = []
            
            for _, row in raw_df.iterrows():
                if pd.notna(row['1y_change']):
                    derived_records.append({
                        "country": self.country,
                        "metric": "1y_change",
                        "value": float(row['1y_change']),
                        "date": row['date'].strftime("%Y-%m-%d"),
                        "source": "Refinitiv",
                        "notes": "1-year change in trade balance (%)",
                        "dataset_code": self.dataset_code,
                        "frequency": "monthly",
                        "last_updated": datetime.now().isoformat()
                    })
                
                if pd.notna(row['deficit_pct_gdp']):
                    derived_records.append({
                        "country": self.country,
                        "metric": "deficit_pct_gdp",
                        "value": float(row['deficit_pct_gdp']),
                        "date": row['date'].strftime("%Y-%m-%d"),
                        "source": "Refinitiv",
                        "notes": "Trade balance as percentage of GDP (%)",
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
            logger.info(f"✅ Calculated derived metrics for {self.country} trade data")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error calculating derived metrics for {self.country} trade: {e}")
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
            
            logger.info(f"✅ Trade balance pipeline completed for {self.country}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Trade balance pipeline failed for {self.country}: {e}")
            return False
    
    def get_data(self):
        """Get the processed data"""
        return self.processed_data