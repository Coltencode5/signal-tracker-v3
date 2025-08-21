"""
Monetary Data datasets - Central Bank and Money Supply
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class MonetaryDataset:
    """Dataset B2: Central Bank Balance Sheet & M2 with derived metrics"""
    
    def __init__(self, country: str, start_date: str, end_date: str):
        self.country = country
        self.start_date = start_date
        self.end_date = end_date
        self.dataset_code = "B2"
        self.raw_data = None
        self.processed_data = None
        
    def fetch_data(self) -> bool:
        """Create dummy monetary data"""
        try:
            logger.info(f" Creating dummy monetary data for {self.country}")
            
            # Create monthly data
            dates = pd.date_range(start=self.start_date, end=self.end_date, freq='M')
            
            np.random.seed(42)  # For reproducible results
            
            # Central bank balance sheet (in millions USD)
            cb_balance = np.random.normal(50000, 5000, len(dates))
            cb_balance = np.abs(cb_balance)
            
            # M2 money supply (in millions USD)
            m2_supply = np.random.normal(100000, 8000, len(dates))
            m2_supply = np.abs(m2_supply)
            
            self.raw_data = pd.DataFrame({
                'date': dates,
                'cb_balance': cb_balance,
                'm2_supply': m2_supply
            })
            
            logger.info(f"✅ Created dummy monetary data for {self.country} ({len(dates)} months)")
            return True
                
        except Exception as e:
            logger.error(f"❌ Error creating dummy monetary data for {self.country}: {e}")
            return False
    
    def process_data(self) -> bool:
        """Process raw data into standardized format"""
        try:
            if self.raw_data is None or self.raw_data.empty:
                return False
            
            processed_records = []
            
            for _, row in self.raw_data.iterrows():
                # Central bank balance sheet
                processed_records.append({
                    "country": self.country,
                    "metric": "cb_balance_sheet",
                    "value": float(row['cb_balance']),
                    "date": row['date'].strftime("%Y-%m-%d"),
                    "source": "Refinitiv",
                    "notes": f"Central bank balance sheet for {self.country} (millions USD)",
                    "dataset_code": self.dataset_code,
                    "frequency": "monthly",
                    "last_updated": datetime.now().isoformat()
                })
                
                # M2 money supply
                processed_records.append({
                    "country": self.country,
                    "metric": "m2",
                    "value": float(row['m2_supply']),
                    "date": row['date'].strftime("%Y-%m-%d"),
                    "source": "Refinitiv",
                    "notes": f"M2 money supply for {self.country} (millions USD)",
                    "dataset_code": self.dataset_code,
                    "frequency": "monthly",
                    "last_updated": datetime.now().isoformat()
                })
            
            self.processed_data = pd.DataFrame(processed_records)
            logger.info(f"✅ Processed monetary data for {self.country}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error processing monetary data for {self.country}: {e}")
            return False
    
    def calculate_derived_metrics(self) -> bool:
        """Calculate derived metrics"""
        try:
            if self.processed_data is None or self.processed_data.empty:
                return False
            
            # Get the raw data for calculations
            raw_df = self.raw_data.copy()
            
            # Calculate year-over-year growth rates
            raw_df['cb_yoy_growth'] = raw_df['cb_balance'].pct_change(12) * 100
            raw_df['m2_yoy_growth'] = raw_df['m2_supply'].pct_change(12) * 100
            
            # Add derived metrics as new records
            derived_records = []
            
            for _, row in raw_df.iterrows():
                if pd.notna(row['cb_yoy_growth']):
                    derived_records.append({
                        "country": self.country,
                        "metric": "cb_yoy_growth",
                        "value": float(row['cb_yoy_growth']),
                        "date": row['date'].strftime("%Y-%m-%d"),
                        "source": "Refinitiv",
                        "notes": "Central bank balance sheet YoY growth (%)",
                        "dataset_code": self.dataset_code,
                        "frequency": "monthly",
                        "last_updated": datetime.now().isoformat()
                    })
                
                if pd.notna(row['m2_yoy_growth']):
                    derived_records.append({
                        "country": self.country,
                        "metric": "m2_yoy_growth",
                        "value": float(row['m2_yoy_growth']),
                        "date": row['date'].strftime("%Y-%m-%d"),
                        "source": "Refinitiv",
                        "notes": "M2 money supply YoY growth (%)",
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
            logger.info(f"✅ Calculated derived metrics for {self.country} monetary data")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error calculating derived metrics for {self.country} monetary: {e}")
            return False
    
    def run_full_pipeline(self) -> bool:
        """Run the complete data pipeline"""
        try:
            logger.info(f"🚀 Starting monetary data pipeline for {self.country}")
            
            # Step 1: Fetch data
            if not self.fetch_data():
                return False
            
            # Step 2: Process data
            if not self.process_data():
                return False
            
            # Step 3: Calculate derived metrics
            if not self.calculate_derived_metrics():
                return False
            
            logger.info(f"✅ Monetary data pipeline completed for {self.country}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Monetary data pipeline failed for {self.country}: {e}")
            return False
    
    def get_data(self):
        """Get the processed data"""
        return self.processed_data