"""
Sovereign Risk datasets - CDS spreads and credit ratings
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Optional

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
        
    def fetch_data(self) -> bool:
        """Fetch CDS data from Refinitiv Workspace"""
        try:
            logger.info(f"�� Creating dummy CDS data for {self.country}")
            
            # For now, create dummy data to test the structure
            dates = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
            dummy_cds = np.random.normal(300, 50, len(dates))  # Random CDS spreads around 300 bps
            
            self.raw_data = pd.DataFrame({
                'date': dates,
                'cds_spread': dummy_cds
            })
            
            logger.info(f"✅ Created dummy CDS data for {self.country} ({len(dates)} records)")
            return True
                
        except Exception as e:
            logger.error(f"❌ Error creating dummy CDS data for {self.country}: {e}")
            return False
    
    def process_data(self) -> bool:
        """Process raw CDS data into standardized format"""
        try:
            if self.raw_data is None or self.raw_data.empty:
                return False
            
            # Create standardized records
            processed_records = []
            
            for _, row in self.raw_data.iterrows():
                processed_records.append({
                    "country": self.country,
                    "metric": "cds_5y_bps",
                    "value": float(row['cds_spread']),
                    "date": row['date'].strftime("%Y-%m-%d"),
                    "source": "Refinitiv",
                    "notes": f"CDS 5Y spread for {self.country}",
                    "dataset_code": self.dataset_code,
                    "frequency": "daily",
                    "last_updated": datetime.now().isoformat()
                })
            
            self.processed_data = pd.DataFrame(processed_records)
            logger.info(f"✅ Processed CDS data for {self.country}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error processing CDS data for {self.country}: {e}")
            return False
    
    def calculate_derived_metrics(self) -> bool:
        """Calculate 7-day changes and 6-month z-scores"""
        try:
            if self.processed_data is None or self.processed_data.empty:
                return False
            
            df = self.processed_data.copy()
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # Calculate 7-day change
            df['7d_change_bps'] = df['value'].diff(7)
            
            # Calculate 6-month z-score (180 days)
            if len(df) >= 180:
                rolling_mean = df['value'].rolling(window=180).mean()
                rolling_std = df['value'].rolling(window=180).std()
                df['z_6m'] = (df['value'] - rolling_mean) / rolling_std
            else:
                df['z_6m'] = np.nan
            
            # Add derived metrics as new records
            derived_records = []
            
            for _, row in df.iterrows():
                if pd.notna(row['7d_change_bps']):
                    derived_records.append({
                        "country": self.country,
                        "metric": "7d_change_bps",
                        "value": row['7d_change_bps'],
                        "date": row['date'].strftime("%Y-%m-%d"),
                        "source": "Refinitiv",
                        "notes": "7-day change in CDS spread (basis points)",
                        "dataset_code": self.dataset_code,
                        "frequency": "daily",
                        "last_updated": datetime.now().isoformat()
                    })
                
                if pd.notna(row['z_6m']):
                    derived_records.append({
                        "country": self.country,
                        "metric": "z_6m",
                        "value": row['z_6m'],
                        "date": row['date'].strftime("%Y-%m-%d"),
                        "source": "Refinitiv",
                        "notes": "6-month z-score of CDS spread",
                        "dataset_code": self.dataset_code,
                        "frequency": "daily",
                        "last_updated": datetime.now().isoformat()
                    })
            
            # Combine original and derived records
            combined_df = pd.concat([
                self.processed_data,
                pd.DataFrame(derived_records)
            ], ignore_index=True)
            
            self.processed_data = combined_df
            logger.info(f"✅ Calculated derived metrics for {self.country} CDS data")
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
            
            logger.info(f"✅ CDS pipeline completed for {self.country}")
            return True
            
        except Exception as e:
            logger.error(f"❌ CDS pipeline failed for {self.country}: {e}")
            return False
    
    def get_data(self):
        """Get the processed data"""
        return self.processed_data