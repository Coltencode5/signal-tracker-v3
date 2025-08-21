"""
Inflation, GDP, and Trade datasets
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Optional

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
        
    def fetch_data(self) -> bool:
        """Create dummy CPI inflation data"""
        try:
            logger.info(f" Creating dummy CPI data for {self.country}")
            
            # Create monthly data
            dates = pd.date_range(start=self.start_date, end=self.end_date, freq='M')
            
            np.random.seed(42)  # For reproducible results
            
            # CPI YoY (realistic inflation rates)
            cpi_yoy = np.random.normal(15, 5, len(dates))  # High inflation for Turkey/Argentina
            cpi_yoy = np.abs(cpi_yoy)  # Make positive
            
            self.raw_data = pd.DataFrame({
                'date': dates,
                'cpi_yoy': cpi_yoy
            })
            
            logger.info(f"✅ Created dummy CPI data for {self.country} ({len(dates)} months)")
            return True
                
        except Exception as e:
            logger.error(f"❌ Error creating dummy CPI data for {self.country}: {e}")
            return False
    
    def process_data(self) -> bool:
        """Process raw data into standardized format"""
        try:
            if self.raw_data is None or self.raw_data.empty:
                return False
            
            processed_records = []
            
            for _, row in self.raw_data.iterrows():
                processed_records.append({
                    "country": self.country,
                    "metric": "cpi_yoy",
                    "value": float(row['cpi_yoy']),
                    "date": row['date'].strftime("%Y-%m-%d"),
                    "source": "Refinitiv",
                    "notes": f"CPI YoY for {self.country} (%)",
                    "dataset_code": self.dataset_code,
                    "frequency": "monthly",
                    "last_updated": datetime.now().isoformat()
                })
            
            self.processed_data = pd.DataFrame(processed_records)
            logger.info(f"✅ Processed CPI data for {self.country}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error processing CPI data for {self.country}: {e}")
            return False
    
    def calculate_derived_metrics(self) -> bool:
        """Calculate 3-month average and acceleration flag"""
        try:
            if self.processed_data is None or self.processed_data.empty:
                return False
            
            # Get the raw data for calculations
            raw_df = self.raw_data.copy()
            
            # Calculate 3-month average
            raw_df['3m_avg_cpi'] = raw_df['cpi_yoy'].rolling(window=3).mean()
            
            # Calculate acceleration flag (is inflation speeding up?)
            raw_df['acceleration_flag'] = raw_df['cpi_yoy'] > raw_df['3m_avg_cpi']
            
            # Add derived metrics as new records
            derived_records = []
            
            for _, row in raw_df.iterrows():
                if pd.notna(row['3m_avg_cpi']):
                    derived_records.append({
                        "country": self.country,
                        "metric": "3m_avg_cpi",
                        "value": float(row['3m_avg_cpi']),
                        "date": row['date'].strftime("%Y-%m-%d"),
                        "source": "Refinitiv",
                        "notes": "3-month average CPI YoY (%)",
                        "dataset_code": self.dataset_code,
                        "frequency": "monthly",
                        "last_updated": datetime.now().isoformat()
                    })
                
                if pd.notna(row['acceleration_flag']):
                    derived_records.append({
                        "country": self.country,
                        "metric": "acceleration_flag",
                        "value": float(row['acceleration_flag']),
                        "date": row['date'].strftime("%Y-%m-%d"),
                        "source": "Refinitiv",
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
            logger.info(f"🚀 Starting CPI pipeline for {self.country}")
            
            # Step 1: Fetch data
            if not self.fetch_data():
                return False
            
            # Step 2: Process data
            if not self.process_data():
                return False
            
            # Step 3: Calculate derived metrics
            if not self.calculate_derived_metrics():
                return False
            
            logger.info(f"✅ CPI pipeline completed for {self.country}")
            return True
            
        except Exception as e:
            logger.error(f"❌ CPI pipeline failed for {self.country}: {e}")
            return False
    
    def get_data(self):
        """Get the processed data"""
        return self.processed_data