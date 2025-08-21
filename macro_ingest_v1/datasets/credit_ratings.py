"""
Credit Ratings datasets
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class CreditRatingsDataset:
    """Dataset A2: Credit ratings with outlook changes"""
    
    def __init__(self, country: str, start_date: str, end_date: str):
        self.country = country
        self.start_date = start_date
        self.end_date = end_date
        self.dataset_code = "A2"
        self.raw_data = None
        self.processed_data = None
        
    def fetch_data(self) -> bool:
        """Create dummy credit rating data"""
        try:
            logger.info(f" Creating dummy credit rating data for {self.country}")
            
            # Create monthly data
            dates = pd.date_range(start=self.start_date, end=self.end_date, freq='M')
            
            np.random.seed(42)  # For reproducible results
            
            # Credit rating (1-22 scale, where 1=AAA, 22=Default)
            rating_numeric = np.random.choice([15, 16, 17, 18], len(dates))  # Realistic for Turkey/Argentina
            
            # Outlook (-1=negative, 0=stable, 1=positive)
            outlook = np.random.choice([-1, 0, 1], len(dates))
            
            self.raw_data = pd.DataFrame({
                'date': dates,
                'rating_numeric': rating_numeric,
                'outlook': outlook
            })
            
            logger.info(f"✅ Created dummy credit rating data for {self.country} ({len(dates)} months)")
            return True
                
        except Exception as e:
            logger.error(f"❌ Error creating dummy credit rating data for {self.country}: {e}")
            return False
    
    def process_data(self) -> bool:
        """Process raw data into standardized format"""
        try:
            if self.raw_data is None or self.raw_data.empty:
                return False
            
            processed_records = []
            
            for _, row in self.raw_data.iterrows():
                # Credit rating
                processed_records.append({
                    "country": self.country,
                    "metric": "rating",
                    "value": float(row['rating_numeric']),
                    "date": row['date'].strftime("%Y-%m-%d"),
                    "source": "Refinitiv",
                    "notes": f"Credit rating for {self.country} (1=AAA, 22=Default)",
                    "dataset_code": self.dataset_code,
                    "frequency": "monthly",
                    "last_updated": datetime.now().isoformat()
                })
                
                # Outlook
                processed_records.append({
                    "country": self.country,
                    "metric": "outlook",
                    "value": float(row['outlook']),
                    "date": row['date'].strftime("%Y-%m-%d"),
                    "source": "Refinitiv",
                    "notes": f"Credit outlook for {self.country} (-1=negative, 0=stable, 1=positive)",
                    "dataset_code": self.dataset_code,
                    "frequency": "monthly",
                    "last_updated": datetime.now().isoformat()
                })
            
            self.processed_data = pd.DataFrame(processed_records)
            logger.info(f"✅ Processed credit rating data for {self.country}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error processing credit rating data for {self.country}: {e}")
            return False
    
    def calculate_derived_metrics(self) -> bool:
        """Calculate rating changes and outlook shifts"""
        try:
            if self.processed_data is None or self.processed_data.empty:
                return False
            
            # Get the raw data for calculations
            raw_df = self.raw_data.copy()
            
            # Calculate rating changes
            raw_df['rating_change'] = raw_df['rating_numeric'].diff()
            
            # Calculate outlook shifts
            raw_df['outlook_shift'] = raw_df['outlook'].diff()
            
            # Add derived metrics as new records
            derived_records = []
            
            for _, row in raw_df.iterrows():
                if pd.notna(row['rating_change']):
                    derived_records.append({
                        "country": self.country,
                        "metric": "rating_change",
                        "value": float(row['rating_change']),
                        "date": row['date'].strftime("%Y-%m-%d"),
                        "source": "Refinitiv",
                        "notes": "Change in credit rating (positive=worse, negative=better)",
                        "dataset_code": self.dataset_code,
                        "frequency": "monthly",
                        "last_updated": datetime.now().isoformat()
                    })
                
                if pd.notna(row['outlook_shift']):
                    derived_records.append({
                        "country": self.country,
                        "metric": "outlook_shift",
                        "value": float(row['outlook_shift']),
                        "date": row['date'].strftime("%Y-%m-%d"),
                        "source": "Refinitiv",
                        "notes": "Change in credit outlook",
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
            logger.info(f"✅ Calculated derived metrics for {self.country} credit ratings")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error calculating derived metrics for {self.country} credit ratings: {e}")
            return False
    
    def run_full_pipeline(self) -> bool:
        """Run the complete data pipeline"""
        try:
            logger.info(f"🚀 Starting credit ratings pipeline for {self.country}")
            
            # Step 1: Fetch data
            if not self.fetch_data():
                return False
            
            # Step 2: Process data
            if not self.process_data():
                return False
            
            # Step 3: Calculate derived metrics
            if not self.calculate_derived_metrics():
                return False
            
            logger.info(f"✅ Credit ratings pipeline completed for {self.country}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Credit ratings pipeline failed for {self.country}: {e}")
            return False
    
    def get_data(self):
        """Get the processed data"""
        return self.processed_data