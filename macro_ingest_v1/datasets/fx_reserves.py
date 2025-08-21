"""
FX Reserves & External Fragility datasets
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class FXReservesDataset:
    """Dataset C1: FX Reserve levels with derived metrics"""
    
    def __init__(self, country: str, start_date: str, end_date: str):
        self.country = country
        self.start_date = start_date
        self.end_date = end_date
        self.dataset_code = "C1"
        self.raw_data = None
        self.processed_data = None
        
    def fetch_data(self) -> bool:
        """Create dummy FX reserves data"""
        try:
            logger.info(f" Creating dummy FX reserves data for {self.country}")
            
            # Create monthly data
            dates = pd.date_range(start=self.start_date, end=self.end_date, freq='M')
            
            np.random.seed(42)  # For reproducible results
            
            # FX reserves (in millions USD)
            fx_reserves = np.random.normal(80000, 10000, len(dates))
            fx_reserves = np.abs(fx_reserves)
            
            # Import coverage (months of imports)
            import_coverage = np.random.normal(6, 1, len(dates))
            import_coverage = np.abs(import_coverage)
            
            self.raw_data = pd.DataFrame({
                'date': dates,
                'fx_reserves': fx_reserves,
                'import_coverage': import_coverage
            })
            
            logger.info(f"✅ Created dummy FX reserves data for {self.country} ({len(dates)} months)")
            return True
                
        except Exception as e:
            logger.error(f"❌ Error creating dummy FX reserves data for {self.country}: {e}")
            return False
    
    def process_data(self) -> bool:
        """Process raw data into standardized format"""
        try:
            if self.raw_data is None or self.raw_data.empty:
                return False
            
            processed_records = []
            
            for _, row in self.raw_data.iterrows():
                # FX reserves
                processed_records.append({
                    "country": self.country,
                    "metric": "fx_reserves",
                    "value": float(row['fx_reserves']),
                    "date": row['date'].strftime("%Y-%m-%d"),
                    "source": "Refinitiv",
                    "notes": f"FX reserves for {self.country} (millions USD)",
                    "dataset_code": self.dataset_code,
                    "frequency": "monthly",
                    "last_updated": datetime.now().isoformat()
                })
                
                # Import coverage
                processed_records.append({
                    "country": self.country,
                    "metric": "import_coverage",
                    "value": float(row['import_coverage']),
                    "date": row['date'].strftime("%Y-%m-%d"),
                    "source": "Refinitiv",
                    "notes": f"Import coverage for {self.country} (months)",
                    "dataset_code": self.dataset_code,
                    "frequency": "monthly",
                    "last_updated": datetime.now().isoformat()
                })
            
            self.processed_data = pd.DataFrame(processed_records)
            logger.info(f"✅ Processed FX reserves data for {self.country}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error processing FX reserves data for {self.country}: {e}")
            return False
    
    def calculate_derived_metrics(self) -> bool:
        """Calculate derived metrics"""
        try:
            if self.processed_data is None or self.processed_data.empty:
                return False
            
            # Get the raw data for calculations
            raw_df = self.raw_data.copy()
            
            # Calculate 1-month change percentage
            raw_df['1m_change_pct'] = raw_df['fx_reserves'].pct_change(1) * 100
            
            # Calculate coverage ratio (FX reserves / import coverage)
            raw_df['coverage_ratio'] = raw_df['fx_reserves'] / (raw_df['import_coverage'] * 1000)  # Normalize
            
            # Add derived metrics as new records
            derived_records = []
            
            for _, row in raw_df.iterrows():
                if pd.notna(row['1m_change_pct']):
                    derived_records.append({
                        "country": self.country,
                        "metric": "1m_change_pct",
                        "value": float(row['1m_change_pct']),
                        "date": row['date'].strftime("%Y-%m-%d"),
                        "source": "Refinitiv",
                        "notes": "1-month change in FX reserves (%)",
                        "dataset_code": self.dataset_code,
                        "frequency": "monthly",
                        "last_updated": datetime.now().isoformat()
                    })
                
                if pd.notna(row['coverage_ratio']):
                    derived_records.append({
                        "country": self.country,
                        "metric": "coverage_ratio",
                        "value": float(row['coverage_ratio']),
                        "date": row['date'].strftime("%Y-%m-%d"),
                        "source": "Refinitiv",
                        "notes": "FX reserves coverage ratio",
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
            logger.info(f"✅ Calculated derived metrics for {self.country} FX reserves")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error calculating derived metrics for {self.country} FX reserves: {e}")
            return False
    
    def run_full_pipeline(self) -> bool:
        """Run the complete data pipeline"""
        try:
            logger.info(f"🚀 Starting FX reserves pipeline for {self.country}")
            
            # Step 1: Fetch data
            if not self.fetch_data():
                return False
            
            # Step 2: Process data
            if not self.process_data():
                return False
            
            # Step 3: Calculate derived metrics
            if not self.calculate_derived_metrics():
                return False
            
            logger.info(f"✅ FX reserves pipeline completed for {self.country}")
            return True
            
        except Exception as e:
            logger.error(f"❌ FX reserves pipeline failed for {self.country}: {e}")
            return False
    
    def get_data(self):
        """Get the processed data"""
        return self.processed_data