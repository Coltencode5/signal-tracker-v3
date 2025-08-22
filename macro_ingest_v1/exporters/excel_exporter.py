"""
Excel export functionality for Macro Ingest Module
"""

import pandas as pd
import logging
from pathlib import Path
from typing import List
from ..config import SHEET_MAPPINGS

logger = logging.getLogger(__name__)

class ExcelExporter:
    """Handles export of macro datasets to Excel format"""
    
    def __init__(self, output_file: str):
        self.output_file = Path(output_file)
        
    def export_datasets(self, datasets: List) -> bool:
        """Export all datasets to Excel workbook"""
        try:
            logger.info(f" Exporting {len(datasets)} datasets to {self.output_file}")
            
            # Group datasets by sheet
            sheet_data = self._group_datasets_by_sheet(datasets)
            
            # Create Excel writer
            with pd.ExcelWriter(self.output_file, engine='openpyxl') as writer:
                # Export each sheet
                for sheet_name, data_list in sheet_data.items():
                    if data_list:
                        self._export_sheet(writer, sheet_name, data_list)
                
                # Create summary sheet
                self._create_summary_sheet(writer, datasets)
            
            logger.info(f"✅ Successfully exported data to {self.output_file}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Export failed: {e}")
            return False
    
    def _group_datasets_by_sheet(self, datasets: List) -> dict:
        """Group datasets by their corresponding Excel sheet"""
        sheet_data = {}
        
        for dataset in datasets:
            sheet_name = SHEET_MAPPINGS.get(dataset.dataset_code, "Other")
            
            if sheet_name not in sheet_data:
                sheet_data[sheet_name] = []
            
            if dataset.processed_data is not None and not dataset.processed_data.empty:
                sheet_data[sheet_name].append(dataset.processed_data)
        
        return sheet_data
    
    def _export_sheet(self, writer, sheet_name: str, data_list: List):
        """Export a single sheet with multiple datasets"""
        try:
            # Combine all datasets for this sheet
            combined_data = pd.concat(data_list, ignore_index=True)
            
            # Handle different column formats for different sheets
            if sheet_name == "CDS":
                # For CDS sheet, sort by country, date, and ric
                if 'country' in combined_data.columns and 'date' in combined_data.columns:
                    combined_data = combined_data.sort_values(['country', 'date', 'ric'])
            else:
                # For other sheets, sort by country, date, and metric
                if 'country' in combined_data.columns and 'date' in combined_data.columns and 'metric' in combined_data.columns:
                    combined_data = combined_data.sort_values(['country', 'date', 'metric'])
            
            # Export to sheet
            combined_data.to_excel(
                writer, 
                sheet_name=sheet_name, 
                index=False,
                float_format='%.4f'
            )
            
            logger.info(f"✅ Exported sheet: {sheet_name} ({len(combined_data)} records)")
            
        except Exception as e:
            logger.error(f"❌ Error exporting sheet {sheet_name}: {e}")
    
    def _create_summary_sheet(self, writer, datasets: List):
        """Create a summary sheet with metadata"""
        try:
            summary_data = []
            
            for dataset in datasets:
                if dataset.processed_data is not None and not dataset.processed_data.empty:
                    # Handle different column formats
                    if 'metric' in dataset.processed_data.columns:
                        metrics = list(dataset.processed_data['metric'].unique())
                    elif 'ric' in dataset.processed_data.columns:
                        metrics = [f"CDS_{dataset.processed_data['ric'].iloc[0]}"]
                    else:
                        metrics = ["Unknown"]
                    
                    summary_data.append({
                        "dataset_code": dataset.dataset_code,
                        "country": dataset.country,
                        "record_count": len(dataset.processed_data),
                        "metrics": metrics,
                        "date_range": f"{dataset.processed_data['date'].min()} to {dataset.processed_data['date'].max()}",
                        "last_updated": pd.Timestamp.now().isoformat()
                    })
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name="Summary", index=False)
                logger.info(f"✅ Created summary sheet")
            
        except Exception as e:
            logger.error(f"❌ Error creating summary sheet: {e}")