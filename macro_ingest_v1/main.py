"""
Main program for Macro Ingest Module v1
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Import our modules
from .config import INITIAL_COUNTRIES, DEFAULT_OUTPUT_FILE
from .datasets.sovereign_risk import SovereignCDSDataset
from .datasets.capital_flows import CapitalFlowsDataset
from .datasets.monetary_data import MonetaryDataset
from .datasets.fx_reserves import FXReservesDataset
from .datasets.inflation_data import InflationDataset
from .datasets.trade_data import TradeBalanceDataset
from .exporters.excel_exporter import ExcelExporter

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Macro Ingest Module v1 - Fetch and process macroeconomic data",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--countries",
        nargs="+",
        default=INITIAL_COUNTRIES,
        help=f"Countries to fetch data for (default: {', '.join(INITIAL_COUNTRIES)})"
    )
    
    parser.add_argument(
        "--start",
        default=(datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
        help="Start date in YYYY-MM-DD format (default: 30 days ago)"
    )
    
    parser.add_argument(
        "--end", 
        default=datetime.now().strftime("%Y-%m-%d"),
        help="End date in YYYY-MM-DD format (default: today)"
    )
    
    parser.add_argument(
        "--excel",
        default=DEFAULT_OUTPUT_FILE,
        help=f"Output Excel file name (default: {DEFAULT_OUTPUT_FILE})"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()

def run_pipeline(countries: list, start_date: str, end_date: str, output_file: str) -> bool:
    """Run the complete data pipeline"""
    logger.info(f"🚀 Starting macro data pipeline")
    logger.info(f"🌍 Countries: {', '.join(countries)}")
    logger.info(f"�� Date range: {start_date} to {end_date}")
    logger.info(f"📁 Output: {output_file}")
    
    # Create datasets for each country
    datasets = []
    for country in countries:
        # Dataset A1: CDS spreads
        cds_dataset = SovereignCDSDataset(country, start_date, end_date)
        datasets.append(cds_dataset)
        
        
        # Dataset B1: Capital flows
        capital_flows_dataset = CapitalFlowsDataset(country, start_date, end_date)
        datasets.append(capital_flows_dataset)
        
        # Dataset B2: Monetary data
        monetary_dataset = MonetaryDataset(country, start_date, end_date)
        datasets.append(monetary_dataset)
        
        # Dataset C1: FX reserves
        fx_reserves_dataset = FXReservesDataset(country, start_date, end_date)
        datasets.append(fx_reserves_dataset)
        
        # Dataset D1: CPI inflation
        inflation_dataset = InflationDataset(country, start_date, end_date)
        datasets.append(inflation_dataset)
        
        # Dataset D2: Trade balance
        trade_balance_dataset = TradeBalanceDataset(country, start_date, end_date)
        datasets.append(trade_balance_dataset)
    
    # Run pipeline for each dataset
    successful_datasets = []
    for dataset in datasets:
        try:
            if dataset.run_full_pipeline():
                successful_datasets.append(dataset)
                logger.info(f"✅ {dataset.dataset_code} - {dataset.country}: Success")
            else:
                logger.error(f"❌ {dataset.dataset_code} - {dataset.country}: Failed")
        except Exception as e:
            logger.error(f"❌ {dataset.dataset_code} - {dataset.country}: Exception - {e}")
    
    if not successful_datasets:
        logger.error("❌ No datasets completed successfully")
        return False
    
    # Export to Excel
    try:
        exporter = ExcelExporter(output_file)
        success = exporter.export_datasets(successful_datasets)
        
        if success:
            logger.info(f"✅ Data exported successfully to {output_file}")
            return True
        else:
            logger.error(f"❌ Failed to export data to {output_file}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Export failed: {e}")
        return False

def main():
    """Main entry point"""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Set verbose logging if requested
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        logger.info("🌍 Macro Ingest Module v1 Starting...")
        
        # Run pipeline
        success = run_pipeline(
            args.countries, 
            args.start, 
            args.end, 
            args.excel
        )
        
        if success:
            logger.info("🎉 Macro data pipeline completed successfully!")
            sys.exit(0)
        else:
            logger.error("❌ Macro data pipeline failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("⏹️ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()