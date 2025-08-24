from datasets.trade_data import TradeBalanceDataset
from datetime import datetime, timedelta

def test_trade_debug():
    try:
        print("🚀 Testing Trade Balance Dataset with Debug Info...")
        
        # Create dataset instance for Turkey
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        
        trade_dataset = TradeBalanceDataset(
            country="Turkey",
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d")
        )
        
        print(f"🌍 Country: {trade_dataset.country}")
        print(f"🏷️ Country code: {trade_dataset.country_codes.get(trade_dataset.country, 'N/A')}")
        print(f"�� Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Step 1: Test data fetching
        print("\n📊 Step 1: Testing data fetch...")
        fetch_success = trade_dataset.fetch_data()
        print(f"Fetch result: {fetch_success}")
        
        if fetch_success:
            print(f"✅ Raw data shape: {trade_dataset.raw_data.shape if trade_dataset.raw_data is not None else 'None'}")
            print(f"📊 Working series: {trade_dataset.working_series}")
            print(f"🔄 Data source: {trade_dataset.data_source}")
            print(f"🔄 Fallback used: {trade_dataset.fallback_used}")
            
            if trade_dataset.raw_data is not None and not trade_dataset.raw_data.empty:
                print(f"📋 Raw data columns: {list(trade_dataset.raw_data.columns)}")
                print(f"📋 Raw data sample:")
                print(trade_dataset.raw_data.head())
        else:
            print("❌ Data fetch failed!")
            return
        
        # Step 2: Test data processing
        print("\n📊 Step 2: Testing data processing...")
        process_success = trade_dataset.process_data()
        print(f"Process result: {process_success}")
        
        if process_success:
            print(f"✅ Processed data shape: {trade_dataset.processed_data.shape if trade_dataset.processed_data is not None else 'None'}")
            
            if trade_dataset.processed_data is not None and not trade_dataset.processed_data.empty:
                print(f"📋 Processed data columns: {list(trade_dataset.processed_data.columns)}")
                print(f"📋 Processed data sample:")
                print(trade_dataset.processed_data.head())
                
                # Check what's in ric_or_series column
                if 'ric_or_series' in trade_dataset.processed_data.columns:
                    print(f"📊 Unique ric_or_series values:")
                    print(trade_dataset.processed_data['ric_or_series'].value_counts())
                else:
                    print("⚠️ No 'ric_or_series' column found!")
        else:
            print("❌ Data processing failed!")
            return
        
        # Step 3: Test derived metrics
        print("\n📊 Step 3: Testing derived metrics...")
        metrics_success = trade_dataset.calculate_derived_metrics()
        print(f"Derived metrics result: {metrics_success}")
        
        if metrics_success:
            print(f"✅ Final data shape: {trade_dataset.processed_data.shape if trade_dataset.processed_data is not None else 'None'}")
            print(f"📊 GDP available: {trade_dataset.gdp_available}")
            if trade_dataset.latest_gdp:
                print(f" Latest GDP: {trade_dataset.latest_gdp:,.0f} USD")
        else:
            print("❌ Derived metrics failed!")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_trade_debug()