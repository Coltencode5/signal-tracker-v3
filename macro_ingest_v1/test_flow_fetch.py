from datasets.capital_flows import CapitalFlowsDataset
from datetime import datetime, timedelta

def test_capital_flows():
    try:
        print("🚀 Testing updated Capital Flows Dataset...")
        
        # Create dataset instance for Turkey
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)  # Start with 90 days for testing (monthly data)
        
        capital_flows_dataset = CapitalFlowsDataset(
            country="Turkey",
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d")
        )
        
        print("📊 Running full pipeline...")
        success = capital_flows_dataset.run_full_pipeline()
        
        if success:
            print("✅ Pipeline completed successfully!")
            print(f"📈 Records: {len(capital_flows_dataset.processed_data)}")
            print(f"�� Country: {capital_flows_dataset.country}")
            print(f"🏷️ Country code: {capital_flows_dataset.country_codes.get(capital_flows_dataset.country, 'N/A')}")
            print(f"📊 Working series: {capital_flows_dataset.working_series}")
            print(f"🔄 Data source: {capital_flows_dataset.data_source}")
            print(f"📊 GDP available: {capital_flows_dataset.gdp_available}")
            if capital_flows_dataset.latest_gdp:
                print(f" Latest GDP: {capital_flows_dataset.latest_gdp:,.0f} USD")
            
            # Show sample data
            if capital_flows_dataset.processed_data is not None and not capital_flows_dataset.processed_data.empty:
                print("\n📋 Sample data:")
                print(capital_flows_dataset.processed_data.head(10))
                
                # Show unique series IDs
                if 'series_id' in capital_flows_dataset.processed_data.columns:
                    print(f"\n📊 Available series:")
                    print(capital_flows_dataset.processed_data['series_id'].value_counts())
        else:
            print("❌ Pipeline failed!")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_capital_flows()