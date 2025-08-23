from datasets.monetary_data import MonetaryDataset
from datetime import datetime, timedelta

def test_monetary_data():
    try:
        print("🚀 Testing updated Monetary Dataset...")
        
        # Create dataset instance for Turkey
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)  # Start with 90 days for testing (monthly data)
        
        monetary_dataset = MonetaryDataset(
            country="Turkey",
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d")
        )
        
        print("📊 Running full pipeline...")
        success = monetary_dataset.run_full_pipeline()
        
        if success:
            print("✅ Pipeline completed successfully!")
            
            # Get the data
            data = monetary_dataset.get_data()
            print(f"📈 Data shape: {data.shape}")
            
            # Show summary by metric type
            print("\n🔍 Data summary by metric:")
            for metric in data['metric'].unique():
                metric_data = data[data['metric'] == metric]
                print(f"  {metric}: {len(metric_data)} records")
            
            print("\n�� First few rows:")
            print(data.head(10))
            
            # Check if we have the right columns
            print(f"\n🔍 Columns: {list(data.columns)}")
            
            # Show some sample values
            print("\n💰 Sample M2 values:")
            m2_data = data[data['metric'] == 'M2_LEVEL_TRY']
            if not m2_data.empty:
                print(m2_data[['date', 'value', 'notes']].head())
            
            # Show derived metrics
            print("\n📈 Sample derived metrics:")
            derived_data = data[data['metric'] == 'M2_YOY_PCT']
            if not derived_data.empty:
                print(derived_data[['date', 'metric', 'value', 'notes']].head())
            
            # Log which RIC was used
            print(f"\n🔗 RIC used: {monetary_dataset.working_ric}")
            print(f"📡 Data source: {monetary_dataset.data_source}")
            print(f"🔄 Fallback used: {monetary_dataset.fallback_used}")
            
        else:
            print("❌ Pipeline failed!")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_monetary_data()