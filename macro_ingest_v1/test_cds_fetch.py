from datasets.sovereign_risk import SovereignCDSDataset
from datetime import datetime, timedelta

def test_sovereign_risk():
    try:
        print("🚀 Testing updated Sovereign Risk Dataset...")
        
        # Create dataset instance for Turkey
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)  # Start with 30 days for testing
        
        cds_dataset = SovereignCDSDataset(
            country="Turkey",
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d")
        )
        
        print("📊 Running full pipeline...")
        success = cds_dataset.run_full_pipeline()
        
        if success:
            print("✅ Pipeline completed successfully!")
            
            # Get the data
            data = cds_dataset.get_data()
            print(f"📈 Data shape: {data.shape}")
            print("\n�� First few rows:")
            print(data.head())
            
            # Check if we have the right columns
            print(f"\n🔍 Columns: {list(data.columns)}")
            
        else:
            print("❌ Pipeline failed!")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_sovereign_risk()