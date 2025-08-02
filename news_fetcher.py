import os
import sys
import logging
from datetime import datetime
from typing import List, Dict, Optional

# === CONSTANTS ===
DEFAULT_QUERY = "protest OR strike OR coup OR military OR default OR cyberattack"
DEFAULT_COUNT = 10

# === LOGGING SETUP ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_workspace_session():
    """
    Set up and authenticate with Refinitiv Workspace session using refinitiv.data SDK.
    
    Returns:
        bool: True if session established successfully, False otherwise
    """
    try:
        import refinitiv.data as rd
        
        logger.info("🔐 Attempting to authenticate with Refinitiv Workspace...")
        
        # Open session using refinitiv.data SDK
        rd.open_session()
        
        logger.info("✅ Successfully authenticated with Refinitiv Workspace")
        return True
        
    except ImportError as e:
        logger.error(f"❌ Failed to import refinitiv.data SDK: {e}")
        logger.error("Please install the Refinitiv Data SDK: pip install refinitiv-data")
        return False
    except Exception as e:
        logger.error(f"❌ Authentication failed: {e}")
        return False

def fetch_lseg_articles(query: str = DEFAULT_QUERY, count: int = DEFAULT_COUNT) -> List[Dict]:
    """
    Fetch articles from Refinitiv Workspace using the provided query.
    
    Args:
        query (str): Search query for articles (default: DEFAULT_QUERY)
        count (int): Number of articles to fetch (default: DEFAULT_COUNT)
    
    Returns:
        List[Dict]: List of article dictionaries with 'headline', 'timestamp', 'sourceCode'
    """
    try:
        import refinitiv.data as rd
        
        # Set up Workspace session
        if not setup_workspace_session():
            logger.error("❌ Failed to establish Workspace session")
            return []
        
        logger.info(f"🔍 Fetching {count} articles with query: '{query}'")
        
        # Create news headlines definition using refinitiv.data SDK
        news_def = rd.content.news.headlines.Definition(
            query=query,
            count=count
        )
        
        # Fetch the data
        response = news_def.get_data()
        df = response.data.df  # extract the actual DataFrame
        
        if df is None or df.empty:
            logger.warning("⚠️ No articles found for the given query")
            return []
        
        # Convert DataFrame to list of dictionaries
        articles = []
        for _, row in df.iterrows():
            article = {
                'headline': row.get('headline', ''),
                'timestamp': row.name.isoformat(),
                'sourceCode': row.get('sourceCode', '')
            }
            articles.append(article)
        
        logger.info(f"✅ Successfully fetched {len(articles)} articles")
        for i, article in enumerate(articles, 1):
            logger.debug(f"  {i}. {article['headline']} ({article['sourceCode']})")
            
        return articles
        
    except Exception as e:
        logger.error(f"❌ Error fetching articles: {e}")
        return []
        
    finally:
        # Ensure session is properly closed
        try:
            import refinitiv.data as rd
            rd.close_session()
            logger.info("🔒 Workspace session closed successfully")
        except Exception as e:
            logger.error(f"❌ Error closing Workspace session: {e}")

def validate_workspace_connection() -> bool:
    """
    Validate that Workspace connection can be established.
    
    Returns:
        bool: True if connection successful, False otherwise
    """
    try:
        import refinitiv.data as rd
        
        logger.info("🔐 Testing Workspace connection...")
        rd.open_session()
        rd.close_session()
        logger.info("✅ Workspace connection validation successful")
        return True
    except Exception as e:
        logger.error(f"❌ Workspace connection validation error: {e}")
        return False

def get_available_sources() -> List[str]:
    """
    Get list of available news sources from Workspace.
    
    Returns:
        List[str]: List of available source codes
    """
    try:
        import refinitiv.data as rd
        
        # Set up session
        if not setup_workspace_session():
            return []
        
        # Fetch available sources using the SDK
        # Note: This is a placeholder - actual implementation depends on available SDK methods
        sources_def = rd.content.news.sources.Definition()
        response = sources_def.get_data()
        
        if response is not None and not response.empty:
            sources = response.get('sourceCode', []).tolist()
        else:
            # Fallback to common sources if API doesn't provide them
            sources = [
                'REUTERS',
                'BLOOMBERG', 
                'CNBC',
                'FINANCIAL_TIMES',
                'WALL_STREET_JOURNAL',
                'ASSOCIATED_PRESS'
            ]
        
        return sources
        
    except Exception as e:
        logger.error(f"❌ Error fetching available sources: {e}")
        return []
    finally:
        try:
            import refinitiv.data as rd
            rd.close_session()
        except:
            pass

if __name__ == "__main__":
    # === TEST THE MODULE ===
    logger.info("🧪 Testing news_fetcher module...")
    
    # Test connection
    if validate_workspace_connection():
        # Test article fetching
        test_articles = fetch_lseg_articles(
            query="protest OR strike",
            count=3
        )
        
        print(f"\n📰 Retrieved {len(test_articles)} test articles:")
        for i, article in enumerate(test_articles, 1):
            print(f"  {i}. {article['headline']}")
            print(f"     Source: {article['sourceCode']}")
            print(f"     Time: {article['timestamp']}")
            print()
    else:
        print("❌ Workspace connection test failed")