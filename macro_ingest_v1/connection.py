"""
Refinitiv Workspace connection management
Reuses the same authentication method from your news_fetcher.py
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

class RefinitivConnection:
    """Manages Refinitiv Workspace connection for macro data fetching"""
    
    def __init__(self):
        self.session_active = False
        self._rd = None
        
    def connect(self) -> bool:
        """Establish connection to Refinitiv Workspace"""
        try:
            import refinitiv.data as rd
            
            logger.info("🔐 Attempting to authenticate with Refinitiv Workspace...")
            rd.open_session()
            self._rd = rd
            self.session_active = True
            logger.info("✅ Successfully authenticated with Refinitiv Workspace")
            return True
            
        except ImportError as e:
            logger.error(f"❌ Failed to import refinitiv.data SDK: {e}")
            logger.error("Please install the Refinitiv Data SDK: pip install refinitiv-data")
            return False
        except Exception as e:
            logger.error(f"❌ Authentication failed: {e}")
            self.session_active = False
            return False
    
    def disconnect(self):
        """Close Refinitiv Workspace session"""
        try:
            if self._rd and self.session_active:
                self._rd.close_session()
                self.session_active = False
                logger.info("🔒 Workspace session closed successfully")
        except Exception as e:
            logger.error(f"❌ Error closing Workspace session: {e}")
    
    def is_connected(self) -> bool:
        """Check if connection is active"""
        return self.session_active and self._rd is not None
    
    def get_session(self):
        """Get the active refinitiv.data session"""
        if not self.is_connected():
            raise ConnectionError("No active Refinitiv session. Call connect() first.")
        return self._rd

# Global connection instance
refinitiv_conn = RefinitivConnection()

def get_refinitiv_session():
    """Get the global Refinitiv session"""
    return refinitiv_conn.get_session()

def validate_connection() -> bool:
    """Validate that Refinitiv connection can be established"""
    try:
        if refinitiv_conn.connect():
            refinitiv_conn.disconnect()
            return True
        return False
    except Exception as e:
        logger.error(f"❌ Connection validation failed: {e}")
        return False