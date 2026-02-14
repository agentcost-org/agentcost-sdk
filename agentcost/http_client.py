"""
AgentCost HTTP Client

Handles communication with the AgentCost backend API.
Features retry logic, timeouts, rate limiting, and error handling.
"""

import requests
import time
import threading
from typing import List, Dict, Optional
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class RateLimiter:
    """Simple rate limiter to prevent overwhelming the backend"""
    
    def __init__(self, max_requests: int = 10, window_seconds: float = 1.0):
        """
        Args:
            max_requests: Maximum requests per window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests = []
        self._lock = threading.Lock()
    
    def acquire(self) -> float:
        """
        Try to acquire a request slot.
        
        Returns:
            Time to wait before making request (0 if no wait needed)
        """
        with self._lock:
            now = time.time()
            
            # Remove old requests outside the window
            self._requests = [t for t in self._requests if now - t < self.window_seconds]
            
            if len(self._requests) < self.max_requests:
                self._requests.append(now)
                return 0.0
            else:
                # Calculate wait time until oldest request expires
                oldest = min(self._requests)
                wait_time = self.window_seconds - (now - oldest)
                return max(0, wait_time)
    
    def wait_and_acquire(self) -> None:
        """Wait if necessary, then acquire a slot"""
        wait_time = self.acquire()
        if wait_time > 0:
            time.sleep(wait_time)
            self.acquire()


class AgentCostHTTPClient:
    """HTTP client for sending telemetry to AgentCost backend"""
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "",
        timeout: float = 10.0,
        max_retries: int = 3,
        debug: bool = False
    ):
        """
        Args:
            api_key: User's AgentCost API key
            base_url: Backend API URL (default: AGENTCOST_API_URL env var or https://api.agentcost.tech)
            timeout: Request timeout in seconds
            max_retries: Number of retry attempts
            debug: Enable debug logging
        """
        self.api_key = api_key
        if not base_url:
            import os
            base_url = os.getenv("AGENTCOST_API_URL", "https://api.agentcost.tech")
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.debug = debug
        self._closed = False
        
        # Rate limiter (10 requests per second max)
        self._rate_limiter = RateLimiter(max_requests=10, window_seconds=1.0)
        
        # Create session with retry logic
        self.session = self._create_session(max_retries)
    
    def _create_session(self, max_retries: int) -> requests.Session:
        """Create requests session with retry logic"""
        session = requests.Session()
        
        # Retry strategy
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,  # Wait 1s, 2s, 4s between retries
            status_forcelist=[429, 500, 502, 503, 504],  # Retry these HTTP codes
            allowed_methods=["POST", "GET"]  # Updated from deprecated method_whitelist
        )
        
        # Mount adapter with retry strategy
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def send_events(self, project_id: str, events: List[Dict]) -> bool:
        """
        Send batch of events to backend
        
        Args:
            project_id: User's project ID
            events: List of event dictionaries
        
        Returns:
            True if successful, False otherwise
        """
        # Apply rate limiting
        self._rate_limiter.wait_and_acquire()
        
        url = f"{self.base_url}/v1/events/batch"
        
        from . import __version__
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'User-Agent': f'AgentCost-SDK/{__version__}',
            'X-AgentCost-SDK-Version': __version__,
        }
        
        payload = {
            'project_id': project_id,
            'events': events
        }
        
        if self.debug:
            print(f"[AgentCost] Sending {len(events)} events to {url}")
        
        try:
            response = self.session.post(
                url,
                json=payload,
                headers=headers,
                timeout=self.timeout
            )
            
            # Check response
            response.raise_for_status()  # Raises exception for 4xx/5xx
            
            # Parse response
            data = response.json()
            
            if data.get('status') == 'ok':
                if self.debug:
                    print(f"[AgentCost] Sent {len(events)} events successfully")
                return True
            else:
                if self.debug:
                    print(f"[AgentCost] Error: Backend returned error: {data}")
                return False
        
        except requests.exceptions.Timeout:
            if self.debug:
                print(f"[AgentCost] Error: Request timed out after {self.timeout}s")
            return False
        
        except requests.exceptions.ConnectionError as e:
            if self.debug:
                print(f"[AgentCost] Error: Connection error: {e}")
            return False
        
        except requests.exceptions.HTTPError as e:
            if self.debug:
                status = e.response.status_code if e.response else "unknown"
                text = e.response.text if e.response else str(e)
                print(f"[AgentCost] Error: HTTP error: {status} - {text}")
            return False
        
        except Exception as e:
            if self.debug:
                print(f"[AgentCost] Error: Unexpected error: {e}")
            return False
    
    def test_connection(self) -> bool:
        """Test if backend is reachable"""
        url = f"{self.base_url}/v1/health"
        
        try:
            response = self.session.get(url, timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def get_project_info(self, project_id: str) -> Optional[Dict]:
        """
        Get project information from backend
        
        Args:
            project_id - Project ID
            
        Returns:
            Project info dict or None if failed
        """
        url = f"{self.base_url}/v1/projects/{project_id}"
        
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }
        
        try:
            response = self.session.get(url, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            if self.debug:
                print(f"[AgentCost] Failed to get project info: {e}")
            return None
    
    def close(self) -> None:
        """Close the session and release resources"""
        if not self._closed:
            self.session.close()
            self._closed = True
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures session is closed"""
        self.close()
        return False
    
    def __del__(self):
        """Destructor - cleanup on garbage collection"""
        try:
            self.close()
        except Exception:
            pass  # Ignore errors during cleanup


class MockHTTPClient:
    """
    Mock HTTP client for testing and offline development.
    Stores events locally instead of sending to backend.
    """
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.sent_events: List[Dict] = []
        self.send_count = 0
    
    def send_events(self, project_id: str, events: List[Dict]) -> bool:
        """Store events locally"""
        self.sent_events.extend(events)
        self.send_count += 1
        
        if self.debug:
            print(f"[AgentCost Mock] Stored {len(events)} events (total: {len(self.sent_events)})")
        
        return True
    
    def test_connection(self) -> bool:
        """Always returns True"""
        return True
    
    def get_all_events(self) -> List[Dict]:
        """Get all stored events"""
        return self.sent_events.copy()
    
    def clear(self) -> None:
        """Clear stored events"""
        self.sent_events = []
        self.send_count = 0
    
    def close(self) -> None:
        """No-op for mock client"""
        pass
