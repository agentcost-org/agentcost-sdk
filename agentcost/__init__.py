"""
AgentCost SDK

Track LLM costs in your LangChain applications with zero code changes.

Usage:
    from agentcost import track_costs
    
    ## Initialize tracking
    track_costs.init(
        api_key="your_api_key",
        project_id="your_project_id"
    )
    
    ## Your LangChain code runs normally - costs are tracked automatically!
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4")
    response = llm.invoke("Hello!")
    
    ## For local testing (no backend required)
    track_costs.init(local_mode=True)
    # ... run your code ...
    events = track_costs.get_local_events()  # See all captured events

For more information, visit: https://agentcost.dev
"""

from pathlib import Path


def _get_version() -> str:
    """Read version from VERSION file (single source of truth)."""
    possible_paths = [
        Path(__file__).parent.parent.parent / "VERSION",  # Project root
        Path(__file__).parent.parent / "VERSION",
        Path("VERSION"),
    ]
    
    for path in possible_paths:
        if path.exists():
            return path.read_text().strip()
    
    return "0.1.0"  # Default fallback


__version__ = _get_version()
__author__ = "AgentCost"

# Import main tracker module as track_costs
from . import tracker as track_costs

# Also expose key functions at package level for convenience
from .tracker import (
    init,
    shutdown,
    flush,
    get_stats,
    get_local_events,
    set_agent_name,
    add_metadata,
    session,
    agent,
    metadata,
    AgentCostTracker,
)

# Expose configuration
from .config import AgentCostConfig, DEFAULT_PRICING

# Expose components for advanced usage
from .token_counter import TokenCounter
from .cost_calculator import (
    CostCalculator,
    calculate_cost,
    get_pricing_manager,
    refresh_pricing,
    update_pricing,
)
from .batcher import HybridBatcher, LocalBatcher
from .http_client import AgentCostHTTPClient, MockHTTPClient
from .interceptor import LangChainInterceptor

__all__ = [
    # Version
    "__version__",
    
    # Main module
    "track_costs",
    
    # Tracker functions
    "init",
    "shutdown", 
    "flush",
    "get_stats",
    "get_local_events",
    "set_agent_name",
    "add_metadata",
    "session",
    "agent",
    "metadata",
    "AgentCostTracker",
    
    # Configuration
    "AgentCostConfig",
    "DEFAULT_PRICING",
    
    # Components (for advanced usage)
    "TokenCounter",
    "CostCalculator",
    "calculate_cost",
    "get_pricing_manager",
    "refresh_pricing",
    "update_pricing",
    "HybridBatcher",
    "LocalBatcher",
    "AgentCostHTTPClient",
    "MockHTTPClient",
    "LangChainInterceptor",
]
