#!/usr/bin/env python3
"""
Service Availability Monitor
Provides real-time service availability checking for graceful degradation

Features:
- Real-time service health monitoring
- Capability-based decision making
- Graceful degradation strategies
- Agent behavior adaptation
- Performance tracking
"""

import os
import sys
import json
import time
import logging
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import threading
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from security.api_key_manager import APIKeyManager, check_system_readiness

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CapabilityLevel(Enum):
    """Capability availability levels"""
    FULL = "full"           # All features available
    DEGRADED = "degraded"   # Limited functionality
    MINIMAL = "minimal"     # Basic functionality only
    UNAVAILABLE = "unavailable"  # Feature not available

@dataclass
class ServiceHealth:
    """Health status of a service"""
    service_name: str
    is_available: bool
    last_check: datetime
    response_time: Optional[float] = None
    error_count: int = 0
    last_error: Optional[str] = None
    consecutive_failures: int = 0

@dataclass
class CapabilityStatus:
    """Status of a system capability"""
    name: str
    level: CapabilityLevel
    required_services: List[str]
    available_services: List[str]
    fallback_strategy: Optional[str] = None
    limitations: List[str] = field(default_factory=list)

class ServiceAvailabilityMonitor:
    """Monitor service availability and provide graceful degradation"""
    
    def __init__(self, check_interval: int = 300):
        self.key_manager = APIKeyManager()
        self.check_interval = check_interval  # seconds
        self.service_health: Dict[str, ServiceHealth] = {}
        self._monitoring = False
        self._monitor_thread = None
        
        # Define capability requirements
        self.capability_requirements = {
            "market_analysis": {
                "required_services": ["llm_provider"],
                "optional_services": ["web_search", "database_storage"],
                "fallback_strategies": {
                    "no_web_search": "Use knowledge-based analysis without real-time data",
                    "no_database": "Return results without storage"
                }
            },
            "code_analysis": {
                "required_services": ["github_api", "llm_provider"],
                "optional_services": ["database_storage"],
                "fallback_strategies": {
                    "no_github": "Use local code analysis only",
                    "no_database": "Return analysis without storage"
                }
            },
            "multimodal_analysis": {
                "required_services": ["vision_llm"],
                "optional_services": ["image_apis", "database_storage"],
                "fallback_strategies": {
                    "no_image_apis": "Use vision LLMs only for image analysis",
                    "no_database": "Return analysis without storage"
                }
            },
            "synthesis": {
                "required_services": ["llm_provider"],
                "optional_services": ["database_storage", "market_data"],
                "fallback_strategies": {
                    "no_market_data": "Use cached/historical data for synthesis",
                    "no_database": "Return synthesis without storage"
                }
            },
            "real_time_processing": {
                "required_services": ["redis", "llm_provider"],
                "optional_services": ["web_search"],
                "fallback_strategies": {
                    "no_redis": "Use polling instead of real-time events",
                    "no_web_search": "Process existing data only"
                }
            }
        }
        
        self._initialize_service_health()
    
    def _initialize_service_health(self):
        """Initialize service health tracking"""
        services = [
            "llm_provider", "database_storage", "github_api", "web_search",
            "vision_llm", "image_apis", "redis", "market_data"
        ]
        
        for service in services:
            self.service_health[service] = ServiceHealth(
                service_name=service,
                is_available=False,
                last_check=datetime.now()
            )
        
        # Initial health check
        self.check_all_services()
    
    def check_all_services(self) -> Dict[str, ServiceHealth]:
        """Check health of all services"""
        logger.info("🔍 Checking service availability...")
        
        # Check LLM providers
        llm_providers = self.key_manager.get_available_llm_providers()
        self.service_health["llm_provider"].is_available = len(llm_providers) > 0
        self.service_health["llm_provider"].last_check = datetime.now()
        
        # Check vision LLM (specific providers with vision capabilities)
        vision_providers = [p for p in llm_providers if p in ["openai", "gemini", "openrouter"]]
        self.service_health["vision_llm"].is_available = len(vision_providers) > 0
        self.service_health["vision_llm"].last_check = datetime.now()
        
        # Check other services
        capabilities = self.key_manager.get_system_capabilities()
        
        service_mapping = {
            "database_storage": "database_storage",
            "github_api": "code_analysis", 
            "web_search": "web_search",
            "image_apis": "image_processing"
        }
        
        for service, capability in service_mapping.items():
            if service in self.service_health:
                self.service_health[service].is_available = capabilities.get(capability, False)
                self.service_health[service].last_check = datetime.now()
        
        # Check Redis (simplified - just check if configured)
        redis_host = os.getenv("REDIS_HOST")
        redis_port = os.getenv("REDIS_PORT")
        self.service_health["redis"].is_available = bool(redis_host and redis_port)
        self.service_health["redis"].last_check = datetime.now()
        
        # Market data availability (depends on web search + LLM)
        self.service_health["market_data"].is_available = (
            self.service_health["web_search"].is_available and 
            self.service_health["llm_provider"].is_available
        )
        self.service_health["market_data"].last_check = datetime.now()
        
        return self.service_health
    
    def get_capability_status(self, capability_name: str) -> CapabilityStatus:
        """Get the current status of a system capability"""
        if capability_name not in self.capability_requirements:
            return CapabilityStatus(
                name=capability_name,
                level=CapabilityLevel.UNAVAILABLE,
                required_services=[],
                available_services=[],
                limitations=["Unknown capability"]
            )
        
        config = self.capability_requirements[capability_name]
        required_services = config["required_services"]
        optional_services = config.get("optional_services", [])
        
        # Check required services
        available_required = [
            service for service in required_services
            if self.service_health.get(service, ServiceHealth(service, False, datetime.now())).is_available
        ]
        
        # Check optional services
        available_optional = [
            service for service in optional_services
            if self.service_health.get(service, ServiceHealth(service, False, datetime.now())).is_available
        ]
        
        # Determine capability level
        if len(available_required) == len(required_services):
            if len(available_optional) == len(optional_services):
                level = CapabilityLevel.FULL
            elif len(available_optional) > 0:
                level = CapabilityLevel.DEGRADED
            else:
                level = CapabilityLevel.MINIMAL
        else:
            level = CapabilityLevel.UNAVAILABLE
        
        # Determine limitations and fallback strategy
        limitations = []
        fallback_strategy = None
        
        missing_required = set(required_services) - set(available_required)
        missing_optional = set(optional_services) - set(available_optional)
        
        if missing_required:
            limitations.extend([f"Missing required: {service}" for service in missing_required])
        
        if missing_optional:
            limitations.extend([f"Missing optional: {service}" for service in missing_optional])
            
            # Get fallback strategy
            fallback_strategies = config.get("fallback_strategies", {})
            for missing_service in missing_optional:
                strategy_key = f"no_{missing_service}"
                if strategy_key in fallback_strategies:
                    fallback_strategy = fallback_strategies[strategy_key]
                    break
        
        return CapabilityStatus(
            name=capability_name,
            level=level,
            required_services=required_services,
            available_services=available_required + available_optional,
            fallback_strategy=fallback_strategy,
            limitations=limitations
        )
    
    def can_perform_capability(self, capability_name: str) -> bool:
        """Check if a capability can be performed (at least minimally)"""
        status = self.get_capability_status(capability_name)
        return status.level != CapabilityLevel.UNAVAILABLE
    
    def get_recommended_action(self, capability_name: str) -> Dict[str, Any]:
        """Get recommended action for using a capability"""
        status = self.get_capability_status(capability_name)
        
        if status.level == CapabilityLevel.UNAVAILABLE:
            return {
                "action": "abort",
                "reason": "Required services unavailable",
                "missing_services": [s for s in status.required_services if s not in status.available_services],
                "recommendation": "Check API keys and service configuration"
            }
        
        elif status.level == CapabilityLevel.MINIMAL:
            return {
                "action": "proceed_with_caution",
                "reason": "Limited functionality available",
                "limitations": status.limitations,
                "fallback_strategy": status.fallback_strategy,
                "recommendation": "Use basic functionality only"
            }
        
        elif status.level == CapabilityLevel.DEGRADED:
            return {
                "action": "proceed_with_fallback",
                "reason": "Some optional services unavailable",
                "limitations": status.limitations,
                "fallback_strategy": status.fallback_strategy,
                "recommendation": "Use fallback strategies for missing services"
            }
        
        else:  # FULL
            return {
                "action": "proceed_normally",
                "reason": "All services available",
                "recommendation": "Full functionality available"
            }
    
    def get_system_overview(self) -> Dict[str, Any]:
        """Get comprehensive system status overview"""
        self.check_all_services()  # Refresh status
        
        capabilities = {}
        for capability_name in self.capability_requirements.keys():
            status = self.get_capability_status(capability_name)
            capabilities[capability_name] = {
                "level": status.level.value,
                "available_services": status.available_services,
                "limitations": status.limitations,
                "fallback_strategy": status.fallback_strategy
            }
        
        # Count services by availability
        total_services = len(self.service_health)
        available_services = sum(1 for health in self.service_health.values() if health.is_available)
        
        # Determine overall system health
        system_ready = check_system_readiness()
        critical_capabilities = ["market_analysis", "synthesis"]
        critical_available = sum(
            1 for cap in critical_capabilities 
            if self.can_perform_capability(cap)
        )
        
        if not system_ready:
            overall_health = "CRITICAL"
        elif critical_available < len(critical_capabilities):
            overall_health = "DEGRADED"
        elif available_services < total_services * 0.8:
            overall_health = "LIMITED"
        else:
            overall_health = "HEALTHY"
        
        return {
            "overall_health": overall_health,
            "system_ready": system_ready,
            "timestamp": datetime.now().isoformat(),
            "services": {
                "total": total_services,
                "available": available_services,
                "percentage": round((available_services / total_services) * 100, 1)
            },
            "capabilities": capabilities,
            "service_details": {
                name: {
                    "available": health.is_available,
                    "last_check": health.last_check.isoformat(),
                    "error_count": health.error_count
                }
                for name, health in self.service_health.items()
            }
        }
    
    def print_status_report(self):
        """Print a comprehensive status report"""
        overview = self.get_system_overview()
        
        print("\\n🔍 Service Availability Monitor Report")
        print("=" * 60)
        
        # Overall health
        health_emoji = {
            "HEALTHY": "🟢",
            "LIMITED": "🟡", 
            "DEGRADED": "🟠",
            "CRITICAL": "🔴"
        }.get(overview["overall_health"], "❓")
        
        print(f"\\n{health_emoji} Overall Health: {overview['overall_health']}")
        print(f"🏛️ System Ready: {'✅' if overview['system_ready'] else '❌'}")
        print(f"⚙️ Services Available: {overview['services']['available']}/{overview['services']['total']} ({overview['services']['percentage']}%)")
        
        # Capabilities status
        print("\\n📊 Capability Status:")
        for cap_name, cap_info in overview["capabilities"].items():
            level_emoji = {
                "full": "🟢",
                "degraded": "🟡",
                "minimal": "🟠",
                "unavailable": "🔴"
            }.get(cap_info["level"], "❓")
            
            print(f"  {level_emoji} {cap_name.replace('_', ' ').title()}: {cap_info['level'].title()}")
            
            if cap_info["limitations"]:
                print(f"    ⚠️ Limitations: {', '.join(cap_info['limitations'])}")
            
            if cap_info["fallback_strategy"]:
                print(f"    🔄 Fallback: {cap_info['fallback_strategy']}")
        
        # Service details
        print("\\n🔧 Service Details:")
        for service_name, service_info in overview["service_details"].items():
            status_emoji = "✅" if service_info["available"] else "❌"
            print(f"  {status_emoji} {service_name.replace('_', ' ').title()}")
            
            if service_info["error_count"] > 0:
                print(f"    🐛 Errors: {service_info['error_count']}")

# Global monitor instance
_monitor = ServiceAvailabilityMonitor()

# Convenience functions for agents
def can_perform(capability: str) -> bool:
    """Check if a capability can be performed"""
    return _monitor.can_perform_capability(capability)

def get_action_plan(capability: str) -> Dict[str, Any]:
    """Get recommended action plan for a capability"""
    return _monitor.get_recommended_action(capability)

def check_service_health() -> Dict[str, Any]:
    """Get current service health overview"""
    return _monitor.get_system_overview()

def require_capability(capability: str) -> bool:
    """Decorator/check function that ensures a capability is available"""
    if not can_perform(capability):
        action_plan = get_action_plan(capability)
        logger.error(f"❌ Capability '{capability}' unavailable: {action_plan['reason']}")
        return False
    return True

# Testing and demonstration
if __name__ == "__main__":
    print("🔍 Running Service Availability Monitor Test")
    print("=" * 70)
    
    monitor = ServiceAvailabilityMonitor()
    
    # Print comprehensive report
    monitor.print_status_report()
    
    # Test capability checking
    print("\\n🧪 Capability Testing:")
    test_capabilities = ["market_analysis", "code_analysis", "multimodal_analysis", "real_time_processing"]
    
    for capability in test_capabilities:
        can_do = can_perform(capability)
        action_plan = get_action_plan(capability)
        print(f"\\n  📋 {capability.replace('_', ' ').title()}:")
        print(f"    Can Perform: {'✅' if can_do else '❌'}")
        print(f"    Action: {action_plan['action']}")
        print(f"    Reason: {action_plan['reason']}")
        
        if 'recommendation' in action_plan:
            print(f"    💡 Recommendation: {action_plan['recommendation']}")
    
    print("\\n🎉 Test Complete!")
