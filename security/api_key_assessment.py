#!/usr/bin/env python3
"""
API Key Assessment and Management System
Analyzes present API keys and provides logic for missing ones

Features:
- Comprehensive API key validation
- Service availability assessment  
- Graceful degradation strategies
- Provider priority management
- Configuration recommendations
"""

import os
import sys
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import requests
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from security.api_key_manager import load_environment

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ServiceStatus(Enum):
    """Service availability status"""
    AVAILABLE = "available"
    CONFIGURED = "configured"  # Key present but not tested
    MISSING = "missing"
    INVALID = "invalid" 
    RATE_LIMITED = "rate_limited"
    ERROR = "error"

class ServicePriority(Enum):
    """Service priority levels"""
    CRITICAL = "critical"      # Essential for core functionality
    HIGH = "high"             # Important for full functionality  
    MEDIUM = "medium"         # Nice to have, alternatives exist
    LOW = "low"              # Optional enhancements
    EXPERIMENTAL = "experimental"  # Future/testing features

@dataclass
class ServiceConfig:
    """Configuration for a service/API"""
    name: str
    key_names: List[str]  # Environment variable names
    description: str
    priority: ServicePriority
    test_url: Optional[str] = None
    test_method: str = "GET"
    test_headers: Optional[Dict[str, str]] = None
    fallback_services: List[str] = field(default_factory=list)
    required_for: List[str] = field(default_factory=list)  # Features that need this service

@dataclass
class ServiceAssessment:
    """Assessment result for a service"""
    config: ServiceConfig
    status: ServiceStatus
    keys_present: Dict[str, bool]
    error_message: Optional[str] = None
    response_time: Optional[float] = None
    recommendations: List[str] = field(default_factory=list)

class APIKeyAssessment:
    """Comprehensive API Key Assessment System"""
    
    def __init__(self):
        # Load environment
        load_environment()
        
        # Define service configurations
        self.services = {
            # Critical Services
            "supabase": ServiceConfig(
                name="Supabase Database",
                key_names=["SUPABASE_URL", "SUPABASE_KEY", "SUPABASE_SERVICE_ROLE_KEY"],
                description="Primary database and authentication service", 
                priority=ServicePriority.CRITICAL,
                required_for=["data_storage", "user_management", "real_time"]
            ),
            
            # High Priority LLM Services
            "openrouter": ServiceConfig(
                name="OpenRouter LLM Gateway",
                key_names=["OPENROUTER_API_KEY"],
                description="Primary LLM access with multiple model support",
                priority=ServicePriority.HIGH,
                test_url="https://openrouter.ai/api/v1/models",
                test_headers={"Authorization": "Bearer {OPENROUTER_API_KEY}"},
                required_for=["market_analysis", "synthesis", "multimodal_analysis"]
            ),
            
            "openai": ServiceConfig(
                name="OpenAI Direct API",
                key_names=["OPENAI_API_KEY"],
                description="Direct OpenAI API access for advanced models",
                priority=ServicePriority.HIGH,
                test_url="https://api.openai.com/v1/models",
                test_headers={"Authorization": "Bearer {OPENAI_API_KEY}"},
                fallback_services=["openrouter", "groq", "together"],
                required_for=["advanced_analysis", "code_generation"]
            ),
            
            # Alternative LLM Providers
            "groq": ServiceConfig(
                name="Groq Ultra-Fast Inference",
                key_names=["GROQ_API_KEY"],
                description="Ultra-fast LLM inference for real-time processing",
                priority=ServicePriority.MEDIUM,
                test_url="https://api.groq.com/openai/v1/models",
                test_headers={"Authorization": "Bearer {GROQ_API_KEY}"},
                required_for=["real_time_analysis", "fast_processing"]
            ),
            
            "together": ServiceConfig(
                name="Together.ai Open Source Models",
                key_names=["TOGETHER_API_KEY"],
                description="Cost-effective access to open-source models",
                priority=ServicePriority.MEDIUM,
                test_url="https://api.together.xyz/v1/models",
                test_headers={"Authorization": "Bearer {TOGETHER_API_KEY}"},
                required_for=["cost_optimization", "open_source_models"]
            ),
            
            "huggingface": ServiceConfig(
                name="Hugging Face Inference API",
                key_names=["HF_API_KEY"],
                description="Access to thousands of open-source models",
                priority=ServicePriority.MEDIUM,
                test_url="https://api-inference.huggingface.co/models/gpt2",
                test_headers={"Authorization": "Bearer {HF_API_KEY}"},
                required_for=["model_variety", "research_models"]
            ),
            
            "gemini": ServiceConfig(
                name="Google Gemini API",
                key_names=["GEMINI_API_KEY"],
                description="Google's advanced AI models",
                priority=ServicePriority.MEDIUM,
                test_url="https://generativelanguage.googleapis.com/v1/models",
                test_headers={"x-goog-api-key": "{GEMINI_API_KEY}"},
                required_for=["multimodal_analysis", "google_integration"]
            ),
            
            # Enterprise Services
            "routellm": ServiceConfig(
                name="RouteLL Intelligent Routing",
                key_names=["ROUTELLM_API_KEY"],
                description="Intelligent model routing and optimization",
                priority=ServicePriority.MEDIUM,
                required_for=["intelligent_routing", "cost_optimization"]
            ),
            
            "abacus": ServiceConfig(
                name="Abacus.ai LLM Teams",
                key_names=["ABACUS_API_KEY", "ABACUS_DEPLOYMENT_ID", "ABACUS_DEPLOYMENT_TOKEN"],
                description="Enterprise-grade AI platform",
                priority=ServicePriority.LOW,
                required_for=["enterprise_features", "custom_models"]
            ),
            
            # Data Source APIs
            "github": ServiceConfig(
                name="GitHub API",
                key_names=["GITHUB_TOKEN"],
                description="Code repository analysis and trend detection",
                priority=ServicePriority.HIGH,
                test_url="https://api.github.com/user",
                test_headers={"Authorization": "token {GITHUB_TOKEN}"},
                required_for=["code_analysis", "repository_intelligence"]
            ),
            
            "serper": ServiceConfig(
                name="Google Search (Serper)",
                key_names=["SERPER_API_KEY"],
                description="Google search results for market intelligence",
                priority=ServicePriority.HIGH,
                test_url="https://google.serper.dev/search",
                test_method="POST",
                required_for=["web_scraping", "market_research"]
            ),
            
            # Media and Content APIs
            "pexels": ServiceConfig(
                name="Pexels Stock Photos",
                key_names=["PEXELS_API_KEY"],
                description="Stock photo access for visual analysis",
                priority=ServicePriority.LOW,
                test_url="https://api.pexels.com/v1/search?query=technology&per_page=1",
                test_headers={"Authorization": "{PEXELS_API_KEY}"},
                required_for=["visual_content", "stock_images"]
            ),
            
            "deepai": ServiceConfig(
                name="DeepAI Services",
                key_names=["DEEPAI_API_KEY"],
                description="AI-powered image and content processing",
                priority=ServicePriority.LOW,
                required_for=["image_processing", "ai_tools"]
            ),
            
            # Social Media APIs
            "reddit": ServiceConfig(
                name="Reddit API",
                key_names=["REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET", "REDDIT_USER_AGENT"],
                description="Social media sentiment and trend analysis",
                priority=ServicePriority.MEDIUM,
                required_for=["social_sentiment", "trend_analysis"]
            ),
            
            # Productivity Integration
            "notion": ServiceConfig(
                name="Notion API",
                key_names=["NOTION_API_KEY", "NOTION_DATABASE_ID"],
                description="Knowledge base and document management",
                priority=ServicePriority.LOW,
                required_for=["documentation", "knowledge_management"]
            ),
            
            "vercel": ServiceConfig(
                name="Vercel Deployment",
                key_names=["VERCEL_TOKEN"],
                description="Application deployment and hosting",
                priority=ServicePriority.LOW,
                required_for=["deployment", "hosting"]
            ),
            
            # Infrastructure
            "redis": ServiceConfig(
                name="Redis Cache/PubSub",
                key_names=["REDIS_HOST", "REDIS_PORT"],
                description="Real-time data processing and caching",
                priority=ServicePriority.MEDIUM,
                required_for=["real_time_processing", "caching"]
            )
        }
        
        self.assessments: Dict[str, ServiceAssessment] = {}
        
    def assess_all_services(self, test_connectivity: bool = True) -> Dict[str, ServiceAssessment]:
        """Assess all configured services"""
        logger.info("🔍 Starting comprehensive API key assessment...")
        
        for service_name, config in self.services.items():
            self.assessments[service_name] = self.assess_service(config, test_connectivity)
        
        return self.assessments
    
    def assess_service(self, config: ServiceConfig, test_connectivity: bool = True) -> ServiceAssessment:
        """Assess a specific service"""
        # Check if required keys are present
        keys_present = {}
        missing_keys = []
        
        for key_name in config.key_names:
            value = os.getenv(key_name)
            is_present = bool(value and value != f"your_{key_name.lower()}" and value != "your-id" and value != "your-secret")
            keys_present[key_name] = is_present
            
            if not is_present:
                missing_keys.append(key_name)
        
        # Determine initial status
        if missing_keys:
            status = ServiceStatus.MISSING
            error_message = f"Missing keys: {missing_keys}"
            recommendations = [f"Add {key} to .env file" for key in missing_keys]
        else:
            status = ServiceStatus.CONFIGURED
            error_message = None
            recommendations = []
        
        # Test connectivity if requested and keys are present
        response_time = None
        if test_connectivity and not missing_keys and config.test_url:
            try:
                status, response_time, test_error = self._test_service_connectivity(config)
                if test_error:
                    error_message = test_error
            except Exception as e:
                status = ServiceStatus.ERROR
                error_message = f"Connectivity test failed: {str(e)}"
        
        # Add service-specific recommendations
        recommendations.extend(self._get_service_recommendations(config, status, keys_present))
        
        return ServiceAssessment(
            config=config,
            status=status,
            keys_present=keys_present,
            error_message=error_message,
            response_time=response_time,
            recommendations=recommendations
        )
    
    def _test_service_connectivity(self, config: ServiceConfig) -> Tuple[ServiceStatus, Optional[float], Optional[str]]:
        """Test connectivity to a service"""
        if not config.test_url:
            return ServiceStatus.CONFIGURED, None, None
        
        try:
            # Prepare headers with API keys
            headers = {}
            if config.test_headers:
                for key, template in config.test_headers.items():
                    if "{" in template and "}" in template:
                        # Extract the environment variable name
                        env_var = template.split("{")[1].split("}")[0]
                        api_key = os.getenv(env_var)
                        if api_key:
                            headers[key] = template.replace(f"{{{env_var}}}", api_key)
                    else:
                        headers[key] = template
            
            # Make request
            start_time = datetime.now()
            
            if config.test_method.upper() == "POST":
                response = requests.post(
                    config.test_url,
                    headers=headers,
                    json={"q": "test"} if "serper" in config.test_url else {},
                    timeout=10
                )
            else:
                response = requests.get(config.test_url, headers=headers, timeout=10)
            
            response_time = (datetime.now() - start_time).total_seconds()
            
            if response.status_code == 200:
                return ServiceStatus.AVAILABLE, response_time, None
            elif response.status_code == 401:
                return ServiceStatus.INVALID, response_time, "Invalid API key"
            elif response.status_code == 429:
                return ServiceStatus.RATE_LIMITED, response_time, "Rate limited"
            else:
                return ServiceStatus.ERROR, response_time, f"HTTP {response.status_code}"
                
        except requests.RequestException as e:
            return ServiceStatus.ERROR, None, f"Connection failed: {str(e)}"
    
    def _get_service_recommendations(self, config: ServiceConfig, status: ServiceStatus, keys_present: Dict[str, bool]) -> List[str]:
        """Get service-specific recommendations"""
        recommendations = []
        
        # Priority-based recommendations
        if config.priority == ServicePriority.CRITICAL and status != ServiceStatus.AVAILABLE:
            recommendations.append(f"🚨 CRITICAL: {config.name} is required for core functionality")
        
        # Service-specific guidance
        if config.name == "OpenRouter LLM Gateway":
            if status == ServiceStatus.AVAILABLE:
                recommendations.append("✅ Use free models only (models containing ':free') to avoid charges")
            elif status == ServiceStatus.MISSING:
                recommendations.append("Get API key from https://openrouter.ai - Essential for LLM access")
        
        elif config.name == "Supabase Database":
            if status != ServiceStatus.AVAILABLE:
                recommendations.append("Set up Supabase project at https://supabase.com - Required for data storage")
        
        elif config.name == "GitHub API":
            if status == ServiceStatus.MISSING:
                recommendations.append("Create GitHub Personal Access Token for repository analysis")
            elif status == ServiceStatus.INVALID:
                recommendations.append("Check GitHub token permissions (needs 'repo' and 'read:user' scopes)")
        
        # Fallback recommendations
        if config.fallback_services and status != ServiceStatus.AVAILABLE:
            available_fallbacks = [
                service for service in config.fallback_services 
                if service in self.assessments and self.assessments[service].status == ServiceStatus.AVAILABLE
            ]
            if available_fallbacks:
                recommendations.append(f"✅ Fallback available: {available_fallbacks[0]}")
            else:
                recommendations.append(f"Consider setting up fallback: {config.fallback_services[0]}")
        
        return recommendations
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status and capabilities"""
        if not self.assessments:
            self.assess_all_services()
        
        # Count services by status
        status_counts = {}
        priority_status = {}
        
        for assessment in self.assessments.values():
            status = assessment.status.value
            priority = assessment.config.priority.value
            
            status_counts[status] = status_counts.get(status, 0) + 1
            
            if priority not in priority_status:
                priority_status[priority] = {}
            priority_status[priority][status] = priority_status[priority].get(status, 0) + 1
        
        # Determine system health
        critical_issues = sum(
            1 for a in self.assessments.values()
            if a.config.priority == ServicePriority.CRITICAL and a.status != ServiceStatus.AVAILABLE
        )
        
        high_priority_issues = sum(
            1 for a in self.assessments.values()
            if a.config.priority == ServicePriority.HIGH and a.status not in [ServiceStatus.AVAILABLE, ServiceStatus.CONFIGURED]
        )
        
        # Determine overall health
        if critical_issues > 0:
            overall_health = "CRITICAL"
        elif high_priority_issues > 2:
            overall_health = "DEGRADED"
        elif high_priority_issues > 0:
            overall_health = "WARNING"
        else:
            overall_health = "HEALTHY"
        
        # Calculate capability scores
        capabilities = self._assess_capabilities()
        
        return {
            "overall_health": overall_health,
            "assessment_timestamp": datetime.now().isoformat(),
            "total_services": len(self.services),
            "status_summary": status_counts,
            "priority_breakdown": priority_status,
            "critical_issues": critical_issues,
            "high_priority_issues": high_priority_issues,
            "capabilities": capabilities,
            "service_details": {
                name: {
                    "status": assessment.status.value,
                    "priority": assessment.config.priority.value,
                    "description": assessment.config.description,
                    "response_time": assessment.response_time
                }
                for name, assessment in self.assessments.items()
            }
        }
    
    def _assess_capabilities(self) -> Dict[str, Dict[str, Any]]:
        """Assess system capabilities based on available services"""
        capabilities = {
            "market_analysis": {"available": False, "providers": [], "quality": "none"},
            "code_analysis": {"available": False, "providers": [], "quality": "none"},
            "multimodal_analysis": {"available": False, "providers": [], "quality": "none"},
            "data_storage": {"available": False, "providers": [], "quality": "none"},
            "real_time_processing": {"available": False, "providers": [], "quality": "none"},
            "web_scraping": {"available": False, "providers": [], "quality": "none"}
        }
        
        # LLM capabilities
        llm_services = ["openrouter", "openai", "groq", "together", "huggingface", "gemini"]
        available_llms = [
            name for name in llm_services
            if name in self.assessments and self.assessments[name].status == ServiceStatus.AVAILABLE
        ]
        
        if available_llms:
            capabilities["market_analysis"]["available"] = True
            capabilities["market_analysis"]["providers"] = available_llms
            capabilities["market_analysis"]["quality"] = "high" if len(available_llms) >= 3 else "medium" if len(available_llms) >= 2 else "basic"
            
            capabilities["multimodal_analysis"]["available"] = True
            capabilities["multimodal_analysis"]["providers"] = available_llms
            capabilities["multimodal_analysis"]["quality"] = "high" if "openai" in available_llms or "gemini" in available_llms else "medium"
        
        # Code analysis
        if "github" in self.assessments and self.assessments["github"].status == ServiceStatus.AVAILABLE:
            capabilities["code_analysis"]["available"] = True
            capabilities["code_analysis"]["providers"] = ["github"]
            capabilities["code_analysis"]["quality"] = "high" if available_llms else "basic"
        
        # Data storage
        if "supabase" in self.assessments and self.assessments["supabase"].status == ServiceStatus.AVAILABLE:
            capabilities["data_storage"]["available"] = True
            capabilities["data_storage"]["providers"] = ["supabase"]
            capabilities["data_storage"]["quality"] = "high"
        
        # Web scraping
        if "serper" in self.assessments and self.assessments["serper"].status == ServiceStatus.AVAILABLE:
            capabilities["web_scraping"]["available"] = True
            capabilities["web_scraping"]["providers"] = ["serper"]
            capabilities["web_scraping"]["quality"] = "high"
        
        # Real-time processing
        if "redis" in self.assessments and self.assessments["redis"].status == ServiceStatus.AVAILABLE:
            capabilities["real_time_processing"]["available"] = True
            capabilities["real_time_processing"]["providers"] = ["redis"]
            capabilities["real_time_processing"]["quality"] = "high"
        
        return capabilities
    
    def generate_recommendations_report(self) -> str:
        """Generate a comprehensive recommendations report"""
        if not self.assessments:
            self.assess_all_services()
        
        status = self.get_system_status()
        
        report = [
            "# 🔐 API Key Assessment Report",
            f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Overall Health**: {status['overall_health']}",
            "",
            "## 📊 System Overview",
            f"- **Total Services**: {status['total_services']}",
            f"- **Available Services**: {status['status_summary'].get('available', 0)}",
            f"- **Configured Services**: {status['status_summary'].get('configured', 0)}",
            f"- **Missing Services**: {status['status_summary'].get('missing', 0)}",
            f"- **Critical Issues**: {status['critical_issues']}",
            f"- **High Priority Issues**: {status['high_priority_issues']}",
            ""
        ]
        
        # Critical issues
        if status['critical_issues'] > 0:
            report.extend([
                "## 🚨 CRITICAL ISSUES (Immediate Action Required)",
                ""
            ])
            
            for name, assessment in self.assessments.items():
                if assessment.config.priority == ServicePriority.CRITICAL and assessment.status != ServiceStatus.AVAILABLE:
                    report.append(f"### ❌ {assessment.config.name}")
                    report.append(f"- **Status**: {assessment.status.value.title()}")
                    report.append(f"- **Description**: {assessment.config.description}")
                    if assessment.error_message:
                        report.append(f"- **Error**: {assessment.error_message}")
                    report.append("- **Actions**:")
                    for rec in assessment.recommendations:
                        report.append(f"  - {rec}")
                    report.append("")
        
        # Service details by priority
        for priority in [ServicePriority.HIGH, ServicePriority.MEDIUM, ServicePriority.LOW]:
            priority_services = [
                (name, assessment) for name, assessment in self.assessments.items()
                if assessment.config.priority == priority
            ]
            
            if priority_services:
                report.extend([
                    f"## {priority.value.title()} Priority Services",
                    ""
                ])
                
                for name, assessment in priority_services:
                    status_emoji = {
                        ServiceStatus.AVAILABLE: "✅",
                        ServiceStatus.CONFIGURED: "⚙️",
                        ServiceStatus.MISSING: "❌",
                        ServiceStatus.INVALID: "🔑",
                        ServiceStatus.RATE_LIMITED: "⏱️",
                        ServiceStatus.ERROR: "⚠️"
                    }.get(assessment.status, "❓")
                    
                    report.append(f"### {status_emoji} {assessment.config.name}")
                    report.append(f"- **Status**: {assessment.status.value.title()}")
                    report.append(f"- **Keys**: {', '.join(assessment.config.key_names)}")
                    
                    if assessment.response_time:
                        report.append(f"- **Response Time**: {assessment.response_time:.2f}s")
                    
                    if assessment.error_message:
                        report.append(f"- **Error**: {assessment.error_message}")
                    
                    if assessment.recommendations:
                        report.append("- **Recommendations**:")
                        for rec in assessment.recommendations:
                            report.append(f"  - {rec}")
                    
                    report.append("")
        
        # Capabilities summary
        report.extend([
            "## 🚀 System Capabilities",
            ""
        ])
        
        for capability_name, capability_info in status['capabilities'].items():
            status_emoji = "✅" if capability_info['available'] else "❌"
            quality_emoji = {"high": "🟢", "medium": "🟡", "basic": "🟠", "none": "🔴"}.get(capability_info['quality'], "❓")
            
            report.append(f"### {status_emoji} {capability_name.replace('_', ' ').title()}")
            report.append(f"- **Available**: {capability_info['available']}")
            report.append(f"- **Quality**: {quality_emoji} {capability_info['quality'].title()}")
            
            if capability_info['providers']:
                report.append(f"- **Providers**: {', '.join(capability_info['providers'])}")
            
            report.append("")
        
        return "\n".join(report)

# Main execution and testing
if __name__ == "__main__":
    print("🔐 Running API Key Assessment...")
    print("=" * 60)
    
    assessor = APIKeyAssessment()
    
    # Run assessment
    assessments = assessor.assess_all_services(test_connectivity=True)
    
    # Get system status
    status = assessor.get_system_status()
    
    # Print summary
    print(f"\n📊 ASSESSMENT SUMMARY")
    print(f"Overall Health: {status['overall_health']}")
    print(f"Services: {status['status_summary']}")
    print(f"Critical Issues: {status['critical_issues']}")
    
    # Print recommendations
    print(f"\n📋 FULL REPORT")
    print(assessor.generate_recommendations_report())
