#!/usr/bin/env python3
"""
OpenRouter Cost Monitor and Budget Management
Track usage and prevent overspending on your OpenRouter account
"""

import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class UsageRecord:
    """Track individual API usage"""
    timestamp: str
    model: str
    input_tokens: int
    output_tokens: int
    estimated_cost: float
    task_type: str

class OpenRouterCostMonitor:
    """Monitor and control OpenRouter costs"""
    
    def __init__(self, daily_budget: float = 5.0):
        self.daily_budget = daily_budget
        self.usage_file = "/Users/kfitz/sentient_venture_engine/logs/openrouter_usage.json"
        self.ensure_log_directory()
        
        # Model costs (per 1M tokens)
        self.model_costs = {
            # FREE MODELS
            "meta-llama/llama-3.1-8b-instruct": (0.0, 0.0),
            "mistralai/mistral-7b-instruct": (0.0, 0.0),
            "microsoft/phi-3-mini-128k-instruct": (0.0, 0.0),
            "google/gemma-7b-it": (0.0, 0.0),
            
            # LOW COST MODELS
            "google/gemini-flash-1.5": (0.075, 0.30),
            "anthropic/claude-3-haiku": (0.25, 1.25),
            "openai/gpt-4o-mini": (0.15, 0.60),
            
            # MODERATE COST MODELS
            "anthropic/claude-3-5-sonnet": (3.0, 15.0),
            "openai/gpt-4o": (2.5, 10.0),
            "google/gemini-pro-1.5": (1.25, 5.0)
        }
    
    def ensure_log_directory(self):
        """Ensure log directory exists"""
        os.makedirs(os.path.dirname(self.usage_file), exist_ok=True)
    
    def load_usage_history(self) -> List[UsageRecord]:
        """Load usage history from file"""
        try:
            if os.path.exists(self.usage_file):
                with open(self.usage_file, 'r') as f:
                    data = json.load(f)
                    return [UsageRecord(**record) for record in data]
            return []
        except Exception as e:
            logger.error(f"Error loading usage history: {e}")
            return []
    
    def save_usage_record(self, record: UsageRecord):
        """Save usage record to file"""
        try:
            history = self.load_usage_history()
            history.append(record)
            
            # Keep only last 30 days
            cutoff = (datetime.now() - timedelta(days=30)).isoformat()
            history = [r for r in history if r.timestamp >= cutoff]
            
            with open(self.usage_file, 'w') as f:
                json.dump([record.__dict__ for record in history], f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving usage record: {e}")
    
    def calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for model usage"""
        if model not in self.model_costs:
            return 0.0  # Unknown model assumed free
        
        input_cost, output_cost = self.model_costs[model]
        total_cost = (input_tokens * input_cost / 1_000_000) + (output_tokens * output_cost / 1_000_000)
        return total_cost
    
    def log_usage(self, model: str, input_tokens: int, output_tokens: int, task_type: str = "general") -> float:
        """Log model usage and return cost"""
        cost = self.calculate_cost(model, input_tokens, output_tokens)
        
        record = UsageRecord(
            timestamp=datetime.now().isoformat(),
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost=cost,
            task_type=task_type
        )
        
        self.save_usage_record(record)
        return cost
    
    def get_daily_usage(self, date: str = None) -> Tuple[float, int]:
        """Get daily usage cost and token count"""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        history = self.load_usage_history()
        daily_records = [r for r in history if r.timestamp.startswith(date)]
        
        total_cost = sum(r.estimated_cost for r in daily_records)
        total_tokens = sum(r.input_tokens + r.output_tokens for r in daily_records)
        
        return total_cost, total_tokens
    
    def check_budget_status(self) -> Dict[str, any]:
        """Check current budget status"""
        daily_cost, daily_tokens = self.get_daily_usage()
        
        remaining_budget = self.daily_budget - daily_cost
        budget_percent_used = (daily_cost / self.daily_budget) * 100 if self.daily_budget > 0 else 0
        
        status = {
            "daily_cost": daily_cost,
            "daily_tokens": daily_tokens,
            "daily_budget": self.daily_budget,
            "remaining_budget": remaining_budget,
            "budget_percent_used": budget_percent_used,
            "over_budget": daily_cost > self.daily_budget,
            "warning_threshold": budget_percent_used > 80
        }
        
        return status
    
    def can_execute_task(self, estimated_cost: float) -> Tuple[bool, str]:
        """Check if task can be executed within budget"""
        status = self.check_budget_status()
        
        if status["over_budget"]:
            return False, f"Daily budget exceeded: ${status['daily_cost']:.4f} > ${self.daily_budget:.2f}"
        
        if status["daily_cost"] + estimated_cost > self.daily_budget:
            return False, f"Task would exceed budget: ${status['daily_cost'] + estimated_cost:.4f} > ${self.daily_budget:.2f}"
        
        if status["warning_threshold"]:
            return True, f"⚠️ Warning: {status['budget_percent_used']:.1f}% of budget used"
        
        return True, "✅ Within budget"\n    \n    def get_usage_summary(self, days: int = 7) -> Dict[str, any]:\n        \"\"\"Get usage summary for specified days\"\"\"\n        history = self.load_usage_history()\n        cutoff = (datetime.now() - timedelta(days=days)).isoformat()\n        recent_records = [r for r in history if r.timestamp >= cutoff]\n        \n        total_cost = sum(r.estimated_cost for r in recent_records)\n        total_tokens = sum(r.input_tokens + r.output_tokens for r in recent_records)\n        \n        # Model usage breakdown\n        model_usage = {}\n        for record in recent_records:\n            if record.model not in model_usage:\n                model_usage[record.model] = {"count": 0, "cost": 0.0, "tokens": 0}\n            model_usage[record.model]["count"] += 1\n            model_usage[record.model]["cost"] += record.estimated_cost\n            model_usage[record.model]["tokens"] += record.input_tokens + record.output_tokens\n        \n        # Task type breakdown\n        task_breakdown = {}\n        for record in recent_records:\n            if record.task_type not in task_breakdown:\n                task_breakdown[record.task_type] = {"count": 0, "cost": 0.0}\n            task_breakdown[record.task_type]["count"] += 1\n            task_breakdown[record.task_type]["cost"] += record.estimated_cost\n        \n        return {\n            "period_days": days,\n            "total_cost": total_cost,\n            "total_tokens": total_tokens,\n            "total_requests": len(recent_records),\n            "model_usage": model_usage,\n            "task_breakdown": task_breakdown,\n            "average_cost_per_request": total_cost / len(recent_records) if recent_records else 0\n        }\n\ndef check_openrouter_budget():\n    \"\"\"Quick budget check for current session\"\"\"\n    monitor = OpenRouterCostMonitor(daily_budget=5.0)  # $5 daily budget\n    status = monitor.check_budget_status()\n    \n    print(\"💰 OpenRouter Budget Status\")\n    print(\"=\" * 40)\n    print(f\"📊 Today's Usage: ${status['daily_cost']:.4f}\")\n    print(f\"🎯 Daily Budget: ${status['daily_budget']:.2f}\")\n    print(f\"💵 Remaining: ${status['remaining_budget']:.4f}\")\n    print(f\"📈 Usage: {status['budget_percent_used']:.1f}%\")\n    \n    if status['over_budget']:\n        print(f\"🚨 OVER BUDGET! Consider using free models only.\")\n    elif status['warning_threshold']:\n        print(f\"⚠️ Warning: Approaching budget limit\")\n    else:\n        print(f\"✅ Budget status: Good\")\n    \n    # Show usage summary\n    summary = monitor.get_usage_summary(days=7)\n    print(f\"\\n📋 7-Day Summary:\")\n    print(f\"   Total Requests: {summary['total_requests']}\")\n    print(f\"   Total Cost: ${summary['total_cost']:.4f}\")\n    print(f\"   Avg Cost/Request: ${summary['average_cost_per_request']:.4f}\")\n    \n    return status\n\nif __name__ == \"__main__\":\n    check_openrouter_budget()
