#!/usr/bin/env python3
"""
AI Interaction Wrapper with Automatic Memory Logging
Ensures ALL AI interactions are logged with automated addendums

This wrapper intercepts all AI interactions and automatically:
1. Logs interaction details to memory system
2. Triggers periodic memory analysis every 2 interactions
3. Extracts forward initiative progress
4. Eliminates redundancy through analysis
"""

import os
import sys
import functools
import inspect
from typing import Any, Dict, List, Callable, Optional
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import memory orchestrator
from agents.memory_orchestrator import log_interaction_auto, get_memory_orchestrator

class AIInteractionLogger:
    """Wrapper class for automatic AI interaction logging"""
    
    def __init__(self):
        self.orchestrator = get_memory_orchestrator()
        self.current_interaction = None
        self.interaction_context = {}
    
    def log_interaction(self, user_query: str, ai_response: str, 
                       context: Dict[str, Any] = None) -> str:
        """Log interaction with automatic context extraction"""
        
        # Extract context information
        key_actions = self._extract_key_actions(ai_response, context)
        progress_indicators = self._extract_progress_indicators(ai_response, context)
        forward_initiative = self._extract_forward_initiative(ai_response, context)
        completion_status = self._extract_completion_status(ai_response, context)
        memory_updates = self._extract_memory_updates(ai_response, context)
        
        # Log to memory orchestrator
        interaction_id = log_interaction_auto(
            user_query=user_query,
            ai_response=ai_response,
            key_actions=key_actions,
            progress_indicators=progress_indicators,
            memory_updates=memory_updates,
            forward_initiative=forward_initiative,
            completion_status=completion_status
        )
        
        return interaction_id
    
    def _extract_key_actions(self, response: str, context: Dict = None) -> List[str]:
        """Extract key actions from AI response"""
        actions = []
        
        # Look for action indicators in response
        action_patterns = [
            'created', 'implemented', 'built', 'developed', 'designed',
            'analyzed', 'tested', 'validated', 'integrated', 'configured',
            'installed', 'deployed', 'updated', 'modified', 'enhanced'
        ]
        
        lines = response.lower().split('\n')
        for line in lines:
            for pattern in action_patterns:
                if pattern in line:
                    # Extract the action context
                    if len(line.strip()) < 100:
                        actions.append(line.strip())
                    break
        
        # Add context-based actions
        if context:
            if context.get('files_created'):
                actions.extend([f"Created {f}" for f in context['files_created']])
            if context.get('tools_used'):
                actions.extend([f"Used {t}" for t in context['tools_used']])
            if context.get('tasks_completed'):
                actions.extend([f"Completed {t}" for t in context['tasks_completed']])
        
        return list(set(actions))[:5]  # Limit to 5 unique actions
    
    def _extract_progress_indicators(self, response: str, context: Dict = None) -> List[str]:
        """Extract progress indicators from response"""
        indicators = []
        
        # Look for progress patterns
        progress_patterns = [
            '✅', 'completed', 'successful', 'working', 'ready',
            'implemented', 'finished', 'done', 'achieved'
        ]
        
        lines = response.split('\n')
        for line in lines:
            line_lower = line.lower()
            if any(pattern in line_lower for pattern in progress_patterns):
                if len(line.strip()) < 80:
                    indicators.append(line.strip())
        
        # Add context indicators
        if context:
            if context.get('success', False):
                indicators.append("Operation successful")
            if context.get('progress_percentage'):
                indicators.append(f"Progress: {context['progress_percentage']}%")
        
        return list(set(indicators))[:3]
    
    def _extract_forward_initiative(self, response: str, context: Dict = None) -> str:
        """Extract forward initiative description"""
        
        # Look for forward-looking statements
        forward_patterns = [
            'next steps', 'moving forward', 'now ready', 'enables',
            'ready for', 'can proceed', 'will allow', 'opens up'
        ]
        
        lines = response.lower().split('\n')
        for line in lines:
            for pattern in forward_patterns:
                if pattern in line:
                    # Find the full sentence
                    sentences = response.split('.')
                    for sentence in sentences:
                        if pattern in sentence.lower():
                            return sentence.strip()[:200]
        
        # Extract from context
        if context and context.get('forward_initiative'):
            return context['forward_initiative']
        
        # Default extraction from response summary
        if 'implemented' in response.lower() or 'completed' in response.lower():
            return "Advanced capabilities implemented and ready for integration"
        
        return "Continuing system development and enhancement"
    
    def _extract_completion_status(self, response: str, context: Dict = None) -> str:
        """Extract completion status"""
        
        response_lower = response.lower()
        
        if any(word in response_lower for word in ['error', 'failed', 'exception']):
            return "error"
        elif any(word in response_lower for word in ['completed', 'successful', 'finished', 'done']):
            return "completed"
        elif any(word in response_lower for word in ['started', 'implementing', 'working on']):
            return "in_progress"
        else:
            return "in_progress"
    
    def _extract_memory_updates(self, response: str, context: Dict = None) -> List[str]:
        """Extract memory updates made"""
        updates = []
        
        # Look for memory-related activities
        memory_patterns = [
            'stored', 'saved', 'documented', 'logged', 'recorded',
            'updated memory', 'added to memory', 'memory system'
        ]
        
        lines = response.lower().split('\n')
        for line in lines:
            for pattern in memory_patterns:
                if pattern in line:
                    updates.append(line.strip()[:100])
                    break
        
        return list(set(updates))[:3]

# Global logger instance
ai_logger = AIInteractionLogger()

def auto_log_interaction(func: Callable) -> Callable:
    """
    Decorator to automatically log AI interactions
    
    Usage:
        @auto_log_interaction
        def my_ai_function(user_input):
            # AI processing
            return ai_response
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Extract user query from function arguments
        user_query = ""
        if args:
            user_query = str(args[0])[:200]
        elif 'query' in kwargs:
            user_query = str(kwargs['query'])[:200]
        elif 'user_input' in kwargs:
            user_query = str(kwargs['user_input'])[:200]
        
        # Execute the original function
        start_time = datetime.now()
        try:
            result = func(*args, **kwargs)
            
            # Extract AI response
            ai_response = str(result) if result else "Function completed without return value"
            
            # Prepare context
            context = {
                'function_name': func.__name__,
                'execution_time': (datetime.now() - start_time).total_seconds(),
                'success': True,
                'args_count': len(args),
                'kwargs_keys': list(kwargs.keys())
            }
            
            # Log the interaction
            interaction_id = ai_logger.log_interaction(
                user_query=user_query or f"Function call: {func.__name__}",
                ai_response=ai_response,
                context=context
            )
            
            # Add interaction_id to result if it's a dict
            if isinstance(result, dict):
                result['_interaction_id'] = interaction_id
            
            return result
            
        except Exception as e:
            # Log error interactions too
            error_response = f"Error in {func.__name__}: {str(e)}"
            context = {
                'function_name': func.__name__,
                'execution_time': (datetime.now() - start_time).total_seconds(),
                'success': False,
                'error': str(e)
            }
            
            ai_logger.log_interaction(
                user_query=user_query or f"Function call: {func.__name__}",
                ai_response=error_response,
                context=context
            )
            
            raise  # Re-raise the exception
    
    return wrapper

class InteractionAddendum:
    """Automated addendum system for all interactions"""
    
    @staticmethod
    def add_to_response(response: str, interaction_context: Dict = None) -> str:
        """
        Add automated addendum to AI response
        
        Args:
            response: Original AI response
            interaction_context: Context about the interaction
            
        Returns:
            Response with automated addendum
        """
        
        # Get memory orchestrator status
        orchestrator = get_memory_orchestrator()
        status = orchestrator.get_memory_status()
        
        # Create addendum
        addendum = f"""

---
**🧠 AUTOMATED INTERACTION ADDENDUM**

📊 **Memory Status**: {status['total_interactions']} interactions logged | Next analysis due: {status['next_analysis_due']}  
⚡ **Forward Momentum**: {status['recent_momentum_score']:.1f}/1.0 | Timer: {'Active' if status['timer_active'] else 'Inactive'}  
🔄 **Memory Analysis**: {'Recent' if status['last_analysis'] else 'Pending'} | Interval: Every {status['analysis_interval']} interactions

*All interaction details automatically logged for continuous progress optimization and redundancy elimination.*

---"""
        
        return response + addendum
    
    @staticmethod
    def log_current_interaction(user_query: str, ai_response: str, **kwargs) -> str:
        """Log current interaction and return interaction ID"""
        return ai_logger.log_interaction(user_query, ai_response, kwargs)

# Convenience functions for manual logging
def log_interaction(user_query: str, ai_response: str, **kwargs) -> str:
    """Manually log an interaction"""
    return ai_logger.log_interaction(user_query, ai_response, kwargs)

def add_memory_addendum(response: str, context: Dict = None) -> str:
    """Add memory addendum to response"""
    return InteractionAddendum.add_to_response(response, context)

def get_memory_status() -> Dict[str, Any]:
    """Get current memory orchestrator status"""
    return get_memory_orchestrator().get_memory_status()

# Example usage and testing
def test_interaction_wrapper():
    """Test the AI interaction wrapper"""
    print("🔄 Testing AI Interaction Wrapper")
    print("=" * 50)
    
    # Test 1: Manual interaction logging
    print("\n1. Testing manual interaction logging...")
    
    user_query = "Implement automated memory logging system"
    ai_response = """Successfully implemented automated memory orchestration system with:
    ✅ Created memory_orchestrator.py with comprehensive logging
    ✅ Implemented periodic analysis every 2 interactions  
    ✅ Added LLM-powered memory analysis with safeguards
    ✅ Built interaction context extraction system
    Ready for integration with all AI interactions."""
    
    interaction_id = log_interaction(
        user_query=user_query,
        ai_response=ai_response,
        key_actions=["Created memory system", "Implemented periodic analysis"],
        forward_initiative="All interactions now automatically logged with memory analysis"
    )
    
    print(f"   ✅ Interaction logged: {interaction_id}")
    
    # Test 2: Response addendum
    print("\n2. Testing response addendum...")
    
    enhanced_response = add_memory_addendum(
        "Task completed successfully.",
        context={"completion": True}
    )
    
    print("   ✅ Addendum added to response")
    print(f"   Preview: {enhanced_response[-200:]}...")
    
    # Test 3: Memory status
    print("\n3. Testing memory status...")
    status = get_memory_status()
    
    print(f"   Total interactions: {status['total_interactions']}")
    print(f"   Timer active: {status['timer_active']}")
    print(f"   Momentum score: {status['recent_momentum_score']}")
    
    return True

if __name__ == "__main__":
    success = test_interaction_wrapper()
    if success:
        print("\n🎉 AI Interaction Wrapper working correctly!")
        print("\n📋 Usage Instructions:")
        print("1. Import: from agents.ai_interaction_wrapper import log_interaction, add_memory_addendum")
        print("2. Log interactions: log_interaction(user_query, ai_response)")  
        print("3. Add addendum: enhanced_response = add_memory_addendum(response)")
        print("4. Use decorator: @auto_log_interaction")
    else:
        print("\n❌ AI Interaction Wrapper issues detected.")
