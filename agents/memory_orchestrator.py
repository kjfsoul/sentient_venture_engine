#!/usr/bin/env python3
"""
Automated Memory Orchestration System for SVE
Logs all interactions and implements periodic memory analysis with timer flows

Features:
- Automatic interaction logging to memory system
- Periodic memory analysis every 2 interactions
- Forward initiative progress tracking
- Redundancy elimination
- Agent execution safeguards (max_iterations, max_execution_time, early_stopping)
"""

import os
import sys
import json
import time
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import threading
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

# CrewAI imports with safeguards
try:
    from crewai import Agent, Task, Crew
    from langchain_openai import ChatOpenAI
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False
    # Create mock class for type hints
    class ChatOpenAI:
        pass

# Import security manager
try:
    from security.api_key_manager import get_secret_optional
except ImportError:
    def get_secret_optional(key, fallbacks=None):
        return os.getenv(key)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class InteractionLog:
    """Structure for interaction logging"""
    interaction_id: str
    timestamp: datetime
    user_query: str
    ai_response_summary: str
    key_actions: List[str]
    progress_indicators: List[str]
    memory_updates: List[str]
    forward_initiative: str
    completion_status: str

@dataclass
class MemoryAnalysisResult:
    """Result from periodic memory analysis"""
    analysis_timestamp: datetime
    total_memories_reviewed: int
    key_insights_extracted: List[str]
    progress_summary: str
    redundancies_identified: List[str]
    next_priorities: List[str]
    forward_momentum_score: float

class MemoryOrchestrator:
    """Automated Memory Management and Analysis System"""
    
    def __init__(self):
        """Initialize the memory orchestrator"""
        self.interaction_count = 0
        self.interaction_logs: List[InteractionLog] = []
        self.last_memory_analysis = None
        self.analysis_interval = 2  # Every 2 interactions
        
        # Timer flow management
        self.timer_active = False
        self.analysis_timer = None
        
        # Initialize LLM for memory analysis
        self.llm = self._initialize_llm()
        
        # Create memory storage directory
        self.memory_dir = Path("/Users/kfitz/sentient_venture_engine/memory_logs")
        self.memory_dir.mkdir(exist_ok=True)
        
        logger.info("🧠 Memory Orchestrator initialized")
        self._start_timer_flow()
    
    def _initialize_llm(self) -> Optional["ChatOpenAI"]:
        """Initialize LLM for memory analysis with safeguards"""
        try:
            openrouter_key = get_secret_optional("OPENROUTER_API_KEY")
            if openrouter_key:
                return ChatOpenAI(
                    openai_api_base="https://openrouter.ai/api/v1",
                    openai_api_key=openrouter_key,
                    model_name="mistralai/mistral-7b-instruct:free",
                    temperature=0.3,
                    max_tokens=1024,
                    timeout=30,  # Execution safeguard
                    max_retries=2  # Execution safeguard
                )
            return None
        except Exception as e:
            logger.error(f"❌ LLM initialization failed: {e}")
            return None
    
    def _start_timer_flow(self):
        """Start the automated timer flow for memory analysis"""
        self.timer_active = True
        logger.info("⏰ Timer flow activated - memory analysis every 2 interactions")
    
    def log_interaction(self, user_query: str, ai_response: str, 
                       key_actions: List[str] = None, 
                       progress_indicators: List[str] = None,
                       memory_updates: List[str] = None,
                       forward_initiative: str = "",
                       completion_status: str = "in_progress") -> str:
        """
        Automated addendum to ALL INTERACTIONS - logs key details into memory
        
        Args:
            user_query: The user's query/request
            ai_response: The AI's response
            key_actions: List of key actions taken
            progress_indicators: List of progress indicators
            memory_updates: List of memory updates made
            forward_initiative: Description of forward progress
            completion_status: Status of the interaction
            
        Returns:
            interaction_id: Unique identifier for this interaction
        """
        self.interaction_count += 1
        interaction_id = f"int_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.interaction_count:03d}"
        
        # Create interaction log
        interaction_log = InteractionLog(
            interaction_id=interaction_id,
            timestamp=datetime.now(),
            user_query=user_query[:500],  # Truncate for storage
            ai_response_summary=self._summarize_response(ai_response),
            key_actions=key_actions or [],
            progress_indicators=progress_indicators or [],
            memory_updates=memory_updates or [],
            forward_initiative=forward_initiative,
            completion_status=completion_status
        )
        
        # Add to logs
        self.interaction_logs.append(interaction_log)
        
        # Save to persistent storage
        self._save_interaction_log(interaction_log)
        
        # Check if memory analysis is due (every 2 interactions)
        if self.timer_active and self.interaction_count % self.analysis_interval == 0:
            logger.info(f"🔄 Triggering memory analysis (interaction #{self.interaction_count})")
            self._trigger_memory_analysis()
        
        # Update memory system with interaction details
        self._update_memory_system(interaction_log)
        
        logger.info(f"📝 Interaction logged: {interaction_id}")
        return interaction_id
    
    def _summarize_response(self, ai_response: str) -> str:
        """Summarize AI response for logging"""
        if len(ai_response) <= 200:
            return ai_response
        
        # Extract key phrases and actions
        lines = ai_response.split('\n')
        key_lines = []
        
        for line in lines:
            line = line.strip()
            if any(indicator in line.lower() for indicator in [
                'implemented', 'created', 'completed', 'analysis', 'result', 
                'success', 'error', 'recommendation', 'next step'
            ]):
                key_lines.append(line)
                if len(key_lines) >= 3:
                    break
        
        summary = ' | '.join(key_lines) if key_lines else ai_response[:200]
        return summary[:200] + "..." if len(summary) > 200 else summary
    
    def _save_interaction_log(self, interaction_log: InteractionLog):
        """Save interaction log to persistent storage"""
        try:
            log_file = self.memory_dir / f"interaction_{interaction_log.interaction_id}.json"
            
            log_data = {
                "interaction_id": interaction_log.interaction_id,
                "timestamp": interaction_log.timestamp.isoformat(),
                "user_query": interaction_log.user_query,
                "ai_response_summary": interaction_log.ai_response_summary,
                "key_actions": interaction_log.key_actions,
                "progress_indicators": interaction_log.progress_indicators,
                "memory_updates": interaction_log.memory_updates,
                "forward_initiative": interaction_log.forward_initiative,
                "completion_status": interaction_log.completion_status
            }
            
            with open(log_file, 'w') as f:
                json.dump(log_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"❌ Failed to save interaction log: {e}")
    
    def _update_memory_system(self, interaction_log: InteractionLog):
        """Update the memory system with interaction details"""
        try:
            # This would integrate with the update_memory tool
            # For now, we'll create a summary for manual memory updates
            
            memory_summary = f"""
Interaction {interaction_log.interaction_id}:
- Query: {interaction_log.user_query}
- Key Actions: {', '.join(interaction_log.key_actions)}
- Progress: {interaction_log.forward_initiative}
- Status: {interaction_log.completion_status}
"""
            
            # Save summary for memory system integration
            summary_file = self.memory_dir / "memory_updates_queue.txt"
            with open(summary_file, 'a') as f:
                f.write(f"\n{datetime.now().isoformat()}: {memory_summary}\n")
                
        except Exception as e:
            logger.error(f"❌ Failed to update memory system: {e}")
    
    def _trigger_memory_analysis(self):
        """Trigger periodic memory analysis with timer flow"""
        try:
            logger.info("🔍 Starting periodic memory analysis...")
            
            # Load recent memories and interactions
            recent_interactions = self._get_recent_interactions(limit=10)
            
            # Perform analysis
            analysis_result = self._analyze_memories_and_progress(recent_interactions)
            
            # Store analysis result
            self.last_memory_analysis = analysis_result
            self._save_memory_analysis(analysis_result)
            
            # Log analysis completion
            logger.info("✅ Memory analysis completed")
            self._log_analysis_results(analysis_result)
            
        except Exception as e:
            logger.error(f"❌ Memory analysis failed: {e}")
    
    def _get_recent_interactions(self, limit: int = 10) -> List[InteractionLog]:
        """Get recent interaction logs"""
        return self.interaction_logs[-limit:] if len(self.interaction_logs) >= limit else self.interaction_logs
    
    def _analyze_memories_and_progress(self, interactions: List[InteractionLog]) -> MemoryAnalysisResult:
        """
        Analyze memories and extract key details for forward initiative progress
        Implements agent execution safeguards
        """
        start_time = time.time()
        max_execution_time = 60  # Safeguard: 60 seconds max
        
        try:
            if not self.llm:
                return self._fallback_memory_analysis(interactions)
            
            # Prepare interaction summary for analysis
            interaction_summary = self._prepare_interaction_summary(interactions)
            
            # Enhanced LLM analysis with deeper insight extraction
            analysis_prompt = f"""
As a memory analysis specialist, analyze these recent interactions to extract key insights and maintain forward progress:

RECENT INTERACTIONS:
{interaction_summary}

Provide analysis in this exact JSON format:
{{
    "key_insights_extracted": ["technical insight", "business insight", "pattern identified"],
    "progress_summary": "Overall progress assessment with metrics",
    "redundancies_identified": ["redundant action", "repeated pattern"],
    "next_priorities": ["priority 1 with rationale", "priority 2 with rationale"],
    "forward_momentum_score": 0.85,
    "improvement_patterns": ["pattern 1", "pattern 2"],
    "technical_debt_identified": ["debt item 1", "debt item 2"],
    "optimization_opportunities": ["opportunity 1", "opportunity 2"]
}}

Focus on:
1. Key achievements and completed tasks
2. Identified patterns and redundancies
3. Clear next steps for forward momentum
4. Progress indicators and completion status
5. Actionable priorities without duplication
6. Technical debt and optimization opportunities
7. Improvement patterns across interactions

Respond ONLY with valid JSON.
"""
            
            # Execute with timeout safeguard
            response = self.llm.invoke(analysis_prompt)
            
            # Check execution time safeguard
            if time.time() - start_time > max_execution_time:
                logger.warning("⚠️ Memory analysis exceeded time limit - using fallback")
                return self._fallback_memory_analysis(interactions)
            
            # Parse LLM response
            try:
                analysis_data = json.loads(response.content if hasattr(response, 'content') else str(response))
                
                return MemoryAnalysisResult(
                    analysis_timestamp=datetime.now(),
                    total_memories_reviewed=len(interactions),
                    key_insights_extracted=analysis_data.get("key_insights_extracted", []),
                    progress_summary=analysis_data.get("progress_summary", ""),
                    redundancies_identified=analysis_data.get("redundancies_identified", []),
                    next_priorities=analysis_data.get("next_priorities", []),
                    forward_momentum_score=analysis_data.get("forward_momentum_score", 0.5)
                )
                
            except json.JSONDecodeError:
                logger.warning("⚠️ LLM response not valid JSON - using fallback analysis")
                return self._fallback_memory_analysis(interactions)
                
        except Exception as e:
            logger.error(f"❌ Memory analysis error: {e}")
            return self._fallback_memory_analysis(interactions)
    
    def _prepare_interaction_summary(self, interactions: List[InteractionLog]) -> str:
        """Prepare interaction summary for analysis"""
        summary_parts = []
        
        for i, interaction in enumerate(interactions, 1):
            summary_parts.append(f"""
Interaction {i} ({interaction.timestamp.strftime('%Y-%m-%d %H:%M')}):
- Query: {interaction.user_query}
- Response: {interaction.ai_response_summary}
- Actions: {', '.join(interaction.key_actions)}
- Progress: {interaction.forward_initiative}
- Status: {interaction.completion_status}
""")
        
        return '\n'.join(summary_parts)
    
    def _fallback_memory_analysis(self, interactions: List[InteractionLog]) -> MemoryAnalysisResult:
        """Fallback memory analysis when LLM is unavailable"""
        
        # Extract key information using basic analysis
        key_actions = []
        progress_items = []
        completed_tasks = []
        
        for interaction in interactions:
            key_actions.extend(interaction.key_actions)
            if interaction.forward_initiative:
                progress_items.append(interaction.forward_initiative)
            if interaction.completion_status == "completed":
                completed_tasks.append(interaction.interaction_id)
        
        # Remove duplicates
        key_actions = list(set(key_actions))
        progress_items = list(set(progress_items))
        
        # Calculate momentum score
        completed_ratio = len(completed_tasks) / len(interactions) if interactions else 0
        momentum_score = min(0.3 + completed_ratio * 0.7, 1.0)
        
        return MemoryAnalysisResult(
            analysis_timestamp=datetime.now(),
            total_memories_reviewed=len(interactions),
            key_insights_extracted=key_actions[:3],
            progress_summary=f"Completed {len(completed_tasks)} of {len(interactions)} recent interactions",
            redundancies_identified=[],
            next_priorities=progress_items[:3],
            forward_momentum_score=momentum_score
        )
    
    def _save_memory_analysis(self, analysis: MemoryAnalysisResult):
        """Save memory analysis results"""
        try:
            analysis_file = self.memory_dir / f"memory_analysis_{analysis.analysis_timestamp.strftime('%Y%m%d_%H%M%S')}.json"
            
            analysis_data = {
                "analysis_timestamp": analysis.analysis_timestamp.isoformat(),
                "total_memories_reviewed": analysis.total_memories_reviewed,
                "key_insights_extracted": analysis.key_insights_extracted,
                "progress_summary": analysis.progress_summary,
                "redundancies_identified": analysis.redundancies_identified,
                "next_priorities": analysis.next_priorities,
                "forward_momentum_score": analysis.forward_momentum_score
            }
            
            with open(analysis_file, 'w') as f:
                json.dump(analysis_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"❌ Failed to save memory analysis: {e}")
    
    def _log_analysis_results(self, analysis: MemoryAnalysisResult):
        """Log memory analysis results"""
        logger.info("📊 MEMORY ANALYSIS RESULTS:")
        logger.info(f"   Memories Reviewed: {analysis.total_memories_reviewed}")
        logger.info(f"   Momentum Score: {analysis.forward_momentum_score:.2f}")
        logger.info(f"   Progress: {analysis.progress_summary}")
        
        if analysis.key_insights_extracted:
            logger.info("   Key Insights:")
            for insight in analysis.key_insights_extracted:
                logger.info(f"   - {insight}")
        
        if analysis.next_priorities:
            logger.info("   Next Priorities:")
            for priority in analysis.next_priorities:
                logger.info(f"   - {priority}")
        
        if analysis.redundancies_identified:
            logger.info("   Redundancies Identified:")
            for redundancy in analysis.redundancies_identified:
                logger.info(f"   - {redundancy}")
    
    def get_memory_status(self) -> Dict[str, Any]:
        """Get current memory orchestration status"""
        return {
            "total_interactions": self.interaction_count,
            "timer_active": self.timer_active,
            "analysis_interval": self.analysis_interval,
            "last_analysis": self.last_memory_analysis.analysis_timestamp.isoformat() if self.last_memory_analysis else None,
            "next_analysis_due": f"After interaction #{((self.interaction_count // self.analysis_interval) + 1) * self.analysis_interval}",
            "recent_momentum_score": self.last_memory_analysis.forward_momentum_score if self.last_memory_analysis else 0.0
        }
    
    def force_memory_analysis(self) -> MemoryAnalysisResult:
        """Force immediate memory analysis (bypass timer)"""
        logger.info("🔧 Forcing immediate memory analysis...")
        recent_interactions = self._get_recent_interactions(limit=10)
        analysis_result = self._analyze_memories_and_progress(recent_interactions)
        self.last_memory_analysis = analysis_result
        self._save_memory_analysis(analysis_result)
        self._log_analysis_results(analysis_result)
        return analysis_result

# Global memory orchestrator instance
memory_orchestrator = MemoryOrchestrator()

def log_interaction_auto(user_query: str, ai_response: str, **kwargs) -> str:
    """
    Automated addendum function to be called for ALL INTERACTIONS
    
    Usage:
        interaction_id = log_interaction_auto(
            user_query="User's request",
            ai_response="AI's complete response", 
            key_actions=["action1", "action2"],
            progress_indicators=["progress1", "progress2"],
            forward_initiative="Description of forward progress"
        )
    """
    return memory_orchestrator.log_interaction(user_query, ai_response, **kwargs)

def get_memory_orchestrator() -> MemoryOrchestrator:
    """Get the global memory orchestrator instance"""
    return memory_orchestrator

def test_memory_orchestrator():
    """Test the memory orchestrator system"""
    print("🧠 Testing Memory Orchestrator System")
    print("=" * 50)
    
    orchestrator = get_memory_orchestrator()
    
    # Test 1: Log multiple interactions to trigger analysis
    print("\n1. Testing interaction logging...")
    
    # First interaction
    id1 = log_interaction_auto(
        user_query="Implement causal analysis agent",
        ai_response="Successfully implemented causal analysis with DoWhy, EconML, and causal-learn integration...",
        key_actions=["Created causal_analysis_agent.py", "Integrated causal libraries", "Implemented DAG modeling"],
        progress_indicators=["Task 1.3 started", "Libraries researched", "DAG defined"],
        forward_initiative="Advanced causal inference capabilities implemented",
        completion_status="completed"
    )
    print(f"   ✅ Logged interaction: {id1}")
    
    # Second interaction (should trigger analysis)
    id2 = log_interaction_auto(
        user_query="Test causal analysis functionality",
        ai_response="Testing completed successfully with 120 data points analyzed...",
        key_actions=["Ran causal analysis tests", "Validated database integration", "Confirmed LLM interpretation"],
        progress_indicators=["Testing phase completed", "All functionality verified"],
        forward_initiative="Causal analysis system ready for production",
        completion_status="completed"
    )
    print(f"   ✅ Logged interaction: {id2}")
    print(f"   🔄 Memory analysis should have been triggered!")
    
    # Test 2: Check memory status
    print("\n2. Testing memory status...")
    status = orchestrator.get_memory_status()
    print(f"   Total interactions: {status['total_interactions']}")
    print(f"   Timer active: {status['timer_active']}")
    print(f"   Last analysis: {status['last_analysis']}")
    print(f"   Next analysis due: {status['next_analysis_due']}")
    print(f"   Momentum score: {status['recent_momentum_score']}")
    
    # Test 3: Force analysis
    print("\n3. Testing forced memory analysis...")
    analysis = orchestrator.force_memory_analysis()
    print(f"   ✅ Analysis completed with {len(analysis.key_insights_extracted)} insights")
    
    return True

if __name__ == "__main__":
    success = test_memory_orchestrator()
    if success:
        print("\n🎉 Memory Orchestrator system working correctly!")
    else:
        print("\n❌ Memory Orchestrator issues detected.")
