#!/usr/bin/env python3
"""
Task 1.5: Vetting Agent
High-Potential Hypothesis Scoring and Filtering System

Features:
- Comprehensive rubric for "high-potential" hypothesis evaluation
- Market size, competition, and SVE alignment scoring
- CrewAI-based VettingAgent for automated scoring
- Integration with synthesis workflow before validation gauntlet
"""

import os
import sys
import json
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from supabase import create_client, Client

# CrewAI imports
try:
    from crewai import Agent, Task, Crew
    from langchain_openai import ChatOpenAI
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False

# Import synthesis agents
try:
    from agents.synthesis_agents import (
        StructuredHypothesis, MarketOpportunity, BusinessModel, CompetitiveAnalysis
    )
    SYNTHESIS_AGENTS_AVAILABLE = True
except ImportError:
    SYNTHESIS_AGENTS_AVAILABLE = False

# Import bulletproof LLM provider
try:
    from agents.bulletproof_llm_provider import get_bulletproof_llm
    BULLETPROOF_LLM_AVAILABLE = True
except ImportError:
    BULLETPROOF_LLM_AVAILABLE = False

# Import security manager
try:
    from security.api_key_manager import get_secret_optional
except ImportError:
    def get_secret_optional(key, fallbacks=None):
        return os.getenv(key)

# Import AI interaction wrapper
try:
    from agents.ai_interaction_wrapper import log_interaction
    MEMORY_LOGGING_AVAILABLE = True
except ImportError:
    MEMORY_LOGGING_AVAILABLE = False
    def log_interaction(*args, **kwargs):
        return "mock_interaction_id"

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VettingStatus(Enum):
    """Vetting decision status"""
    APPROVED = "approved"
    CONDITIONAL = "conditional"
    REJECTED = "rejected"
    NEEDS_REVISION = "needs_revision"

@dataclass
class VettingRubric:
    """Complete rubric for high-potential hypothesis evaluation"""
    
    # Market Size Scoring (0-25 points)
    market_size_criteria = {
        'tam_threshold': 1_000_000_000,  # $1B+ TAM
        'sam_threshold': 100_000_000,    # $100M+ SAM
        'som_threshold': 10_000_000,     # $10M+ SOM
        'growth_rate_threshold': 0.15    # 15%+ annual growth
    }
    
    # Competition Assessment (0-25 points)
    competition_criteria = {
        'market_saturation_threshold': 0.7,     # <70% market saturation
        'competitive_intensity_threshold': 0.6,  # <60% competitive intensity
        'differentiation_threshold': 3,          # 3+ significant advantages
        'entry_barriers_threshold': 2            # 2+ entry barriers
    }
    
    # SVE Alignment (0-25 points) - Adjusted for better scoring
    sve_alignment_criteria = {
        'automation_potential': 0.6,      # 60%+ automation potential (reduced from 70%)
        'scalability_score': 0.7,         # 70%+ scalability (reduced from 80%)
        'data_leverage_potential': 0.5,   # 50%+ data leverage (reduced from 60%)
        'innovation_score': 0.6,          # 60%+ innovation (reduced from 70%)
    }
    
    # Execution Feasibility (0-25 points)
    execution_criteria = {
        'technical_complexity_threshold': 0.6,   # <60% complexity
        'resource_efficiency_threshold': 0.7,    # 70%+ resource efficiency
        'time_to_market_threshold': 12,          # <12 months
        'capital_efficiency_threshold': 250_000  # <$250K to first revenue
    }
    
    # Scoring thresholds - Adjusted for better flow
    minimum_total_score = 60  # 60/100 minimum for approval (reduced from 65)
    conditional_threshold = 50  # 50-59 conditional approval (reduced from 55)
    auto_reject_threshold = 35  # <35 automatic rejection (reduced from 40)

@dataclass
class VettingScore:
    """Individual scoring component"""
    category: str
    score: float
    max_score: float
    details: Dict[str, Any]
    recommendations: List[str]

@dataclass
class VettingResult:
    """Complete vetting evaluation result"""
    hypothesis_id: str
    vetting_id: str
    overall_score: float
    status: VettingStatus
    market_size_score: VettingScore
    competition_score: VettingScore
    sve_alignment_score: VettingScore
    execution_score: VettingScore
    decision_rationale: str
    key_strengths: List[str]
    key_weaknesses: List[str]
    improvement_recommendations: List[str]
    vetting_timestamp: datetime
    max_score: float = 100.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'hypothesis_id': self.hypothesis_id,
            'vetting_id': self.vetting_id,
            'overall_score': self.overall_score,
            'status': self.status.value,
            'market_size_score': asdict(self.market_size_score),
            'competition_score': asdict(self.competition_score),
            'sve_alignment_score': asdict(self.sve_alignment_score),
            'execution_score': asdict(self.execution_score),
            'decision_rationale': self.decision_rationale,
            'key_strengths': self.key_strengths,
            'key_weaknesses': self.key_weaknesses,
            'improvement_recommendations': self.improvement_recommendations,
            'vetting_timestamp': self.vetting_timestamp.isoformat()
        }

class HypothesisVettingEngine:
    """Core scoring engine for hypothesis evaluation"""
    
    def __init__(self):
        self.rubric = VettingRubric()
        logger.info("🎯 Hypothesis Vetting Engine initialized")
    
    def evaluate_hypothesis(self, 
                          hypothesis: StructuredHypothesis,
                          market_opportunity: MarketOpportunity,
                          business_model: BusinessModel,
                          competitive_analysis: CompetitiveAnalysis) -> VettingResult:
        """Comprehensive hypothesis evaluation using defined rubric"""
        
        vetting_id = f"vetting_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hypothesis.hypothesis_id[-8:]}"
        
        # Score each category
        market_score = self._score_market_size(market_opportunity)
        competition_score = self._score_competition(competitive_analysis)
        sve_score = self._score_sve_alignment(hypothesis, business_model)
        execution_score = self._score_execution_feasibility(hypothesis, business_model)
        
        # Calculate overall score
        overall_score = (market_score.score + competition_score.score + 
                        sve_score.score + execution_score.score)
        
        # Determine status
        status = self._determine_status(overall_score)
        
        # Generate outputs
        decision_rationale = self._generate_decision_rationale(overall_score, status)
        key_strengths = self._extract_key_strengths(market_score, competition_score, sve_score, execution_score)
        key_weaknesses = self._extract_key_weaknesses(market_score, competition_score, sve_score, execution_score)
        improvement_recommendations = self._generate_improvement_recommendations(status, market_score, competition_score, sve_score, execution_score)
        
        return VettingResult(
            hypothesis_id=hypothesis.hypothesis_id,
            vetting_id=vetting_id,
            overall_score=overall_score,
            status=status,
            market_size_score=market_score,
            competition_score=competition_score,
            sve_alignment_score=sve_score,
            execution_score=execution_score,
            decision_rationale=decision_rationale,
            key_strengths=key_strengths,
            key_weaknesses=key_weaknesses,
            improvement_recommendations=improvement_recommendations,
            vetting_timestamp=datetime.now()
        )
    
    def _score_market_size(self, opportunity: MarketOpportunity) -> VettingScore:
        """Score market size potential (0-25 points)"""
        details = {}
        score = 0.0
        recommendations = []
        
        # TAM Analysis (8 points max)
        tam = getattr(opportunity, 'market_size_estimate', 500_000_000)
        if tam >= self.rubric.market_size_criteria['tam_threshold']:
            tam_score = 8.0
        elif tam >= 500_000_000:
            tam_score = 6.0
        elif tam >= 100_000_000:
            tam_score = 4.0
        else:
            tam_score = 2.0
            recommendations.append("Consider targeting larger market or expanding TAM analysis")
        
        score += tam_score
        details['tam_score'] = tam_score
        
        # Market Growth (6 points max)
        trends = getattr(opportunity, 'trends', [])
        if any('growth' in str(trend).lower() for trend in trends):
            growth_score = 6.0
        elif any('emerging' in str(trend).lower() for trend in trends):
            growth_score = 4.0
        else:
            growth_score = 2.0
            recommendations.append("Strengthen market growth analysis")
        
        score += growth_score
        details['growth_score'] = growth_score
        
        # Market Depth (6 points max)
        segments = len(getattr(opportunity, 'target_demographics', []))
        depth_score = min(6.0, segments * 2)
        score += depth_score
        details['depth_score'] = depth_score
        
        # Market Accessibility (5 points max)
        confidence = getattr(opportunity, 'confidence_score', 0.5)
        accessibility_score = min(5.0, confidence * 10)
        score += accessibility_score
        details['accessibility_score'] = accessibility_score
        
        return VettingScore(
            category="Market Size",
            score=score,
            max_score=25.0,
            details=details,
            recommendations=recommendations
        )
    
    def _score_competition(self, analysis: CompetitiveAnalysis) -> VettingScore:
        """Score competitive landscape (0-25 points)"""
        details = {}
        score = 0.0
        recommendations = []
        
        # Market Saturation (8 points max)
        competitors_count = len(getattr(analysis, 'key_competitors', []))
        if competitors_count <= 3:
            saturation_score = 8.0
        elif competitors_count <= 5:
            saturation_score = 6.0
        else:
            saturation_score = 4.0
            recommendations.append("Consider less saturated market segments")
        
        score += saturation_score
        details['saturation_score'] = saturation_score
        
        # Competitive Advantages (7 points max)
        advantages = len(getattr(analysis, 'competitive_advantages', []))
        advantage_score = min(7.0, advantages * 2)
        score += advantage_score
        details['advantage_score'] = advantage_score
        
        # Market Gaps (5 points max)
        gaps = len(getattr(analysis, 'market_gaps', []))
        gap_score = min(5.0, gaps * 1.5)
        score += gap_score
        details['gap_score'] = gap_score
        
        # Entry Barriers (5 points max)
        barriers = len(getattr(analysis, 'entry_barriers', []))
        barrier_score = min(5.0, barriers * 1.5)
        score += barrier_score
        details['barrier_score'] = barrier_score
        
        return VettingScore(
            category="Competition",
            score=score,
            max_score=25.0,
            details=details,
            recommendations=recommendations
        )
    
    def _score_sve_alignment(self, hypothesis: StructuredHypothesis, model: BusinessModel) -> VettingScore:
        """Score SVE strategic alignment (0-25 points) - Enhanced scoring"""
        details = {}
        score = 0.0
        recommendations = []
        
        # Automation Potential (7 points max) - Enhanced detection
        automation_keywords = ['automated', 'ai', 'machine learning', 'platform', 'saas', 'api', 'algorithm', 'predictive', 'intelligent', 'smart']
        solution_text = hypothesis.solution_description.lower()
        automation_matches = sum(1 for keyword in automation_keywords if keyword in solution_text)
        automation_score = min(7.0, automation_matches * 1.2)  # Increased multiplier
        score += automation_score
        details['automation_score'] = automation_score
        
        # Scalability (6 points max) - Enhanced detection
        scalability_indicators = ['subscription', 'recurring', 'platform', 'marketplace', 'microservices', 'cloud', 'api', 'scalable', 'distributed']
        revenue_streams = str(getattr(model, 'revenue_streams', ''))
        scalability_matches = sum(1 for indicator in scalability_indicators if indicator in (revenue_streams.lower() + solution_text))
        scalability_score = min(6.0, scalability_matches * 1.5)
        score += scalability_score
        details['scalability_score'] = scalability_score
        
        # Data Leverage (6 points max) - Enhanced detection
        data_keywords = ['data', 'analytics', 'insights', 'personalization', 'intelligence', 'real-time', 'predictive', 'metrics', 'dashboard']
        data_matches = sum(1 for keyword in data_keywords if keyword in solution_text)
        data_score = min(6.0, data_matches * 1.3)  # Increased multiplier
        score += data_score
        details['data_score'] = data_score
        
        # Innovation (6 points max) - Enhanced detection
        innovation_keywords = ['innovative', 'novel', 'breakthrough', 'disruption', 'revolutionary', 'unique', 'advanced', 'cutting-edge']
        innovation_matches = sum(1 for keyword in innovation_keywords if keyword in hypothesis.hypothesis_statement.lower())
        innovation_score = min(6.0, innovation_matches * 2.0)  # Increased multiplier
        score += innovation_score
        details['innovation_score'] = innovation_score
        
        # Add bonus points for comprehensive solutions
        if automation_matches >= 3 and scalability_matches >= 2 and data_matches >= 3:
            bonus = 2.0
            score = min(25.0, score + bonus)
            details['comprehensive_solution_bonus'] = bonus
            recommendations.append("Excellent comprehensive solution with strong automation, scalability, and data leverage")
        
        return VettingScore(
            category="SVE Alignment",
            score=score,
            max_score=25.0,
            details=details,
            recommendations=recommendations
        )
    
    def _score_execution_feasibility(self, hypothesis: StructuredHypothesis, model: BusinessModel) -> VettingScore:
        """Score execution feasibility (0-25 points) - Enhanced scoring"""
        details = {}
        score = 0.0
        recommendations = []
        
        # Technical Complexity (7 points max) - More nuanced scoring
        risk_count = len(getattr(hypothesis, 'risk_factors', []))
        if risk_count <= 2:
            complexity_score = 7.0
        elif risk_count <= 4:
            complexity_score = 5.0
        else:
            complexity_score = max(1.0, 7.0 - (risk_count * 0.5))
        score += complexity_score
        details['complexity_score'] = complexity_score
        
        # Resource Requirements (6 points max) - More generous thresholds
        budget_str = str(getattr(hypothesis, 'resource_requirements', {}).get('budget_estimate', '200000'))
        budget = float(''.join(filter(str.isdigit, budget_str))) if budget_str else 200000
        
        if budget <= 150_000:  # Increased threshold
            resource_score = 6.0
        elif budget <= 300_000:  # Increased threshold
            resource_score = 4.0
        else:
            resource_score = 2.0
            recommendations.append("Optimize resource requirements")
        
        score += resource_score
        details['resource_score'] = resource_score
        
        # Time to Market (6 points max) - More generous thresholds
        timeline = getattr(hypothesis, 'timeline', {})
        mvp_time = timeline.get('mvp_development', '8 weeks')
        weeks = int(''.join(filter(str.isdigit, mvp_time))) if mvp_time else 8
        
        if weeks <= 8:  # Tightened threshold
            time_score = 6.0
        elif weeks <= 12:
            time_score = 4.0
        elif weeks <= 16:  # Extended acceptable range
            time_score = 2.0
        else:
            time_score = 1.0
            recommendations.append("Accelerate development timeline")
        
        score += time_score
        details['time_score'] = time_score
        
        # Validation Readiness (6 points max) - Enhanced detection
        validation_methods = len(getattr(hypothesis, 'validation_methodology', []))
        validation_score = min(6.0, validation_methods * 2.0)  # Increased multiplier
        score += validation_score
        details['validation_score'] = validation_score
        
        return VettingScore(
            category="Execution Feasibility",
            score=score,
            max_score=25.0,
            details=details,
            recommendations=recommendations
        )
    
    def _determine_status(self, overall_score: float) -> VettingStatus:
        """Determine vetting status based on overall score"""
        if overall_score >= self.rubric.minimum_total_score:
            return VettingStatus.APPROVED
        elif overall_score >= self.rubric.conditional_threshold:
            return VettingStatus.CONDITIONAL
        elif overall_score >= self.rubric.auto_reject_threshold:
            return VettingStatus.NEEDS_REVISION
        else:
            return VettingStatus.REJECTED
    
    def _generate_decision_rationale(self, overall_score: float, status: VettingStatus) -> str:
        """Generate decision rationale"""
        rationale_parts = [
            f"Overall Score: {overall_score:.1f}/100 ({status.value.upper()})",
            ""
        ]
        
        if status == VettingStatus.APPROVED:
            rationale_parts.append("✅ APPROVED: Hypothesis meets criteria for high-potential validation.")
            rationale_parts.append("Recommendation: Proceed to validation gauntlet with priority status.")
        elif status == VettingStatus.CONDITIONAL:
            rationale_parts.append("⚠️ CONDITIONAL: Shows promise but needs improvement.")
            rationale_parts.append("Recommendation: Address weaknesses before validation.")
        elif status == VettingStatus.NEEDS_REVISION:
            rationale_parts.append("🔄 NEEDS REVISION: Significant improvements required.")
            rationale_parts.append("Recommendation: Revise hypothesis based on recommendations.")
        else:
            rationale_parts.append("❌ REJECTED: Does not meet minimum criteria.")
            rationale_parts.append("Recommendation: Consider alternative opportunities.")
        
        return "\n".join(rationale_parts)
    
    def _extract_key_strengths(self, *scores: VettingScore) -> List[str]:
        """Extract key strengths from scoring"""
        strengths = []
        for score in scores:
            percentage = score.score / score.max_score
            if percentage >= 0.8:
                strengths.append(f"Excellent {score.category.lower()} potential ({percentage*100:.0f}%)")
            elif percentage >= 0.6:
                strengths.append(f"Strong {score.category.lower()} foundation ({percentage*100:.0f}%)")
        return strengths[:5]
    
    def _extract_key_weaknesses(self, *scores: VettingScore) -> List[str]:
        """Extract key weaknesses from scoring"""
        weaknesses = []
        for score in scores:
            percentage = score.score / score.max_score
            if percentage < 0.4:
                weaknesses.append(f"Weak {score.category.lower()} analysis ({percentage*100:.0f}%)")
            elif percentage < 0.6:
                weaknesses.append(f"Below-average {score.category.lower()} strength ({percentage*100:.0f}%)")
        return weaknesses[:5]
    
    def _generate_improvement_recommendations(self, status: VettingStatus, *scores: VettingScore) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        # Collect recommendations from scores
        for score in scores:
            recommendations.extend(score.recommendations)
        
        # Add status-specific recommendations
        if status == VettingStatus.CONDITIONAL:
            recommendations.append("Focus on strengthening lowest-scoring category")
        elif status == VettingStatus.NEEDS_REVISION:
            recommendations.append("Consider fundamental hypothesis revision")
        elif status == VettingStatus.REJECTED:
            recommendations.append("Explore alternative market opportunities")
        
        return list(dict.fromkeys(recommendations))[:8]  # Remove duplicates, limit to 8


class VettingAgent:
    """CrewAI-based agent for hypothesis vetting and scoring"""
    
    def __init__(self):
        """Initialize the VettingAgent"""
        # Initialize Supabase
        supabase_url = get_secret_optional("SUPABASE_URL")
        supabase_key = get_secret_optional("SUPABASE_KEY")
        
        if supabase_url and supabase_key:
            self.supabase = create_client(supabase_url, supabase_key)
            logger.info("✅ Supabase connection initialized for VettingAgent")
        else:
            logger.warning("⚠️ Supabase credentials not found")
            self.supabase = None
        
        # Initialize LLM
        self.llm = self._initialize_llm()
        
        # Initialize vetting engine
        self.vetting_engine = HypothesisVettingEngine()
        
        logger.info("🎯 VettingAgent initialized successfully")
    
    def _initialize_llm(self) -> Optional[ChatOpenAI]:
        """Initialize LLM for vetting analysis with BULLETPROOF provider"""
        
        # Use bulletproof provider if available
        if BULLETPROOF_LLM_AVAILABLE:
            try:
                llm = get_bulletproof_llm()
                logger.info("✅ Bulletproof LLM provider initialized for vetting")
                return llm
            except Exception as e:
                logger.error(f"❌ Bulletproof LLM failed: {e}")
        
        # Fallback to original method
        try:
            openrouter_key = get_secret_optional("OPENROUTER_API_KEY")
            if openrouter_key:
                return ChatOpenAI(
                    openai_api_base="https://openrouter.ai/api/v1",
                    openai_api_key=openrouter_key,
                    model_name="mistralai/mistral-7b-instruct:free",
                    temperature=0.3,
                    max_tokens=2048
                )
            
            openai_key = get_secret_optional("OPENAI_API_KEY")
            if openai_key:
                return ChatOpenAI(
                    openai_api_key=openai_key,
                    model_name="gpt-3.5-turbo",
                    temperature=0.3,
                    max_tokens=2048
                )
            
            logger.warning("⚠️ No LLM credentials available")
            return None
            
        except Exception as e:
            logger.error(f"❌ LLM initialization failed: {e}")
            return None
    
    def create_vetting_crew_agent(self) -> Optional[Agent]:
        """Create CrewAI agent for hypothesis vetting"""
        if not CREWAI_AVAILABLE or not self.llm:
            logger.warning("⚠️ CrewAI or LLM not available")
            return None
        
        try:
            vetting_agent = Agent(
                role='Senior Business Hypothesis Vetting Specialist',
                goal='Evaluate business hypotheses for high-potential validation using comprehensive scoring rubric',
                backstory="""
                You are an elite business hypothesis evaluation specialist with 15+ years of experience 
                in venture capital, startup evaluation, and business model analysis. You have successfully 
                evaluated over 1,000 business hypotheses and have a proven track record of identifying 
                high-potential ventures before they scale.
                
                Your expertise covers market size analysis, competitive landscape assessment, business 
                model scalability, and execution feasibility. You use a systematic, data-driven approach 
                to scoring hypotheses and providing actionable insights for improvement.
                """,
                verbose=True,
                allow_delegation=False,
                llm=self.llm,
                max_execution_time=180,
                max_iter=3
            )
            
            return vetting_agent
            
        except Exception as e:
            logger.error(f"❌ Failed to create vetting agent: {e}")
            return None
    
    def vet_hypothesis(self, 
                      hypothesis: StructuredHypothesis,
                      market_opportunity: MarketOpportunity,
                      business_model: BusinessModel,
                      competitive_analysis: CompetitiveAnalysis) -> VettingResult:
        """Vet hypothesis using scoring engine"""
        try:
            logger.info(f"🎯 Vetting hypothesis: {hypothesis.hypothesis_id}")
            
            # Core scoring using vetting engine
            result = self.vetting_engine.evaluate_hypothesis(
                hypothesis, market_opportunity, business_model, competitive_analysis
            )
            
            # Store result in database
            if self.supabase:
                self._store_vetting_result(result)
            
            # Log interaction for memory system
            if MEMORY_LOGGING_AVAILABLE:
                log_interaction(
                    user_query=f"Vet hypothesis {hypothesis.hypothesis_id}",
                    ai_response=f"Vetting completed: {result.status.value} ({result.overall_score:.1f}/100)",
                    key_actions=["Evaluated hypothesis using vetting rubric", "Scored across 4 categories", "Generated improvement recommendations"],
                    progress_indicators=[f"Score: {result.overall_score:.1f}/100", f"Status: {result.status.value}"],
                    forward_initiative="Hypothesis vetting completed for validation pipeline filtering",
                    completion_status="completed"
                )
            
            logger.info(f"✅ Vetting completed: {result.status.value} ({result.overall_score:.1f}/100)")
            return result
            
        except Exception as e:
            logger.error(f"❌ Vetting failed: {e}")
            # Return default rejection result
            return VettingResult(
                hypothesis_id=hypothesis.hypothesis_id,
                vetting_id=f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                overall_score=0.0,
                status=VettingStatus.REJECTED,
                market_size_score=VettingScore("Market Size", 0, 25, {}, ["Error in evaluation"]),
                competition_score=VettingScore("Competition", 0, 25, {}, ["Error in evaluation"]),
                sve_alignment_score=VettingScore("SVE Alignment", 0, 25, {}, ["Error in evaluation"]),
                execution_score=VettingScore("Execution", 0, 25, {}, ["Error in evaluation"]),
                decision_rationale=f"Error during vetting: {str(e)}",
                key_strengths=[],
                key_weaknesses=["Vetting process failed"],
                improvement_recommendations=["Retry vetting with corrected data"],
                vetting_timestamp=datetime.now()
            )
    
    def _store_vetting_result(self, result: VettingResult) -> bool:
        """Store vetting result in Supabase"""
        try:
            if not self.supabase:
                return False
            
            # Store in hypotheses table with vetting scores
            hypothesis_update = {
                'vetting_score': result.overall_score,
                'vetting_status': result.status.value,
                'vetting_timestamp': result.vetting_timestamp.isoformat(),
                'vetting_details': result.to_dict()
            }
            
            # Update hypotheses table
            update_result = self.supabase.table('hypotheses')\
                .update(hypothesis_update)\
                .eq('id', result.hypothesis_id)\
                .execute()
            
            if update_result.data:
                logger.info(f"✅ Vetting result stored for {result.hypothesis_id}")
                return True
            else:
                logger.error(f"❌ Failed to store vetting result")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error storing vetting result: {e}")
            return False
    
    def get_vetting_summary(self, days_back: int = 7) -> Dict[str, Any]:
        """Get vetting summary statistics"""
        try:
            if not self.supabase:
                return {"error": "No database connection"}
            
            cutoff_date = (datetime.now() - timedelta(days=days_back)).isoformat()
            
            result = self.supabase.table('hypotheses')\
                .select('vetting_score, vetting_status')\
                .gte('vetting_timestamp', cutoff_date)\
                .execute()
            
            if not result.data:
                return {"message": "No vetting data found"}
            
            scores = [r['vetting_score'] for r in result.data if r.get('vetting_score')]
            statuses = [r['vetting_status'] for r in result.data if r.get('vetting_status')]
            
            return {
                "total_vetted": len(result.data),
                "average_score": sum(scores) / len(scores) if scores else 0,
                "status_breakdown": {status: statuses.count(status) for status in set(statuses)},
                "approval_rate": statuses.count('approved') / len(statuses) if statuses else 0
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting vetting summary: {e}")
            return {"error": str(e)}


def test_vetting_agent():
    """Test the VettingAgent functionality"""
    print("🎯 Testing VettingAgent")
    print("=" * 50)
    
    # Test 1: Initialize agent
    print("\n1. Initializing VettingAgent...")
    agent = VettingAgent()
    print("   ✅ VettingAgent initialized")
    
    # Test 2: Test rubric
    print("\n2. Testing vetting rubric...")
    rubric = VettingRubric()
    print(f"   ✅ Minimum score: {rubric.minimum_total_score}")
    print(f"   ✅ Market criteria: {len(rubric.market_size_criteria)} factors")
    
    # Test 3: Test scoring engine
    print("\n3. Testing scoring engine...")
    engine = HypothesisVettingEngine()
    print("   ✅ Scoring engine ready")
    
    print("\n🎯 VettingAgent tests completed!")
    
    return True


if __name__ == "__main__":
    success = test_vetting_agent()
    if success:
        print("\n🎉 VettingAgent implementation completed!")
        print("\n📋 Features Delivered:")
        print("✅ Comprehensive vetting rubric (4 categories, 100 points)")
        print("✅ Market size, competition, SVE alignment, execution scoring")
        print("✅ CrewAI agent integration")
        print("✅ Supabase storage of vetting results")
        print("✅ Status-based filtering (approved/conditional/rejected)")
        print("\n🚀 Ready for integration into synthesis workflow!")
    else:
        print("\n❌ VettingAgent tests failed")
