#!/usr/bin/env python3
"""
Unified Intelligence Agent
Integrates text, code, and multimodal analysis with multiple LLM providers

Features:
- Text analysis with market intelligence agents
- Code analysis with GitHub repository intelligence
- Multimodal analysis with commercial APIs and local tools
- Support for Gemini Advanced, ChatGPT Plus, and other providers
- Unified reporting and insights generation
"""

import os
import sys
import json
import logging
from typing import Dict, List, Any
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from supabase import create_client, Client

# Import all specialized agents
try:
    from agents.market_intel_agents import run_market_analysis
    from agents.code_analysis_agent import CodeAnalysisAgent
    from agents.enhanced_multimodal_agent import EnhancedMultimodalAgent
except ImportError as e:
    print(f"❌ Failed to import specialized agents: {e}")
    sys.exit(1)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UnifiedIntelligenceAgent:
    """Unified agent that orchestrates all intelligence gathering capabilities"""
    
    def __init__(self):
        # Supabase setup
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_key = os.getenv('SUPABASE_KEY')
        self.supabase = None
        
        if self.supabase_url and self.supabase_key:
            try:
                self.supabase = create_client(self.supabase_url, self.supabase_key)
                logger.info("✅ Supabase client initialized successfully")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Supabase: {e}")
        
        # Initialize specialized agents
        self.code_agent = CodeAnalysisAgent()
        self.multimodal_agent = EnhancedMultimodalAgent()
        
        logger.info("🤖 Unified Intelligence Agent initialized")

    def run_unified_analysis(self) -> Dict[str, Any]:
        """Run comprehensive analysis across all intelligence domains"""
        logger.info("🚀 Starting Unified Intelligence Analysis")
        
        try:
            # Step 1: Run text-based market analysis
            logger.info("📰 Running text-based market analysis...")
            text_analysis = run_market_analysis()
            
            # Step 2: Run code intelligence analysis
            logger.info("💻 Running code intelligence analysis...")
            code_analysis = self.code_agent.run_code_intelligence_analysis()
            
            # Step 3: Run multimodal analysis
            logger.info("🎨🎬 Running multimodal analysis...")
            multimodal_analysis = self.multimodal_agent.run_enhanced_visual_intelligence_analysis()
            
            # Step 4: Generate unified insights
            logger.info("📊 Generating unified insights report...")
            unified_insights = self._generate_unified_insights(
                text_analysis, code_analysis, multimodal_analysis
            )
            
            # Step 5: Store comprehensive results
            logger.info("💾 Storing unified intelligence...")
            stored = self._store_unified_intelligence(unified_insights)
            
            # Return final results
            final_results = {
                'success': True,
                'text_analysis': text_analysis,
                'code_analysis': code_analysis,
                'multimodal_analysis': multimodal_analysis,
                'unified_insights': unified_insights,
                'stored_successfully': stored,
                'execution_timestamp': datetime.now().isoformat()
            }
            
            logger.info("✅ Unified Intelligence Analysis completed successfully")
            return final_results
            
        except Exception as e:
            logger.error(f"❌ Unified Intelligence Analysis failed: {e}")
            return {"error": str(e), "success": False}

    def _generate_unified_insights(self, text_data: Dict, code_data: Dict, multimodal_data: Dict) -> Dict[str, Any]:
        """Generate unified insights by correlating all analysis domains"""
        try:
            # Extract key elements from each analysis
            text_trends = text_data.get('trends', [])
            text_pain_points = text_data.get('pain_points', [])
            
            code_insights = code_data.get('insights', {})
            code_trends = code_insights.get('technological_trends', {})
            
            multimodal_insights = multimodal_data.get('insights', {})
            visual_trends = multimodal_insights.get('emerging_trends', {})
            
            # Identify cross-domain correlations
            cross_domain_insights = []
            
            # Look for technology trends mentioned in text that match code trends
            text_trend_titles = [t.get('title', '').lower() for t in text_trends]
            code_trend_names = [t.lower() for t in code_trends.keys()]
            
            for text_trend in text_trend_titles:
                for code_trend in code_trend_names:
                    if text_trend in code_trend or code_trend in text_trend:
                        cross_domain_insights.append(
                            f"Text trend '{text_trend}' aligns with code trend '{code_trend}'"
                        )
            
            # Look for visual trends that match text or code trends
            visual_trend_names = [t.lower() for t in visual_trends.keys()]
            
            for visual_trend in visual_trend_names:
                for text_trend in text_trend_titles:
                    if visual_trend in text_trend or text_trend in visual_trend:
                        cross_domain_insights.append(
                            f"Visual trend '{visual_trend}' aligns with text trend '{text_trend}'"
                        )
                
                for code_trend in code_trend_names:
                    if visual_trend in code_trend or code_trend in visual_trend:
                        cross_domain_insights.append(
                            f"Visual trend '{visual_trend}' aligns with code trend '{code_trend}'"
                        )
            
            # Generate unified report
            unified_report = {
                'executive_summary': {
                    'total_text_trends': len(text_trends),
                    'total_code_trends': len(code_trends),
                    'total_visual_trends': len(visual_trends),
                    'cross_domain_correlations': len(cross_domain_insights),
                    'analysis_timestamp': datetime.now().isoformat()
                },
                'market_intelligence': {
                    'trends': text_trends,
                    'pain_points': text_pain_points
                },
                'technology_intelligence': {
                    'code_trends': code_trends,
                    'framework_adoption': code_insights.get('framework_adoption', {}),
                    'language_popularity': code_insights.get('language_popularity', {})
                },
                'visual_intelligence': {
                    'trending_objects': multimodal_insights.get('trending_objects', {}),
                    'emerging_visual_trends': visual_trends,
                    'brand_presence': multimodal_insights.get('popular_brands', {}),
                    'color_palettes': multimodal_insights.get('dominant_colors', {})
                },
                'cross_domain_insights': cross_domain_insights,
                'actionable_recommendations': self._generate_recommendations(
                    text_trends, code_trends, visual_trends
                )
            }
            
            return unified_report
            
        except Exception as e:
            logger.error(f"❌ Failed to generate unified insights: {e}")
            return {"error": str(e)}

    def _generate_recommendations(self, text_trends: List, code_trends: Dict, visual_trends: Dict) -> List[str]:
        """Generate actionable recommendations based on all intelligence domains"""
        recommendations = []
        
        # High-priority recommendations based on convergence across domains
        text_trend_titles = [t.get('title', '').lower() for t in text_trends]
        code_trend_names = [t.lower() for t in code_trends.keys()]
        visual_trend_names = [t.lower() for t in visual_trends.keys()]
        
        # Find trends that appear in multiple domains
        all_trends = set(text_trend_titles + code_trend_names + visual_trend_names)
        trend_frequency = {}
        
        for trend in all_trends:
            frequency = 0
            if trend in text_trend_titles:
                frequency += 1
            if trend in code_trend_names:
                frequency += 1
            if trend in visual_trend_names:
                frequency += 1
            trend_frequency[trend] = frequency
        
        # Recommend trends that appear in 2 or more domains
        for trend, frequency in trend_frequency.items():
            if frequency >= 2:
                recommendations.append(
                    f"High-priority opportunity: '{trend}' appears in {frequency} intelligence domains"
                )
        
        # Add general recommendations
        if text_trends:
            recommendations.append("Market validation recommended for top text-based trends")
        
        if code_trends:
            recommendations.append("Technology stack evaluation based on emerging code patterns")
        
        if visual_trends:
            recommendations.append("Visual brand strategy alignment with emerging design trends")
        
        return recommendations

    def _store_unified_intelligence(self, insights: Dict[str, Any]) -> bool:
        """Store unified intelligence insights in Supabase"""
        if not self.supabase:
            logger.warning("Supabase not available - insights not stored")
            return False
        
        try:
            # Prepare data for storage
            storage_data = {
                'analysis_type': 'unified_intelligence',
                'insights': insights,
                'timestamp': datetime.now().isoformat(),
                'source': 'unified_intelligence_agent'
            }
            
            # Store in market_intelligence table
            result = self.supabase.table('market_intelligence').insert(storage_data).execute()
            
            if result.data:
                logger.info("✅ Unified intelligence stored successfully")
                return True
            else:
                logger.error("❌ Failed to store unified intelligence")
                return False
                
        except Exception as e:
            logger.error(f"Error storing unified intelligence: {e}")
            return False

def main():
    """Main execution function"""
    print("🤖 Starting Unified Intelligence Agent")
    print("=" * 60)
    
    # Initialize agent
    agent = UnifiedIntelligenceAgent()
    
    # Run analysis
    results = agent.run_unified_analysis()
    
    # Display results
    if results.get('success'):
        print("\n✅ UNIFIED INTELLIGENCE ANALYSIS COMPLETE")
        print(f"💾 Data stored: {results['stored_successfully']}")
        
        insights = results['unified_insights']
        summary = insights.get('executive_summary', {})
        print(f"\n📈 EXECUTIVE SUMMARY:")
        print(f"   📰 Text trends analyzed: {summary.get('total_text_trends', 0)}")
        print(f"   💻 Code trends analyzed: {summary.get('total_code_trends', 0)}")
        print(f"   🎨 Visual trends analyzed: {summary.get('total_visual_trends', 0)}")
        print(f"   🔗 Cross-domain correlations: {summary.get('cross_domain_correlations', 0)}")
        
        recommendations = insights.get('actionable_recommendations', [])
        if recommendations:
            print(f"\n💡 TOP RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations[:5], 1):
                print(f"   {i}. {rec}")
        
        print(f"\n🎉 Unified intelligence gathering complete!")
    else:
        print(f"❌ Analysis failed: {results.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()
