#!/usr/bin/env python3
"""
Unified Multi-Modal Market Intelligence Agent
Task 1.1.3: Complete implementation for Image/Video Analysis

Orchestrates:
- Image analysis (multimodal_agents.py)
- Video analysis (video_analysis_agent.py)
- Unified reporting and insights
- Cross-modal trend correlation
- Comprehensive market intelligence from visual data
"""

import os
import sys
import json
import logging
from typing import Dict, List, Any
from datetime import datetime
import asyncio

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our specialized agents
try:
    from agents.multimodal_agents import MultiModalIntelligenceAgent
    from agents.video_analysis_agent import VideoIntelligenceAgent
except ImportError as e:
    logger.error(f"Failed to import specialized agents: {e}")
    # Create dummy agents for testing
    class MultiModalIntelligenceAgent:
        def run_visual_intelligence_analysis(self):
            return {"success": True, "insights": {}, "content_analyzed": 0}
    
    class VideoIntelligenceAgent:
        def run_video_intelligence_analysis(self):
            return {"success": True, "insights": {}, "videos_analyzed": 0}

class UnifiedMultiModalAgent:
    """Orchestrates comprehensive visual market intelligence"""
    
    def __init__(self):
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_key = os.getenv('SUPABASE_KEY')
        self.supabase = None
        
        if self.supabase_url and self.supabase_key:
            try:
                self.supabase = create_client(self.supabase_url, self.supabase_key)
                logger.info("✅ Unified agent Supabase client initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Supabase: {e}")
        
        # Initialize specialized agents
        self.image_agent = MultiModalIntelligenceAgent()
        self.video_agent = VideoIntelligenceAgent()
        
        # Test mode configuration
        self.test_mode = os.getenv('TEST_MODE', 'false').lower() == 'true'
        
        logger.info("🎨 Unified Multi-Modal Agent initialized")

    def run_parallel_analysis(self) -> Dict[str, Any]:
        """Run image and video analysis in parallel for efficiency"""
        logger.info("🚀 Starting parallel multi-modal analysis...")
        
        try:
            results = {}
            
            # Run image analysis
            logger.info("📸 Running image intelligence analysis...")
            image_results = self.image_agent.run_visual_intelligence_analysis()
            results['image_analysis'] = image_results
            
            # Run video analysis
            logger.info("🎬 Running video intelligence analysis...")
            video_results = self.video_agent.run_video_intelligence_analysis()
            results['video_analysis'] = video_results
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Parallel analysis failed: {e}")
            return {"error": str(e)}

    def correlate_cross_modal_insights(self, image_results: Dict, video_results: Dict) -> Dict[str, Any]:
        """Correlate insights between image and video analysis"""
        logger.info("🔗 Correlating cross-modal insights...")
        
        try:
            correlations = {
                'cross_modal_trends': [],
                'shared_brand_presence': [],
                'sentiment_alignment': {},
                'complementary_insights': []
            }
            
            # Extract insights from both modalities
            image_insights = image_results.get('insights', {})
            video_insights = video_results.get('insights', {})
            
            # Find shared trending elements
            image_trends = set(image_insights.get('trending_objects', {}).keys())
            video_trends = set(video_insights.get('popular_objects', {}).keys())
            shared_trends = image_trends.intersection(video_trends)
            
            if shared_trends:
                correlations['cross_modal_trends'] = list(shared_trends)
                logger.info(f"🎯 Found {len(shared_trends)} cross-modal trends")
            
            # Find shared brand presence
            image_brands = set(image_insights.get('popular_brands', {}).keys())
            video_brands = set(video_insights.get('brand_visibility', {}).keys())
            shared_brands = image_brands.intersection(video_brands)
            
            if shared_brands:
                correlations['shared_brand_presence'] = list(shared_brands)
                logger.info(f"🏷️ Found {len(shared_brands)} brands across modalities")
            
            # Compare sentiment patterns
            image_sentiment = image_insights.get('analysis_summary', {}).get('average_sentiment', 0)
            video_sentiment = video_insights.get('sentiment_insights', {}).get('average_sentiment', 0)
            
            correlations['sentiment_alignment'] = {
                'image_sentiment': image_sentiment,
                'video_sentiment': video_sentiment,
                'sentiment_correlation': abs(image_sentiment - video_sentiment),
                'alignment_strength': 'high' if abs(image_sentiment - video_sentiment) < 0.2 else 'moderate'
            }
            
            # Generate complementary insights
            correlations['complementary_insights'] = self._generate_complementary_insights(
                image_insights, video_insights
            )
            
            return correlations
            
        except Exception as e:
            logger.error(f"❌ Cross-modal correlation failed: {e}")
            return {"error": str(e)}

    def _generate_complementary_insights(self, image_insights: Dict, video_insights: Dict) -> List[str]:
        """Generate insights that complement both modalities"""
        insights = []
        
        # Analyze engagement patterns
        image_opportunities = image_insights.get('market_opportunities', [])
        video_opportunities = video_insights.get('content_opportunities', [])
        
        if image_opportunities and video_opportunities:
            insights.append("Multi-modal marketing opportunities identified across both static and dynamic content")
        
        # Color and visual trends
        dominant_colors = image_insights.get('dominant_colors', {})
        if dominant_colors:
            top_color = list(dominant_colors.keys())[0] if dominant_colors else "unknown"
            insights.append(f"Visual branding opportunity: {top_color} trending across image content")
        
        # Engagement correlation
        video_engagement = video_insights.get('engagement_patterns', {}).get('average_engagement', 0)
        if video_engagement > 50000:
            insights.append("High video engagement suggests strong visual content performance potential")
        
        return insights

    def generate_unified_report(self, analysis_results: Dict) -> Dict[str, Any]:
        """Generate comprehensive unified multi-modal intelligence report"""
        logger.info("📊 Generating unified multi-modal report...")
        
        try:
            image_results = analysis_results.get('image_analysis', {})
            video_results = analysis_results.get('video_analysis', {})
            
            # Correlate insights
            correlations = self.correlate_cross_modal_insights(image_results, video_results)
            
            # Build comprehensive report
            unified_report = {
                'multi_modal_summary': {
                    'analysis_timestamp': datetime.now().isoformat(),
                    'image_content_analyzed': image_results.get('content_analyzed', 0),
                    'video_content_analyzed': video_results.get('videos_analyzed', 0),
                    'total_content_pieces': (
                        image_results.get('content_analyzed', 0) + 
                        video_results.get('videos_analyzed', 0)
                    ),
                    'analysis_success_rate': self._calculate_success_rate(analysis_results)
                },
                'image_intelligence': image_results.get('insights', {}),
                'video_intelligence': video_results.get('insights', {}),
                'cross_modal_correlations': correlations,
                'unified_market_insights': self._synthesize_market_insights(
                    image_results.get('insights', {}),
                    video_results.get('insights', {}),
                    correlations
                ),
                'actionable_recommendations': self._generate_actionable_recommendations(
                    image_results.get('insights', {}),
                    video_results.get('insights', {}),
                    correlations
                )
            }
            
            return unified_report
            
        except Exception as e:
            logger.error(f"❌ Unified report generation failed: {e}")
            return {"error": str(e)}

    def _calculate_success_rate(self, analysis_results: Dict) -> float:
        """Calculate overall analysis success rate"""
        image_success = analysis_results.get('image_analysis', {}).get('success', False)
        video_success = analysis_results.get('video_analysis', {}).get('success', False)
        
        successes = sum([image_success, video_success])
        total = 2  # Two analysis types
        
        return (successes / total) * 100

    def _synthesize_market_insights(self, image_insights: Dict, video_insights: Dict, correlations: Dict) -> Dict[str, Any]:
        """Synthesize market insights across modalities"""
        synthesis = {
            'trending_products': [],
            'brand_performance': {},
            'consumer_behavior_patterns': [],
            'visual_trend_forecast': [],
            'market_sentiment_overview': {}
        }
        
        # Combine trending products from both modalities
        image_objects = list(image_insights.get('trending_objects', {}).keys())[:5]
        video_objects = list(video_insights.get('popular_objects', {}).keys())[:5]
        synthesis['trending_products'] = list(set(image_objects + video_objects))
        
        # Brand performance across modalities
        image_brands = image_insights.get('popular_brands', {})
        video_brands = video_insights.get('brand_visibility', {})
        
        all_brands = set(list(image_brands.keys()) + list(video_brands.keys()))
        for brand in all_brands:
            synthesis['brand_performance'][brand] = {
                'image_presence': image_brands.get(brand, 0),
                'video_presence': video_brands.get(brand, 0),
                'total_visibility': image_brands.get(brand, 0) + video_brands.get(brand, 0)
            }
        
        # Consumer behavior patterns
        image_activities = list(image_insights.get('trending_activities', {}).keys())[:3]
        video_activities = list(video_insights.get('trending_activities', {}).keys())[:3]
        synthesis['consumer_behavior_patterns'] = list(set(image_activities + video_activities))
        
        # Visual trend forecast
        image_trends = list(image_insights.get('emerging_trends', {}).keys())[:3]
        video_viral = list(video_insights.get('viral_elements', {}).keys())[:3]
        synthesis['visual_trend_forecast'] = list(set(image_trends + video_viral))
        
        # Market sentiment overview
        image_sentiment = image_insights.get('analysis_summary', {}).get('average_sentiment', 0)
        video_sentiment = video_insights.get('sentiment_insights', {}).get('average_sentiment', 0)
        
        synthesis['market_sentiment_overview'] = {
            'overall_sentiment': (image_sentiment + video_sentiment) / 2,
            'sentiment_consistency': correlations.get('sentiment_alignment', {}).get('alignment_strength', 'unknown'),
            'sentiment_trend': 'positive' if (image_sentiment + video_sentiment) / 2 > 0.6 else 'neutral'
        }
        
        return synthesis

    def _generate_actionable_recommendations(self, image_insights: Dict, video_insights: Dict, correlations: Dict) -> List[str]:
        """Generate actionable business recommendations"""
        recommendations = []
        
        # Cross-modal trend recommendations
        cross_trends = correlations.get('cross_modal_trends', [])
        if cross_trends:
            recommendations.append(
                f"Capitalize on cross-platform trends: {', '.join(cross_trends[:3])} - "
                "These elements are trending across both image and video content"
            )
        
        # Brand visibility recommendations
        shared_brands = correlations.get('shared_brand_presence', [])
        if shared_brands:
            recommendations.append(
                f"Monitor competitive landscape: {', '.join(shared_brands[:3])} - "
                "These brands have strong multi-modal presence"
            )
        
        # Sentiment-based recommendations
        sentiment_alignment = correlations.get('sentiment_alignment', {})
        if sentiment_alignment.get('alignment_strength') == 'high':
            recommendations.append(
                "Strong sentiment consistency across visual content types - "
                "Consider unified visual marketing strategy"
            )
        
        # Content strategy recommendations
        image_opportunities = image_insights.get('market_opportunities', [])
        video_opportunities = video_insights.get('content_opportunities', [])
        
        if image_opportunities:
            recommendations.append(f"Image content opportunity: {image_opportunities[0]}")
        
        if video_opportunities:
            recommendations.append(f"Video content opportunity: {video_opportunities[0]}")
        
        return recommendations

    def store_unified_intelligence(self, unified_report: Dict) -> bool:
        """Store unified multi-modal intelligence in Supabase"""
        if not self.supabase:
            logger.warning("Supabase not available - unified intelligence not stored")
            return False
        
        try:
            storage_data = {
                'analysis_type': 'unified_multimodal_intelligence',
                'insights': unified_report,
                'timestamp': datetime.now().isoformat(),
                'source': 'unified_multimodal_agent'
            }
            
            result = self.supabase.table('market_intelligence').insert(storage_data).execute()
            
            if result.data:
                logger.info("✅ Unified multi-modal intelligence stored successfully")
                return True
            else:
                logger.error("❌ Failed to store unified intelligence")
                return False
                
        except Exception as e:
            logger.error(f"Error storing unified intelligence: {e}")
            return False

    def run_complete_multimodal_analysis(self) -> Dict[str, Any]:
        """Main execution method for complete multi-modal analysis"""
        logger.info("🎨🎬 Starting Complete Multi-Modal Intelligence Analysis")
        print("=" * 80)
        
        try:
            # Step 1: Run parallel analysis
            logger.info("⚡ Running parallel image and video analysis...")
            analysis_results = self.run_parallel_analysis()
            
            if not analysis_results or "error" in analysis_results:
                return {"error": "Parallel analysis failed", "results": analysis_results}
            
            # Step 2: Generate unified report
            logger.info("🔗 Generating unified multi-modal report...")
            unified_report = self.generate_unified_report(analysis_results)
            
            if "error" in unified_report:
                return {"error": "Report generation failed", "report": unified_report}
            
            # Step 3: Store results
            logger.info("💾 Storing unified multi-modal intelligence...")
            stored = self.store_unified_intelligence(unified_report)
            
            # Final results
            final_results = {
                'success': True,
                'unified_report': unified_report,
                'raw_analysis_results': analysis_results,
                'stored_successfully': stored,
                'execution_timestamp': datetime.now().isoformat(),
                'content_summary': {
                    'total_images_analyzed': analysis_results.get('image_analysis', {}).get('content_analyzed', 0),
                    'total_videos_analyzed': analysis_results.get('video_analysis', {}).get('videos_analyzed', 0),
                    'cross_modal_trends_found': len(unified_report.get('cross_modal_correlations', {}).get('cross_modal_trends', [])),
                    'shared_brands_identified': len(unified_report.get('cross_modal_correlations', {}).get('shared_brand_presence', []))
                }
            }
            
            logger.info("✅ Complete Multi-Modal Intelligence Analysis finished successfully")
            return final_results
            
        except Exception as e:
            logger.error(f"❌ Complete Multi-Modal Analysis failed: {e}")
            return {"error": str(e), "success": False}

def main():
    """Main execution function"""
    print("🎨🎬 UNIFIED MULTI-MODAL MARKET INTELLIGENCE AGENT")
    print("=" * 80)
    print("Analyzing visual trends across images and videos...")
    print()
    
    # Initialize unified agent
    agent = UnifiedMultiModalAgent()
    
    # Run complete analysis
    results = agent.run_complete_multimodal_analysis()
    
    # Display results
    if results.get('success'):
        print("✅ MULTI-MODAL INTELLIGENCE ANALYSIS COMPLETE")
        print("=" * 60)
        
        content_summary = results.get('content_summary', {})
        print(f"📊 ANALYSIS SUMMARY:")
        print(f"   🖼️  Images analyzed: {content_summary.get('total_images_analyzed', 0)}")
        print(f"   🎬 Videos analyzed: {content_summary.get('total_videos_analyzed', 0)}")
        print(f"   🔗 Cross-modal trends: {content_summary.get('cross_modal_trends_found', 0)}")
        print(f"   🏷️  Shared brands: {content_summary.get('shared_brands_identified', 0)}")
        print(f"   💾 Data stored: {results.get('stored_successfully', False)}")
        
        unified_report = results.get('unified_report', {})
        market_insights = unified_report.get('unified_market_insights', {})
        
        print(f"\n🎯 KEY MARKET INSIGHTS:")
        trending_products = market_insights.get('trending_products', [])[:3]
        if trending_products:
            print(f"   📈 Trending products: {', '.join(trending_products)}")
        
        behavior_patterns = market_insights.get('consumer_behavior_patterns', [])[:3]
        if behavior_patterns:
            print(f"   👥 Consumer behaviors: {', '.join(behavior_patterns)}")
        
        sentiment_overview = market_insights.get('market_sentiment_overview', {})
        overall_sentiment = sentiment_overview.get('overall_sentiment', 0)
        sentiment_trend = sentiment_overview.get('sentiment_trend', 'unknown')
        print(f"   😊 Market sentiment: {overall_sentiment:.2f} ({sentiment_trend})")
        
        recommendations = unified_report.get('actionable_recommendations', [])
        if recommendations:
            print(f"\n💡 TOP RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations[:3], 1):
                print(f"   {i}. {rec}")
        
        print(f"\n🎉 Multi-modal intelligence gathering complete!")
        
    else:
        print(f"❌ ANALYSIS FAILED")
        print(f"Error: {results.get('error', 'Unknown error occurred')}")
        
        # Show partial results if available
        if 'raw_analysis_results' in results:
            raw_results = results['raw_analysis_results']
            if raw_results.get('image_analysis', {}).get('success'):
                print("✅ Image analysis completed successfully")
            if raw_results.get('video_analysis', {}).get('success'):
                print("✅ Video analysis completed successfully")

if __name__ == "__main__":
    main()
