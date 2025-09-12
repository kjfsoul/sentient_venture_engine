#!/usr/bin/env python3
"""
Test script to verify achievement tracking in the enhanced vetting agent
"""

import os
import sys
import asyncio

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from agents.enhanced_vetting_agent import EnhancedVettingAgent, AchievementTracker
    print("✅ Enhanced Vetting Agent imported successfully")
    
    async def test_achievement_tracking():
        print("\n🎯 Testing Achievement Tracking...")
        
        # Initialize the achievement tracker
        tracker = AchievementTracker()
        print("✅ Achievement Tracker initialized")
        
        # Record a test achievement
        achievement = tracker.record_achievement(
            category="Test Category",
            title="Test Achievement",
            description="Testing achievement recording functionality",
            metrics_before={'test_metric': 10.0},
            metrics_after={'test_metric': 25.0},
            business_impact="150% improvement in test metric",
            technical_details={
                'test_feature': 'Achievement tracking',
                'implementation': 'Automated logging system'
            }
        )
        
        print(f"✅ Achievement recorded: {achievement.title}")
        print(f"   Improvement: {achievement.improvement_percentage:.1f}%")
        print(f"   Business Impact: {achievement.business_impact}")
        
        # Generate achievement report
        report = tracker.generate_achievement_report()
        print(f"✅ Achievement report generated ({len(report)} characters)")
        
        # Export to memory system
        success = tracker.export_to_memory_system()
        if success:
            print("✅ Achievements exported to PROJECT_MEMORY_SYSTEM.md")
        else:
            print("❌ Failed to export achievements")
        
        return success

    if __name__ == "__main__":
        success = asyncio.run(test_achievement_tracking())
        if success:
            print("\n🎉 Achievement tracking test completed successfully!")
        else:
            print("\n❌ Achievement tracking test failed!")
        sys.exit(0 if success else 1)
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
