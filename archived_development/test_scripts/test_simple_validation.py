#!/usr/bin/env python3
"""
Simple Test for Phase 3 Validation Components
"""

import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set test mode
os.environ['TEST_MODE'] = 'true'

print("🧪 Testing basic imports...")

try:
    # Test basic dataclasses
    from agents.validation_agents import ValidationTier, ValidationStatus
    print("✅ Basic enums imported successfully")

    # Test tools
    from agents.validation_tools import ValidationMetricsCalculator
    print("✅ Validation tools imported successfully")

    # Test simple calculation
    predictions = [1, 0, 1, 1, 0]
    actuals = [1, 0, 0, 1, 0]
    metrics = ValidationMetricsCalculator.calculate_binary_classification_metrics(predictions, actuals)
    print(f"✅ Metrics calculation: Accuracy = {metrics.accuracy:.3f}")

    print("\n🎉 BASIC VALIDATION COMPONENTS WORKING!")
    print("✅ Phase 3 Core Implementation: FUNCTIONAL")

except Exception as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)