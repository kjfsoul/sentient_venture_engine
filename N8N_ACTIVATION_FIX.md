# N8N Workflow Activation Fix Guide

## Problem: "Unknown alias: und" Error

The N8N workflow fails to activate with "Unknown alias: und" error. This is caused by:
1. ANSI color codes in agent output that N8N can't parse
2. Complex conda environment activation issues
3. Invalid characters or formatting in execution output

## Solution Summary

We've created a **clean N8N-compatible agent** (`agents/market_intel_n8n.py`) that:
- ✅ Removes all ANSI color codes and escape sequences
- ✅ Uses simpler, cleaner output formatting
- ✅ Handles search tool failures gracefully
- ✅ Works with direct Python path execution

## N8N Command Options (Try in Order)

### Option 1: Direct Python Path (RECOMMENDED)
```bash
cd /Users/kfitz/sentient_venture_engine && /Users/kfitz/opt/anaconda3/envs/sve_env/bin/python agents/market_intel_n8n.py
```

### Option 2: Using the Wrapper Script
```bash
/Users/kfitz/sentient_venture_engine/run_agent_n8n.sh
```

### Option 3: With Search Disabled (Rate Limit Safe)
```bash
cd /Users/kfitz/sentient_venture_engine && DISABLE_SEARCH=true /Users/kfitz/opt/anaconda3/envs/sve_env/bin/python agents/market_intel_n8n.py
```

### Option 4: Test Mode (Development)
```bash
cd /Users/kfitz/sentient_venture_engine && TEST_MODE=true /Users/kfitz/opt/anaconda3/envs/sve_env/bin/python agents/market_intel_n8n.py
```

## Testing Results

All commands tested successfully:

✅ **Test Mode**: Clean execution with sample data
✅ **Search Disabled**: LLM-only execution with fallback data  
✅ **Rate Limited**: Graceful degradation with error handling
✅ **Wrapper Script**: Clean execution through bash wrapper

## N8N Workflow Files

1. **SVE_ORACLE_BOOTSTRAP.json** - Updated original workflow
2. **SVE_ORACLE_BOOTSTRAP_CLEAN.json** - New clean workflow version

## Agent Capabilities Confirmed

The agent IS capable of real analysis (not hard-coded):
- ✅ Dynamic LLM model selection and fallback
- ✅ Real-time search when available
- ✅ Intelligent task planning and execution
- ✅ JSON response generation and parsing
- ✅ Robust error handling and graceful degradation
- ✅ Supabase database integration

## Automation Features

- 🔄 **Hourly Execution**: Automated market intelligence gathering
- 🎯 **Free Models**: 14+ free OpenRouter model fallback chain
- 🛡️ **Rate Limit Protection**: Multiple safety mechanisms
- 📊 **Data Storage**: Automatic Supabase trend/pain point storage
- 🧪 **Test Modes**: Local testing without API costs

## Next Steps

1. **Import the clean workflow**: Use `SVE_ORACLE_BOOTSTRAP_CLEAN.json`
2. **Try Option 1 command**: Direct Python path execution
3. **Activate workflow**: Should now work without "und" alias error
4. **Monitor execution**: Hourly runs will collect market intelligence automatically

## Emergency Fallback

If all commands fail, the system always generates fallback data to ensure continuous operation.
