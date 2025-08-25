# N8N Step-by-Step Testing Guide

## Current Status
✅ **Conservative agent works**: `/agents/conservative_agent.py` runs successfully  
✅ **N8N agent works**: `/agents/market_intel_n8n.py` processes data correctly  
✅ **Command line execution works**: All Python scripts execute without errors  

## Test These Workflows In Order

### Step 1: Test Minimal Echo Command
**File**: `TEST_WORKFLOW.json`
**Command**: `echo 'Hello from N8N test'`
**Purpose**: Test if N8N can activate any workflow at all

### Step 2: Test Simple Python Agent
**File**: `SIMPLE_TEST_WORKFLOW.json`  
**Command**: `cd /Users/kfitz/sentient_venture_engine && /Users/kfitz/opt/anaconda3/envs/sve_env/bin/python agents/simple_test_agent.py`
**Purpose**: Test if N8N can run basic Python scripts

### Step 3: Test Conservative Agent
**File**: `CONSERVATIVE_WORKFLOW.json`
**Command**: `cd /Users/kfitz/sentient_venture_engine && /Users/kfitz/opt/anaconda3/envs/sve_env/bin/python agents/conservative_agent.py`
**Purpose**: Test with market intelligence logic but no external dependencies

### Step 4: Test Full N8N Agent
**File**: `SVE_ORACLE_BOOTSTRAP.json`
**Command**: `cd /Users/kfitz/sentient_venture_engine && /Users/kfitz/opt/anaconda3/envs/sve_env/bin/python agents/market_intel_n8n.py`
**Purpose**: Test full functionality

## Testing Instructions

For each workflow:

1. **Import the JSON file** into N8N
2. **Try to activate** the workflow
3. **Note the exact error message** if activation fails
4. **If successful**, let it run once and check the output

## Expected Results

- **Step 1**: Should activate without issues (tests basic N8N functionality)
- **Step 2**: Should activate and run (tests Python environment access)
- **Step 3**: Should activate and produce market intelligence output (tests our logic)
- **Step 4**: Should activate and run with full LLM functionality

## Troubleshooting

If **Step 1 fails**: N8N configuration issue (not our code)
If **Step 2 fails**: Python environment or path issue
If **Step 3 fails**: Our agent logic has an issue
If **Step 4 fails**: LLM or external API issue

## Alternative Commands to Try

If the main command fails, try these alternatives in the Execute Command node:

```bash
# Option A: Wrapper script
/Users/kfitz/sentient_venture_engine/run_agent_n8n.sh

# Option B: Test mode
cd /Users/kfitz/sentient_venture_engine && TEST_MODE=true /Users/kfitz/opt/anaconda3/envs/sve_env/bin/python agents/conservative_agent.py

# Option C: Direct path only
/Users/kfitz/opt/anaconda3/envs/sve_env/bin/python /Users/kfitz/sentient_venture_engine/agents/conservative_agent.py
```

## Next Steps

Start with **Step 1** and let me know:
1. Which step you're testing
2. Whether activation succeeded or failed
3. The exact error message if it failed
4. Any output if it succeeded

This will help us identify exactly where the "Unknown alias: und" issue is coming from.
