# N8N "Alias und" Activation Error - Systematic Fix

## Problem Analysis
- ✅ Commands work when executed manually
- ✅ Test workflow executes successfully 
- ❌ Workflow activation fails with "Unknown alias: und"

This indicates an **N8N workflow structure issue**, not a command execution problem.

## Root Cause
The "alias und" error typically occurs when:
1. N8N tries to parse corrupted execution data
2. Workflow JSON contains problematic references
3. Environment variable parsing issues during activation
4. Node ID conflicts or malformed structure

## Solution Approaches (Try in Order)

### Approach 1: Minimal Clean Workflow
**File**: `MINIMAL_WORKFLOW.json`
- Ultra-simple structure with explicit node IDs
- Basic echo command only
- Clean JSON structure with proper sections

### Approach 2: Fresh UUID-based Workflow  
**File**: `FRESH_WORKFLOW.json`
- Fresh UUID node IDs to avoid conflicts
- Direct Python path execution
- Clean metadata and settings

### Approach 3: Simple Wrapper Script
**File**: `WRAPPER_WORKFLOW.json` + `simple_wrapper.sh`
- Uses bash wrapper to isolate command execution
- Avoids complex command strings in N8N
- Tested and working wrapper script

### Approach 4: Clean SVE Workflow
**File**: `SVE_CLEAN_WORKFLOW.json`
- Conservative agent with proper structure
- Explicit node IDs and clean connections
- Minimal complexity

## Testing Steps

1. **Start with MINIMAL_WORKFLOW.json**
   - Import and try to activate
   - If this fails, the issue is with N8N itself

2. **Try WRAPPER_WORKFLOW.json** 
   - Uses the tested `simple_wrapper.sh`
   - Should eliminate command parsing issues

3. **Use FRESH_WORKFLOW.json**
   - Fresh node IDs avoid any cached conflicts
   - Clean structure with proper metadata

4. **Final: SVE_CLEAN_WORKFLOW.json**
   - Full functionality with clean structure

## Additional Troubleshooting

If ALL workflows fail to activate:

### Clear N8N Data
```bash
# Stop N8N
# Clear execution data/cache if possible
# Restart N8N
```

### Alternative Commands to Try
```bash
# Option 1: Wrapper script
/Users/kfitz/sentient_venture_engine/simple_wrapper.sh

# Option 2: Direct execution
/Users/kfitz/opt/anaconda3/envs/sve_env/bin/python /Users/kfitz/sentient_venture_engine/agents/conservative_agent.py

# Option 3: With environment isolation
env -i /Users/kfitz/opt/anaconda3/envs/sve_env/bin/python /Users/kfitz/sentient_venture_engine/agents/conservative_agent.py
```

## Expected Results

- **MINIMAL_WORKFLOW**: Should activate (tests N8N basic functionality)
- **WRAPPER_WORKFLOW**: Should activate (tests wrapper approach)  
- **FRESH_WORKFLOW**: Should activate (tests clean structure)
- **SVE_CLEAN_WORKFLOW**: Should activate (tests full functionality)

## Next Action

Try importing and activating **MINIMAL_WORKFLOW.json** first. This will tell us if the issue is:
- N8N configuration problem (if minimal fails)
- Workflow structure problem (if minimal works but others fail)
- Command-specific problem (if wrapper fails but minimal works)

Let me know which workflow you test and the exact result!
