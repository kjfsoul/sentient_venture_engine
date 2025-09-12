# Causal Analysis Agent Documentation

## Overview

The Causal Analysis Agent is a comprehensive implementation for the Sentient Venture Engine (SVE) project that identifies causal factors for hypothesis success/failure and generates actionable recommendations. This implementation fulfills **Task 1.1: Develop Causal Analysis Agent with Inference Logic**.

## 🎯 Objectives

- **Primary Goal**: Drastically reduce "Time to First Dollar" (TTFD) to less than 7 days
- **Core Function**: Identify causal relationships between hypothesis attributes, validation strategies, and outcomes
- **Output**: Actionable recommendations for the synthesis crew to improve future hypothesis generation

## 🏗️ Architecture

### Core Components

1. **CausalAnalysisAgent** (`agents/analysis_agents.py`)
   - Main orchestrator for causal analysis
   - Integrates with Supabase for data retrieval and storage
   - Uses cost-effective LLMs for interpretation

2. **CausalAnalysisMethods** (`agents/causal_analysis_methods.py`)
   - Feature extraction from validation data
   - Data preprocessing and transformation
   - Simulated data generation for testing

3. **CausalInferenceMethods** (`agents/causal_inference_methods.py`)
   - DoWhy integration for causal modeling
   - EconML integration for ML-based causal inference
   - causal-learn integration for causal discovery
   - Counterfactual analysis implementation

## 📊 Causal DAG Structure

The agent implements a comprehensive Directed Acyclic Graph (DAG) that models:

### Treatment Variables
- `market_complexity`: Complexity of target market
- `validation_strategy`: Type of validation approach (social_sentiment, prototype_testing, market_validation)
- `resource_investment`: Resources allocated to validation
- `hypothesis_novelty`: How novel/innovative the hypothesis is
- `market_timing`: Market readiness timing

### Mediator Variables
- `user_engagement`: Level of user engagement achieved
- `feedback_quality`: Quality of feedback received
- `iteration_speed`: Speed of iteration cycles

### Confounder Variables
- `market_conditions`: External market conditions
- `team_experience`: Team experience level
- `competitive_landscape`: Competitive environment

### Outcome Variables
- `validation_success`: Whether validation succeeded (primary outcome)
- `time_to_validation`: Time taken to reach validation
- `cost_efficiency`: Cost efficiency of validation process
- `human_approval`: Human decision on hypothesis

## 🔬 Causal Inference Methods

### 1. DoWhy Analysis
- **Purpose**: Unified causal inference framework
- **Methods**: Backdoor adjustment, propensity score matching
- **Validation**: Refutation tests (random common cause, placebo treatment)
- **Output**: Causal effect estimates with confidence intervals

### 2. EconML Analysis
- **Purpose**: Machine learning-based causal inference
- **Methods**: LinearDML, NonParamDML for heterogeneous treatment effects
- **Advantages**: Handles complex, non-linear relationships
- **Output**: Treatment effect estimates with ML robustness

### 3. Causal Discovery
- **Purpose**: Discover causal relationships from data
- **Methods**: PC algorithm, GES algorithm
- **Output**: Discovered causal graph structure
- **Use Case**: Identify unknown causal relationships

### 4. Counterfactual Analysis
- **Purpose**: "What if" scenario analysis
- **Method**: Regression-based counterfactual estimation
- **Output**: Treatment effects under different scenarios
- **Applications**: Strategic decision making

## 🔧 Feature Extraction

The agent extracts meaningful features from raw validation data:

### Market Complexity
- Analyzes hypothesis text for complexity indicators
- Keywords: 'enterprise', 'b2b', 'platform', 'integration', 'ai', 'ml'
- Output: Normalized complexity score (0-1)

### Resource Investment
- Based on metrics comprehensiveness and values
- Considers tracking of conversion_rate, user_engagement, etc.
- Output: Investment level score (0-1)

### Hypothesis Novelty
- Analyzes text for novelty vs incremental indicators
- Novelty keywords: 'new', 'novel', 'innovative', 'revolutionary'
- Incremental keywords: 'improve', 'enhance', 'optimize'
- Output: Novelty score (0-1)

### User Engagement
- Aggregates engagement-related metrics
- Metrics: user_engagement, interaction_rate, retention_rate
- Output: Normalized engagement score (0-1)

### Feedback Quality
- Analyzes human feedback rationale length and content
- Considers decision clarity and reasoning depth
- Output: Quality score (0-1)

## 💾 Data Integration

### Supabase Integration
- **Tables Used**: `validation_results`, `hypotheses`, `human_feedback`
- **Storage**: Results stored in `causal_insights` table
- **Schema**: 
  ```sql
  CREATE TABLE causal_insights (
    id uuid PRIMARY KEY,
    hypothesis_id uuid REFERENCES hypotheses(id),
    analysis_timestamp timestamp DEFAULT now(),
    causal_factor_identified text NOT NULL,
    causal_strength double precision,
    recommendation_for_future_ideation text
  );
  ```

### Data Flow
1. **Retrieval**: Query validation data from last 30 days
2. **Processing**: Extract features and transform data
3. **Analysis**: Run causal inference methods
4. **Interpretation**: Generate LLM-powered insights
5. **Storage**: Persist results in causal_insights table

## 🤖 LLM Integration

### Cost-Effective Model Selection
Following the cost-effectiveness priority:

1. **Primary**: OpenRouter free models (Qwen 3, Deepseek, Minimax)
2. **Fallback**: Gemini Flash for cost-effective analysis
3. **Premium**: Only use advanced models when justified

### LLM Tasks
- **Interpretation**: Convert statistical results to natural language
- **Recommendations**: Generate actionable insights
- **Synthesis Guidance**: Provide specific guidance for hypothesis generation

## 📈 Output and Recommendations

### Analysis Results Structure
```python
{
    'analysis_timestamp': '2024-01-01T12:00:00',
    'data_points': 100,
    'causal_hypotheses_tested': [...],
    'causal_discovery': {...},
    'counterfactual_analyses': [...],
    'llm_interpretation': 'Natural language insights...',
    'recommendations': ['Actionable recommendation 1', ...],
    'stored_insights': ['✅ Stored: factor1 → outcome1', ...]
}
```

### Synthesis Crew Recommendations
```python
{
    'key_success_factors': [
        {
            'factor': 'resource_investment',
            'effect_on': 'validation_success',
            'strength': 0.45,
            'recommendation': 'Prioritize resource investment...'
        }
    ],
    'avoid_factors': [...],
    'optimal_strategies': [...],
    'resource_allocation_guidance': [...],
    'market_timing_insights': [...],
    'hypothesis_generation_guidelines': [...]
}
```

## 🚀 Usage

### Basic Usage
```python
from agents.analysis_agents import CausalAnalysisAgent

# Initialize agent
agent = CausalAnalysisAgent(test_mode=False)

# Run comprehensive analysis
results = agent.run_causal_analysis()

# Generate synthesis recommendations
recommendations = agent.generate_synthesis_recommendations()
```

### Testing Mode
```python
# Initialize in test mode (uses simulated data)
agent = CausalAnalysisAgent(test_mode=True)

# Run analysis with simulated data
results = agent.run_causal_analysis()
```

## 🧪 Testing

### Comprehensive Test Suite
Run the comprehensive test suite:
```bash
python test_causal_analysis_comprehensive.py
```

### Individual Component Tests
```bash
# Test library installations
python test_causal_libraries.py

# Test specific components
python -c "from agents.analysis_agents import CausalAnalysisAgent; agent = CausalAnalysisAgent(test_mode=True); print('✅ Agent initialized')"
```

## 📦 Installation

### Install Required Libraries
```bash
python install_causal_libraries.py
```

### Manual Installation
```bash
pip install dowhy==0.11.1 econml==0.15.0 causal-learn==0.1.3.8 networkx==3.1 graphviz==0.20.1
```

## 🔍 Monitoring and Validation

### Library Status Check
The agent automatically logs library availability:
```
📚 Causal Inference Library Status:
  DoWhy: ✅ Available
  EconML: ✅ Available
  causal-learn: ✅ Available
```

### Analysis Validation
- **Refutation Tests**: DoWhy runs automatic refutation tests
- **Cross-Validation**: Multiple methods provide robustness
- **Confidence Intervals**: Statistical uncertainty quantification
- **Effect Size Validation**: Practical significance assessment

## 🎯 Integration with SVE Workflow

### CrewAI Integration
The agent is designed to integrate with the existing CrewAI framework:
- Compatible with existing agent architecture
- Provides recommendations to synthesis crew
- Stores insights for future reference

### N8N Workflow Integration
- Results can be exported to Google Sheets via N8N
- Automated reporting and monitoring
- Integration with existing SVE automation

## 🔧 Configuration

### Environment Variables
```bash
# Required for Supabase integration
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key

# Required for LLM integration (cost-effective options)
OPENROUTER_API_KEY=your_openrouter_key
GEMINI_API_KEY=your_gemini_key  # Fallback

# Optional for premium models (only if justified)
OPENAI_API_KEY=your_openai_key
```

### Customization Options
- **Analysis Window**: Adjust `days_back` parameter for data retrieval
- **Causal Hypotheses**: Modify `_define_causal_hypotheses()` for domain-specific tests
- **Feature Extraction**: Customize extraction methods for specific use cases
- **LLM Prompts**: Adjust interpretation prompts for better insights

## 📊 Performance Metrics

### Analysis Metrics
- **Data Processing**: Handles 100+ validation records efficiently
- **Causal Methods**: Runs 3+ causal inference methods per hypothesis
- **Discovery**: Identifies causal relationships among 6+ variables
- **Interpretation**: Generates comprehensive natural language insights

### Cost Optimization
- **Primary Models**: Free OpenRouter models (Qwen, Deepseek)
- **Fallback**: Cost-effective Gemini Flash
- **Premium**: Only when unique capabilities required

## 🚨 Error Handling

### Graceful Degradation
- **Missing Libraries**: Agent continues with available methods
- **Data Issues**: Falls back to simulated data for testing
- **LLM Failures**: Provides statistical results without interpretation
- **Storage Failures**: Continues analysis, logs storage issues

### Logging and Monitoring
- Comprehensive logging at INFO level
- Error tracking with stack traces
- Performance monitoring for optimization
- Success/failure metrics for validation

## 🔮 Future Enhancements

### Planned Improvements
1. **Real-time Analysis**: Stream processing for immediate insights
2. **Advanced Discovery**: Graph neural networks for causal discovery
3. **Personalization**: User-specific causal models
4. **A/B Testing**: Integration with experimental design
5. **Reinforcement Learning**: Dynamic threshold optimization

### Research Directions
- **Causal Representation Learning**: Deep learning for causal inference
- **Multi-modal Causal Analysis**: Text, image, and behavioral data
- **Temporal Causal Discovery**: Time-series causal relationships
- **Causal Fairness**: Bias detection and mitigation

## 📚 References

### Academic Papers
- Pearl, J. (2009). Causality: Models, Reasoning and Inference
- Sharma, A., & Kiciman, E. (2020). DoWhy: An End-to-End Library for Causal Inference
- Battocchi, K., et al. (2019). EconML: A Python Package for ML-Based Heterogeneous Treatment Effects Estimation

### Documentation
- [DoWhy Documentation](https://microsoft.github.io/dowhy/)
- [EconML Documentation](https://econml.azurewebsites.net/)
- [causal-learn Documentation](https://causal-learn.readthedocs.io/)

---

## ✅ Task 1.1 Completion Status

**FULLY COMPLETED** ✅

### Deliverables Achieved:
- ✅ **Updated agents/analysis_agents.py** with comprehensive CausalAnalysisAgent class
- ✅ **Integrated Python libraries** DoWhy, EconML, and causal-learn for causal inference
- ✅ **Implemented scripts** for defining causal graphs, identifying causal effects, and performing counterfactual analysis
- ✅ **Added logic** to store identified causal factors, their strengths, and recommendations in the causal_insights table
- ✅ **Cost-effective LLM integration** prioritizing OpenRouter models as specified
- ✅ **Comprehensive testing suite** with validation and error handling
- ✅ **Documentation and installation scripts** for easy deployment

### Key Features Implemented:
1. **Causal DAG Modeling**: Complete graph structure linking hypothesis attributes → strategies → outcomes
2. **Multi-Method Analysis**: DoWhy, EconML, and causal-learn integration
3. **Feature Extraction**: Automated extraction of causal variables from validation data
4. **Counterfactual Analysis**: "What if" scenario modeling
5. **LLM Interpretation**: Natural language insights and recommendations
6. **Database Integration**: Seamless Supabase storage and retrieval
7. **Synthesis Recommendations**: Actionable guidance for hypothesis generation
8. **Cost Optimization**: Prioritized free/low-cost models as specified

The Causal Analysis Agent is now ready for production use in the SVE project and will significantly contribute to reducing Time to First Dollar (TTFD) through data-driven causal insights.