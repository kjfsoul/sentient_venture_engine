# LLM Prompts for API Provider Gap Analysis

This document contains specific prompts designed to be used with LLMs to identify which features in your system lack API providers and suggest alternatives.

## Prompt 1: Feature Gap Analysis

```
You are an AI systems analyst. I have a multi-modal intelligence gathering system with the following components:

1. Text Analysis Agent - Uses market intelligence agents with LLMs
2. Code Analysis Agent - Analyzes GitHub repositories for technological trends
3. Multimodal Analysis Agent - Analyzes visual content (images and videos)

Based on the API keys available in my .env file, I have access to:
- OPENROUTER_API_KEY
- OPENAI_API_KEY
- GITHUB_TOKEN
- GEMINI_API_KEY
- TOGETHER_API_KEY
- GROQ_API_KEY
- HF_API_KEY
- SERPER_API_KEY
- COMFYUI_API_KEY
- Videodb_api_key
- CURSOR_API_KEY
- FIRECRAWL_API_KEY
- PEXELS_API_KEY
- DEEPAI_API_KEY

However, my system has placeholders for the following commercial APIs that I don't have keys for:
- Veo 3 (video analysis)
- SORA (video generation)
- Claude Code Max (code analysis)
- Qwen 3 Coder (code analysis)
- Deepseek Coder (code analysis)

Your task is to:
1. Identify which system features currently lack API providers based on the available keys
2. For each missing feature, suggest 2-3 free/open-source or alternative API providers that could be used
3. Rank the suggestions by ease of implementation (1 = easiest, 3 = hardest)
4. Provide specific reasons why each suggestion would work for the feature

Respond in the following JSON format:
{
  "missing_features": [
    {
      "feature": "feature name",
      "current_placeholder": "API name",
      "reason_missing": "why it's missing",
      "alternatives": [
        {
          "name": "alternative name",
          "type": "free/open-source/commercial",
          "implementation_difficulty": 1,
          "reason": "why this would work"
        }
      ]
    }
  ]
}
```

## Prompt 2: Rate Limiting Strategy Analysis

```
You are a software architecture expert. I have a multi-agent system that makes API calls to various services. I need to implement robust rate limiting to handle failures gracefully.

My system has these characteristics:
- Multiple agents running concurrently
- Various API providers (OpenAI, Gemini, OpenRouter, GitHub, etc.)
- Different rate limits per provider
- Need to handle both HTTP 429 (rate limit) and other failures

Please provide:
1. A rate limiting strategy that works across different API providers
2. Specific implementation approaches for handling rate limit errors
3. How to implement exponential backoff with jitter
4. How to queue requests when rate limits are hit
5. How to monitor and log rate limiting events
6. Best practices for graceful degradation when rate limits are consistently hit

Respond with a detailed technical approach that includes:
- Pseudocode for the rate limiting mechanism
- Examples of how to handle different types of rate limit responses
- Recommendations for configuration parameters
```

## Prompt 3: Alternative Implementation Research

```
You are a research assistant helping to find open-source alternatives for commercial APIs. Based on the following requirements, suggest specific open-source projects or free APIs that could replace the commercial services.

Required Capabilities:
1. Video analysis (object detection, activity recognition, sentiment analysis)
2. Advanced image analysis (object detection, brand recognition, color analysis)
3. Code analysis (pattern recognition, trend identification in repositories)
4. Advanced LLM capabilities (code understanding, complex reasoning)

For each capability, provide:
1. 3 specific open-source projects or free APIs
2. Links to documentation or repositories
3. Implementation difficulty (Easy, Medium, Hard)
4. Key features that match the requirements
5. Any limitations or gaps compared to the commercial alternatives

Format your response as a structured report with sections for each capability.
```

## Prompt 4: System Integration Assessment

```
You are a system integration expert. I have a Python-based intelligence gathering system with multiple agents. The system currently has placeholders for commercial APIs that I don't have access to.

I need to understand:
1. Which parts of my system will fail or be degraded without these commercial APIs
2. How to implement fallback mechanisms for each missing API
3. What features I can implement immediately with my existing API keys
4. Which missing APIs are critical vs. nice-to-have

The system architecture includes:
- Text analysis using LLMs (OpenRouter, OpenAI, Gemini already available)
- Code repository analysis using GitHub API
- Multimodal analysis with placeholders for Veo 3, SORA, ComfyUI, etc.

Provide a detailed assessment that identifies:
- Critical failure points
- Immediate opportunities with existing keys
- Fallback strategies for each missing API
- Recommendations for prioritizing implementation efforts

Respond with a structured analysis organized by system component.
```
