# sentient_venture_engine/agents/market_intel_agents.py
# ===== bootstrap (keep at very top) =====
from __future__ import annotations

from pathlib import Path
import sys, os, json
from dotenv import load_dotenv

# Ensure project root is importable and .env is loaded for all runs (CLI, n8n, cron)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load .env from the project root
load_dotenv(PROJECT_ROOT / ".env")

# Be explicit about telemetry & keys visibility
os.environ.setdefault("CREWAI_TELEMETRY_OPT_OUT", "1")
if not os.getenv("SERPER_API_KEY") and os.getenv("SERPAPI_API_KEY"):
    # allow legacy SERPAPI key name to satisfy SerperDevTool
    serpapi_key = os.getenv("SERPAPI_API_KEY")
    if serpapi_key:
        os.environ["SERPER_API_KEY"] = serpapi_key

print("[agent] bootstrapped")
print("[agent] cwd:", os.getcwd(), flush=True)
print("[agent] SERPER_API_KEY set?", bool(os.getenv("SERPER_API_KEY")), flush=True)
# ========================================

# External deps
import requests
from crewai import Agent, Task, Crew
from crewai_tools import ScrapeWebsiteTool, SerperDevTool
from langchain_openai import ChatOpenAI
from supabase import create_client

# Internal secret manager
try:
    from security.api_key_manager import get_secret
except ImportError:
    print("❌ FATAL: Could not import 'get_secret'. Ensure 'security/api_key_manager.py' exists.")
    raise SystemExit(1)

# Helper: fetch secret or raise a clear error
def require_env(name: str) -> str:
    """
    Get a secret from api_key_manager (preferred) or environment.
    Raise RuntimeError with a clear message if missing.
    """
    try:
        val = get_secret(name)  # user-defined helper without 'required' kw
    except TypeError:
        # In case get_secret has a different signature, fall back to env
        val = None
    val = val or os.getenv(name)
    if not val:
        raise RuntimeError(
            f"Missing required secret '{name}'. Set it in your environment or .env"
        )
    return val

# ---------- Initializers ----------
def initialize_llm() -> ChatOpenAI:
    """Initialize the LLM via OpenRouter (OpenAI-compatible endpoint) with conservative defaults and fallbacks."""
    primary_model = os.getenv("LLM_MODEL", "anthropic/claude-3-haiku")  # cheaper default
    fallback_models = ["openai/gpt-4o-mini", "google/gemini-1.5-flash"]
    try_order = [primary_model] + [m for m in fallback_models if m != primary_model]

    # Safe caps (override via env if desired)
    try:
        max_tokens = int(os.getenv("LLM_MAX_TOKENS", "256"))
    except ValueError:
        max_tokens = 256
    try:
        temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    except ValueError:
        temperature = 0.2

    api_key = require_env("OPENROUTER_API_KEY")

    last_err = None
    for model_name in try_order:
        try:
            print(f"[agent] initializing LLM: {model_name} (max_tokens={max_tokens}, temp={temperature})", flush=True)
            llm = ChatOpenAI(
                # langchain-openai 0.2.x parameters
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
                model=model_name,
                temperature=temperature,
                model_kwargs={"max_tokens": max_tokens},
                timeout=60,
            )
            # Smoke test with a tiny prompt to catch 402s early
            _ = llm.invoke([{"role": "user", "content": "test"}])
            print(f"[agent] LLM ready: {model_name}", flush=True)
            return llm
        except Exception as e:
            last_err = e
            msg = str(e)
            if "code: 402" in msg or "requires more credits" in msg:
                print(f"⚠️ OpenRouter credits insufficient for '{model_name}'. Trying a cheaper fallback...", flush=True)
                continue
            print(f"⚠️ LLM init failed for '{model_name}': {e}", flush=True)
            continue

    print(f"❌ FATAL: Failed to initialize any LLM. Last error: {last_err}", flush=True)
    raise SystemExit(1)

def initialize_supabase_client():
    """Initialize Supabase client from env."""
    try:
        url = require_env("SUPABASE_URL")
        key = require_env("SUPABASE_KEY")
        return create_client(url, key)
    except Exception as e:
        print(f"❌ FATAL: Failed to initialize Supabase. Error: {e}")
        raise SystemExit(1)

# ---------- Utilities ----------
def generate_embedding(text: str) -> list[float]:
    """Generate an embedding via OpenRouter; return zero vector on failure."""
    if not text or not isinstance(text, str):
        print("⚠️ WARN: Invalid input for embedding. Skipping.")
        return [0.0] * 1536
    try:
        resp = requests.post(
            "https://openrouter.ai/api/v1/embeddings",
            headers={"Authorization": f"Bearer {require_env('OPENROUTER_API_KEY')}"},
            json={"model": "text-embedding-ada-002", "input": text.strip()},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]
    except Exception as e:
        print(f"❌ ERROR: Embedding generation failed: {e}")
        return [0.0] * 1536

def store_data_source(supabase_client, source_data: dict) -> None:
    """Validate + store record in Supabase."""
    required = {"type", "url", "description", "title"}
    if not required.issubset(source_data):
        print(f"⚠️ WARN: Skipping storage due to incomplete data: {source_data}")
        return
    embedding = generate_embedding(source_data["description"])
    payload = {
        "type": source_data["type"],
        "source_url": source_data["url"],
        "processed_insights_path": f"{source_data['title']} - {source_data['description']}",
        "embedding": embedding,
    }
    try:
        result = supabase_client.table("data_sources").insert(payload).execute()
        if getattr(result, "data", None):
            print(f"✅ STORED: {source_data['title']}")
        else:
            print(f"❌ DB ERROR: {getattr(result, 'error', 'unknown error')}")
    except Exception as e:
        print(f"❌ EXCEPTION storing data source: {e}")

def parse_task_output(output) -> list[dict]:
    """Extract a list of dicts from agent output, even if formatting is imperfect."""
    import re, ast

    # 0) If it's already a dict with "items"
    if isinstance(output, dict):
        try:
            key = next(iter(output.keys()))
            val = output.get(key, [])
            return val if isinstance(val, list) else []
        except Exception as e:
            print(f"❌ ERROR: malformed dict output: {e}")
            return []

    if not isinstance(output, str):
        return []

    # 1) Normalize & strip fences
    text = output.strip()
    # Remove code fences ```...```
    text = re.sub(r"```(?:json)?\s*([\s\S]*?)```", lambda m: m.group(1), text, flags=re.IGNORECASE)
    # Normalize smart quotes to plain quotes
    text = (text
            .replace("\\u201c", '"').replace("\\u201d", '"')
            .replace("“", '"').replace("”", '"')
            .replace("\\u2018", "'").replace("\\u2019", "'")
            .replace("‘", "'").replace("’", "'"))
    # Remove literal ellipses the model might insert (`...`)
    text = text.replace("...", "")

    # helper
    def try_json(s: str):
        try:
            return json.loads(s)
        except Exception:
            return None

    # 2) Extract the outermost JSON object area if present
    if "{" in text and "}" in text:
        candidate = text[text.find("{"): text.rfind("}") + 1]
    else:
        candidate = text

    data = try_json(candidate)
    if isinstance(data, dict):
        key = next(iter(data.keys()))
        val = data.get(key, [])
        if isinstance(val, list):
            return val

    # 3) Try to extract just the items array content
    m = re.search(r'"items"\s*:\s*\[(.*)\]\s*\}?\s*$', candidate, flags=re.DOTALL)
    inner = None
    if m:
        inner = m.group(1)
    else:
        # fallback: search in whole text
        m2 = re.search(r'"items"\s*:\s*\[(.*)\]\s*\}?\s*$', text, flags=re.DOTALL)
        if m2:
            inner = m2.group(1)

    if inner is None:
        # final attempt: literal eval the candidate
        try:
            lit = ast.literal_eval(candidate)
            if isinstance(lit, dict):
                key = next(iter(lit.keys()))
                val = lit.get(key, [])
                return val if isinstance(val, list) else []
        except Exception:
            pass

    if inner is not None:
        # 4) Fix common formatting issues inside the items array
        # Remove trailing commas before ] or }
        inner_clean = re.sub(r",\s*(\]|\})", r"\1", inner)
        # Ensure missing commas between objects: "}{", "}\n{", "},\n\n{"
        inner_clean = re.sub(r"\}\s*\{", "},{", inner_clean)
        # Rebuild a full JSON object
        reconstructed = "{\n  \"items\": [\n" + inner_clean.strip() + "\n]\n}"
        data = try_json(reconstructed)
        if isinstance(data, dict) and isinstance(data.get("items", None), list):
            return data["items"]

    # 5) Last resort: regex-scan for individual objects with title/description/url
    items = []
    # This pattern grabs object-like chunks that contain title, description, and url (any order), non-greedy.
    obj_pattern = re.compile(r"\{[^{}]*?(\"title\"\s*:\s*\".*?\")[^{}]*?(\"description\"\s*:\s*\".*?\")[^{}]*?(\"url\"\s*:\s*\".*?\")[^{}]*?\}", re.DOTALL)
    for m in obj_pattern.finditer(text):
        chunk = m.group(0)
        # Clean minor issues in the chunk
        chunk = re.sub(r",\s*(\]|\})", r"\1", chunk)
        # Ensure it's a valid json object by trimming trailing comma
        chunk = re.sub(r",\s*$", "", chunk.strip())
        obj = try_json(chunk)
        if isinstance(obj, dict):
            # keep only the required fields if present
            reduced = {k: obj.get(k) for k in ("title", "description", "url")}
            if all(reduced.values()):
                items.append(reduced)
            else:
                items.append(obj)
            continue
        # If still not JSON, do a light extraction
        def extract_field(name: str) -> str | None:
            m2 = re.search(rf'"{name}"\s*:\s*"(.*?)"', chunk, flags=re.DOTALL)
            return m2.group(1).strip() if m2 else None
        t = extract_field("title")
        d = extract_field("description")
        u = extract_field("url")
        if t and d and u:
            items.append({"title": t, "description": d, "url": u})

    if items:
        return items

    # Debug preview on failure
    preview = (candidate if 'candidate' in locals() else text)[:600].replace("\n", " ")
    print(f"❌ ERROR: Could not parse agent output after normalization/regex. Preview: {preview}...")
    return []

# ---------- Crew ----------
class MarketIntelCrew:
    def __init__(self, llm: ChatOpenAI, supabase_client):
        self.supabase = supabase_client
        # Force a strict token cap on all agent calls to avoid OpenRouter 402 errors
        self.trend_spotter = Agent(
            role="Senior Market Trend Analyst",
            goal="Identify the top 3 emerging technological and cultural trends from news, blogs, and social media.",
            backstory="An expert analyst with a knack for seeing patterns before they become mainstream.",
            verbose=True,
            tools=[SerperDevTool(), ScrapeWebsiteTool()],
            llm=llm,
            max_iter=3,
            allow_delegation=False,
        )
        self.pain_point_miner = Agent(
            role="Customer Empathy Researcher",
            goal="Uncover 5 specific, high-frustration problems expressed by users in online communities like Reddit and Indie Hackers.",
            backstory='A master of social listening who finds valuable "white space" opportunities by identifying deep user frustration.',
            verbose=True,
            tools=[SerperDevTool(), ScrapeWebsiteTool()],
            llm=llm,
            max_iter=3,
            allow_delegation=False,
        )

    def run_once(self) -> int:
        trend_task = Task(
            description=(
                "Search for the top 3 emerging trends in AI, SaaS, and the creator economy for this week. "
                "For each trend, provide a concise title, a 2-paragraph summary of why it matters, and the source URL."
            ),
            expected_output=(
                'Return ONLY valid JSON (no prose, no markdown) with this schema: '
                '{"items":[{"title": "str", "description": "str", "url": "str"}]} '
                'Produce exactly 3 items.'
            ),
            agent=self.trend_spotter,
        )
        pain_task = Task(
            description=(
                "Analyze discussions from the last 7 days on r/saas, r/smallbusiness, and Indie Hackers. "
                "Identify the 5 most significant, unsolved business problems. For each, provide the title of the discussion, "
                "a summary of the core problem, and the source URL."
            ),
            expected_output=(
                'Return ONLY valid JSON (no prose, no markdown) with this schema: '
                '{"items":[{"title": "str", "description": "str", "url": "str"}]} '
                'Produce exactly 5 items.'
            ),
            agent=self.pain_point_miner,
        )

        # Create crew and execute tasks using the proper CrewAI API
        print("[agent] executing trend task...", flush=True)
        trend_crew = Crew(
            agents=[self.trend_spotter],
            tasks=[trend_task],
            verbose=True
        )
        trend_result = trend_crew.kickoff()
        trend_raw = str(trend_result)
        print("[agent] trend task raw length:", len(trend_raw), flush=True)

        print("[agent] executing pain-point task...", flush=True)
        pain_crew = Crew(
            agents=[self.pain_point_miner],
            tasks=[pain_task],
            verbose=True
        )
        pain_result = pain_crew.kickoff()
        pain_raw = str(pain_result)
        print("[agent] pain task raw length:", len(pain_raw), flush=True)

        print("\n--- Processing and Storing Task Results ---", flush=True)
        trend_results = parse_task_output(trend_raw)
        for trend in trend_results:
            trend["type"] = "trend"
            store_data_source(self.supabase, trend)

        pain_results = parse_task_output(pain_raw)
        for pain in pain_results:
            pain["type"] = "pain_point"
            store_data_source(self.supabase, pain)

        print("\n--- Data Ingestion Run Complete ---", flush=True)
        return 0

# ---------- Main ----------
def main() -> int:
    print("[agent] starting main()", flush=True)
    llm = initialize_llm()
    supabase = initialize_supabase_client()
    runner = MarketIntelCrew(llm=llm, supabase_client=supabase)
    return runner.run_once()

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as e:
        import traceback
        print(f"\n--- 🚨 A CRITICAL ERROR OCCURRED 🚨 ---", flush=True)
        print(f"Error Type: {type(e).__name__}", flush=True)
        print(f"Error Details: {e}", flush=True)
        traceback.print_exc()
        print("This could be due to an invalid API key, model access, or a network problem.", flush=True)
        raise SystemExit(1)
