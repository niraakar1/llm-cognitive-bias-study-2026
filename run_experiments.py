"""
LLM Cognitive Bias Experiment Pipeline (FREE TIER)
====================================================
Runs multiple models IN PARALLEL to avoid taking 50 minutes.
Includes automatic exponential back-off for 429 quota errors.

Setup:
    1. Get a FREE Gemini key:  https://aistudio.google.com/apikey
    2. Get a FREE Groq key:    https://console.groq.com/keys
    3. (Optional) Install Ollama: https://ollama.com
    4. pip install google-generativeai groq python-dotenv pandas

Cost: $0
"""

import os
import re
import time
import hashlib
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ═══════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
RAW_PATH = RESULTS_DIR / "raw_responses.csv"
SUMMARY_PATH = RESULTS_DIR / "summary_stats.csv"

N_TRIALS = 20             # Responses per condition
TEMPERATURES = [1.0]

SYSTEM_PROMPT = (
    "You are a participant in a psychology study. "
    "Please respond naturally and give your honest first reaction. "
    "Answer concisely."
)

# ═══════════════════════════════════════════════════════════════════
#  Model Registry (Current APIs)
# ═══════════════════════════════════════════════════════════════════

# Rate limits: 
# Gemini 1.5 Flash is 15 RPM (4 seconds between requests)
# Groq is 30 RPM per model class (2 seconds between requests)

ALL_MODELS = {
    # ── GOOGLE (Free Tier)
    "gemini-2.5-flash": {
        "key_env": "GOOGLE_API_KEY",
        "provider": "Google",
        "delay": 12.5,   # 5 RPM max
        "n_trials": 2    # Cap at 20 requests per day limit (10 conditions * 2)
    },
    # ── GROQ (Free Tier)
    "llama-3.1-8b-instant": {
        "key_env": "GROQ_API_KEY",
        "provider": "Groq",
        "delay": 2.1,    # 30 RPM max
        "n_trials": N_TRIALS
    },
    "llama-3.3-70b-versatile": {
        "key_env": "GROQ_API_KEY",
        "provider": "Groq",
        "delay": 2.1,    # 30 RPM max
        "n_trials": N_TRIALS
    },
    # ── COHERE (Trial API, No CC)
    "command-r-08-2024": {
        "key_env": "COHERE_API_KEY",
        "provider": "Cohere",
        "delay": 6.5,    # 10 RPM max
        "n_trials": N_TRIALS
    },
    "command-r-plus-08-2024": {
        "key_env": "COHERE_API_KEY",
        "provider": "Cohere",
        "delay": 6.5,    # 10 RPM max
        "n_trials": N_TRIALS
    },
    # ── GITHUB MODELS (Free Tier, No CC, 15 RPM limits)
    "gpt-4o-mini": {
        "key_env": "GITHUB_API_KEY",
        "provider": "GitHub",
        "delay": 4.1,    # ~15 RPM max
        "n_trials": N_TRIALS
    },
    # ── OLLAMA (Local)
    "ollama:llama3.1": {
        "key_env": None,
        "provider": "Ollama (local)",
        "delay": 0.0,
        "n_trials": N_TRIALS
    },
}

# ═══════════════════════════════════════════════════════════════════
#  Stimuli — Classic Cognitive Bias Experiments
# ═══════════════════════════════════════════════════════════════════

EXPERIMENTS = {
    # ── 1. ANCHORING (Tversky & Kahneman, 1974)
    "anchoring": {
        "description": "Anchoring effect on numerical estimates",
        "conditions": {
            "high_anchor": {
                "prompt": (
                    "Do you think the percentage of African countries "
                    "in the United Nations is higher or lower than 65%?\n\n"
                    "Now, what is your best estimate of the actual percentage "
                    "of African countries in the United Nations?\n\n"
                    "Please respond with ONLY a number (your percentage estimate)."
                ),
                "parse": "numeric",
            },
            "low_anchor": {
                "prompt": (
                    "Do you think the percentage of African countries "
                    "in the United Nations is higher or lower than 10%?\n\n"
                    "Now, what is your best estimate of the actual percentage "
                    "of African countries in the United Nations?\n\n"
                    "Please respond with ONLY a number (your percentage estimate)."
                ),
                "parse": "numeric",
            },
        },
    },

    # ── 2. FRAMING EFFECT (Tversky & Kahneman, 1981)
    "framing": {
        "description": "Asian Disease Problem — gain vs. loss frame",
        "conditions": {
            "gain_frame": {
                "prompt": (
                    "Imagine that the U.S. is preparing for an outbreak of an "
                    "unusual disease, which is expected to kill 600 people. "
                    "Two programs have been proposed:\n\n"
                    "Program A: 200 people will be saved.\n"
                    "Program B: There is a 1/3 probability that 600 people "
                    "will be saved, and a 2/3 probability that no one will be saved.\n\n"
                    "Which program do you prefer? Respond with ONLY 'A' or 'B'."
                ),
                "parse": "choice_AB",
            },
            "loss_frame": {
                "prompt": (
                    "Imagine that the U.S. is preparing for an outbreak of an "
                    "unusual disease, which is expected to kill 600 people. "
                    "Two programs have been proposed:\n\n"
                    "Program A: 400 people will die.\n"
                    "Program B: There is a 1/3 probability that nobody will die, "
                    "and a 2/3 probability that 600 people will die.\n\n"
                    "Which program do you prefer? Respond with ONLY 'A' or 'B'."
                ),
                "parse": "choice_AB",
            },
        },
    },

    # ── 3. SUNK COST FALLACY (Arkes & Blumer, 1985)
    "sunk_cost": {
        "description": "Sunk cost effect on continuation decisions",
        "conditions": {
            "high_sunk_cost": {
                "prompt": (
                    "You have bought a $100 ticket for a weekend ski trip to Michigan. "
                    "Several weeks later you buy a $50 ticket for a weekend ski trip "
                    "to Wisconsin. You think you will enjoy the Wisconsin trip more.\n\n"
                    "As you are putting your tickets away, you notice that the two "
                    "trips are for the same weekend! You must choose one.\n\n"
                    "Which trip do you go on? Respond with ONLY 'Michigan' or 'Wisconsin'."
                ),
                "parse": "choice_MW",
            },
            "no_sunk_cost": {
                "prompt": (
                    "You are planning a weekend ski trip. You can go to Michigan "
                    "or Wisconsin. You think you will enjoy the Wisconsin trip more.\n\n"
                    "Which trip do you go on? Respond with ONLY 'Michigan' or 'Wisconsin'."
                ),
                "parse": "choice_MW",
            },
        },
    },

    # ── 4. DECOY EFFECT (Huber, Payne & Puto, 1982)
    "decoy": {
        "description": "Asymmetric dominance / attraction effect",
        "conditions": {
            "no_decoy": {
                "prompt": (
                    "You are choosing between two restaurants:\n\n"
                    "Restaurant A: Quality rating 8/10, 5-minute drive\n"
                    "Restaurant B: Quality rating 6/10, 1-minute drive\n\n"
                    "Which restaurant do you choose? Respond with ONLY 'A' or 'B'."
                ),
                "parse": "choice_AB",
            },
            "with_decoy": {
                "prompt": (
                    "You are choosing between three restaurants:\n\n"
                    "Restaurant A: Quality rating 8/10, 5-minute drive\n"
                    "Restaurant B: Quality rating 6/10, 1-minute drive\n"
                    "Restaurant C: Quality rating 7/10, 8-minute drive\n\n"
                    "Which restaurant do you choose? Respond with ONLY 'A', 'B', or 'C'."
                ),
                "parse": "choice_ABC",
            },
        },
    },

    # ── 5. BASE RATE NEGLECT (Kahneman & Tversky, 1973)
    "base_rate": {
        "description": "Lawyer/Engineer problem — base rate neglect",
        "conditions": {
            "high_base_rate": {
                "prompt": (
                    "A study interviewed 30 engineers and 70 lawyers.\n"
                    "One participant, Jack, was randomly selected. Here is his description:\n\n"
                    "Jack is a 45-year-old man. He is married and has four children. "
                    "He is generally conservative, careful, and ambitious. He shows "
                    "no interest in political or social issues and spends most of his "
                    "free time on home carpentry, sailing, and mathematical puzzles.\n\n"
                    "What is the probability (0-100) that Jack is an engineer?\n"
                    "Respond with ONLY a number."
                ),
                "parse": "numeric",
            },
            "low_base_rate": {
                "prompt": (
                    "A study interviewed 70 engineers and 30 lawyers.\n"
                    "One participant, Jack, was randomly selected. Here is his description:\n\n"
                    "Jack is a 45-year-old man. He is married and has four children. "
                    "He is generally conservative, careful, and ambitious. He shows "
                    "no interest in political or social issues and spends most of his "
                    "free time on home carpentry, sailing, and mathematical puzzles.\n\n"
                    "What is the probability (0-100) that Jack is an engineer?\n"
                    "Respond with ONLY a number."
                ),
                "parse": "numeric",
            },
        },
    },
}

# ═══════════════════════════════════════════════════════════════════
#  Response Parsers
# ═══════════════════════════════════════════════════════════════════

def parse_numeric(text: str) -> Optional[float]:
    match = re.search(r"(\d+\.?\d*)\s*%?", text.strip())
    if match:
        return float(match.group(1))
    return None

def parse_choice(text: str, options: list[str]) -> Optional[str]:
    text_clean = text.strip()
    for opt in options:
        if text_clean.upper() == opt.upper():
            return opt
        if text_clean.upper().startswith(opt.upper()):
            return opt
    for opt in options:
        if opt.upper() in text_clean.upper():
            return opt
    return None

PARSERS = {
    "numeric": parse_numeric,
    "choice_AB": lambda t: parse_choice(t, ["A", "B"]),
    "choice_MW": lambda t: parse_choice(t, ["Michigan", "Wisconsin"]),
    "choice_ABC": lambda t: parse_choice(t, ["A", "B", "C"]),
}

# ═══════════════════════════════════════════════════════════════════
#  API Callers (With Retry)
# ═══════════════════════════════════════════════════════════════════

def _call_gemini(prompt: str, model: str, temperature: float) -> str:
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    gm = genai.GenerativeModel(model_name=model, system_instruction=SYSTEM_PROMPT)
    resp = gm.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            temperature=temperature, max_output_tokens=100
        ),
    )
    return resp.text.strip()

def _call_groq(prompt: str, model: str, temperature: float) -> str:
    from groq import Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=100,
    )
    return resp.choices[0].message.content.strip()

def _call_ollama(prompt: str, model: str, temperature: float) -> str:
    import urllib.request
    import json as _json
    body = _json.dumps({
        "model": model.replace("ollama:", ""),
        "prompt": prompt,
        "system": SYSTEM_PROMPT,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": 100},
    }).encode()
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = _json.loads(resp.read())
    return data["response"].strip()

def _call_openrouter(prompt: str, model: str, temperature: float) -> str:
    from openai import OpenAI
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=100,
    )
    return resp.choices[0].message.content.strip()

def _call_github(prompt: str, model: str, temperature: float) -> str:
    from openai import OpenAI
    client = OpenAI(
        base_url="https://models.inference.ai.azure.com",
        api_key=os.getenv("GITHUB_API_KEY"),
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=100,
    )
    return resp.choices[0].message.content.strip()

def _call_huggingface(prompt: str, model: str, temperature: float) -> str:
    from openai import OpenAI
    client = OpenAI(
        base_url="https://api-inference.huggingface.co/v1/",
        api_key=os.getenv("HUGGINGFACE_API_KEY"),
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=100,
    )
    return resp.choices[0].message.content.strip()

def _call_cohere(prompt: str, model: str, temperature: float) -> str:
    import cohere
    client = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))
    resp = client.chat(
        model=model,
        preamble=SYSTEM_PROMPT,
        message=prompt,
        temperature=temperature,
        max_tokens=100,
    )
    return resp.text.strip()

def safe_api_call(provider: str, prompt: str, model: str, temperature: float) -> str:
    """Executes the API call with exponential backoff for 429 Rate Limit errors."""
    
    caller_fn = None
    if provider == "Google": caller_fn = _call_gemini
    elif provider == "Groq": caller_fn = _call_groq
    elif provider == "OpenRouter": caller_fn = _call_openrouter
    elif provider == "Cohere": caller_fn = _call_cohere
    elif provider == "GitHub": caller_fn = _call_github
    elif provider == "HuggingFace": caller_fn = _call_huggingface
    else: caller_fn = _call_ollama

    max_retries = 5
    for attempt in range(max_retries):
        try:
            return caller_fn(prompt, model, temperature)
        except Exception as e:
            err_msg = str(e).lower()
            if "429" in err_msg or "quota" in err_msg or "rate" in err_msg:
                # Calculate exponential backoff: 10s, 20s, 40s...
                sleep_time = min(10 * (2 ** attempt), 60)
                print(f"    [!] {model} hit Rate Limit (429). Retrying in {sleep_time}s... (Attempt {attempt+1}/{max_retries})")
                time.sleep(sleep_time)
            else:
                return f"ERROR: {e}"
    return "ERROR: Max retries exceeded due to rate limits."

# ═══════════════════════════════════════════════════════════════════
#  Multithreaded Runner
# ═══════════════════════════════════════════════════════════════════

def process_model_queue(model_name: str, tasks: list, config: dict, shared_results: list, completed_ids: set):
    """
    Worker thread payload. Processes a specific model's requests sequentially 
    to strictly adhere to that API's RPM limit, avoiding 429 errors.
    """
    provider = config["provider"]
    delay = config["delay"]
    
    tasks_to_run = [t for t in tasks if t["id"] not in completed_ids]
    if not tasks_to_run:
        return
        
    print(f"▶ Thread started for {model_name} ({len(tasks_to_run)} trials left)")

    for idx, t in enumerate(tasks_to_run):
        start_ms = time.time() * 1000
        
        # 1. Dispatch safe API call
        raw_text = safe_api_call(provider, t["prompt"], model_name, t["temperature"])
        latency_ms = time.time() * 1000 - start_ms

        # 2. Parse results
        parser = PARSERS[t["parse_type"]]
        parsed = parser(raw_text) if "ERROR" not in raw_text else None

        row = {
            "trial_id": t["id"],
            "bias": t["bias"],
            "condition": t["condition"],
            "model": model_name,
            "provider": provider,
            "temperature": t["temperature"],
            "trial_num": t["trial_num"],
            "prompt": t["prompt"][:120] + "...",
            "raw_response": raw_text,
            "parsed_value": parsed,
            "parse_success": parsed is not None,
            "latency_ms": round(latency_ms, 1),
            "timestamp": datetime.now().isoformat(),
        }
        
        shared_results.append(row)
        
        # Print progress ~every 5 requests on this thread
        if (idx+1) % 5 == 0 or idx == len(tasks_to_run) - 1:
            print(f"  [{model_name}] Completed {idx+1}/{len(tasks_to_run)}  ({'✓' if row['parse_success'] else '✗'})")

        # 3. Enforce strict rate limit delay BEFORE next request
        if idx < len(tasks_to_run) - 1:
            time.sleep(delay)


def run_all_experiments(models: list[str], completed_ids: set, all_rows: list):
    # Build queues per model
    queues = {m: [] for m in models}
    
    for bias_name, bias_data in EXPERIMENTS.items():
        for cond_name, cond_data in bias_data["conditions"].items():
            for model_name in models:
                model_trials = ALL_MODELS[model_name].get("n_trials", N_TRIALS)
                for temp in TEMPERATURES:
                    for trial in range(1, model_trials + 1):
                        trial_str = f"{bias_name}-{cond_name}-{model_name}-{trial}"
                        trial_id = hashlib.md5(trial_str.encode()).hexdigest()[:12]
                        
                        queues[model_name].append({
                            "id": trial_id,
                            "bias": bias_name,
                            "condition": cond_name,
                            "trial_num": trial,
                            "temperature": temp,
                            "prompt": cond_data["prompt"],
                            "parse_type": cond_data["parse"]
                        })

    # Start saving thread to continually flush results to disk
    stop_saving = False
    def save_worker():
        while not stop_saving:
            if all_rows:
                pd.DataFrame(all_rows).to_csv(RAW_PATH, index=False)
            time.sleep(5)
            
    saver_thread = threading.Thread(target=save_worker, daemon=True)
    saver_thread.start()

    # Launch model worker threads
    threads = []
    for model_name in models:
        config = ALL_MODELS[model_name]
        tasks = queues[model_name]
        t = threading.Thread(
            target=process_model_queue, 
            args=(model_name, tasks, config, all_rows, completed_ids)
        )
        threads.append(t)
        t.start()
        
    # Wait for all experiments to finish
    for t in threads:
        t.join()

    # Stop saving thread, guarantee final write
    stop_saving = True
    saver_thread.join(timeout=10)
    
    df = pd.DataFrame(all_rows)
    df.to_csv(RAW_PATH, index=False)
    print(f"\n✅ All tasks finished! Raw data saved ({len(df)} rows)")

    generate_summary(df)


# ═══════════════════════════════════════════════════════════════════
#  Summary Statistics
# ═══════════════════════════════════════════════════════════════════

def generate_summary(df: pd.DataFrame):
    if len(df) == 0:
        return
        
    valid = df[df["parse_success"] == True].copy()
    summaries = []

    for (bias, cond, model), group in valid.groupby(["bias", "condition", "model"]):
        total_for_group = len(df[(df.bias == bias) & (df.condition == cond) & (df.model == model)])
        row = {
            "bias": bias,
            "condition": cond,
            "model": model,
            "n": len(group),
            "parse_rate": f"{len(group) / total_for_group * 100:.1f}%" if total_for_group > 0 else "0%",
        }

        vals = group["parsed_value"]

        if bias in ("anchoring", "base_rate"):
            numeric_vals = vals.astype(float)
            row["mean"] = round(numeric_vals.mean(), 2)
            row["median"] = round(numeric_vals.median(), 2)
            row["std"] = round(numeric_vals.std(), 2)
        else:
            counts = vals.value_counts()
            row["choice_distribution"] = dict(counts)
            if len(counts) > 0:
                row["most_common"] = counts.index[0]
                row["most_common_pct"] = f"{counts.iloc[0] / len(group) * 100:.1f}%"

        summaries.append(row)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(SUMMARY_PATH, index=False)
    
    print("\n" + "=" * 60)
    print("RESULTS OVERVIEW")
    print("=" * 60)
    for bias_name, bias_data in EXPERIMENTS.items():
        print(f"\n📊 {bias_name.upper()} — {bias_data['description']}")
        bias_rows = summary_df[summary_df["bias"] == bias_name]
        for _, r in bias_rows.iterrows():
            if "mean" in r and pd.notna(r.get("mean")):
                print(f"   {r['model']:30s} | {r['condition']:20s} | mean={r['mean']}")
            elif "most_common_pct" in r and pd.notna(r.get("most_common_pct")):
                print(f"   {r['model']:30s} | {r['condition']:20s} | {r.get('most_common','?')}={r['most_common_pct']}")


def detect_available_models() -> list[str]:
    available = []
    for model_name, info in ALL_MODELS.items():
        key_env = info["key_env"]
        if key_env is None:
            # Check Ollama
            try:
                import urllib.request
                urllib.request.urlopen("http://localhost:11434/api/tags", timeout=1)
                available.append(model_name)
                print(f"  ✅ {model_name:30s} — {info['provider']}")
            except Exception:
                print(f"  ⬚  {model_name:30s} — Ollama not running")
        elif os.getenv(key_env):
            available.append(model_name)
            print(f"  ✅ {model_name:30s} — {info['provider']}")
        else:
            print(f"  ⬚  {model_name:30s} — {key_env} not set")
    return available


if __name__ == "__main__":
    print("╔═══════════════════════════════════════════════════════╗")
    print("║  LLM Cognitive Bias Experiment Pipeline (FREE TIER)  ║")
    print("╚═══════════════════════════════════════════════════════╝")
    print("\nChecking APIs...")

    available = detect_available_models()

    if not available:
        print("\n❌ No models available. Update your .env file!")
        exit(1)

    max_trials = max(ALL_MODELS[m].get("n_trials", N_TRIALS) for m in available)
    total_calls = sum(ALL_MODELS[m].get("n_trials", N_TRIALS) * len(EXPERIMENTS) * 2 for m in available)
    # The longest duration thread dictates the total time
    worst_case_delay = max(ALL_MODELS[m]["delay"] for m in available)
    est_minutes = (max_trials * len(EXPERIMENTS) * 2 * worst_case_delay) / 60

    print("\nScanning for existing data...")
    completed_ids = set()
    all_rows = []
    
    if RAW_PATH.exists():
        try:
            existing = pd.read_csv(RAW_PATH)
            if "trial_id" in existing.columns:
                completed_ids = set(existing["trial_id"])
                all_rows = existing.to_dict("records")
                print(f"📂 Resuming gracefully — {len(completed_ids)} trials found in results/raw_responses.csv. These will be skipped!")
        except Exception as e:
            print(f"⚠️ Could not load existing results data: {e}")
    else:
        print("📁 No existing results found, starting a fresh dataset.")

    print(f"\n▶ Models: {len(available)} (running simultaneously!)")
    print(f"▶ Tokens & Requests: Optimized to safely match Free Tier API limits.")
    print(f"▶ Total API calls: {total_calls} (minus the {len(completed_ids)} already done)")
    print(f"▶ Estimated time: ~{est_minutes:.1f} minutes")
    print()

    input("Press Enter to start (Ctrl+C to cancel)... ")
    print("\nStarting execution threads...")
    run_all_experiments(models=available, completed_ids=completed_ids, all_rows=all_rows)
