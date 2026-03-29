"""
Self-improving research loop — gets smarter over time.

Unlike most "self-improvement" demos that just log to JSON,
this one ACTUALLY modifies how the engine operates:

  1. Search arxiv for new techniques
  2. Pick the most promising one
  3. Generate experiment code
  4. Run it on this machine
  5. Score results — keep or discard
  6. If kept → ask model to generate a CONFIG PATCH
  7. Apply patch to engine config → behavior actually changes

Config patches can modify:
  - max_tokens per task type (learned optimal lengths)
  - temperature per task type
  - RLM routing rules (which tasks need code execution)
  - System prompt tweaks
  - Custom few-shot examples that worked well
"""
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, List

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
KNOWLEDGE_PATH = os.path.join(DATA_DIR, "knowledge.json")
HISTORY_PATH = os.path.join(DATA_DIR, "history.json")
CONFIG_PATH = os.path.join(DATA_DIR, "learned_config.json")
EXPERIMENTS_DIR = os.path.join(DATA_DIR, "experiments")


def _load(path):
    return json.load(open(path)) if os.path.exists(path) else []

def _load_dict(path):
    return json.load(open(path)) if os.path.exists(path) else {}

def _save(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_learned_config() -> Dict:
    """Load the config that self-improvement has built over time."""
    return _load_dict(CONFIG_PATH)


class SelfImprover:

    def __init__(self, llm_client, topic: str = "LLM inference optimization"):
        self.llm = llm_client
        self.topic = topic
        os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

    def run_cycle(self) -> Dict:
        """One full improvement cycle that ACTUALLY changes engine behavior."""
        cycle_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"[improve] Cycle {cycle_id}")

        # 1. Search
        papers = self._search()
        if not papers:
            return {"cycle_id": cycle_id, "status": "no_papers"}

        # 2. Pick technique
        technique = self._pick(papers)
        print(f"  Technique: {technique['name']}")

        # 3. Generate experiment
        code = self._generate_experiment(technique)

        # 4. Run
        result = self._run_experiment(code, cycle_id)
        print(f"  Experiment: {'PASS' if result.get('passed') else 'FAIL'}")

        # 5. Evaluate
        evaluation = self._evaluate(technique, result)
        print(f"  Score: {evaluation['score']}, Action: {evaluation['action']}")

        # 6. If KEPT → generate and apply config patch
        applied_patch = None
        if "KEEP" in evaluation["action"] and evaluation["finding"]:
            applied_patch = self._apply_finding(evaluation["finding"], result)
            if applied_patch:
                print(f"  CONFIG PATCHED: {applied_patch}")

        summary = {
            "cycle_id": cycle_id,
            "technique": technique["name"],
            "passed": result.get("passed", False),
            "score": evaluation["score"],
            "finding": evaluation["finding"],
            "action": evaluation["action"],
            "config_patch": applied_patch,
        }
        history = _load(HISTORY_PATH)
        history.append(summary)
        _save(HISTORY_PATH, history)

        return summary

    def run_cycles(self, n: int = 3) -> List[Dict]:
        results = []
        for i in range(n):
            print(f"\n--- Cycle {i+1}/{n} ---")
            results.append(self.run_cycle())
        return results

    # ------------------------------------------------------------------
    # Step 6: THE KEY PART — Apply findings to engine config
    # ------------------------------------------------------------------

    def _apply_finding(self, finding: str, result: Dict) -> Dict:
        """
        Convert a finding into a config patch and apply it.

        This is what makes self-improvement REAL — the finding
        changes how the engine operates.
        """
        # Ask model to generate a config patch from the finding
        resp = self.llm.chat([{"role": "user", "content": f"""You found: "{finding}"

Convert this into a JSON config patch for an AI engine. Available keys:

  "max_tokens_math": int (default 200, how many tokens for math answers)
  "max_tokens_code": int (default 500)
  "max_tokens_chat": int (default 500)
  "temperature_math": float (default 0.3)
  "temperature_code": float (default 0.3)
  "temperature_chat": float (default 0.7)
  "rlm_for_math": bool (default true, use code execution for math)
  "rlm_for_search": bool (default true, use web search)
  "system_prompt_suffix": str (extra instruction added to all prompts)
  "few_shot_example": str (a worked example to prepend)

Output ONLY valid JSON. Only include keys that should change.
Example: {{"temperature_math": 0.1, "system_prompt_suffix": "Double-check all math."}}"""}], max_tokens=200)

        # Extract JSON from response
        patch = self._extract_json(resp)
        if not patch:
            return None

        # Validate — only allow known keys
        allowed = {
            "max_tokens_math", "max_tokens_code", "max_tokens_chat",
            "temperature_math", "temperature_code", "temperature_chat",
            "rlm_for_math", "rlm_for_search",
            "system_prompt_suffix", "few_shot_example",
        }
        patch = {k: v for k, v in patch.items() if k in allowed}
        if not patch:
            return None

        # Apply to learned config
        config = _load_dict(CONFIG_PATH)
        config.update(patch)
        config["_last_updated"] = datetime.now().isoformat()
        config["_total_patches"] = config.get("_total_patches", 0) + 1
        _save(CONFIG_PATH, config)

        return patch

    def _extract_json(self, text: str) -> Dict:
        """Extract JSON object from model response."""
        # Try to find JSON block
        match = re.search(r"\{[^{}]+\}", text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        # Try the whole response
        try:
            return json.loads(text.strip())
        except:
            return None

    # ------------------------------------------------------------------
    # Steps 1-5: Search, pick, experiment, evaluate
    # ------------------------------------------------------------------

    def _search(self) -> List[Dict]:
        print("  Searching...")
        try:
            from ddgs import DDGS
            with DDGS() as d:
                results = list(d.text(f"arxiv 2025 2026 {self.topic}", max_results=5))
            papers = [{"title": r.get("title", ""), "url": r.get("href", ""),
                       "snippet": r.get("body", "")} for r in results]
            print(f"  Found {len(papers)} papers")
            return papers
        except Exception as e:
            print(f"  Search failed: {e}")
            return []

    def _pick(self, papers: List[Dict]) -> Dict:
        # Include what we already know to avoid repeating
        knowledge = _load(KNOWLEDGE_PATH)
        known = ", ".join(k.get("technique", "") for k in knowledge[-5:]) if knowledge else "nothing yet"

        text = "\n".join(f"{i+1}. {p['title']}: {p['snippet'][:150]}" for i, p in enumerate(papers))
        resp = self.llm.chat([{"role": "user", "content":
            f"We already know: {known}\n\n"
            f"Pick ONE NEW technique to test:\n{text}\n\n"
            f"Reply:\nTECHNIQUE: name\nHYPOTHESIS: what to test"}], max_tokens=200)
        name = "unknown"
        for line in resp.split("\n"):
            if line.startswith("TECHNIQUE:"):
                name = line.split(":", 1)[1].strip()
        return {"name": name, "raw": resp}

    def _generate_experiment(self, technique: Dict) -> str:
        resp = self.llm.chat([{"role": "user", "content":
            f"Write a Python experiment for: {technique['name']}\n"
            "Requirements:\n"
            "- Print RESULT: metric_name = value\n"
            "- Print PASS or FAIL at the end\n"
            "- Complete in under 30 seconds\n"
            "- Use only stdlib (no pip install)\n"
            "- Must actually test something measurable\n"
            "Output ONLY Python code."}], max_tokens=800)
        match = re.search(r"```python\n(.*?)```", resp, re.DOTALL)
        return match.group(1) if match else resp

    def _run_experiment(self, code: str, cycle_id: str) -> Dict:
        exp_dir = os.path.join(EXPERIMENTS_DIR, cycle_id)
        os.makedirs(exp_dir, exist_ok=True)
        path = os.path.join(exp_dir, "experiment.py")
        with open(path, "w") as f:
            f.write(code)
        try:
            p = subprocess.run([sys.executable, path], capture_output=True,
                               text=True, timeout=30, cwd=exp_dir)
            return {"success": p.returncode == 0, "output": p.stdout[:500],
                    "error": p.stderr[:300], "passed": "PASS" in p.stdout}
        except subprocess.TimeoutExpired:
            return {"success": False, "output": "", "error": "Timeout", "passed": False}

    def _evaluate(self, technique: Dict, result: Dict) -> Dict:
        resp = self.llm.chat([{"role": "user", "content":
            f"Evaluate experiment:\n"
            f"Technique: {technique['name']}\n"
            f"Passed: {result.get('passed')}\n"
            f"Output: {result.get('output', '')[:300]}\n"
            f"Error: {result.get('error', '')[:200]}\n\n"
            "Reply EXACTLY:\nSCORE: 1-10\nFINDING: one actionable sentence\nACTION: KEEP or DISCARD"}], max_tokens=150)

        score, finding, action = 0, "", "DISCARD"
        for line in resp.split("\n"):
            if line.startswith("SCORE:"):
                try:
                    score = float(line.split(":")[1].strip().split("/")[0])
                except:
                    pass
            elif line.startswith("FINDING:"):
                finding = line.split(":", 1)[1].strip()
            elif line.startswith("ACTION:"):
                action = line.split(":", 1)[1].strip().upper()

        if "KEEP" in action and finding:
            knowledge = _load(KNOWLEDGE_PATH)
            knowledge.append({"finding": finding, "score": score,
                               "technique": technique["name"],
                               "date": datetime.now().isoformat()})
            _save(KNOWLEDGE_PATH, knowledge)

        return {"score": score, "finding": finding, "action": action}

    def report(self) -> str:
        knowledge = _load(KNOWLEDGE_PATH)
        history = _load(HISTORY_PATH)
        config = _load_dict(CONFIG_PATH)
        patches = config.get("_total_patches", 0)
        return (f"Self-Improvement: {len(history)} cycles, "
                f"{len(knowledge)} findings, {patches} config patches applied")
