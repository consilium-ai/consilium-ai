"""
HyperAgent — Self-modifying AI that improves its own code.

Based on: Meta's HyperAgents (arXiv 2603.19461)

The agent maintains task_code (solution) and meta_prompt (improvement strategy).
Every cycle it tests, improves, and keeps/reverts. Every 3 cycles it improves
HOW it improves (meta-learning).

State persists to disk — survives restarts, improves over days.
"""
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from typing import Callable, Dict, List, Optional


class HyperAgent:

    def __init__(self, llm_client, task_name: str = "default", data_dir: str = "data/hyper"):
        self.llm = llm_client
        self.task_name = task_name
        self.task_dir = os.path.join(data_dir, task_name)
        os.makedirs(self.task_dir, exist_ok=True)

        self.task_code: str = ""
        self.meta_prompt: str = self._default_meta()
        self.best_score: float = 0
        self.history: List[Dict] = []
        self._load()

    def _default_meta(self) -> str:
        return ("You improve code to score higher. Fix errors first, then optimize logic. "
                "Output ONLY improved Python code with result = 'answer'.")

    def improve(self, benchmark_fn: Callable) -> Dict:
        """One improvement cycle: test → improve → test → keep/revert."""
        cycle = len(self.history) + 1
        print(f"[hyper] Cycle {cycle}")

        # Test current
        cur = self._run(self.task_code, benchmark_fn) if self.task_code else {"score": 0}
        print(f"  Current: {cur['score']}/10")

        # Ask for improvement
        new_code = self._ask_improve(cur)
        if not new_code:
            return {"cycle": cycle, "action": "skip"}

        # Test improved
        new = self._run(new_code, benchmark_fn)
        print(f"  Improved: {new['score']}/10")

        if new["score"] >= cur["score"]:
            self.task_code = new_code
            self.best_score = new["score"]
            action = "KEEP"
        else:
            action = "REVERT"

        entry = {"cycle": cycle, "action": action,
                 "old": cur["score"], "new": new["score"],
                 "time": datetime.now().isoformat()}
        self.history.append(entry)

        if cycle % 3 == 0:
            self._improve_meta()

        self._save()
        print(f"  → {action}")
        return entry

    def run_cycles(self, benchmark_fn: Callable, n: int = 5) -> List[Dict]:
        return [self.improve(benchmark_fn) for _ in range(n)]

    def _ask_improve(self, result: Dict) -> Optional[str]:
        resp = self.llm.chat([{"role": "user", "content":
            f"{self.meta_prompt}\n\nCode:\n```python\n"
            f"{self.task_code or '# Write initial solution'}\n```\n"
            f"Output: {result.get('output', '')[:200]}\n"
            f"Error: {result.get('error', '')[:200]}\n"
            f"Score: {result.get('score', 0)}/10\n\n"
            f"Write improved Python code. MUST set result = answer."}], max_tokens=800)

        match = re.search(r"```python\n(.*?)```", resp, re.DOTALL)
        if match:
            return match.group(1)
        if "result" in resp and "=" in resp:
            return resp
        return None

    def _improve_meta(self):
        kept = sum(1 for h in self.history[-6:] if h["action"] == "KEEP")
        resp = self.llm.chat([{"role": "user", "content":
            f"Current strategy: {self.meta_prompt}\n"
            f"Recent: {kept} kept of {min(6, len(self.history))} cycles. Best: {self.best_score}/10.\n"
            f"Write a better 2-sentence improvement strategy."}], max_tokens=200)
        if len(resp) > 20:
            self.meta_prompt = resp
            print("  [meta] Strategy updated")

    def _run(self, code: str, benchmark_fn: Callable) -> Dict:
        path = os.path.join(self.task_dir, "run.py")
        with open(path, "w") as f:
            f.write(code)
        try:
            p = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=15)
            result_line = ""
            for line in p.stdout.split("\n"):
                if line.startswith("RESULT:"):
                    result_line = line[7:].strip()
            score = benchmark_fn(result_line, p.stdout, p.stderr)
            return {"success": p.returncode == 0, "output": p.stdout[:500],
                    "error": p.stderr[:300], "score": score}
        except subprocess.TimeoutExpired:
            return {"success": False, "output": "", "error": "Timeout", "score": 0}

    def _save(self):
        with open(os.path.join(self.task_dir, "state.json"), "w") as f:
            json.dump({"code": self.task_code, "meta": self.meta_prompt,
                        "best": self.best_score, "history": self.history}, f, indent=2)

    def _load(self):
        path = os.path.join(self.task_dir, "state.json")
        if os.path.exists(path):
            with open(path) as f:
                s = json.load(f)
            self.task_code = s.get("code", "")
            self.meta_prompt = s.get("meta", self._default_meta())
            self.best_score = s.get("best", 0)
            self.history = s.get("history", [])

    def report(self) -> str:
        scores = " → ".join(str(h["new"]) for h in self.history[-10:])
        return (f"HyperAgent [{self.task_name}]: {len(self.history)} cycles, "
                f"best {self.best_score}/10, progression: {scores}")
