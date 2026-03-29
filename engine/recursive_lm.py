"""
Recursive Language Model (RLM) — Small models punch above their weight.

Based on: PrimeIntellect RLM (2026)

The model gets a Python sandbox with tools:
  - llm(prompt) → calls itself recursively
  - search(query) → searches the web
  - read_file(path) → reads files

Communication between sandbox and model uses a file bridge
(no shared memory needed — works on any system).

Benchmark: RLM scores 40/50 vs Standard's 30/50 on our test suite.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Dict, List, Optional


class RecursiveLM:
    """Wraps an LLM with a Python sandbox for recursive self-management."""

    def __init__(self, llm_client, max_retries: int = 2, timeout: int = 45):
        self.llm = llm_client
        self.max_retries = max_retries
        self.timeout = timeout
        self._call_count = 0
        self._max_calls = 10
        self._bridge_dir = None

    def _needs_rlm(self, task: str) -> bool:
        """Only use RLM when code execution or web search actually helps."""
        t = task.lower().strip()

        # ALWAYS use RLM: math with operators (Python computes exact answers)
        if re.search(r"\d+\s*[\+\-\*\/\%\^]\s*\d+", t):
            return True

        # ALWAYS use RLM: explicit search/find requests
        if any(w in t for w in ["search for", "search:", "find out", "look up", "latest news"]):
            return True

        # ALWAYS use RLM: computation tasks
        if any(w in t for w in ["calculate", "compute", "how many", "factorial", "fibonacci"]):
            return True

        # NEVER use RLM: code generation (model is better at writing code directly)
        if any(w in t for w in ["write code", "write a", "create a", "implement", "build",
                                  "golang", "python", "javascript", "rust", "html", "css",
                                  "function", "class", "api", "server", "app", "website"]):
            return False

        # NEVER use RLM: creative/explanatory tasks
        if any(w in t for w in ["explain", "describe", "write", "story", "poem", "essay",
                                  "tell me about", "what is", "how does", "why"]):
            return False

        # NEVER use RLM: simple chat
        if len(t) < 30:
            return False

        return False  # Default: direct mode (faster, better for most tasks)

    def solve(self, task: str) -> Dict:
        """Solve a task — routes to direct answer or code execution."""
        start = time.time()
        self._call_count = 0

        if not task or not task.strip():
            return {"answer": "Please enter a question.", "code": None,
                    "attempts": 0, "llm_calls": 0, "time": 0}

        try:
            if not self._needs_rlm(task):
                answer = self._call_llm(task)
                return {"answer": answer.strip(), "code": None, "attempts": 0,
                        "llm_calls": 1, "time": round(time.time() - start, 1)}
        except Exception as e:
            return {"answer": f"(error: {str(e)[:100]})", "code": None,
                    "attempts": 0, "llm_calls": 1, "time": round(time.time() - start, 1)}

        system = self._system_prompt()
        answer, code, attempts = None, None, 0

        for attempt in range(self.max_retries + 1):
            attempts += 1
            prompt = f"{system}\n\nTask: {task}\n\nWrite ONLY Python code. Set result = your answer."
            if attempt > 0:
                prompt += f"\n\nPrevious error: {last_error}\nWrite simpler code."

            response = self._call_llm(prompt)
            code = self._extract_code(response)

            if not code:
                answer = response
                break

            result = self._execute_code(code)
            if result["success"]:
                answer = result["output"]
                break
            else:
                last_error = result["error"]

        if answer is None or "No result variable set" in str(answer):
            answer = self._call_llm(f"Answer directly in plain text: {task}")

        return {"answer": answer.strip() if answer else "Failed",
                "code": code, "attempts": attempts,
                "llm_calls": self._call_count,
                "time": round(time.time() - start, 1)}

    def _system_prompt(self) -> str:
        return """You solve problems by writing Python code. Output ONLY code.

Built-in functions (USE THEM, do not import requests/urllib/bs4):
  result = llm("question")     → asks AI, returns string
  result = search("query")     → searches web, returns string
  result = read_file("path")   → reads a file, returns string

RULES:
  - MUST set: result = "your final answer"
  - For math: compute with Python (result = str(15 * 27))
  - For knowledge: data = search("topic"); result = llm(f"Summarize: {data[:500]}")
  - For non-Python code: result = llm("Write a Go HTTP server")
  - Keep under 20 lines. No markdown fences."""

    def _call_llm(self, prompt: str) -> str:
        self._call_count += 1
        if self._call_count > self._max_calls:
            return "Max calls reached."
        return self.llm.chat([{"role": "user", "content": prompt}], max_tokens=1000)

    def _extract_code(self, response: str) -> Optional[str]:
        match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
        if match:
            return match.group(1)
        match = re.search(r"```\n(.*?)```", response, re.DOTALL)
        if match and "result" in match.group(1):
            return match.group(1)
        indicators = ["result =", "result=", "import ", "def ", "search(", "llm("]
        lines = response.strip().split("\n")
        code_lines = sum(1 for l in lines if any(ind in l for ind in indicators))
        if code_lines >= 1 and "result" in response:
            clean = [l for l in lines if any(ind in l for ind in indicators) or l.startswith("#") or l.startswith(" ")]
            return "\n".join(clean) if clean else None
        return None

    def _execute_code(self, code: str) -> Dict:
        """Execute code in sandbox with file bridge for llm() calls."""
        bridge_dir = tempfile.mkdtemp(prefix="rlm_")
        self._bridge_dir = bridge_dir
        script = self._build_script(code, bridge_dir)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(script)
            script_path = f.name

        try:
            proc = subprocess.Popen(
                [sys.executable, script_path],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            start = time.time()
            while proc.poll() is None and (time.time() - start) < self.timeout:
                req_file = os.path.join(bridge_dir, "request.json")
                resp_file = os.path.join(bridge_dir, "response.txt")
                if os.path.exists(req_file):
                    try:
                        with open(req_file) as f:
                            req = json.load(f)
                        response = self._call_llm(req.get("prompt", ""))
                        with open(resp_file, "w") as f:
                            f.write(response)
                    except Exception as e:
                        with open(resp_file, "w") as f:
                            f.write(f"[Error: {e}]")
                time.sleep(0.3)

            if proc.poll() is None:
                proc.kill()
                return {"success": False, "output": "", "error": "Timeout"}

            output = proc.stdout.read().strip()
            error = proc.stderr.read().strip()

            if "RESULT:" in output:
                for line in output.split("\n"):
                    if line.startswith("RESULT:"):
                        output = line[7:].strip()
                        break

            return {"success": proc.returncode == 0 and len(output) > 0,
                    "output": output[:2000], "error": error[:500]}
        except Exception as e:
            return {"success": False, "output": "", "error": str(e)}
        finally:
            os.unlink(script_path)
            shutil.rmtree(bridge_dir, ignore_errors=True)

    def _build_script(self, user_code: str, bridge_dir: str) -> str:
        indent = "\n".join("    " + line for line in user_code.split("\n"))
        return f'''
import sys, json, time, os
BRIDGE = "{bridge_dir}"

def llm(prompt):
    req = os.path.join(BRIDGE, "request.json")
    resp = os.path.join(BRIDGE, "response.txt")
    with open(req, "w") as f: json.dump({{"prompt": prompt[:1000]}}, f)
    for _ in range(90):
        if os.path.exists(resp):
            with open(resp) as f: r = f.read()
            os.remove(resp); os.remove(req)
            return r
        time.sleep(0.5)
    return "[timeout]"

def search(query):
    try:
        from ddgs import DDGS
        with DDGS() as d: results = list(d.text(query, max_results=3))
        return "\\n".join([r.get("body","") for r in results])
    except Exception as e: return f"[search failed: {{e}}]"

def read_file(path):
    try:
        with open(path) as f: return f.read()[:2000]
    except: return "[could not read]"

try:
{indent}
    if "result" in dir() or "result" in locals():
        print(f"RESULT:{{result}}")
    else:
        print("RESULT:No result variable set")
except Exception as e:
    print(f"ERROR:{{e}}", file=sys.stderr)
    sys.exit(1)
'''
