#!/usr/bin/env python3
"""
Consilium — Local AI agent that runs on 8GB RAM.

No cloud. No API keys. No Ollama. Just your model, your machine.

Usage:
  python consilium.py          # Interactive chat
  python consilium.py --bench  # Run benchmark
"""
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from engine.fast_mlx import FastMLXModel
from engine.recursive_lm import RecursiveLM
from engine.length_predict import predict_length


BANNER = """\033[95m
  ██████╗ ██████╗ ███╗   ██╗███████╗██╗██╗     ██╗██╗   ██╗███╗   ███╗
  ██╔════╝██╔═══██╗████╗  ██║██╔════╝██║██║     ██║██║   ██║████╗ ████║
  ██║     ██║   ██║██╔██╗ ██║███████╗██║██║     ██║██║   ██║██╔████╔██║
  ██║     ██║   ██║██║╚██╗██║╚════██║██║██║     ██║██║   ██║██║╚██╔╝██║
  ╚██████╗╚██████╔╝██║ ╚████║███████║██║███████╗██║╚██████╔╝██║ ╚═╝ ██║
   ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝╚══════╝╚═╝╚══════╝╚═╝ ╚═════╝ ╚═╝     ╚═╝
\033[0m\033[90m  Qwen3.5-4B + Claude Opus Reasoning | 31 tok/s | Fully Offline
  /help for commands | /rlm to toggle agent mode\033[0m
"""


def main():
    model_path = os.path.join(ROOT, "model")

    if not os.path.exists(os.path.join(model_path, "model.safetensors")):
        print("Model not found. Run: python setup.py")
        return

    print(BANNER)
    print("\033[93m  Loading...\033[0m", end="", flush=True)
    model = FastMLXModel(model_path)
    print(f"\033[92m ready ({model.load_time:.1f}s)\033[0m\n")

    rlm = RecursiveLM(model, timeout=45)
    rlm_mode = False
    history = []

    while True:
        try:
            tag = "\033[96m[RLM]\033[0m " if rlm_mode else ""
            user = input(f"{tag}\033[97mYou:\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\033[90mBye!\033[0m")
            break

        if not user:
            continue

        # --- Commands ---
        cmd = user.lower()

        if cmd in ("/quit", "/exit", "/q"):
            print("\033[90mBye!\033[0m")
            break

        if cmd == "/help":
            _help()
            continue

        if cmd == "/rlm":
            rlm_mode = not rlm_mode
            s = "\033[92mON\033[0m (code + search)" if rlm_mode else "\033[91mOFF\033[0m (fast)"
            print(f"  RLM: {s}\n")
            continue

        if cmd == "/clear":
            history = []
            model.clear_cache()
            print("  \033[90mHistory + KV cache cleared\033[0m\n")
            continue

        if cmd == "/stats":
            s = model.stats()
            print(f"  Requests:      {s['requests']} | Avg: {s['avg_response_time']}s")
            print(f"  Weights:       {s['weights']}")
            print(f"  KV Cache:      {s['kv_cache']}")
            print(f"  OOM recovered: {s['oom_recoveries']}")
            print(f"  Errors:        {s['errors']}")
            print()
            continue

        if cmd == "/bench":
            _bench(model)
            continue

        if cmd == "/improve":
            _improve(model)
            continue

        if cmd.startswith("/hyper"):
            _hyper(model, user[6:].strip())
            continue

        if cmd.startswith("/search "):
            _search(user[8:])
            continue

        if cmd.startswith("/run "):
            _run(user[5:])
            continue

        # --- Generate ---
        history.append({"role": "user", "content": user})

        try:
            start = time.time()
            if rlm_mode:
                r = rlm.solve(user)
                response = r["answer"]
                elapsed = time.time() - start
                meta = f"{elapsed:.1f}s, {r['llm_calls']} calls"
            else:
                # Smart history: only send last 4 messages for short questions
                # Full history for complex questions (saves prefill time)
                if len(user.split()) < 10 and len(history) > 4:
                    recent = history[-4:]
                else:
                    recent = history[-12:]  # Cap at 6 exchanges max
                response = model.chat(recent, max_tokens=1500)
                elapsed = time.time() - start
                meta = f"{elapsed:.1f}s"

            clean = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
            if not clean:
                clean = response.strip()

            print(f"\n\033[95mConsilium\033[0m \033[90m({meta})\033[0m\033[95m:\033[0m")
            print(f"  {chr(10).join('  ' + l if i > 0 else l for i, l in enumerate(clean.split(chr(10))))}\n")

            history.append({"role": "assistant", "content": response})

        except KeyboardInterrupt:
            print("\n  \033[93m(interrupted)\033[0m\n")
            # Remove the unanswered user message
            history.pop()
            continue

        except RuntimeError as e:
            if "memory" in str(e).lower() or "kIOGPU" in str(e):
                print(f"\n  \033[91mOut of memory. Try:\033[0m")
                print(f"    - Close other apps")
                print(f"    - Use /clear to reset conversation")
                print(f"    - Ask a shorter question\n")
                history.pop()
                # Clear memory
                import gc
                gc.collect()
            else:
                print(f"\n  \033[91mError: {str(e)[:100]}\033[0m\n")
                history.pop()
            continue

        except Exception as e:
            print(f"\n  \033[91mError: {str(e)[:100]}\033[0m\n")
            history.pop()
            continue

        if len(history) > 20:
            history = history[-20:]


def _help():
    print("""
  \033[95mCommands:\033[0m
    /rlm          Toggle agent mode (code + web search)
    /search Q     Search the web
    /run CODE     Execute Python code
    /bench        Speed benchmark
    /improve      Run self-improvement cycle
    /hyper TASK   Run HyperAgent on a task
    /stats        Model statistics
    /clear        Clear history
    /quit         Exit
""")


def _bench(model):
    tests = [("Math", "What is 15 * 27?"), ("Code", "Python is_prime. Short."), ("Quick", "Capital of France?")]
    for name, q in tests:
        s = time.time()
        r = model.chat_simple(q, max_tokens=200)
        print(f"  {name:8s} | {time.time()-s:.1f}s | {r[:60]}")
    print()


def _improve(model):
    from engine.self_improve import SelfImprover
    imp = SelfImprover(model)
    imp.run_cycle()
    print(f"  {imp.report()}\n")


def _hyper(model, task):
    if not task:
        task = "math_solver"
    from engine.hyper_agent import HyperAgent
    def bench(result, output, error):
        return 10 if result and len(result) > 0 and "error" not in result.lower() else 0
    agent = HyperAgent(model, task_name=task)
    agent.improve(bench)
    print(f"  {agent.report()}\n")


def _search(query):
    try:
        from ddgs import DDGS
        with DDGS() as d:
            for r in d.text(query, max_results=3):
                print(f"\n  \033[97m{r.get('title','')}\033[0m")
                print(f"  \033[90m{r.get('body','')[:150]}\033[0m")
                print(f"  \033[94m{r.get('href','')}\033[0m")
    except Exception as e:
        print(f"  \033[91m{e}\033[0m")
    print()


def _run(code):
    try:
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=10)
        if r.stdout:
            print(f"  \033[92m{r.stdout.strip()}\033[0m")
        if r.stderr:
            print(f"  \033[91m{r.stderr.strip()}\033[0m")
    except subprocess.TimeoutExpired:
        print("  \033[91mTimeout\033[0m")
    print()


if __name__ == "__main__":
    if "--bench" in sys.argv:
        model = FastMLXModel(os.path.join(ROOT, "model"))
        _bench(model)
    else:
        main()
