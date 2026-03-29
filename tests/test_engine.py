"""Tests for the core inference engine."""
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestLengthPredictor:
    def test_math_is_short(self):
        from engine.length_predict import predict_length
        assert predict_length("What is 15 * 27?") < 600

    def test_code_is_medium(self):
        from engine.length_predict import predict_length
        assert predict_length("Write a Python quicksort function") > 200

    def test_yes_no_is_tiny(self):
        from engine.length_predict import predict_length
        assert predict_length("Yes or no: is 7 prime?") < 200

    def test_long_input_increases(self):
        from engine.length_predict import predict_length
        short = predict_length("Hi")
        long = predict_length("Explain " + "very " * 50 + "complex topic")
        assert long > short

    def test_task_detection(self):
        from engine.length_predict import detect_task_type
        assert detect_task_type("What is 15 * 27?") == "math"
        assert detect_task_type("Write a Python function") == "code"
        assert detect_task_type("Search for TurboQuant") == "research"
        assert detect_task_type("Hi") == "chat"

    def test_history_learning(self):
        from engine.length_predict import predict_length, record_actual_length
        record_actual_length("What is 1+1?", 50)
        record_actual_length("What is 2+2?", 55)
        record_actual_length("What is 3+3?", 60)
        pred = predict_length("What is 4+4?")
        assert 30 < pred < 200  # Should be influenced by history


class TestTurboKV:
    def test_compress_decompress(self):
        import mlx.core as mx
        from engine.turbo_kv import TurboQuantKV

        tq = TurboQuantKV(bits=4)
        kv = mx.random.normal((1, 4, 10, 32))
        idx, scales, shape = tq.compress(kv)
        recon = tq.decompress(idx, scales, shape)
        assert recon.shape == kv.shape

    def test_compression_ratio(self):
        from engine.turbo_kv import TurboQuantKV
        tq = TurboQuantKV(bits=4)
        assert tq.compression_ratio() > 3.0

    def test_indices_in_range(self):
        import mlx.core as mx
        from engine.turbo_kv import TurboQuantKV

        tq = TurboQuantKV(bits=4)
        kv = mx.random.normal((1, 4, 10, 32))
        idx, _, _ = tq.compress(kv)
        assert idx.max().item() < 16  # 4-bit = 16 levels


class TestRLM:
    def test_simple_skips_rlm(self):
        from engine.recursive_lm import RecursiveLM

        class FakeLLM:
            def chat(self, messages, **kw):
                return "Hello!"

        rlm = RecursiveLM(FakeLLM())
        assert not rlm._needs_rlm("hi")
        assert not rlm._needs_rlm("hello")
        assert not rlm._needs_rlm("capital of France")

    def test_math_uses_rlm(self):
        from engine.recursive_lm import RecursiveLM

        class FakeLLM:
            def chat(self, messages, **kw):
                return "result = str(15 * 27)"

        rlm = RecursiveLM(FakeLLM())
        assert rlm._needs_rlm("15 + 27")
        assert rlm._needs_rlm("56567 * 76678")

    def test_code_extraction(self):
        from engine.recursive_lm import RecursiveLM

        rlm = RecursiveLM(None)
        code = rlm._extract_code("```python\nresult = 42\n```")
        assert code.strip() == "result = 42"

    def test_code_extraction_no_fences(self):
        from engine.recursive_lm import RecursiveLM

        rlm = RecursiveLM(None)
        code = rlm._extract_code("result = str(15 * 27)")
        assert "result" in code
