"""Test with RAG-length prompt to measure prompt eval + generation separately."""
from llama_cpp import Llama
import time

MODEL = "D:/Offline_Rag_V2/models/llm/qwen2.5-1.5b-instruct-q4_k_m.gguf"

# Simulate a ~3500 token prompt (RAG context)
FILLER = "The video editing software provides advanced features for professional content creation. " * 80
PROMPT = (
    "<|im_start|>system\nYou are a helpful assistant. Answer using ONLY the provided context. "
    "Cite sources. Be thorough.<|im_end|>\n"
    f"<|im_start|>user\nCONTEXT:\n{FILLER}\n\n"
    "QUESTION: What are the main features of the video editing software?<|im_end|>\n"
    "<|im_start|>assistant\n"
)

for layers in [28, -1]:
    print(f"\n=== gpu_layers={layers} ===")
    m = Llama(
        model_path=MODEL, n_gpu_layers=layers, n_ctx=4096,
        n_batch=512, flash_attn=True, n_threads=6, verbose=False,
    )
    prompt_tokens = len(m.tokenize(PROMPT.encode("utf-8")))
    print(f"  Prompt tokens: {prompt_tokens}")

    # Warm up
    m.create_completion(prompt="Hello", max_tokens=5, temperature=0.3)

    # Benchmark with timing
    t0 = time.perf_counter()
    first_token_time = None
    count = 0
    for chunk in m.create_completion(
        prompt=PROMPT, max_tokens=200, temperature=0.3,
        stop=["<|im_end|>"], stream=True
    ):
        if first_token_time is None:
            first_token_time = time.perf_counter() - t0
        count += 1
    total = time.perf_counter() - t0

    gen_time = total - first_token_time if first_token_time else total
    print(f"  TTFT (prompt eval): {first_token_time:.2f}s ({prompt_tokens/first_token_time:.0f} tok/s prompt eval)")
    print(f"  Generation: {count} tok in {gen_time:.2f}s = {count/gen_time:.1f} tok/s")
    print(f"  Total: {total:.2f}s")

    del m
    import gc; gc.collect()
    time.sleep(2)
