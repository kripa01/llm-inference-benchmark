import time
import torch

from config import (
    MODEL_NAME,
    PROMPT,
    MAX_NEW_TOKENS,
    RESULTS_FILE
)

from model_loader import load_model

from metrics import (
    get_memory_usage,
    calculate_tokens_per_second
)

from utils import save_results


def run_benchmark():

    print("=" * 50)
    print("LLM Inference Benchmark")
    print("=" * 50)

    print(f"Model: {MODEL_NAME}")


    # Choose device
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Using device: {device}")


    memory_before = get_memory_usage()


    # Load model
    start = time.perf_counter()

    tokenizer, model = load_model(MODEL_NAME)

    model = model.to(device)
    model.eval()

    load_time = time.perf_counter() - start


    # Tokenize input
    inputs = tokenizer(
        PROMPT,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }


    print("\nDebug:")
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Input device: {inputs['input_ids'].device}")


    # Inference
    print("\nStarting inference...")

    if device == "mps":
        torch.mps.synchronize()

    start = time.perf_counter()


    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS
        )


    if device == "mps":
        torch.mps.synchronize()


    inference_time = time.perf_counter() - start


    generated_tokens = (
        output.shape[1]
        -
        inputs["input_ids"].shape[1]
    )


    tokens_per_second = calculate_tokens_per_second(
        generated_tokens,
        inference_time
    )


    memory_after = get_memory_usage()


    text = tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )


    results = {

        "model": MODEL_NAME,

        "device": device,

        "load_time_seconds":
            round(load_time, 3),

        "inference_time_seconds":
            round(inference_time, 3),

        "generated_tokens":
            generated_tokens,

        "tokens_per_second":
            tokens_per_second,

        "memory_used_mb":
            round(memory_after - memory_before, 2)

    }


    print("\nResults")
    print("-" * 30)

    for key, value in results.items():
        print(f"{key}: {value}")


    print("\nGenerated:")
    print(text)


    save_results(
        results,
        RESULTS_FILE
    )


if __name__ == "__main__":
    run_benchmark()