import pandas as pd
import matplotlib.pyplot as plt
import os


RESULTS_FILE = "results/benchmark_results.csv"
PLOT_DIR = "results/plots"


def create_plots():

    os.makedirs(
        PLOT_DIR,
        exist_ok=True
    )

    df = pd.read_csv(RESULTS_FILE)

    print(df)


    # Tokens per second graph
    plt.figure(figsize=(8,5))

    plt.bar(
        df["model"],
        df["tokens_per_second"]
    )

    plt.xlabel("Model")
    plt.ylabel("Tokens per Second")
    plt.title("LLM Inference Throughput")

    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.savefig(
        f"{PLOT_DIR}/tokens_per_second.png"
    )

    plt.close()



    # Inference latency graph
    plt.figure(figsize=(8,5))

    plt.bar(
        df["model"],
        df["inference_time_seconds"]
    )

    plt.xlabel("Model")
    plt.ylabel("Seconds")

    plt.title(
        "LLM Inference Latency"
    )

    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.savefig(
        f"{PLOT_DIR}/latency.png"
    )

    plt.close()


    print("Plots created successfully!")


if __name__ == "__main__":
    create_plots()