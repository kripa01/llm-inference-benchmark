import pandas as pd
import os


def save_results(data, filename):

    os.makedirs(
        os.path.dirname(filename),
        exist_ok=True
    )

    df = pd.DataFrame([data])

    if os.path.exists(filename):

        old = pd.read_csv(filename)

        df = pd.concat(
            [old, df],
            ignore_index=True
        )

    df.to_csv(
        filename,
        index=False
    )