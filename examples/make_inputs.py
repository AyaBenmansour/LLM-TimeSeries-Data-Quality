"""Generate the small, deterministic synthetic input tables used in the CLI examples."""
from pathlib import Path

import numpy as np
import pandas as pd


def make_tables():
    rng = np.random.default_rng(23)
    n = 112
    index = np.arange(n)
    regular = pd.DataFrame({"record_id": [f"r{i}" for i in index], "timestamp": index,
                            "value": np.sin(index * .4) + rng.normal(0, .08, n)})
    channels = []
    for name, step, count in (("meter_a", .95, 96), ("meter_b", 1.4, 69)):
        t = np.round(np.cumsum(rng.uniform(.85*step, 1.15*step, count)), 4)
        channels.append(pd.DataFrame({"timestamp": t, "sensor_id": name,
                                      "value": 20 + np.sin(t / 6) + rng.normal(0, .1, count)}))
    asynchronous = pd.concat(channels).sort_values("timestamp")
    return {"regular": regular, "asynchronous": asynchronous}


if __name__ == "__main__":
    folder = Path(__file__).parent
    for name, frame in make_tables().items():
        frame.to_csv(folder / f"{name}.csv", index=False, float_format="%.12g")
