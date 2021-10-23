from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np


def plot_aggregate(
        save_name='aggregated',
        model_name='cartpole_balance',
):
    p = Path('./data')
    fig, ax = plt.subplots()

    for x in p.iterdir():
        if model_name in x.name and x.suffix == '.json':
            f = x.open()
            g = json.load(f)
            n = len(g['avg_rewards'])
            ax.plot(np.linspace(1, n, n), g['avg_rewards'], label=g['label'])
    ax.legend()
    fig.savefig(save_name)
    plt.close(fig)
