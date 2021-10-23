from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np

p = Path('./data')
fig, ax = plt.subplots()

for x in p.iterdir():
    if x.suffix == '.json':
        f = x.open()
        g = json.load(f)
        ax.plot(np.linspace(1, g['num_episodes'], g['num_episodes']), g['avg_rewards'], label=g['label'])
ax.legend()
fig.savefig('aggregated')
plt.close(fig)

