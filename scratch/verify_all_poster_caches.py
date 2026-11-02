import numpy as np

files = {
    'Panel 3: Ophelia': 'Data/Poster_Cache_Ophelia.npz',
    'Panel 4: Pollard': 'Data/Poster_Cache_Pollard.npz',
    'Panel 5: German Bioreactor': 'Data/Poster_Cache_German.npz',
    'Panel 6: Gore Limit Cycles': 'Data/Gore_Comm5_Oscillation_Cache.npz',
}

for title, path in files.items():
    print("=" * 60)
    print(f"{title} -> {path}")
    print("=" * 60)
    data = np.load(path, allow_pickle=True)
    for k in data.files:
        val = data[k]
        print(f"  {k:18s}: shape={val.shape}, dtype={val.dtype}")
print("\nALL POSTER DATA CACHES VERIFIED AND READY!")
