"""Run in a FRESH Python process (each process gets its own string-hash seed).
Prints the code assigned to zone 132 (JFK) and a hash of all codes, for the
naive set-based vocabulary and for the fixed sorted vocabulary."""
import hashlib
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from b_features import load_cleaned, zone_codes, zone_codes_naive  # noqa: E402

zones = load_cleaned(sys.argv[1])['pickup_zone']
h = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:12]
naive, fixed = zone_codes_naive(zones), zone_codes(zones)
jfk = (zones.astype(str) == '132').to_numpy()
print(f'{int(naive[jfk][0])},{h(naive)},{int(fixed[jfk][0])},{h(fixed)}')
