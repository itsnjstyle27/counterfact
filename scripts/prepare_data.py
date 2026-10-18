"""Convert Open Bandit Dataset CSVs into compact .npz files.

# full release, read straight from the downloaded zip (no 11.7 GB extraction)
python scripts/prepare_data.py --source data/download/open_bandit_dataset.zip

# the 10,000-row sample bundled with obp, written to data/sample/processed
python scripts/prepare_data.py --source data/sample/obd --out data/sample/processed
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from counterfact.data import CAMPAIGNS, POLICIES, convert_one, data_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True, help="release zip or directory with {policy}/{campaign}/")
    parser.add_argument("--out", type=Path, default=None, help="output directory (default: data/processed)")
    parser.add_argument("--policies", nargs="+", default=list(POLICIES), choices=POLICIES)
    parser.add_argument("--campaigns", nargs="+", default=list(CAMPAIGNS), choices=CAMPAIGNS)
    parser.add_argument("--chunksize", type=int, default=500_000)
    args = parser.parse_args()

    out = args.out or data_dir() / "processed"
    for policy in args.policies:
        for campaign in args.campaigns:
            start = time.perf_counter()
            card = convert_one(args.source, policy, campaign, out, chunksize=args.chunksize)
            card["seconds"] = round(time.perf_counter() - start, 1)
            print(json.dumps(card), flush=True)


if __name__ == "__main__":
    main()
