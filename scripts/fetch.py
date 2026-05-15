"""Pull college basketball team_box parquet files."""

import argparse
import os
import urllib.request

URL = (
    "https://github.com/sportsdataverse/hoopR-mbb-data/raw/main/"
    "mbb/team_box/parquet/team_box_{year}.parquet"
)


def fetch(years, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for y in years:
        path = os.path.join(out_dir, f"team_box_{y}.parquet")
        if os.path.exists(path) and os.path.getsize(path) > 100_000:
            print(f"  have {y}")
            continue
        try:
            urllib.request.urlretrieve(URL.format(year=y), path)
            sz = os.path.getsize(path) / 1e6
            print(f"  wrote {y} ({sz:.2f} MB)")
        except Exception as e:
            print(f"  skip {y}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+",
                    default=[2018, 2019, 2022, 2023, 2024])
    ap.add_argument("--out", default="data")
    args = ap.parse_args()
    fetch(args.years, args.out)


if __name__ == "__main__":
    main()
