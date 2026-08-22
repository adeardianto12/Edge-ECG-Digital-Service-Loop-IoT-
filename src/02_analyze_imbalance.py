"""Report the local MIT-BIH label counts for the N-versus-V experiment."""

import argparse
from collections import Counter
from pathlib import Path

import wfdb


WINDOW_SIZE = 300
HALF_WINDOW = WINDOW_SIZE // 2
TRAINING_SYMBOLS = {"N", "V"}
KNOWN_BEAT_SYMBOLS = {
    "N", "L", "R", "B", "A", "a", "J", "S", "V", "r", "F", "e", "j",
    "n", "E", "f", "/", "Q", "?",
}


def count_symbols(data_dir: Path) -> tuple[Counter, Counter, int]:
    accepted = Counter()
    excluded_beats = Counter()
    nonbeat_annotations = 0
    record_names = sorted(path.stem for path in data_dir.glob("*.hea"))
    if not record_names:
        raise FileNotFoundError(f"Expected local MIT-BIH .hea files in {data_dir}")

    for name in record_names:
        header = wfdb.rdheader(str(data_dir / name))
        annotation = wfdb.rdann(str(data_dir / name), "atr")
        for peak, symbol in zip(annotation.sample, annotation.symbol):
            if peak < HALF_WINDOW or peak + HALF_WINDOW >= header.sig_len:
                continue
            if symbol in TRAINING_SYMBOLS:
                accepted[symbol] += 1
            elif symbol in KNOWN_BEAT_SYMBOLS:
                excluded_beats[symbol] += 1
            else:
                nonbeat_annotations += 1
    return accepted, excluded_beats, nonbeat_annotations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/mitdb"))
    args = parser.parse_args()

    accepted, excluded_beats, nonbeat_annotations = count_symbols(args.data_dir)
    normal_count = accepted["N"]
    pvc_count = accepted["V"]
    accepted_total = normal_count + pvc_count
    print("MIT-BIH N-versus-PVC data report")
    print("=" * 40)
    print(f"Accepted beats: {accepted_total}")
    print(f"Normal N: {normal_count} ({normal_count / accepted_total:.2%})")
    print(f"PVC V: {pvc_count} ({pvc_count / accepted_total:.2%})")
    print(f"N:V ratio: {normal_count / pvc_count:.2f}:1")
    print(f"Excluded other beat annotations: {sum(excluded_beats.values())}")
    print(f"Excluded non-beat annotations: {nonbeat_annotations}")
    print("Excluded beat symbols:")
    for symbol, count in excluded_beats.most_common():
        print(f"  {symbol}: {count}")


if __name__ == "__main__":
    main()
