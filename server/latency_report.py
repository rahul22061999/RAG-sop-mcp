import re
import sys
import urllib.request
from collections import defaultdict

BUCKET = re.compile(r"^(rag_\w+_seconds)_bucket\{(.*)\}\s+(\S+)$")
LABEL = re.compile(r'(\w+)="([^"]*)"')


def quantile(q: float, buckets: list[tuple[float, float]]) -> float:
    total = buckets[-1][1]
    if total == 0:
        return float("nan")
    rank, prev_le, prev_count = q * total, 0.0, 0.0
    for le, count in buckets:
        if count >= rank:
            if le == float("inf"):
                return prev_le
            span = count - prev_count
            return prev_le + (le - prev_le) * ((rank - prev_count) / span if span else 0)
        prev_le, prev_count = le, count
    return prev_le


def main(url: str) -> None:
    text = urllib.request.urlopen(url, timeout=10).read().decode()
    series: dict[tuple, dict[float, float]] = defaultdict(lambda: defaultdict(float))
    for line in text.splitlines():
        m = BUCKET.match(line)
        if not m:
            continue
        metric, raw_labels, value = m.groups()
        labels = dict(LABEL.findall(raw_labels))
        le = float("inf") if labels["le"] == "+Inf" else float(labels["le"])
        key = (metric, labels.get("rag_mode", ""), labels.get("rag_outcome", ""), labels.get("rag_cache", ""))
        series[key][le] += float(value)

    print(f"{'metric':36} {'mode':7} {'outcome':10} {'cache':5} {'n':>5} {'p50':>8} {'p90':>8} {'p99':>8}  (ms)")
    for (metric, mode, outcome, cache), by_le in sorted(series.items()):
        buckets = sorted(by_le.items())
        n = int(buckets[-1][1])
        p50, p90, p99 = (quantile(q, buckets) for q in (0.5, 0.9, 0.99))
        print(f"{metric:36} {mode:7} {outcome:10} {cache:5} {n:5d} {p50 * 1000:8.0f} {p90 * 1000:8.0f} {p99 * 1000:8.0f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000/metrics")
