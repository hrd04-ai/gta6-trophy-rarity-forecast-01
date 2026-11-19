#!/usr/bin/env python3
"""GTA VI trophy rarity forecaster for make.fun.

Reads one case per run from MAKEFUN_INPUT_DIR (default /input): every
valid read the house made of the Exophase trophy page before the case's
checkpoint, plus the trophy list. Writes one forecast per case to
MAKEFUN_OUTPUT_DIR (default /output): for each of the 55 trophies, five
quantiles [q05, q25, q50, q75, q95] of its earn rate (percent) at the
case's target, on the logit scale.

Model, in one paragraph: an Exophase earn rate is earned / tracked, and
both sides move. Tracked players flood in after launch while earned
counts grow only as players actually reach each trophy, so early on the
rates fall even though more people earn the trophies every hour. I model
each trophy's rate directly on the logit scale as a line against
log(1 + hours since release), fitted by least squares over the last 24
hours of the case's history, anchored at the last read, with the slope
shrunk toward zero and damped for the extrapolation beyond the last
read (a straight line in log-time is a trend, not a promise). When the whole history is one frozen
page -- the launch-night situation, where Exophase had not refreshed for
hours -- the line is flat and useless, so I fall back on the refresh
prior: the next refresh multiplies the tracked count several-fold while
earned counts grow only modestly, which pushes every rate down. The
size of that push depends on the trophy class: story/chapter trophies
are earned fastest, grind and collection trophies slowest, and the
platinum slowest of all. Spreads widen with the horizon like a random
walk, plus an honest model-uncertainty term that is largest when the
history is frozen (a refresh we have not seen yet can move every rate)
and for rare trophies. Everything is deterministic: no randomness, no
clock, no network, standard library only.
"""

import json
import math
import os
import sys
from datetime import datetime

# Standard-normal quantiles for the five levels [0.05, 0.25, 0.5, 0.75, 0.95].
Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722)

# How much the tracked count jumps at one Exophase refresh after a long
# frozen stretch. The first refresh of launch night took the page from
# 638 to 3,868 tracked players (about 6x); 5x is the conservative centre
# of that prior, with the extra uncertainty carried in the spread.
REFRESH_TRACKED_MULT = 5.0

# Earned-count growth across one refresh, by trophy class. Story
# trophies are earned by everyone pushing through the campaign; grind /
# collection trophies need hours of side activity; the platinum needs
# everything. All well below the tracked jump, so every rate falls.
EARNED_MULT = {
    "story": 1.6,    # chapter / final-mission trophies
    "oneoff": 1.5,   # single-action trophies (fishing duel, arcade, ...)
    "grind": 1.3,    # counts, collections, money, "every location"
    "platinum": 1.2,  # 100% completion / the platinum itself
}

GRIND_WORDS = (
    "250 stunts", "50 challenges", "8 mission strands", "every spot",
    "every location", "100 locations", "collectible card", "win 20",
    "earn $", "spend $", "purchase all", "register 10", "discover 100",
    "rob 5", "100% completion", "5 bank", "all garage",
)


def parse_time(text):
    """Parse an ISO-8601 timestamp (with Z or offset) to an aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def logit(p):
    """logit of a probability strictly inside (0, 1)."""
    p = clamp(p, 1e-6, 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def trophy_class(trophy):
    """Classify a trophy by how fast its earned count grows."""
    desc = trophy.get("description", "").lower()
    tier = trophy.get("tier", "")
    if tier == "platinum" or "100% completion" in desc:
        return "platinum"
    if desc.startswith("complete chapter") or "final mission" in desc:
        return "story"
    if any(word in desc for word in GRIND_WORDS):
        return "grind"
    return "oneoff"


def least_squares_slope(xs, ys):
    """Slope of the least-squares line through (xs, ys); 0 if degenerate."""
    n = len(xs)
    if n < 3:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0.0:
        return 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / sxx


def forecast_trophy(trophy, reads, x_target, x_last, horizon_days,
                    history_stale, n_reads):
    """Five quantiles for one trophy, in percent, non-decreasing."""
    idx = trophy["id"] - 1
    last_rate = None
    for read in reversed(reads):
        rate = read["rates"][idx]
        if rate is not None:
            last_rate = float(rate)
            break
    if last_rate is None:
        last_rate = 1.0  # no read ever listed this trophy; neutral fallback

    # A trophy every tracked player already has stays (nearly) universal:
    # a refresh cannot dilute it, because every new player earns it too.
    if last_rate >= 99.99:
        median_logit = logit(0.999)
        sigma = 0.35
    elif history_stale:
        # Frozen page: the next refresh is the whole story. Tracked jumps
        # several-fold, earned grows only by its class multiplier, so the
        # rate drops by earned_mult / tracked_mult in logit terms.
        mult = EARNED_MULT[trophy_class(trophy)] / REFRESH_TRACKED_MULT
        median_logit = logit(last_rate / 100.0) + math.log(mult)
        walk = 0.5 * math.sqrt(horizon_days)
        sigma = math.sqrt(walk * walk + 1.15 * 1.15)
    else:
        # Real history: fit logit(rate) against log(1 + hours since
        # release), anchor at the last read, shrink the slope toward zero
        # and damp the extrapolation past the last read. The slope is fit
        # on the LAST 24 HOURS only: over a launch week the early reads
        # are dominated by the denominator flood (tracked exploding while
        # earned lags), but the trend that matters for the target is the
        # recent one -- rates bottom out and turn up as progression
        # outpaces new buyers. A fit over the whole history would drag
        # the median the wrong way.
        xs, ys = [], []
        recent = [read for read in reads
                  if (reads[-1]["at"] - read["at"]).total_seconds()
                  <= 24 * 3600]
        window = recent if len(recent) >= 3 else reads
        for read in window:
            rate = read["rates"][idx]
            if rate is not None:
                xs.append(read["x"])
                ys.append(logit(clamp(float(rate), 0.01, 99.99) / 100.0))
        slope = least_squares_slope(xs, ys)
        slope = clamp(slope, -2.0, 2.0)
        shrink = len(xs) / (len(xs) + 4.0)  # weak evidence -> slope near 0
        shift = slope * shrink * 0.6 * (x_target - x_last)
        shift = clamp(shift, -3.5, 2.5)
        median_logit = logit(clamp(last_rate, 0.01, 99.99) / 100.0) + shift
        walk = 0.5 * math.sqrt(horizon_days)
        unc = 0.30 + 0.20 * math.sqrt(horizon_days)
        if last_rate < 0.5:
            unc += 0.25  # rare trophies: earned counts are tiny, noisy
        if n_reads < 10:
            unc += 0.15  # short history: the trend is barely identified
        sigma = math.sqrt(walk * walk + unc * unc)

    quantiles = []
    for z in Z:
        q = 100.0 * sigmoid(median_logit + z * sigma)
        quantiles.append(round(clamp(q, 0.01, 99.99), 4))
    # Rounding can only tie neighbours, never invert them, but enforce
    # monotonicity explicitly so the output is valid by construction.
    for i in range(1, len(quantiles)):
        if quantiles[i] < quantiles[i - 1]:
            quantiles[i] = quantiles[i - 1]
    return quantiles


def forecast_case(case):
    trophies = case["trophies"]
    history = case.get("history", [])

    release = parse_time(case["game"]["release_date"] + "T00:00:00Z"
                         if len(case["game"]["release_date"]) == 10
                         else case["game"]["release_date"])
    target = parse_time(case["target"])

    def x_of(moment):
        hours = (moment - release).total_seconds() / 3600.0
        return math.log1p(max(hours, 0.0))

    reads = []
    for read in history:
        reads.append({
            "at": parse_time(read["at"]),
            "x": x_of(parse_time(read["at"])),
            "rates": read["rates"],
            "tracked": read["players_tracked"],
        })
    x_target = x_of(target)
    x_last = reads[-1]["x"] if reads else x_target
    horizon_days = float(case["horizon_minutes"]) / 1440.0

    # Stale page? Every read shows the same tracked count and the same
    # rates: Exophase has not refreshed once in the whole history.
    history_stale = False
    if len(reads) >= 2:
        tracked_set = {r["tracked"] for r in reads}
        rates_set = {tuple(r["rates"]) for r in reads}
        history_stale = len(tracked_set) == 1 and len(rates_set) == 1

    rows = []
    for trophy in trophies:
        rows.append({
            "id": trophy["id"],
            "q": forecast_trophy(trophy, reads, x_target, x_last,
                                 horizon_days, history_stale, len(reads)),
        })
    return {"trophies": rows}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(output_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(input_dir) if n.endswith(".in"))
    for name in names:
        with open(os.path.join(input_dir, name), "r", encoding="utf-8") as f:
            case = json.load(f)
        forecast = forecast_case(case)
        stem = name[:-3]
        with open(os.path.join(output_dir, stem + ".out"), "w",
                  encoding="utf-8") as f:
            f.write(json.dumps(forecast, separators=(",", ":")))
            f.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
