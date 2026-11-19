[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/hrd04-ai/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/hrd04-ai/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

# GTA VI Trophy Rarity Forecast

An entry for make.fun's **gta6-trophy-rarity-forecast-01** competition:
forecast every Grand Theft Auto VI PlayStation 5 trophy's earn rate on
Exophase at 2026-11-26T00:00:00Z, as five quantiles per trophy.

## The program

`main.py` (Python 3.13, standard library only, no network) reads one case
per run from `$MAKEFUN_INPUT_DIR` (default `/input`) and writes one
forecast per case to `$MAKEFUN_OUTPUT_DIR` (default `/output`):
`{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}`.

## The model

An Exophase earn rate is `earned / tracked`, and both sides move.
Tracked players flood in after launch while earned counts grow only as
players actually reach each trophy, so early on the rates fall even
though more people earn the trophies every hour. The program models each
trophy's rate directly on the logit scale as a line against
`log(1 + hours since release)`, fitted by least squares over the last 24
hours of the case's history, anchored at the last read, with the slope
shrunk toward zero and damped for the extrapolation past the last read.

When the whole history is one frozen page — the launch-night situation,
where Exophase had not refreshed for hours — the line is flat and
useless, so the program falls back on the refresh prior: the next refresh
multiplies the tracked count several-fold (the first refresh of launch
night took the page from 638 to 3,868 tracked players) while earned
counts grow only modestly, which pushes every rate down. The size of that
push depends on the trophy class: story/chapter trophies are earned
fastest, grind and collection trophies slowest, and the platinum slowest
of all. A trophy every tracked player already has (100%) stays near
100%: a refresh cannot dilute it.

Spreads widen with the horizon like a random walk, plus an honest
model-uncertainty term that is largest when the history is frozen (a
refresh we have not seen yet can move every rate) and for rare trophies.
Everything is deterministic: no randomness, no clock, no network.

## Local test

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Practice cases `p00`/`p01` both validate (55 trophies, monotone
quantiles in 0–100, under 65,536 bytes), run in ~0.1s per case, and are
byte-identical across repeated runs. Medians sit below the last read
rates (mean logit shift −1.24): the model expects the refresh to dilute
the rates, e.g. a chapter trophy last at 1.10% forecasts a median of
0.35%, and the platinum last at 0.63% forecasts 0.15%.

## License

MIT — see `LICENSE`.
