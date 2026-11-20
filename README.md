# GTA VI trophy rarity forecast

[![Make.fun bounty](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![Rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/rafiehamas/fun/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard)
[![Reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/rafiehamas/fun/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

An offline Python 3.13 forecaster for the Exophase tracker. It collapses unchanged reads, recovers integer earn-count ranges consistent with the published rounded rates, and projects player counts and earn counts separately. Recent refreshes help estimate whether player growth is slowing. The model samples discrete unlock counts for rare trophies, pools sparse history across related trophies, and accounts for uncertain cache delays. Chapter and platinum requirements constrain the resulting quantiles.

Run it with the `input/` directory from the competition kit.

```sh
MAKEFUN_INPUT_DIR=/path/to/kit/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

The program writes one JSON `.out` file per `.in` case. It uses only the Python standard library and produces deterministic results.

Built for the week-one trophy forecasting bounty on [make.fun](https://make.fun/).
