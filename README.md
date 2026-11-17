# gta6-twitch-ccu-forecast-01

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/thespaceyasin/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/thespaceyasin/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

A Twitch launch-day forecaster: it reads a game's first-hour Twitch viewer count plus the hourly curves of past big launches, and forecasts hours 2–24 of the release day as five quantiles each. The model fits, per hour, how an hour's viewers (relative to hour 1) depend on the size of the hour-1 print across rolling-midnight launches — a huge hour-1 print means a front-loaded launch, a small one keeps ramping as time zones unlock (US East 05:00Z, US West 08:00Z) — and widens its bands when the print sits far from anything seen before. Standard library only, no network, deterministic. On the practice set it scores about 0.16 mean pinball loss against the house baseline's 0.39.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=4dhdkr5y&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
