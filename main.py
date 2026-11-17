"""Forecast GTA VI's launch-day Twitch viewers, hour by hour, as five quantiles.

Entry for make.fun competition gta6-twitch-ccu-forecast-01. Python 3.13,
standard library only, no network, deterministic: the same case always
writes the same forecast bytes.

The idea
--------
A launch day's shape is mostly set by *when* the world unlocks, and by how
much of the audience is already watching in hour 1. Compare 38 past big
launches (the case's own comparables): launches whose hour-1 print is huge
are front-loaded -- their audience is already there, so later hours grow
little or fall -- while launches with a small hour-1 print keep ramping
through the day as time zones unlock (US East at 05:00Z, US West at 08:00Z)
and word spreads. Across the rolling-midnight launches in the case, the
log-ratio of an hour's viewers to hour 1's print falls almost linearly in
log(hour-1 print), with a slope near -0.4 from hour 6 on (correlation about
-0.7 to -0.8), so this program fits that line fresh for every hour, from the
case's own comparables, and reads its forecast off it.

Concretely, for each hour h = 2..24, over the rolling-midnight comparables
(those whose unlock notes say "rolling" or "midnight local"; a launch whose
hour 1 is before its unlock, like a single global unlock at 14:00Z, has a
near-zero hour-1 print and a meaningless ratio, so it is left out):

    ln(v_h / v_1) = a_h + b_h * ln(v_1) + noise

fitted by a Huber-robust regression (a few reweighted rounds, so one
launch with a wildly different shape cannot drag the line for everyone
else). The median forecast is p1 * exp(a_h + b_h * x), where p1 is this
case's hour-1 print and x is ln(p1), capped at the training pool's largest
ln(v_1) plus 0.3 so an unprecedented print (GTA VI may well be one) cannot
drag the forecast far past anything ever seen. The five quantiles come
from the fit's own residuals, not from a normal curve: the residuals have
a tight core and a long upper tail (launches that ramp late), so the
empirical residual quantiles, centred on their median and widened by the
regression leverage  sqrt(1 + 1/n + (x - xbar)^2 / Sxx), put the 75th
percentile where the data puts it instead of where a bell curve would.

If too few comparables carry an hour, the program falls back to that
hour's empirical quantiles of the log-ratios, and if there is no usable
pool at all it falls back to the house persistence baseline (every hour at
hour 1's print, spread widening like a random walk). Every fallback keeps
the output valid: 23 rows, hours 2..24, five finite numbers > 0 and at
most 1e9, non-decreasing, at most 65,536 bytes.

Local validation (leave-one-out over the 16 rolling-midnight launches the
final case pools, and the 4 practice cases against SullyGnome's own
figures): the regression scores about 0.11 mean pinball loss on the log
scale where the persistence baseline scores about 0.18, and 0.080 on the
practice set against the baseline's 0.19. One launch (Resident Evil
Requiem) ramps far harder than its hour-1 print predicts and stays the
worst case under every variant tried; no second feature tested (pre-launch
baseline, hour-1 channels per viewer) predicted that shape.

How the grader runs it: `python3 -P -s /bot/main.py` in python:3.13-slim.
Outside the grader, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR stand in
for /input and /output:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
from datetime import datetime
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level (persistence fallback only).
Z = [-1.645, -0.674, 0, 0.674, 1.645]
# Persistence baseline's spread, in log units per square-root hour (the
# house baseline's fixed sigma; used only as a fallback).
BASE_SIGMA = 0.15
# How far past the training pool's largest log hour-1 print the forecast
# is allowed to extrapolate, in log units.
CAP_EXTRA = 0.3
# Minimum training launches for the regression; below this, use the
# empirical quantiles of the log-ratios instead.
MIN_REGRESSION_POINTS = 4


def round2(x):
    """x rounded to 2 decimals, half up, as JavaScript's Math.round does it.

    A whole number comes back as an int, so json.dumps writes 1240 where
    JavaScript writes 1240 (not 1240.0).
    """
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value == int(value) else value


def parse_ts(text):
    """An ISO 8601 UTC timestamp to epoch seconds."""
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def release_day_series(comp):
    """Hour (1..24) -> viewers for a comparable's release day (UTC).

    The comparable's hourly points run from 14 days before its release day
    to 3 days after; only the 24 hours of the release day itself are kept.
    """
    day_start = parse_ts(comp["day_start"])
    out = {}
    for point in comp.get("hourly") or []:
        viewers = point.get("viewers")
        if viewers is None:
            continue
        delta = (parse_ts(point["from"]) - day_start) / 3600
        if 0 <= delta < 24 and abs(delta - round(delta)) < 1e-6:
            out[int(delta) + 1] = viewers
    return out


def is_rolling(comp):
    """True when a comparable unlocked at midnight local time, rolling.

    The unlock note says so ("rolling", "midnight local"). A launch with
    no usable note is judged by its data instead: hour 1 must already hold
    at least 5% of the day's peak, which a pre-unlock hour 1 (a single
    global unlock later in the day) never does.
    """
    note = (comp.get("unlock") or "").lower()
    if "rolling" in note or "midnight local" in note:
        return True
    if note.strip():
        return False
    series = release_day_series(comp)
    if 1 not in series or not series[1]:
        return False
    peak = max(series.values())
    return peak > 0 and series[1] / peak >= 0.05


def ols(xs, ys):
    """Least-squares fit y = a + b*x; returns (a, b, xbar, sxx)."""
    n = len(xs)
    xbar = sum(xs) / n
    ybar = sum(ys) / n
    sxx = sum((x - xbar) ** 2 for x in xs)
    if sxx <= 0:
        return ybar, 0.0, xbar, 0.0
    b = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / sxx
    return ybar - b * xbar, b, xbar, sxx


def _wols(xs, ys, ws):
    """Weighted least squares y = a + b*x; returns (a, b)."""
    sw = sum(ws)
    xbar = sum(w * x for w, x in zip(ws, xs)) / sw
    ybar = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - xbar) ** 2 for w, x in zip(ws, xs))
    if sxx <= 0:
        return ybar, 0.0
    b = sum(w * (x - xbar) * (y - ybar) for w, x, y in zip(ws, xs, ys)) / sxx
    return ybar - b * xbar, b


def huber_fit(xs, ys, rounds=3, k=1.345):
    """Huber IRLS fit y = a + b*x; returns (a, b, xbar, sxx) like ols.

    A few reweighted rounds: each round's residuals set a robust scale
    (1.4826 * MAD), and points further than k scales out get downweighted
    instead of dropped. One launch with an extreme shape -- a late ramp
    twice the pool's -- then bends the line far less, while staying in
    the pool for the spread.
    """
    a, b, xbar, sxx = ols(xs, ys)
    for _ in range(rounds):
        res = [y - (a + b * x) for x, y in zip(xs, ys)]
        mad = sorted(abs(r) for r in res)[len(res) // 2]
        scale = 1.4826 * mad
        if scale <= 0:
            break
        ws = [1.0 if abs(r) <= k * scale else k * scale / abs(r) for r in res]
        a, b = _wols(xs, ys, ws)
    return a, b, xbar, sxx


def _quantile(sorted_values, tau):
    n = len(sorted_values)
    if n == 0:
        return float("nan")
    pos = tau * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def _stddev(values):
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))


def hour_forecast(p1, hour, pool):
    """Five quantiles for one hour, from the pool of (name, series)."""
    xs = []
    ys = []
    for _name, series in pool:
        if hour in series and series[hour] > 0 and series[1] > 0:
            xs.append(math.log(series[1]))
            ys.append(math.log(series[hour] / series[1]))
    if len(xs) >= MIN_REGRESSION_POINTS:
        a, b, xbar, sxx = huber_fit(xs, ys)
        x = math.log(p1)
        xmax = max(xs) + CAP_EXTRA
        if x > xmax:
            x = xmax
        median_log = a + b * x
        residuals = sorted(y - (a + b * xx) for xx, y in zip(xs, ys))
        # The quantiles are the fit's own residuals, not a bell curve:
        # centred on the residual median (so the 50th stays on the fitted
        # line) and widened by the regression leverage, so a print far
        # from the pool's centre honestly gets a wider band.
        med_res = _quantile(residuals, 0.5)
        lev = 1.0
        if sxx > 0:
            lev = math.sqrt(1.0 + 1.0 / len(xs) + (x - xbar) ** 2 / sxx)
        return [p1 * math.exp(median_log + (_quantile(residuals, tau) - med_res) * lev)
                for tau in LEVELS]
    if xs:
        ratios = sorted(ys)
        return [p1 * math.exp(_quantile(ratios, tau)) for tau in LEVELS]
    # No usable comparable for this hour: the persistence baseline.
    return [p1 * math.exp(z * BASE_SIGMA * math.sqrt(hour - 1)) for z in Z]


def forecast_case(case):
    """The forecast object for one case dict."""
    known = case.get("known") or []
    first = next((k for k in known if k.get("hour") == 1), None)
    if first is not None and first.get("viewers"):
        p1 = max(1, first["viewers"])
    else:
        # No hour-1 print: fall back to the last point of the history.
        history = case.get("history") or []
        p1 = 1
        for point in history:
            if point.get("viewers"):
                p1 = max(1, point["viewers"])
    pool = []
    for comp in case.get("comparables") or []:
        if "Grand Theft" in (comp.get("name") or ""):
            # The target category's own pre-release history is not a launch.
            continue
        series = release_day_series(comp)
        if 1 not in series or not series[1] or len(series) < 20:
            continue
        if is_rolling(comp):
            pool.append((comp.get("name") or "?", series))
    rows = []
    for row in case["forecast"]:
        hour = row["hour"]
        quantiles = hour_forecast(p1, hour, pool)
        # Enforce the output contract: finite, > 0, <= 1e9, non-decreasing.
        clean = []
        prev = 0.01
        for q in quantiles:
            if not math.isfinite(q) or q <= 0:
                q = prev
            q = min(q, 1e9)
            q = max(q, prev)
            clean.append(round2(q))
            prev = clean[-1]
        rows.append({"hour": hour, "q": clean})
    return {"hours": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        if len(text.encode("utf-8")) > 65536:
            raise ValueError(f"{path.name}: forecast over 65,536 bytes")
        (OUTPUT_DIR / f"{path.stem}.out").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
