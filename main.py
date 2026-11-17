"""Calibrated log-curve forecaster for GTA VI launch-day Twitch viewers (v2).

Forecasts, for each hour 2..24 of the release day (UTC), five quantiles of the
Twitch category's concurrent viewers, from the case file alone: hour 1's print
plus the past big-launch curves shipped in the case (`comparables`).

Method (v2 — winner pattern: comparables-weighted log-curve, closed form):

1. Shape from history. For every comparable launch, take its release-day
   hourly viewers as ratios to its own hour-1 print. Each hour's log-ratio is
   regressed (OLS) on log10(hour-1 print): log r_h = a_h + b_h * log10(p1).
   The regression is fit twice: (a) on the rolling-midnight-local launches
   only -- the same unlock pattern as GTA VI and as the four practice
   launches; (b) on every comparable with hour-1 >= 1000 viewers (the full
   set). The two fits are blended 60/40 in (mu, sd) space: the similar
   launches drive the shape, the full set stabilises it.
2. Calibration (chosen by a 4-regime sweep, ~70 variants; see BUILD-NOTES-v2):
   the median is shifted +0.05 in log space (rolling launches are
   systematically under-predicted by the pooled fit -- leave-one-out
   residuals of rolling comparables run +0.3 in log-ratio from hour 6 on,
   against ~0.0 for the rest), and the spread is NOT one global factor:
   early hours (2..8, where the unlock surge timing varies most across
   launches) use residual SD x 0.7, late hours (9..24, where the evening
   shape is far more consistent) use residual SD x 0.5. The residual SD
   itself is inflated by a few extreme launches, so both factors are below
   1. No median smoothing: smoothing blunts the hour-5 surge and the
   midday dip and cost ~0.001-0.002 in the sweep.
3. Closed-form quantiles. The quantiles are the normal quantiles of the
   blended (mu, sd) -- no Monte Carlo. The sweep proved MC identical to
   closed form to ~0.0003 (rho 0.3/0.6/0.8, 10k/20k/40k pairs all within
   noise, seed replication 0.1464 vs 0.1466), so MC only added runtime
   (~1.5 s) and Monte Carlo noise. Closed form runs in under 0.5 s per
   case end to end (interpreter startup and I/O included), is perfectly
   deterministic, and leaves a wide margin under the 8 s cap.
4. Calibration guards. The hour-1 log is clamped to the comparables' range
   so a record-breaking hour 1 cannot extrapolate the slope wild; median
   ratios are clamped to [0.15, 20]; quantiles are floored at 1 viewer,
   capped at 1e9, and forced non-decreasing. If the case carries too few
   usable comparables, or anything unexpected happens, the forecast falls
   back to the house persistence baseline (hour 1 held, widening spread) so
   the case is still valid -- an invalid case forfeits the prize outright.

What was tried and REJECTED by the sweep (all numbers = mean pinball loss
on the 4 practice cases, workbench scorer reproducing the server to ~0.001):
- Kernel-weighted comparables (bandwidth 0.3/0.5/0.8 dex on log10(p1),
  rolling bonus x2): worse everywhere (best 0.1471 vs 0.1413) and fragile
  under comparable dropout (0.172 vs 0.155). Small-p1 targets depend on a
  few small comparables; down-weighting the rest destabilises the slope.
- Rolling-only blend (1.0): 0.1608 -- the 12-curve roll fit alone is too
  noisy; the full set stabilises it. Blend 0.5: 0.1470; blend 0.6 wins.
- One global spread: k=0.5 gives 0.1454, k=0.6 gives 0.1463, k=0.7 gives
  0.1503; the hour split (0.7 early / 0.5 late) beats all of them at 0.1413.
- Slope shrinkage (0.5-0.85), quadratic log10(p1), trimmed OLS, channels
  as a second regressor, early-access filter on the roll set, p1_min
  raises, empirical (non-normal) quantiles, LOO-bias correction, and a
  p1-agnostic median stabiliser (0.1883 -- catastrophic): all rejected.
- Tail stretch x1.25: buys stress-regime coverage (0.793 -> 0.848) but
  costs practice loss (0.1413 -> 0.1452); rejected -- loss is what ranks.

Practice backtest (ground truth = SullyGnome hourly averages for the four
practice launches, scored with the contest's own log-scale pinball loss,
entry run end-to-end on each practice case): mean loss 0.1413 vs house
baseline 0.3871 and vs v1's 0.1517 (p00 0.134, p01 0.113, p02 0.115,
p03 0.204), typical miss 25.0% vs 26.0% (v1) and 82.8% (baseline),
90% coverage 0.957. Mimics-final check (other 3 practice games added to
training, as the final's 42 comparables include them): 0.1377. Dropout
stress (25% of comparables removed, two seeds): 0.1551 / 0.1395.
Honest note: hyperparameter selection on 4 cases carries winner's curse --
leave-one-launch-out selection averages 0.149-0.154 held-out, so the final
loss on one new game is plausibly ~0.14-0.15, not exactly 0.1413.

Standard library only, no network, deterministic, under 0.5 s per case
end to end.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a helper
# module shipped beside this file would import. (This entry is one file.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# --- tuned constants (sweep of ~70 variants over 4 regimes; see BUILD-NOTES-v2) ---
BLEND_W = 0.6         # weight of the rolling-midnight fit in the blend
SPREAD_K_EARLY = 0.7  # residual-SD factor, hours 2..8 (surge timing varies)
SPREAD_K_LATE = 0.5   # residual-SD factor, hours 9..24 (evening shape stable)
SPREAD_SPLIT_HOUR = 8  # hours <= this use the early factor
MEDIAN_SHIFT = 0.05   # additive log shift of the median (rolling under-prediction)
P1_MIN_FULL = 1000    # full set: comparables with hour-1 viewers >= this
P1_MIN_ROLL = 10000   # rolling set: comparables with hour-1 viewers >= this
SD_FLOOR = 0.15       # never claim a log spread tighter than this
SD_CAP = 1.5          # never claim a log spread wider than this
RATIO_LO = 0.15       # median ratio clamp, low end
RATIO_HI = 20.0       # median ratio clamp, high end
BASE_SIGMA = 0.15     # fallback (house baseline) spread per sqrt-hour


def norm_ppf(p):
    """Standard normal quantile (Acklam's approximation, ~1e-9 accurate)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    plow, phigh = 0.02425, 1.0 - 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


def parse_dt(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def round2(x):
    """Round to 2 decimals, half up (matches the house baseline's output)."""
    scaled = x * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100.0
    return int(value) if value == int(value) else value


def comp_curves(case):
    """Usable comparables: (hour-1 viewers, {hour: ratio}, unlock text).

    A comparable is usable when its release day has an hour-1 print. Its
    ratio curve is its hourly viewers divided by that print. The GTA VI
    pre-release comparable has no release-day hour 1 and drops out here.
    """
    out = []
    for comp in case.get("comparables") or []:
        day0 = parse_dt(comp["day_start"])
        by_hour = {}
        for row in comp.get("hourly") or []:
            viewers = row.get("viewers")
            if viewers is None:
                continue
            hour = int((parse_dt(row["from"]) - day0).total_seconds() // 3600) + 1
            if 1 <= hour <= 24:
                by_hour[hour] = viewers
        if 1 in by_hour and by_hour[1] >= 1:
            p1 = by_hour[1]
            out.append((p1, {h: v / p1 for h, v in by_hour.items()},
                        comp.get("unlock") or ""))
    return out


def is_rolling(unlock_text):
    """True when the launch unlocked at midnight local time, rolling."""
    text = unlock_text.lower()
    return "midnight local" in text or "rolling" in text


def ols(xs, ys):
    """Least squares y = a + b*x; returns (a, b, residual SD)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx if sxx > 0 else 0.0
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    sd = math.sqrt(sum(r * r for r in resid) / max(1, n - 2))
    return a, b, sd


def regression_mu_sd(curves, p1, p1_min):
    """Per-hour (mu, sd) of log-ratio ~ log10(hour-1 print).

    Returns {hour: (mu, sd)} in log-ratio space for the hours with enough
    data, or None when too few curves carry any hour. The hour-1 log is
    clamped to the training range so the fitted slope cannot extrapolate
    wildly for a record launch; mu is clamped to the median-ratio range.
    """
    lp1 = math.log10(max(1.0, p1))
    train = [(p, r) for p, r, _ in curves if p >= p1_min]
    if len(train) < 8:
        return None
    lo = min(math.log10(p) for p, _ in train)
    hi = max(math.log10(p) for p, _ in train)
    lp1 = min(max(lp1, lo), hi)
    out = {}
    for h in range(2, 25):
        pts = [(math.log10(p), math.log(r[h]))
               for p, r in train if h in r]
        if len(pts) < 8:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        a, b, sd = ols(xs, ys)
        mu = a + b * lp1
        mu = min(max(mu, math.log(RATIO_LO)), math.log(RATIO_HI))
        sd = min(max(sd, SD_FLOOR), SD_CAP)
        out[h] = (mu, sd)
    return out or None


def baseline_quantiles(p1):
    """House persistence baseline: hour 1 held, spread widening by hour."""
    return {h: [round2(p1 * math.exp(z * BASE_SIGMA * math.sqrt(h - 1)))
                for z in Z] for h in range(2, 25)}


def clean_quantiles(raw, p1):
    """Floor at 1 viewer, cap at 1e9, force non-decreasing, round to 2dp."""
    out = {}
    for h in sorted(raw):
        qs = [min(max(float(q), 1.0), 1e9) for q in raw[h]]
        for i in range(1, 5):
            qs[i] = max(qs[i], qs[i - 1])
        out[h] = [round2(q) for q in qs]
    return out


def forecast_case(case):
    """Forecast one case: closed-form calibrated quantiles, baseline on failure."""
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError("case has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))
    try:
        curves = comp_curves(case)
        roll = [(p, r, u) for p, r, u in curves
                if is_rolling(u) and p >= P1_MIN_ROLL]
        full = regression_mu_sd(curves, p1, P1_MIN_FULL)
        if full is None:
            raise ValueError("too few usable comparables")
        roll_fit = regression_mu_sd(roll, p1, 1) if len(roll) >= 8 else None
        base = baseline_quantiles(p1)
        raw = {}
        for h in range(2, 25):
            if roll_fit is not None and h in roll_fit and h in full:
                mu = BLEND_W * roll_fit[h][0] + (1.0 - BLEND_W) * full[h][0]
                sd = BLEND_W * roll_fit[h][1] + (1.0 - BLEND_W) * full[h][1]
            elif h in full:
                mu, sd = full[h]
            elif roll_fit is not None and h in roll_fit:
                mu, sd = roll_fit[h]
            else:
                raw[h] = base[h]
                continue
            mu = mu + MEDIAN_SHIFT
            k = SPREAD_K_EARLY if h <= SPREAD_SPLIT_HOUR else SPREAD_K_LATE
            sd = sd * k
            raw[h] = [p1 * math.exp(mu + norm_ppf(tau) * sd) for tau in LEVELS]
        return clean_quantiles(raw, p1)
    except Exception:
        return clean_quantiles(baseline_quantiles(p1), p1)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        out = {"hours": [{"hour": h, "q": forecast[h]}
                         for h in range(2, 25)]}
        (OUTPUT_DIR / (path.stem + ".out")).write_text(
            json.dumps(out, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
