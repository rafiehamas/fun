"""Offline quantile forecasts for scheduled reads of a trophy tracker."""

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import random
import re
from statistics import median


DAY = 86400
SAMPLES = 768


@dataclass
class Snapshot:
    time: float
    players: int
    rates: list
    earned: int


@lru_cache(maxsize=1024)
def timestamp(value):
    return datetime.fromisoformat(value).timestamp()


def clamp(x, low, high):
    return min(high, max(low, x))


def ratio_log(a, b):
    return math.log(a / b)


def snapshot_series(history):
    snapshots = []
    for read in history:
        snap = Snapshot(timestamp(read.get('slot') or read['at']) / DAY,
                        read['players_tracked'], read['rates'],
                        read.get('trophies_earned', 0))
        if snap.players <= 0:
            continue
        if snapshots:
            last = snapshots[-1]
            if (snap.players, snap.earned, snap.rates) == (last.players, last.earned, last.rates):
                continue
        snapshots.append(snap)
    return snapshots


def count_bounds(rate, players):
    scaled = round(rate * 100)
    first = max(0, ((2 * scaled - 1) * players + 19999) // 20000)
    last = min(players, ((2 * scaled + 1) * players - 1) // 20000)
    if first <= last:
        return first, last
    nearest = round(rate * players / 100)
    return nearest, nearest


def series_for(snapshots, index):
    return [(snap, count_bounds(snap.rates[index], snap.players)) for snap in snapshots
            if snap.rates[index] is not None]


def midpoint(bounds):
    return sum(bounds) * 0.5


def evidence(snapshots, anchor, count):
    """Estimate a time elasticity with one vote per refresh and recency weighting."""
    active = [(snap.time, count(snap)) for snap in snapshots]
    if len(active) < 2:
        return 0.0, 0.0, 0.0
    last_time, last_count = active[-1]
    slopes, weights = [], []
    for when, n in active[:-1]:
        span = last_time - when
        if span < 0.035:
            continue
        dx = ratio_log(last_time - anchor, when - anchor)
        slopes.append(clamp(ratio_log(last_count + 0.5, n + 0.5) / dx, -2.0, 4.0))
        weight = min(1.0, span / 0.22) * math.exp(-span / 1.6)
        weights.append(weight)
    if not weights:
        return 0.0, 0.0, 0.0
    center = median(slopes)
    robust = [w / max(1.0, abs(s - center) / 0.65) for s, w in zip(slopes, weights)]
    slope = sum(w * s for w, s in zip(robust, slopes)) / sum(robust)
    span = active[-1][0] - active[0][0]
    reliability = min(1.0, span / 0.55) * min(1.0, (len(slopes) + 0.45) / 3)
    disagreement = median(abs(s - center) for s in slopes)
    return slope, reliability, disagreement


def group(trophy, start_rate):
    label = trophy['description'].lower()
    if start_rate >= 85:
        return 'opening'
    if trophy['tier'] == 'platinum' or re.search(r'100% completion|every trophy', label):
        return 'completion'
    if re.search(r'complete chapter|complete the final mission\.', label):
        return 'story'
    if re.search(r'\b(?:all|every)\b|\b(?:100|250|50)\b|collectible', label):
        return 'long'
    return 'other'


def launch_quantiles(bounds, players, kind, elapsed_since_change, horizon, rng):
    """Account for the discontinuity when an old launch cache is first replaced."""
    refresh_probability = clamp(0.53 + 0.095 * elapsed_since_change + 0.10 * horizon,
                                0.55, 0.94)
    forecasts = []
    for _ in range(SAMPLES):
        earned = rng.randint(*bounds)
        if rng.random() >= refresh_probability:
            forecasts.append(100 * earned / players)
            continue
        incoming = max(1, round(players * math.exp(rng.gauss(math.log(6), 0.31))))
        if kind == 'opening':
            missing = players - earned
            missing += rng.choices((0, 1, 2, 4), (0.68, 0.2, 0.09, 0.03))[0]
            rate = 100 * (1 - missing / incoming)
        else:
            mean_new = (earned + 0.4) * (0.15 if kind == 'story' else 0.045)
            delta = round(rng.expovariate(1 / max(0.04, mean_new)))
            rate = 100 * (earned + delta) / incoming
        forecasts.append(round(clamp(rate, 0, 100), 2))
    return forecasts


def growth_path(snapshots, anchor, target, rng):
    last = snapshots[-1]
    age = last.time - anchor
    observed, trust, disagreement = evidence(snapshots, anchor, lambda snap: snap.players)
    recent = [snap for snap in snapshots if last.time - snap.time <= 0.9]
    recent_slope, recent_trust, _ = evidence(recent, anchor, lambda snap: snap.players)
    if len(recent) >= 4:
        # The full window can otherwise extrapolate launch acceleration long
        # after recent refreshes show the tracked population leveling off.
        recent_weight = 0.55 * recent_trust
        observed += recent_weight * clamp(recent_slope - observed, -1.1, 1.1)
    observed = clamp(observed, 0, 2.7)
    exponent = 0.78 * (1 - 0.84 * trust) + observed * 0.84 * trust
    cadence = median(b.time - a.time for a, b in zip(snapshots, snapshots[1:])) if len(snapshots) > 1 else 0.25
    cadence = clamp(cadence, 0.025, 0.75)
    remaining = max(0, target - last.time)
    for _ in range(SAMPLES):
        staleness = cadence * rng.random() * (0.5 + rng.random())
        effective_age = max(age, target - anchor - staleness)
        dx = ratio_log(effective_age, age)
        drift = rng.gauss(0, 0.14 + min(0.4, disagreement * 0.25) + 0.22 * (1 - trust))
        grow = clamp(exponent + drift, 0.0, 2.9)
        multiplier = math.exp(grow * dx)
        yield dx, max(0, remaining - staleness), max(last.players, round(last.players * multiplier))


def quantiles(values, levels):
    values.sort()
    return [values[round(level * (len(values) - 1))] for level in levels]


def constrain_prerequisites(trophies, rows):
    chapters = []
    platinum = []
    for trophy, row in zip(trophies, rows):
        description = trophy['description'].lower().strip().rstrip('.')
        chapter = re.fullmatch(r'complete chapter (\d+)', description)
        if chapter:
            chapters.append((int(chapter[1]), row))
        elif description == 'complete the final mission':
            chapters.append((math.inf, row))
        if trophy['tier'] == 'platinum':
            platinum.append(row)
    chapters.sort(key=lambda item: item[0])
    for (_, earlier), (_, later) in zip(chapters, chapters[1:]):
        later['q'] = [min(a, b) for a, b in zip(earlier['q'], later['q'])]
    if len(platinum) == 1:
        platinum[0]['q'] = [min(level) for level in zip(*(row['q'] for row in rows))]


def adjudicate(case):
    snapshots = snapshot_series(case['history'])
    levels = case['levels']
    if not snapshots:
        return {'trophies': [{'id': trophy['id'], 'q': [0] * len(levels)} for trophy in case['trophies']]}
    last = snapshots[-1]
    target = timestamp(case['target']) / DAY
    checkpoint = timestamp(case['at']) / DAY
    anchor = snapshots[0].time - 0.18
    launch = (len(snapshots) == 1 and last.players < 2500 and
              target - snapshots[0].time < 0.45 and target - checkpoint < 0.14)
    rng = random.Random(49979687)
    paths = list(growth_path(snapshots, anchor, target, rng))
    per_trophy = []
    groups = defaultdict(list)
    for index, trophy in enumerate(case['trophies']):
        history = series_for(snapshots, index)
        if not history:
            per_trophy.append(None)
            continue
        n, bounds = history[-1]
        kind = group(trophy, n.rates[index])
        values = {snap.time: midpoint(interval) for snap, interval in history}
        refreshes = [snap for snap, _ in history]
        if len(refreshes) >= 4 and refreshes[0].players * 3 < refreshes[1].players:
            # The abruptly diluted launch cohort is a poor proxy for subsequent growth.
            refreshes = refreshes[1:]
        estimate = evidence(refreshes, anchor, lambda snap: values[snap.time])
        if estimate[1] > 0:
            groups[kind].append(max(0, estimate[0]))
        last_intervals = [interval for _, interval in history[1:] or history]
        stable_count = (len(last_intervals) >= 2 and
                        max(lo for lo, _ in last_intervals) <= min(hi for _, hi in last_intervals))
        per_trophy.append((n, bounds, kind, estimate, stable_count))

    defaults = {'opening': 0.0, 'completion': 0.07, 'story': 0.70, 'long': 0.24, 'other': 0.41}
    group_centers = {kind: (2 * prior + len(groups[kind]) * median(groups[kind])) / (2 + len(groups[kind]))
                     if groups[kind] else prior for kind, prior in defaults.items()}
    result = []
    for trophy, fit in zip(case['trophies'], per_trophy):
        if fit is None:
            values = [0.0] * SAMPLES
        else:
            read, bounds, kind, (slope, trust, error), stable_count = fit
            trophy_rng = random.Random(15485863 + trophy['id'] * 1009)
            if launch:
                values = launch_quantiles(bounds, read.players, kind,
                                          (checkpoint - read.time) * 24, (target - checkpoint) * 24, trophy_rng)
            else:
                group_slope = group_centers[kind]
                earned = midpoint(bounds)
                size_trust = earned / (earned + 5)
                own_weight = (0.25 + 0.68 * size_trust) * trust
                if stable_count:
                    own_weight = max(own_weight, 0.65 * trust)
                    group_slope *= 1 - 0.45 * trust
                center = clamp(own_weight * slope + (1 - own_weight) * group_slope, 0.0, 2.75)
                sigma = 0.15 + 0.35 * (1 - trust) + min(0.58, error * 0.6)
                values = project(paths, read, bounds, kind, center, sigma, trophy_rng)
        result.append({'id': trophy['id'], 'q': quantiles(values, levels)})
    if all(fit is not None for fit in per_trophy):
        constrain_prerequisites(case['trophies'], result)
    return {'trophies': result}


def additional_earners(mean, rng):
    # Small unlock counts are discrete and right-skewed. The gamma mixture
    # allows extra variation in activity among otherwise similar cohorts.
    overdispersion = 0.12
    if mean < 12:
        rate = rng.gammavariate(1 / overdispersion, overdispersion * mean) if mean > 0 else 0
        time = rng.expovariate(1)
        count = 0
        while time <= rate:
            count += 1
            time += rng.expovariate(1)
        return count
    deviation = math.sqrt(mean + overdispersion * mean * mean)
    return max(0, round(rng.gauss(mean, deviation)))


def project(paths, read, bounds, kind, center, sigma, rng):
    values = []
    for dx, elapsed, players in paths:
        count = rng.randint(*bounds)
        if kind == 'opening':
            missing = read.players - count
            exposure = max(0, players - read.players) / read.players
            # A tracked player who missed the opener can still unlock it later.
            recovery = round(missing * (1 - math.exp(-0.23 * elapsed)))
            new_missing = round((missing + 0.35) * exposure * math.exp(rng.gauss(-0.2, 0.65)))
            future = players - missing + recovery - new_missing
        else:
            slope = clamp(rng.gauss(center, sigma), 0, 3.4)
            trend = (count + 0.35) * math.expm1(slope * dx)
            background = (0.07 if kind == 'completion' else 0.20) * elapsed
            future = count + additional_earners(trend + background, rng)
        values.append(round(clamp(100 * future / players, 0, 100), 2))
    return values


def main():
    source = Path(os.environ.get('MAKEFUN_INPUT_DIR', '/input'))
    output = Path(os.environ.get('MAKEFUN_OUTPUT_DIR', '/output'))
    output.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.glob('*.in')):
        case = json.loads(path.read_text(encoding='utf-8'))
        result = json.dumps(adjudicate(case), separators=(',', ':'), allow_nan=False)
        (output / f'{path.stem}.out').write_text(result + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
