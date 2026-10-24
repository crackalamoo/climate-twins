// A one-line verdict on how a match's climate differs from the query's. Pure functions of the API's
// metric data: callers add place names and display units. (Comparisons with well-known cities near
// the query come from the server: /similar's `ruler_comparison`.)

import type { Match, Monthly, QueryPoint } from "./api.js";

// How big a gap is, from its part of the squared climate difference (`difference_d2`: the weighted,
// standardized units the search ranks by), so a word never outsizes what the search counts.
// Thresholds picked from the top 20 matches of 21 varied places; for scale, the same gap in every
// month costs about:
//            temperature (highs and lows)  dewpoint  precipitation  clearness  wind
//   0.02     0.5 °C                        1.4 °C    21%            0.027      0.6 m/s
//   0.05     0.8 °C                        2.1 °C    36%            0.042      0.9 m/s
//   0.15     1.3 °C                        3.7 °C    70%            0.073      1.5 m/s
//   0.4      2.2 °C                        6 °C      2.4×           0.12       2.5 m/s
// Below NEGLIGIBLE_D2 a gap isn't mentioned (temperature is then "nearly identical"). Three sizes, as
// readers can't tell more apart ("slightly" from "a bit"), and a bare "wetter" must read as more
// than "slightly wetter".
const NEGLIGIBLE_D2 = 0.02;
const CLEAR_D2 = 0.15;
const MUCH_D2 = 0.4;
const SIZE_WORDS: [number, string][] = [
  [MUCH_D2, "much"],
  [CLEAR_D2, ""],
  [NEGLIGIBLE_D2, "slightly"],
];
// A factor under this fraction of the largest one is left out (the bars below show it): terse wins.
const MINOR_FRACTION = 1 / 3;
// A factor's monthly gap is split into orthogonal patterns: an overall shift, a bigger or smaller
// seasonal swing, seasons arriving later or earlier (temperature), and days vs nights (temperature).
// The one explaining most of it is described, and sized by the part it explains. Under MIN_EXPLAINED
// of the gap (month-to-month wobbles, no one pattern), the factor is left to the charts.
const MIN_EXPLAINED = 0.35;
// A seasonal or day/night phrase must explain this many times what the overall shift alone does, or
// the plain "warmer" is said instead.
const RICHER_GAIN = 1.25;
// Summer or the rainy season "comes later" only when its peak shifts by at least this many months.
const MIN_LAG_MONTHS = 0.5;
// The query has a summer and winter when its warmest and coldest months differ by this much (°C,
// daily mean); else its seasons, if any, are wet and dry.
const SEASONAL_RANGE_C = 6;
// ...and wet and dry seasons when its wettest 3 months get this many times its driest 3's rain (each
// plus the precipitation feature's 15 mm, so a desert's 2 mm vs 0 mm isn't a wet season).
const WET_DRY_RATIO = 2;
// Winters are milder or colder when the colder of the two averages below this (°C, daily mean); both
// warmer, they're just warmer or cooler.
const COLD_WINTER_C = 15;
// The smaller part of a two-part pattern ("warmer summers, colder winters") is named only when it's
// at least this fraction of the larger.
const PART_FRACTION = 0.4;

// What a factor's pattern says: a phrase, or how it differs by season, to merge with the other
// factors' ("cooler, drier summers and milder, wetter winters").
type Said = string | SeasonWord[];
type Phrase = (size: string) => Said;

interface SeasonWord {
  season: 0 | 1; // the query's high season, or its low one
  word: string;
  size: string;
}

interface Seasons {
  cycle: number[];
  hi: number[];
  lo: number[];
  names: [string, string];
  temperate: boolean;
}

interface Words {
  key?: string;
  up: string;
  down: string;
  noun: string;
  seasonal?: [string, string, string, string]; // up and down words in summer, then in winter
  summerOnly?: boolean;
}

interface Part {
  v: number;
  up: string;
  down: string;
  when: string;
  season?: 0 | 1;
}

// When in the year a factor's season comes (temperature's summer, rain's rainy season): the
// direction of the query's annual cycle (a unit [a, b]) and how many months later the match's peaks.
interface Timing {
  cycle: number[];
  lag: number;
  season: string;
}

interface Factor {
  d2: number;
  series: number[][];
  timing?: Timing;
  words: Words;
  wind?: boolean;
}

interface Pattern {
  explained: number;
  phrase?: Phrase;
}

type ScoredFactor = Factor & Pattern & { clear: number };

/** "much" + "warmer" -> "much warmer"; "" + "warmer" -> "warmer". */
const sized = (size: string, phrase: string) => (size ? `${size} ${phrase}` : phrase);
const sum = (xs: number[]) => xs.reduce((a, b) => a + b, 0);
const mean = (xs: number[]) => sum(xs) / xs.length;
const dot = (a: number[], b: number[]) => sum(a.map((x, i) => x * b[i]));
const monthsBy = (values: number[], desc: boolean) =>
  values.map((v, i) => [v, i]).sort((a, b) => (desc ? b[0] - a[0] : a[0] - b[0])).slice(0, 3).map(([, i]) => i);
const COS = Array.from({ length: 12 }, (_, i) => Math.cos((2 * Math.PI * i) / 12));
const SIN = Array.from({ length: 12 }, (_, i) => Math.sin((2 * Math.PI * i) / 12));
/** The annual cycle of 12 monthly values as [a, b]: values ≈ mean + a·cos + b·sin. */
const harmonic = (v: number[]): [number, number] => [dot(v, COS) / 6, dot(v, SIN) / 6];
/** When in the year (in months, 0 = mid-January) a cycle [a, b] peaks. */
const peakMonth = ([a, b]: [number, number]) => (Math.atan2(b, a) * 6) / Math.PI;
/** In season-aligned monthly diffs d, the mean over some months. */
const over = (d: number[], months: number[]) => mean(months.map((i) => d[i]));
const unit = (h: number[]) => h.map((x) => x / (Math.hypot(...h) || 1));
/** How many months later (-6..6) a match's annual cycle peaks than the query's. */
const lagMonths = (match: number[], query: number[]) => ((((peakMonth(harmonic(match)) - peakMonth(harmonic(query))) % 12) + 18) % 12) - 6;
const logPpt = (mm: number) => Math.log(Math.max(mm, 0) + 15); // the feature's own log(mm + 15)

/**
 * The query's wet and dry seasons, when its wettest 3 months get WET_DRY_RATIO times its driest 3's
 * rain: the direction of its rain's annual cycle (a unit [a, b]) and the seasons' months.
 */
function wetSeasons(q: Monthly): { cycle: number[]; hi: number[]; lo: number[] } | null {
  const wet = monthsBy(q.ppt, true);
  const dry = monthsBy(q.ppt, false);
  if (mean(wet.map((i) => q.ppt[i])) + 15 < WET_DRY_RATIO * (mean(dry.map((i) => q.ppt[i])) + 15)) return null;
  return { cycle: unit(harmonic(q.ppt.map(logPpt))), hi: wet, lo: dry };
}

/**
 * The query's seasons: the direction of its annual cycle (a unit [a, b]), its high and low seasons'
 * months, and their names. Summer and winter, from its daily mean temperature; or in the tropics, wet
 * and dry seasons from its rain; or null when neither stands out. The match is compared month by month
 * after aligning its seasons (as the charts do), so its months under the query's summer are its own
 * summer, in either hemisphere.
 */
function seasons(q: Monthly): Seasons | null {
  const tmean = q.tmax.map((hi, i) => (hi + q.tmin[i]) / 2);
  if (Math.max(...tmean) - Math.min(...tmean) >= SEASONAL_RANGE_C) {
    return { cycle: unit(harmonic(tmean)), hi: monthsBy(tmean, true), lo: monthsBy(tmean, false), names: ["summer", "winter"], temperate: true };
  }
  const wet = wetSeasons(q);
  return wet && { ...wet, names: ["the wet season", "the dry season"], temperate: false };
}

/**
 * A pattern of two parts (summer then winter, days then nights, in that order) as a phrase.
 * Opposite ways: both, when the smaller is at least PART_FRACTION of the larger ("drier summers,
 * much wetter winters"). The same way: the overall word, and which part most ("warmer, especially
 * at night"). parts: [{v, up, down, when}].
 */
function twoParts(parts: Part[], words: Words, size: string): Said {
  const larger = Math.abs(parts[1].v) > Math.abs(parts[0].v) ? 1 : 0;
  const [a, b] = [parts[larger], parts[1 - larger]];
  const word = (p: Part) => (p.v > 0 ? p.up : p.down);
  const said = (named: Part[], sizeOf: (p: Part) => string): Said => {
    if (a.season !== undefined) return named.map((p) => ({ season: p.season!, word: word(p), size: sizeOf(p) }));
    const sizes = new Set(named.map(sizeOf));
    return sizes.size === 1
      ? sized(sizeOf(a), named.map(word).join(" and "))
      : named.map((p) => sized(sizeOf(p), word(p))).join(" and ");
  };
  if (Math.abs(b.v) < PART_FRACTION * Math.abs(a.v)) return said([a], () => size);
  if (Math.sign(a.v) !== Math.sign(b.v)) {
    // "much" belongs to the larger part only; "slightly" covers both (a bare part would read larger).
    return said(parts, (p) => (size !== "much" || p === a ? size : ""));
  }
  return `${sized(size, a.v > 0 ? words.up : words.down)}, especially ${a.when}`;
}

/**
 * The factors' seasonal words as one phrase, by season, in the factors' order: "cooler summers and
 * milder winters", or with a semicolon when a season has more than one: "cooler, drier summers;
 * milder, wetter winters". One size word for all of them leads the phrase ("slightly
 * cooler summers and milder winters"); else each covers its own words, joined by "and", after the
 * unsized ones: "drier, much cooler summers", "cooler, slightly drier and sunnier summers".
 */
function bySeason(words: SeasonWord[], s: Seasons): string {
  const sizes = new Set(words.map((w) => w.size));
  const shared = sizes.size === 1 ? words[0].size : null;
  const counts = ([0, 1] as const).map((season) => words.filter((w) => w.season === season).length);
  const phrase = ([0, 1] as const).flatMap((season) => {
    const these = words.filter((w) => w.season === season);
    if (!these.length) return [];
    const groups = shared === null ? [...sizes].sort((x, y) => (x ? 1 : 0) - (y ? 1 : 0)) : [shared];
    const adjectives = groups.flatMap((size) => {
      const ws = these.filter((w) => w.size === size).map((w) => w.word);
      if (!ws.length) return [];
      return shared !== null || !size ? [ws.join(", ")] : [sized(size, ws.join(" and "))];
    }).join(", ");
    return [s.temperate ? `${adjectives} ${s.names[season]}s` : `${adjectives} in ${s.names[season]}`];
  }).join(Math.max(...counts) > 1 ? "; " : " and ");
  return sized(shared ?? "", phrase);
}

/**
 * How a factor's monthly diffs differ, as {explained, phrase}: the pattern explaining most of them,
 * and its share (0-1) of their sum of squares, which is its share of the factor's part of the climate
 * difference, as the patterns are orthogonal parts of the diffs. f.series is [d] or, for temperature,
 * [dHighs, dLows]; f.timing is when its season comes, for temperature and rain.
 */
function pattern(f: Factor, s: Seasons | null): Pattern {
  const { series, words, timing } = f;
  const all = series.flat();
  const total = dot(all, all);
  if (!total) return { explained: 0 };
  const k = series.length; // each pattern's sum of squares counts once per series
  const d = k === 1 ? series[0] : series[0].map((x, i) => (x + series[1][i]) / 2);
  const m = mean(all);
  const shift = k * 12 * m * m;
  const isTemp = words.key === "temperature";
  // The seasonal and day/night phrases use the actual means (summer's, the highs'), so they also say
  // any overall shift: they explain it as well as their own pattern.
  const candidates: { ss: number; phrase: Phrase; richer?: boolean }[] = [{ ss: shift, phrase: (size) => sized(size, m > 0 ? words.up : words.down) }];
  if (s && (s.temperate || !isTemp)) { // a tropical place's temperature has no seasons to speak of
    const [a, b] = harmonic(d);
    const [ua, ub] = s.cycle;
    const swing = a * ua + b * ub; // a bigger (+) or smaller seasonal cycle
    const [hi, lo] = s.names;
    const [hiUp, hiDown, loUp, loDown] = words.seasonal ?? [words.up, words.down, words.up, words.down];
    const parts: Part[] = [
      { v: over(d, s.hi), up: hiUp, down: hiDown, when: `in ${hi}`, season: 0 },
      { v: over(d, s.lo), up: loUp, down: loDown, when: `in ${lo}`, season: 1 },
    ];
    // Winter humidity isn't felt: a humidity pattern that's mostly winter's says nothing worth saying.
    const summerOnly = words.summerOnly && s.temperate;
    if (!summerOnly || Math.abs(parts[0].v) >= PART_FRACTION * Math.abs(parts[1].v)) {
      candidates.push({
        ss: shift + k * 6 * swing * swing,
        phrase: (size) => (summerOnly ? [{ season: 0, word: parts[0].v > 0 ? hiUp : hiDown, size }] : twoParts(parts, words, size)),
        richer: true,
      });
    }
  }
  if (timing && Math.abs(timing.lag) >= MIN_LAG_MONTHS) {
    const [a, b] = harmonic(d);
    const [ua, ub] = timing.cycle;
    const shifted = b * ua - a * ub; // the same cycle, shifted later (+) or earlier
    const n = Math.round(Math.abs(timing.lag));
    candidates.push({
      ss: k * 6 * shifted * shifted,
      phrase: () => `${timing.season} comes about ${n < 2 ? "a month" : `${n} months`} ${timing.lag > 0 ? "later" : "earlier"}`,
    });
  }
  if (k === 2) {
    const [highs, lows] = series.map(mean);
    const r = (highs - lows) / 2;
    candidates.push({
      ss: shift + k * 12 * r * r,
      phrase: (size) => twoParts([
        { v: highs, up: "warmer days", down: "cooler days", when: "by day" },
        { v: lows, up: "warmer nights", down: "cooler nights", when: "at night" },
      ], words, size),
      richer: true,
    });
  }
  let best = candidates.reduce((a, b) => (b.ss > a.ss ? b : a));
  if (best.richer && best.ss < RICHER_GAIN * shift) best = candidates[0];
  return { explained: best.ss / total, phrase: best.phrase };
}

const andList = (xs: string[]) => (xs.length > 1 ? `${xs.slice(0, -1).join(", ")} and ${xs.at(-1)}` : xs[0]);
const sizeWord = (d2: number) => SIZE_WORDS.find(([min]) => d2 >= min)?.[1] ?? "slightly";

/**
 * A terse verdict on how the match's climate differs from the query's, e.g. "Nearly identical
 * temperatures. Slightly sunnier." One sentence per factor with a clear pattern, the largest first,
 * sized on the search's own scale, but the factors that differ by season share one ("Cooler, drier
 * summers; milder, wetter winters."). A clear difference with no simple pattern, bigger than any
 * named, is named too: "Rainfall differs through the year." A near-twin, judged on the whole
 * difference, is "Very similar." and its largest difference.
 * Null without `monthly` on both and the match's `difference_d2`.
 */
export function verdict(query: QueryPoint, match: Match): string | null {
  const q = query.monthly;
  const m = match.monthly;
  const d2 = match.difference_d2;
  if (!q || !m || !d2) return null;
  const shift = match.seasons_shifted_6_months ? 6 : 0; // as the charts align them
  const aligned = (p: Monthly, v: keyof Monthly) => q[v].map((_, i) => p[v][p === m ? (i + shift) % 12 : i]);
  const diff = (v: keyof Monthly, f = (x: number) => x) => aligned(m, v).map((x, i) => f(x) - f(q[v][i]));
  const tmean = (p: Monthly) => aligned(p, "tmax").map((hi, i) => (hi + aligned(p, "tmin")[i]) / 2);
  const s = seasons(q);
  const wet = wetSeasons(q);

  const factors = [
    {
      d2: d2.tmax + d2.tmin,
      series: [diff("tmax"), diff("tmin")],
      timing: s?.temperate ? { cycle: s.cycle, lag: lagMonths(tmean(m), tmean(q)), season: "summer" } : undefined,
      words: {
        key: "temperature", up: "warmer", down: "cooler", noun: "temperature",
        seasonal: s?.temperate && Math.min(over(tmean(q), s.lo), over(tmean(m), s.lo)) < COLD_WINTER_C
          ? ["warmer", "cooler", "milder", "colder"]
          : ["warmer", "cooler", "warmer", "cooler"],
      },
    },
    {
      d2: d2.ppt,
      series: [diff("ppt", logPpt)],
      timing: wet ? { cycle: wet.cycle, lag: lagMonths(aligned(m, "ppt").map(logPpt), q.ppt.map(logPpt)), season: "the rainy season" } : undefined,
      words: { up: "wetter", down: "drier", noun: "rainfall" },
    },
    { d2: d2.dewpoint, series: [diff("dewpoint")], words: { up: "more humid", down: "less humid", noun: "humidity", summerOnly: true } },
    { d2: d2.clearness, series: [diff("clearness")], words: { up: "sunnier", down: "cloudier", noun: "sunshine" } },
    { d2: d2.ws, series: [diff("ws")], words: { up: "windier", down: "less windy", noun: "wind" }, wind: true },
  ] as ScoredFactor[];
  const temperature = factors[0];
  // Each factor's clearest pattern, and the part of the climate difference it accounts for.
  for (const f of factors) {
    Object.assign(f, pattern(f, s));
    f.clear = f.explained >= MIN_EXPLAINED ? f.d2 * f.explained : 0;
  }
  const ranked = [...factors].sort((a, b) => b.clear - a.clear);
  let shown = ranked.filter((f) => f.clear >= NEGLIGIBLE_D2 && f.clear >= MINOR_FRACTION * ranked[0].clear);
  // Wind rarely shapes how a climate feels: only a big difference, or the only one, is worth saying.
  shown = shown.filter((f) => !f.wind || f.clear >= MUCH_D2 || shown.length === 1);
  // How alike the climates are is judged on the whole difference, not just what has a pattern to name.
  const total = sum(factors.map((f) => f.d2));
  const largest = factors.reduce((a, b) => (b.d2 > a.d2 ? b : a));
  // A clear difference with no simple pattern is named when it's bigger than any the verdict names,
  // and the charts show it.
  const named = Math.max(0, ...shown.map((f) => f.d2));
  let unpatterned = factors.filter((f) => f.explained < MIN_EXPLAINED && f.d2 > named
    && f.d2 >= (f.wind ? MUCH_D2 : CLEAR_D2));
  if (!shown.length && !unpatterned.length) {
    if (total < NEGLIGIBLE_D2) return "Nearly identical climates.";
    if (total < CLEAR_D2) return "Very similar climates.";
    unpatterned = [largest];
  }
  // A near-twin: say so, and give only its largest difference.
  const near = total < CLEAR_D2;
  if (near) shown = shown.slice(0, 1);
  const said = new Map(shown.map((f) => [f, f.phrase!(sizeWord(f.clear))]));
  // The seasonal words, in the factors' order, said where the largest of them would be.
  const seasonal = factors.flatMap((f) => { const x = said.get(f); return Array.isArray(x) ? x : []; });
  const first = shown.find((f) => Array.isArray(said.get(f)));
  const phrases = shown.flatMap((f) => {
    const x = said.get(f)!;
    if (!Array.isArray(x)) return [x];
    return f === first ? [bySeason(seasonal, s!)] : [];
  });
  const sentences = [
    ...(near ? ["very similar"] : temperature.d2 < NEGLIGIBLE_D2 ? ["nearly identical temperatures"] : []),
    ...phrases,
    ...(unpatterned.length ? [`${andList(unpatterned.map((f) => f.words.noun))} ${unpatterned.length > 1 ? "differ" : "differs"} through the year`] : []),
  ];
  return sentences.map((t) => `${t[0].toUpperCase()}${t.slice(1)}.`).join(" ");
}
