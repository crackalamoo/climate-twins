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
// Seasons "arrive later" only when the peak shifts by at least this many months.
const MIN_LAG_MONTHS = 0.5;
// The query has a summer and winter when its warmest and coldest months differ by this much (°C,
// daily mean); else its seasons, if any, are wet and dry.
const SEASONAL_RANGE_C = 6;
// ...and wet and dry seasons when its wettest 3 months get this many times its driest 3's rain (each
// plus the precipitation feature's 15 mm, so a desert's 2 mm vs 0 mm isn't a wet season).
const WET_DRY_RATIO = 2;
// The smaller part of a two-part pattern ("warmer summers, colder winters") is named only when it's
// at least this fraction of the larger.
const PART_FRACTION = 0.4;

type Phrase = (size: string) => string;

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
  seasonal?: [string, string, string, string];
  summerOnly?: boolean;
}

interface Part {
  v: number;
  up: string;
  down: string;
  when: string;
}

interface Factor {
  d2: number;
  series: number[][];
  lag?: number;
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

/**
 * The query's seasons: the direction of its annual cycle (a unit [a, b]), its high and low seasons'
 * months, and their names. Summer and winter, from its daily mean temperature; or in the tropics, wet
 * and dry seasons from its rain; or null when neither stands out. The match is compared month by month
 * after aligning its seasons (as the charts do), so its months under the query's summer are its own
 * summer, in either hemisphere.
 */
function seasons(q: Monthly): Seasons | null {
  const tmean = q.tmax.map((hi, i) => (hi + q.tmin[i]) / 2);
  const unit = (h: number[]) => h.map((x) => x / (Math.hypot(...h) || 1));
  if (Math.max(...tmean) - Math.min(...tmean) >= SEASONAL_RANGE_C) {
    return { cycle: unit(harmonic(tmean)), hi: monthsBy(tmean, true), lo: monthsBy(tmean, false), names: ["summer", "winter"], temperate: true };
  }
  const wet = monthsBy(q.ppt, true);
  const dry = monthsBy(q.ppt, false);
  if (mean(wet.map((i) => q.ppt[i])) + 15 >= WET_DRY_RATIO * (mean(dry.map((i) => q.ppt[i])) + 15)) {
    const rain = q.ppt.map((mm) => Math.log(Math.max(mm, 0) + 15));
    return { cycle: unit(harmonic(rain)), hi: wet, lo: dry, names: ["the wet season", "the dry season"], temperate: false };
  }
  return null;
}

/**
 * A pattern of two parts (summer then winter, days then nights, in that order) as a phrase.
 * Opposite ways: both, when the smaller is at least PART_FRACTION of the larger ("drier summers,
 * much wetter winters"). The same way: the overall word, and which part most ("warmer, especially
 * at night"). parts: [{v, up, down, when}].
 */
function twoParts(parts: Part[], words: Words, size: string) {
  const larger = Math.abs(parts[1].v) > Math.abs(parts[0].v) ? 1 : 0;
  const [a, b] = [parts[larger], parts[1 - larger]];
  const word = (p: Part) => (p.v > 0 ? p.up : p.down);
  if (Math.abs(b.v) < PART_FRACTION * Math.abs(a.v)) return sized(size, word(a));
  if (Math.sign(a.v) !== Math.sign(b.v)) {
    // "much" belongs to the larger part only; "slightly" covers both (a bare part would read larger).
    if (size !== "much") return sized(size, parts.map(word).join(", "));
    return parts.map((p, i) => (i === larger ? sized(size, word(p)) : word(p))).join(", ");
  }
  return `${sized(size, a.v > 0 ? words.up : words.down)}, especially ${a.when}`;
}

/**
 * How a factor's monthly diffs differ, as {explained, phrase, sized}: the pattern explaining most of
 * them; its share (0-1) of their sum of squares, which is its share of the factor's part of the
 * climate difference, as the patterns are orthogonal parts of the diffs; and whether the phrase takes
 * a size word. f.series is [d] or, for temperature, [dHighs, dLows]; f.lag is how many months later
 * the match's temperature peaks.
 */
function pattern(f: Factor, s: Seasons | null): Pattern {
  const { series, words, lag = 0 } = f;
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
    const named = (word: string, season: string) => (s.temperate ? `${word} ${season}s` : `${word} in ${season}`);
    const [hiUp, hiDown, loUp, loDown] = words.seasonal ?? [
      named(words.up, hi), named(words.down, hi), named(words.up, lo), named(words.down, lo),
    ];
    const parts = [
      { v: over(d, s.hi), up: hiUp, down: hiDown, when: `in ${hi}` },
      { v: over(d, s.lo), up: loUp, down: loDown, when: `in ${lo}` },
    ];
    // Winter humidity isn't felt: a humidity pattern that's mostly winter's says nothing worth saying.
    const summerOnly = words.summerOnly && s.temperate;
    if (!summerOnly || Math.abs(parts[0].v) >= PART_FRACTION * Math.abs(parts[1].v)) {
      candidates.push({
        ss: shift + k * 6 * swing * swing,
        phrase: (size) => (summerOnly ? sized(size, parts[0].v > 0 ? hiUp : hiDown) : twoParts(parts, words, size)),
        richer: true,
      });
    }
    if (isTemp && Math.abs(lag) >= MIN_LAG_MONTHS) {
      const timing = b * ua - a * ub; // the same cycle, shifted later (+) or earlier
      const n = Math.round(Math.abs(lag));
      candidates.push({
        ss: k * 6 * timing * timing,
        phrase: () => `summer comes about ${n < 2 ? "a month" : `${n} months`} ${lag > 0 ? "later" : "earlier"}`,
      });
    }
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

const sizeWord = (d2: number) => SIZE_WORDS.find(([min]) => d2 >= min)?.[1] ?? "slightly";

/**
 * A terse verdict on how the match's climate differs from the query's, e.g. "Nearly identical
 * temperatures. Slightly sunnier." One sentence per factor with a clear pattern, the largest first,
 * sized on the search's own scale; a near-twin is "Very similar." and its largest difference. Null
 * without `monthly` on both and the match's `difference_d2`, or when the differences have no simple
 * pattern (the charts show them).
 */
export function verdict(query: QueryPoint, match: Match): string | null {
  const q = query.monthly;
  const m = match.monthly;
  const d2 = match.difference_d2;
  if (!q || !m || !d2) return null;
  const shift = match.seasons_shifted_6_months ? 6 : 0; // as the charts align them
  const aligned = (p: Monthly, v: keyof Monthly) => q[v].map((_, i) => p[v][p === m ? (i + shift) % 12 : i]);
  const diff = (v: keyof Monthly, f = (x: number) => x) => aligned(m, v).map((x, i) => f(x) - f(q[v][i]));
  const logPpt = (mm: number) => Math.log(Math.max(mm, 0) + 15); // the feature's own log(mm + 15)
  const tmean = (p: Monthly) => aligned(p, "tmax").map((hi, i) => (hi + aligned(p, "tmin")[i]) / 2);
  const lag = ((((peakMonth(harmonic(tmean(m))) - peakMonth(harmonic(tmean(q)))) % 12) + 18) % 12) - 6; // -6..6
  const s = seasons(q);

  const factors = [
    {
      d2: d2.tmax + d2.tmin,
      series: [diff("tmax"), diff("tmin")],
      lag,
      words: {
        key: "temperature", up: "warmer", down: "cooler", noun: "temperatures",
        seasonal: ["warmer summers", "cooler summers", "milder winters", "colder winters"],
      },
    },
    { d2: d2.dewpoint, series: [diff("dewpoint")], words: { up: "more humid", down: "less humid", noun: "humidity", summerOnly: true } },
    { d2: d2.ppt, series: [diff("ppt", logPpt)], words: { up: "wetter", down: "drier", noun: "rainfall" } },
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
  if (!shown.length) {
    // Differences with no simple pattern are left to the charts; small ones mean the climates match.
    const largest = Math.max(...factors.map((f) => f.d2));
    if (largest < NEGLIGIBLE_D2) return "Nearly identical climates.";
    return largest < CLEAR_D2 ? "Very similar climates." : null;
  }
  const phrase = (f: ScoredFactor) => f.phrase!(sizeWord(f.clear));
  // A near-twin (every difference slight): say so, and give only its largest difference.
  const sentences = shown.every((f) => f.clear < CLEAR_D2)
    ? ["very similar", phrase(shown[0])]
    : [...(temperature.d2 < NEGLIGIBLE_D2 ? ["nearly identical temperatures"] : []), ...shown.map(phrase)];
  return sentences.map((t) => `${t[0].toUpperCase()}${t.slice(1)}.`).join(" ");
}
