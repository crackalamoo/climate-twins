// The comparison panel: query vs one match, drawn with Observable Plot.

import * as Plot from "@observablehq/plot";
import type { Feature, Match, Monthly, QueryPoint, Summary } from "./api.js";
import { DAYS_PER_MONTH, PRECIP_STOPS, type Stop, TEMP_STOPS, gradientStops } from "./ramps.js";
import { fmtAnnualPrecip, fmtPrecip, fmtTemp, fmtWind, precip, precipUnit, temp, tempUnit } from "./units.js";
import { verdict } from "./verdict.js";

// Colour says which place (the query's ink, the match's red). The climate ramps (ramps.js) are a
// strip beside each chart's y axis, like a relief map's elevation key: charts have their own scales,
// and the strip shows at a glance whether a range is cold or hot, dry or wet.
const STRIP_PX = 6;
const STRIP_GAP_PX = 2; // between the strip and the plot area

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const DEFAULT_WIDTH = 368; // when the panel has no layout yet

type Who = "Query" | "Match";
type Names = Record<Who, string>;

const FEATURE_NAMES: Record<string, string> = {
  tmax: "Daytime highs",
  tmin: "Nighttime lows",
  dewpoint: "Humidity (dewpoint)",
  ppt: "Precipitation",
  clearness: "Sunniness (sky clearness)",
  ws: "Wind",
};

// How muggy the muggiest month feels, from its average dewpoint. Our own bands, a judgment call:
// hourly scales like the US National Weather Service's (sticky from 55 °F) run lower, but a monthly
// average smooths muggy spells away. Most people don't notice humidity below about 60 °F. Dewpoints
// come in ~1 °C steps (TerraClimate's 0.1 kPa), so no edge sits where cool-summer coasts land
// (San Francisco, London: 12-13 °C). 60 / 65 / 70 °F = 15.6 / 18.3 / 21.1 °C.
const DEWPOINT_COMFORT: [number, string][] = [
  [21.1, "Oppressive"],
  [18.3, "Muggy"],
  [15.6, "Humid"],
  [-Infinity, "Comfortable"],
];

/** A dewpoint as how it feels, keeping the number: "Humid (62.1 °F)" (HTML: the number doesn't wrap). */
function fmtDewpoint(c: number) {
  return `${DEWPOINT_COMFORT.find(([min]) => c >= min)![1]} <span class="nowrap">(${fmtTemp(c)})</span>`;
}

// [summary key, label, formatter, tooltip explaining the label (optional)].
const SUMMARY_ROWS: [keyof Summary, string, (v: number) => string, string?][] = [
  ["annual_mean_temp_c", "Annual mean temp", fmtTemp],
  ["warmest_month_high_c", "Warmest-month high", fmtTemp],
  ["coldest_month_low_c", "Coldest-month low", fmtTemp],
  ["annual_precip_mm", "Annual precipitation", fmtAnnualPrecip],
  [
    "muggiest_month_dewpoint_c", "Humidity (muggiest month)", fmtDewpoint,
    "The average dewpoint of the most humid month, and how it feels: comfortable below 60 °F " +
      "(16 °C), humid to 65 °F (18 °C), muggy to 70 °F (21 °C), oppressive above.",
  ],
  [
    "mean_clearness", "Sky clearness", (v) => v.toFixed(2),
    "The share of possible sunlight that reaches the ground, averaged over the year: " +
      "about 0.75 under clear skies, 0.3 under overcast.",
  ],
  ["mean_wind_m_s", "Average wind", fmtWind],
];

const rowLabel = (name: string, tip?: string) => (tip ? `<span class="tip" title="${escapeHtml(tip)}">${name}</span>` : name);

/** One-line headline stats, used under the query title. */
export function statsHtml(summary: Summary) {
  return SUMMARY_ROWS.map(([key, name, fmt, tip]) => `<span>${rowLabel(name, tip)} <b>${fmt(summary[key])}</b></span>`).join("");
}

interface Row {
  i: number;
  who: Who;
  month: string;
  tmin: number;
  tmax: number;
  tmin_c: number;
  tmax_c: number;
  ppt: number;
  ppt_mm: number;
}

/**
 * Rows {i, who, tmin, tmax, ppt} with the match rolled by `shift` months to line up with the query.
 * tmin/tmax/ppt are in display units; *_c and ppt_mm keep the metric values for colours and tooltips.
 */
function monthlyRows(query: Monthly, match: Monthly, shift: number) {
  const row = (i: number, who: Who, m: Monthly, j: number): Row => ({
    i, who, month: MONTHS[j],
    tmin: temp(m.tmin[j]), tmax: temp(m.tmax[j]), tmin_c: m.tmin[j], tmax_c: m.tmax[j],
    ppt: precip(m.ppt[j]), ppt_mm: m.ppt[j],
  });
  const rows: Row[] = [];
  for (let i = 0; i < 12; i++) {
    rows.push(row(i, "Query", query, i));
    rows.push(row(i, "Match", match, (i + shift) % 12));
  }
  return rows;
}

const xAxis: Plot.ScaleOptions = {
  domain: [-0.5, 11.5],
  ticks: 12,
  tickFormat: (i: number) => MONTHS[i] ?? "",
  label: null,
};
let stripId = 0;

type Chart = ReturnType<typeof Plot.plot>;
interface LinearScale {
  domain: [number, number];
  range: [number, number];
  apply(v: number): number;
}

/**
 * Add a climate ramp's strip to `chart`, just left of the plot area, in place of the y axis's tick
 * marks: `toMetric` maps the y scale's display units to the ramp's units.
 */
function withStrip(chart: Chart, stops: Stop[], toMetric: (v: number) => number) {
  const id = `strip-${stripId++}`;
  const y = chart.scale("y") as LinearScale;
  const [lo, hi] = y.domain;
  const [top, bottom] = [y.apply(hi), y.apply(lo)];
  const left = (chart.scale("x") as LinearScale).range[0] - STRIP_GAP_PX - STRIP_PX;
  const stopsSvg = gradientStops(stops, toMetric(hi), toMetric(lo))
    .map(([offset, color]) => `<stop offset="${offset}" stop-color="${color}"/>`)
    .join("");
  const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
  g.innerHTML = `<defs><linearGradient id="${id}" gradientUnits="userSpaceOnUse" x1="0" x2="0" y1="${top}" y2="${bottom}">${stopsSvg}</linearGradient></defs>`
    + `<rect x="${left}" y="${top}" width="${STRIP_PX}" height="${bottom - top}" fill="url(#${id})"/>`;
  (chart.tagName === "svg" ? chart : chart.querySelector("svg")!).append(g);
  return chart;
}

// The y axis for a chart with a strip: no tick marks, labels clear of the strip.
const stripAxis: Plot.ScaleOptions = { tickSize: 0, tickPadding: STRIP_GAP_PX + STRIP_PX + 4 };

const toCelsius = (v: number) => (tempUnit() === "°F" ? ((v - 32) * 5) / 9 : v);
const toMmPerDay = (v: number) => (precipUnit() === "in" ? v * 25.4 : v) / DAYS_PER_MONTH;

/** Daily high and low for both places, beside a strip of the temperature ramp. Each place's high is
 * the upper of its two lines, so they share one style. */
function temperatureChart(rows: Row[], color: Plot.ScaleOptions, names: Names, width: number) {
  const line: Plot.LineYOptions = { x: "i", stroke: "who", strokeWidth: 2, curve: "monotone-x" };
  return withStrip(Plot.plot({
    width,
    height: 170,
    marginLeft: 36,
    x: xAxis,
    y: { label: tempUnit(), grid: true, nice: true, ...stripAxis },
    color,
    marks: [
      Plot.lineY(rows, { ...line, y: "tmax" }),
      Plot.lineY(rows, { ...line, y: "tmin" }),
      Plot.tip(rows, Plot.pointerX({
        x: "i",
        y: "tmax",
        stroke: "who",
        title: (d: Row) => `${names[d.who]} (${d.month})\nhigh ${d.tmax.toFixed(1)} ${tempUnit()}\nlow ${d.tmin.toFixed(1)} ${tempUnit()}`,
      })),
    ],
  }), TEMP_STOPS, toCelsius);
}

/** Monthly totals side by side (query left, match right), beside a strip of the dry-to-wet ramp. */
function precipitationChart(rows: Row[], color: Plot.ScaleOptions, names: Names, width: number) {
  const bar = 0.38; // bar width in months; query left of center, match right
  return withStrip(Plot.plot({
    width,
    height: 140,
    marginLeft: 36,
    x: xAxis,
    y: { label: precipUnit(), grid: true, ...stripAxis },
    color,
    marks: [
      Plot.rectY(rows, {
        x1: (d: Row) => d.i + (d.who === "Query" ? -bar : 0.02),
        x2: (d: Row) => d.i + (d.who === "Query" ? -0.02 : bar),
        y: "ppt",
        fill: "who",
        tip: true,
        title: (d: Row) => `${names[d.who]} (${d.month}): ${fmtPrecip(d.ppt_mm)}`,
      }),
      Plot.ruleY([0]),
    ],
  }), PRECIP_STOPS, toMmPerDay);
}

function differenceChart(share: Record<Feature, number>, width: number, { bar, ink }: { bar: string; ink: string }) {
  const rows = Object.entries(share)
    .map(([key, value]) => ({ name: FEATURE_NAMES[key] ?? key, value }))
    .sort((a, b) => b.value - a.value);
  return Plot.plot({
    width,
    height: 22 * rows.length,
    marginTop: 0,
    marginBottom: 0,
    marginLeft: 150,
    marginRight: 40,
    x: { axis: null, domain: [0, 1] },
    y: { label: null, domain: rows.map((d) => d.name), tickSize: 0 },
    marks: [
      Plot.barX(rows, { x: "value", y: "name", fill: bar, insetTop: 4, insetBottom: 4 }),
      Plot.text(rows, { x: "value", y: "name", text: (d: { value: number }) => `${Math.round(d.value * 100)}%`, dx: 4, textAnchor: "start", fill: ink }),
    ],
  });
}

function summaryTable(query: Summary, match: Summary, names: Names) {
  const body = SUMMARY_ROWS.map(
    ([key, name, fmt, tip]) => `<tr><td>${rowLabel(name, tip)}</td><td>${fmt(query[key])}</td><td>${fmt(match[key])}</td></tr>`,
  ).join("");
  return `<table class="summary">
    <thead><tr><th></th><th class="query">★ ${escapeHtml(names.Query)}</th><th class="match">${escapeHtml(names.Match)}</th></tr></thead>
    <tbody>${body}</tbody></table>`;
}

function keyHtml(names: Names) {
  return `<div class="key"><span class="query">★ ${escapeHtml(names.Query)}</span><span class="match">${escapeHtml(names.Match)}</span></div>`;
}

interface Labels {
  query: string;
  queryName: string | null;
  matchName: string | null;
  verdictName: string;
  matchLines: [string, string | null];
}

/**
 * What to call the two places in the key, table and tooltips: their short names (labels.queryName /
 * labels.matchName), or "Query" and "Match" when a name is missing or both are the same.
 */
function seriesNames(labels: Labels): Names {
  const { queryName: q, matchName: m } = labels;
  return q && m && q !== m ? { Query: q, Match: m } : { Query: "Query", Match: "Match" };
}

/**
 * Render the comparison into `el`.
 * labels: query, the query's full place label; {queryName, matchName} short names (may be null);
 * verdictName: what the verdict is "Compared to" (the query's name, or "your spot"); and matchLines:
 * the match's label as [primary, secondary-or-null], for the two-line heading.
 * `el` must be visible: the charts take its width.
 */
export function renderComparison(el: HTMLElement, query: QueryPoint, match: Match, labels: Labels) {
  const shift = match.seasons_shifted_6_months ? 6 : 0;
  const rows = monthlyRows(query.monthly, match.monthly, shift);
  const style = getComputedStyle(el);
  const [queryColor, matchColor, bar, ink] = ["--query", "--match", "--diff-bar", "--text"].map((v) => style.getPropertyValue(v).trim());
  const color: Plot.ScaleOptions = { domain: ["Query", "Match"], range: [queryColor, matchColor] };
  const names = seriesNames(labels);

  const said = verdict(query, match);
  el.innerHTML = `
    ${headingLines(match.rank, labels.matchLines)}
    <p class="note">vs ${escapeHtml(labels.query)}</p>
    ${said ? `<p class="verdict" title="The largest differences, sized by how much each adds to the climate difference.">Compared to ${escapeHtml(labels.verdictName)}: ${escapeHtml(said[0].toLowerCase() + said.slice(1))}</p>` : ""}
    ${keyHtml(names)}
    ${shift ? `<p class="note">⇄ ${escapeHtml(names.Match)} shifted 6 months: its Jul is shown under Jan, and so on.</p>` : ""}
    <h4>Temperature</h4><p class="note">Average daily high and low.</p><figure class="temp"></figure>
    <h4>Precipitation</h4><figure class="ppt"></figure>
    <h4>Summary</h4>${summaryTable(query.summary, match.summary, names)}
    <h4>What makes up the difference</h4><figure class="diff"></figure>
  `;
  const width = Math.floor(el.clientWidth) || DEFAULT_WIDTH;
  el.querySelector("figure.temp")!.append(temperatureChart(rows, color, names, width));
  el.querySelector("figure.ppt")!.append(precipitationChart(rows, color, names, width));
  el.querySelector("figure.diff")!.append(differenceChart(match.difference_share, width, { bar, ink }));
}

/** The match's heading on two lines: its rank and name, then where it is. */
function headingLines(rank: number, [primary, secondary]: Labels["matchLines"]) {
  return `<h3><span class="rank">${rank}</span> <span class="primary">${escapeHtml(primary)}</span>${
    secondary ? `<span class="secondary">${escapeHtml(secondary)}</span>` : ""}</h3>`;
}

export function escapeHtml(s: string) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" } as Record<string, string>)[c]);
}
