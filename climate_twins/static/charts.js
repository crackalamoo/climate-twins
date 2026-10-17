// The comparison panel: query vs one match, drawn with Observable Plot.

import * as Plot from "https://cdn.jsdelivr.net/npm/@observablehq/plot@0.6.17/+esm";
import { DAYS_PER_MONTH, PRECIP_STOPS, TEMP_STOPS, gradientStops } from "./ramps.js";
import { fmtAnnualPrecip, fmtPrecip, fmtTemp, fmtWind, precip, precipUnit, temp, tempUnit } from "./units.js";
import { verdict } from "./verdict.js";

// Colour says which place (the query's blue, the match's pin colour). The climate ramps (ramps.js)
// are a faint wash behind each chart, so the plot area itself reads as cold/hot or dry/wet.
const WASH_OPACITY = 0.22;

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const DEFAULT_WIDTH = 368; // when the panel has no layout yet

const FEATURE_NAMES = {
  tmax: "Daytime highs",
  tmin: "Nighttime lows",
  dewpoint: "Humidity (dewpoint)",
  ppt: "Precipitation",
  clearness: "Sunniness (sky clearness)",
  ws: "Wind",
};

// How muggy the muggiest month feels, from its average dewpoint. Our own bands, a judgment call:
// moment-to-moment scales like the US National Weather Service's (sticky from 55 °F) describe a
// single hour, and a monthly average smooths the muggy spells away, so ours start higher. Most
// people don't notice humidity below a dewpoint of about 60 °F. TerraClimate's vapor pressure
// comes in 0.1 kPa steps (~1 °C of dewpoint here), so a band edge shouldn't sit where typical
// cool-summer coasts land (San Francisco, London: 12-13 °C).
// In °C: 60 °F = 15.6 °C, 65 °F = 18.3 °C, 70 °F = 21.1 °C.
const DEWPOINT_COMFORT = [
  [21.1, "Oppressive"],
  [18.3, "Muggy"],
  [15.6, "Humid"],
  [-Infinity, "Comfortable"],
];

/** A dewpoint as how it feels, keeping the number: "Humid (62.1 °F)" (HTML: the number doesn't wrap). */
function fmtDewpoint(c) {
  return `${DEWPOINT_COMFORT.find(([min]) => c >= min)[1]} <span class="nowrap">(${fmtTemp(c)})</span>`;
}

// [summary key, label, formatter, tooltip explaining the label (optional)].
const SUMMARY_ROWS = [
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

const rowLabel = (name, tip) => (tip ? `<span class="tip" title="${escapeHtml(tip)}">${name}</span>` : name);

/** One-line headline stats, used under the query title. */
export function statsHtml(summary) {
  return SUMMARY_ROWS.map(([key, name, fmt, tip]) => `<span>${rowLabel(name, tip)} <b>${fmt(summary[key])}</b></span>`).join("");
}

/**
 * Rows {i, who, tmin, tmax, ppt} with the match rolled by `shift` months to line up with the query.
 * tmin/tmax/ppt are in display units; *_c and ppt_mm keep the metric values for colours and tooltips.
 */
function monthlyRows(query, match, shift) {
  const row = (i, who, m, j) => ({
    i, who, month: MONTHS[j],
    tmin: temp(m.tmin[j]), tmax: temp(m.tmax[j]), tmin_c: m.tmin[j], tmax_c: m.tmax[j],
    ppt: precip(m.ppt[j]), ppt_mm: m.ppt[j],
  });
  const rows = [];
  for (let i = 0; i < 12; i++) {
    rows.push(row(i, "Query", query, i));
    rows.push(row(i, "Match", match, (i + shift) % 12));
  }
  return rows;
}

const xAxis = {
  domain: [-0.5, 11.5],
  ticks: 12,
  tickFormat: (i) => MONTHS[i] ?? "",
  label: null,
};
let washId = 0;

/**
 * A faint vertical wash of a climate ramp behind the plot area: `toMetric` maps the y scale's display
 * units to the ramp's units. Returns the frame mark to put first and a function to add its gradient.
 */
function wash(stops, toMetric) {
  const id = `wash-${washId++}`;
  const mark = Plot.frame({ fill: `url(#${id})`, fillOpacity: WASH_OPACITY });
  const addTo = (chart) => {
    const y = chart.scale("y");
    const [lo, hi] = y.domain;
    const stopsSvg = gradientStops(stops, toMetric(hi), toMetric(lo))
      .map(([offset, color]) => `<stop offset="${offset}" stop-color="${color}"/>`)
      .join("");
    const defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
    defs.innerHTML = `<linearGradient id="${id}" gradientUnits="userSpaceOnUse" x1="0" x2="0" y1="${y.apply(hi)}" y2="${y.apply(lo)}">${stopsSvg}</linearGradient>`;
    (chart.tagName === "svg" ? chart : chart.querySelector("svg")).prepend(defs);
    return chart;
  };
  return { mark, addTo };
}

const toCelsius = (v) => (tempUnit() === "°F" ? ((v - 32) * 5) / 9 : v);
const toMmPerDay = (v) => (precipUnit() === "in" ? v * 25.4 : v) / DAYS_PER_MONTH;

/** Daily high and low for both places, over a wash of the temperature ramp. Each place's high is
 * the upper of its two lines, so they share one style. */
function temperatureChart(rows, color, names, width) {
  const bg = wash(TEMP_STOPS, toCelsius);
  const line = { x: "i", stroke: "who", strokeWidth: 2, curve: "monotone-x" };
  return bg.addTo(Plot.plot({
    width,
    height: 170,
    marginLeft: 36,
    x: xAxis,
    y: { label: tempUnit(), grid: true, nice: true },
    color,
    marks: [
      bg.mark,
      Plot.lineY(rows, { ...line, y: "tmax" }),
      Plot.lineY(rows, { ...line, y: "tmin" }),
      Plot.tip(rows, Plot.pointerX({
        x: "i",
        y: "tmax",
        stroke: "who",
        title: (d) => `${names[d.who]} (${d.month})\nhigh ${d.tmax.toFixed(1)} ${tempUnit()}\nlow ${d.tmin.toFixed(1)} ${tempUnit()}`,
      })),
    ],
  }));
}

/** Monthly totals side by side (query left, match right), over a wash of the dry-to-wet ramp. */
function precipitationChart(rows, color, names, width) {
  const bg = wash(PRECIP_STOPS, toMmPerDay);
  const bar = 0.38; // bar width in months; query left of center, match right
  return bg.addTo(Plot.plot({
    width,
    height: 140,
    marginLeft: 36,
    x: xAxis,
    y: { label: precipUnit(), grid: true },
    color,
    marks: [
      bg.mark,
      Plot.rectY(rows, {
        x1: (d) => d.i + (d.who === "Query" ? -bar : 0.02),
        x2: (d) => d.i + (d.who === "Query" ? -0.02 : bar),
        y: "ppt",
        fill: "who",
        tip: true,
        title: (d) => `${names[d.who]} (${d.month}): ${fmtPrecip(d.ppt_mm)}`,
      }),
      Plot.ruleY([0]),
    ],
  }));
}

function differenceChart(share, width) {
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
      Plot.barX(rows, { x: "value", y: "name", fill: "#8f8d85", insetTop: 4, insetBottom: 4 }),
      Plot.text(rows, { x: "value", y: "name", text: (d) => `${Math.round(d.value * 100)}%`, dx: 4, textAnchor: "start", fill: "#1f1f1d" }),
    ],
  });
}

function summaryTable(query, match, names) {
  const body = SUMMARY_ROWS.map(
    ([key, name, fmt, tip]) => `<tr><td>${rowLabel(name, tip)}</td><td>${fmt(query[key])}</td><td>${fmt(match[key])}</td></tr>`,
  ).join("");
  return `<table class="summary">
    <thead><tr><th></th><th class="query">★ ${escapeHtml(names.Query)}</th><th class="match">${escapeHtml(names.Match)}</th></tr></thead>
    <tbody>${body}</tbody></table>`;
}

function keyHtml(names) {
  return `<div class="key"><span class="query">★ ${escapeHtml(names.Query)}</span><span class="match">${escapeHtml(names.Match)}</span></div>`;
}

/**
 * What to call the two places in the key, table and tooltips: their short names (labels.queryName /
 * labels.matchName), or "Query" and "Match" when a name is missing or both are the same.
 */
function seriesNames(labels) {
  const { queryName: q, matchName: m } = labels;
  return q && m && q !== m ? { Query: q, Match: m } : { Query: "Query", Match: "Match" };
}

/**
 * Render the comparison into `el`.
 * query: the /similar `query` object (with monthly); match: one of its `matches` (with monthly).
 * labels: {query, match} full place labels, {queryName, matchName} short names (may be null), and
 * verdictName: what the verdict is "Compared to" (the query's name, or "your spot").
 * `el` must be visible: the charts take its width.
 */
export function renderComparison(el, query, match, labels) {
  const shift = match.seasons_shifted_6_months ? 6 : 0;
  const rows = monthlyRows(query.monthly, match.monthly, shift);
  const style = getComputedStyle(el);
  const [queryColor, matchColor] = ["--query", "--match"].map((v) => style.getPropertyValue(v).trim());
  const color = { domain: ["Query", "Match"], range: [queryColor, matchColor] };
  const names = seriesNames(labels);

  const said = verdict(query, match);
  el.innerHTML = `
    <h3>#${match.rank} ${escapeHtml(labels.match)}</h3>
    <p class="note">vs ${escapeHtml(labels.query)}</p>
    ${said ? `<p class="verdict" title="The largest differences, sized by how much each adds to the climate difference.">Compared to ${escapeHtml(labels.verdictName)}: ${escapeHtml(said[0].toLowerCase() + said.slice(1))}</p>` : ""}
    ${keyHtml(names)}
    ${shift ? `<p class="note">⇄ ${escapeHtml(names.Match)} shifted 6 months: its Jul is shown under Jan, and so on.</p>` : ""}
    <h4>Temperature</h4><p class="note">Average daily high and low.</p><figure class="temp"></figure>
    <h4>Precipitation</h4><figure class="ppt"></figure>
    <h4>Summary</h4>${summaryTable(query.summary, match.summary, names)}
    <h4>What makes up the difference</h4><figure class="diff"></figure>
  `;
  // Charts fill the panel's width (a phone's, or the desktop sidebar's); `el` must be laid out.
  const width = Math.floor(el.clientWidth) || DEFAULT_WIDTH;
  el.querySelector("figure.temp").append(temperatureChart(rows, color, names, width));
  el.querySelector("figure.ppt").append(precipitationChart(rows, color, names, width));
  el.querySelector("figure.diff").append(differenceChart(match.difference_share, width));
}

export function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}
