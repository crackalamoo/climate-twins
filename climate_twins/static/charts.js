// The comparison panel: query vs one match, drawn with Observable Plot.

import * as Plot from "https://cdn.jsdelivr.net/npm/@observablehq/plot@0.6.17/+esm";
import { fmtAnnualPrecip, fmtPrecip, fmtTemp, precip, precipUnit, temp, tempUnit } from "./units.js";

export const QUERY_COLOR = "#2a78d6";
export const MATCH_COLOR = "#1baf7a";

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const WIDTH = 368;

const FEATURE_NAMES = {
  tmax: "Daytime highs",
  tmin: "Nighttime lows",
  dewpoint: "Humidity (dewpoint)",
  ppt: "Precipitation",
  clearness: "Sunniness (sky clearness)",
  ws: "Wind",
};

const SUMMARY_ROWS = [
  ["annual_mean_temp_c", "Annual mean temp", fmtTemp],
  ["warmest_month_high_c", "Warmest-month high", fmtTemp],
  ["coldest_month_low_c", "Coldest-month low", fmtTemp],
  ["annual_precip_mm", "Annual precipitation", fmtAnnualPrecip],
  ["muggiest_month_dewpoint_c", "Muggiest-month dewpoint", fmtTemp],
  ["mean_clearness", "Sky clearness", (v) => v.toFixed(2)],
];

/** One-line headline stats, used under the query title. */
export function statsHtml(summary) {
  return SUMMARY_ROWS.map(([key, name, fmt]) => `<span>${name} <b>${fmt(summary[key])}</b></span>`).join("");
}

/**
 * Rows {i, who, tmin, tmax, ppt} with the match rolled by `shift` months to line up with the query.
 * tmin/tmax/ppt are in display units; ppt_mm keeps the metric value for tooltips.
 */
function monthlyRows(query, match, shift) {
  const row = (i, who, m, j) => ({
    i, who, month: MONTHS[j], tmin: temp(m.tmin[j]), tmax: temp(m.tmax[j]), ppt: precip(m.ppt[j]), ppt_mm: m.ppt[j],
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
const color = { domain: ["Query", "Match"], range: [QUERY_COLOR, MATCH_COLOR] };

function temperatureChart(rows) {
  return Plot.plot({
    width: WIDTH,
    height: 170,
    marginLeft: 36,
    x: xAxis,
    y: { label: tempUnit(), grid: true },
    color,
    marks: [
      Plot.areaY(rows, { x: "i", y1: "tmin", y2: "tmax", fill: "who", fillOpacity: 0.18, curve: "monotone-x" }),
      Plot.lineY(rows, { x: "i", y: "tmax", stroke: "who", strokeWidth: 2, curve: "monotone-x" }),
      Plot.lineY(rows, { x: "i", y: "tmin", stroke: "who", strokeWidth: 2, strokeDasharray: "4,2", curve: "monotone-x" }),
      Plot.tip(rows, Plot.pointerX({
        x: "i",
        y: "tmax",
        stroke: "who",
        title: (d) => `${d.who} (${d.month})\nhigh ${d.tmax.toFixed(1)} ${tempUnit()}\nlow ${d.tmin.toFixed(1)} ${tempUnit()}`,
      })),
    ],
  });
}

function precipitationChart(rows) {
  const bar = 0.38; // bar width in months; query left of center, match right
  return Plot.plot({
    width: WIDTH,
    height: 140,
    marginLeft: 36,
    x: xAxis,
    y: { label: precipUnit(), grid: true },
    color,
    marks: [
      Plot.rectY(rows, {
        x1: (d) => d.i + (d.who === "Query" ? -bar : 0.02),
        x2: (d) => d.i + (d.who === "Query" ? -0.02 : bar),
        y: "ppt",
        fill: "who",
        tip: true,
        title: (d) => `${d.who} (${d.month}): ${fmtPrecip(d.ppt_mm)}`,
      }),
      Plot.ruleY([0]),
    ],
  });
}

function differenceChart(share) {
  const rows = Object.entries(share)
    .map(([key, value]) => ({ name: FEATURE_NAMES[key] ?? key, value }))
    .sort((a, b) => b.value - a.value);
  return Plot.plot({
    width: WIDTH,
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

function summaryTable(query, match) {
  const body = SUMMARY_ROWS.map(
    ([key, name, fmt]) => `<tr><td>${name}</td><td>${fmt(query[key])}</td><td>${fmt(match[key])}</td></tr>`,
  ).join("");
  return `<table class="summary">
    <thead><tr><th></th><th style="color:${QUERY_COLOR}">Query</th><th style="color:${MATCH_COLOR}">Match #</th></tr></thead>
    <tbody>${body}</tbody></table>`;
}

/**
 * Render the comparison into `el`.
 * query: the /similar `query` object (with monthly); match: one of its `matches` (with monthly).
 */
export function renderComparison(el, query, match, labels) {
  const shift = match.seasons_shifted_6_months ? 6 : 0;
  const rows = monthlyRows(query.monthly, match.monthly, shift);

  el.innerHTML = `
    <h3>#${match.rank} ${escapeHtml(labels.match)}</h3>
    <p class="note">vs ${escapeHtml(labels.query)}</p>
    <div class="key"><span style="--c:${QUERY_COLOR}">Query</span><span style="--c:${MATCH_COLOR}">Match</span></div>
    ${shift ? `<p class="note">⇄ Match shifted 6 months: its Jul is shown under Jan, and so on.</p>` : ""}
    <h4>Temperature</h4><p class="note">Band from average daily low (dashed) to high (solid).</p><figure class="temp"></figure>
    <h4>Precipitation</h4><figure class="ppt"></figure>
    <h4>Summary</h4>${summaryTable(query.summary, match.summary)}
    <h4>What makes up the remaining difference</h4><figure class="diff"></figure>
  `;
  el.querySelector("thead th:last-child").textContent = `Match #${match.rank}`;
  el.querySelector("figure.temp").append(temperatureChart(rows));
  el.querySelector("figure.ppt").append(precipitationChart(rows));
  el.querySelector("figure.diff").append(differenceChart(match.difference_share));
}

export function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}
