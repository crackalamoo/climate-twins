// Climate Twins UI: wires the search box, controls, map, results list and comparison panel.
// Query state lives in the URL (?lat&lon&exclude&suppress&align); the display units
// (units.js) live only in memory.

import { ClimateMap, matchColors, renderLegend } from "./map.js";
import { escapeHtml, renderComparison, statsHtml } from "./charts.js";
import { fmtDistance, setUnits, units } from "./units.js";

const N_RESULTS = 20;
const $ = (id) => document.getElementById(id);

const ui = {
  search: $("search"),
  suggestions: $("suggestions"),
  locate: $("locate"),
  exclude: $("exclude"),
  excludeOut: $("exclude-out"),
  suppress: $("suppress"),
  suppressOut: $("suppress-out"),
  align: $("align"),
  units: $("units"),
  message: $("message"),
  empty: $("empty"),
  querySummary: $("query-summary"),
  results: $("results"),
  compare: $("compare"),
  legend: $("legend"),
};

let current = null; // last successful /similar response
let queryPoint = null; // {lat, lon} of the last requested query
let selectedRank = null;
let queryAbort = null;

const map = new ClimateMap("map", {
  onClick: (lat, lon) => runQuery(lat, lon),
  onMarkerClick: (rank) => select(rank),
  onMarkerHover: (rank) => hover(rank),
  overlays: [ui.legend],
});

// ---------------------------------------------------------------- formatting

function fmtCoord(lat, lon) {
  const ns = lat >= 0 ? "N" : "S";
  const ew = lon >= 0 ? "E" : "W";
  return `${Math.abs(lat).toFixed(2)}°${ns}, ${Math.abs(lon).toFixed(2)}°${ew}`;
}

/**
 * A Place label (docs/ui-api.md) in the active units, rebuilt from its structured fields. With
 * `in_area` (the point is across a border from the city), `base_label` is just the city and the label
 * ends with where the point is; an exact-tier city is then only "near", as the point isn't in it.
 */
function placeLabel(place, cell) {
  if (!place) return fmtCoord(cell.lat, cell.lon);
  if (!place.tier) return place.label; // no structured fields: only the metric string
  if (place.tier === "region") return `in ${place.base_label}`;
  const where = place.in_area ? `, in ${place.in_area}` : "";
  if (place.tier === "exact") return place.in_area ? `near ${place.base_label}${where}` : place.base_label;
  if (place.tier === "near") return `near ${place.base_label}${where}`;
  return `${fmtDistance(place.km)} ${place.direction} of ${place.base_label}${where}`;
}

/**
 * A short name for a place: the city if the point is in or near it; otherwise the point's own region
 * where labels show it, else the point's own country (`area`: where the point itself is, as the
 * labelling city can be across a border). Labels never name a region the point isn't in, so this
 * never contradicts the label. Null if there is no place.
 */
function shortPlaceName(place) {
  if (!place?.tier) return null;
  if (place.tier === "exact" || place.tier === "near") return place.name;
  const { admin1, country } = place.area ?? {};
  return admin1 ?? country ?? null;
}

/** more_similar_than_pct_of_land (e.g. 99.994) -> "top 0.006%". */
function fmtTop(pct) {
  const top = 100 - pct;
  if (top < 0.001) return "top <0.001%";
  if (top >= 1) return `top ${+top.toFixed(1)}%`;
  return `top ${+top.toPrecision(1)}%`;
}

/** A slider's km value (the input's string), to the nearest 10 mi in US units. */
function fmtSliderKm(km) {
  return fmtDistance(Number(km), { step: units() === "us" ? 10 : 1 });
}

// ---------------------------------------------------------------- messages

function showMessage(text, kind = "error") {
  ui.message.textContent = text;
  ui.message.className = `message ${kind === "info" ? "info" : ""}`;
  ui.message.hidden = false;
}

function clearMessage() {
  ui.message.hidden = true;
}

// ---------------------------------------------------------------- query

function params() {
  return {
    exclude: Number(ui.exclude.value),
    suppress: Number(ui.suppress.value),
    align: ui.align.checked ? "auto" : "calendar",
  };
}

function updateUrl(lat, lon) {
  const p = params();
  const qs = new URLSearchParams({
    lat: lat.toFixed(4),
    lon: lon.toFixed(4),
    exclude: p.exclude,
    suppress: p.suppress,
    align: p.align,
  });
  history.replaceState(null, "", `?${qs}`);
}

/**
 * Run a similarity query; only the latest request's response is rendered. Then the map shows the
 * star and all the matches (`fit`): "always" for a new place, "ifHidden" when a slider changes which
 * matches a place has (move only if one is out of view), "never" to leave the camera be.
 */
async function runQuery(lat, lon, { fit = "always" } = {}) {
  queryAbort?.abort();
  const abort = (queryAbort = new AbortController());
  const p = params();
  const qs = new URLSearchParams({
    lat: lat.toFixed(4),
    lon: lon.toFixed(4),
    n: N_RESULTS,
    exclude_km: p.exclude,
    suppress_km: p.suppress,
    align: p.align,
    detail: "true",
  });
  showMessage("Searching…", "info");
  let res;
  try {
    res = await fetch(`/similar?${qs}`, { signal: abort.signal });
  } catch (err) {
    if (err.name === "AbortError") return;
    showMessage(`Couldn't reach the server: ${err.message}`);
    return;
  }
  if (res.status === 404) {
    if (abort.signal.aborted) return;
    showMessage(
      current
        ? "That point isn't on land (or is in Antarctica, which isn't covered). Showing the previous results."
        : "That point isn't on land (or is in Antarctica, which isn't covered). Try clicking on land.",
    );
    return;
  }
  let data;
  try {
    if (!res.ok) throw new Error((await res.json()).detail ?? res.statusText);
    data = await res.json();
  } catch (err) {
    if (abort.signal.aborted) return;
    showMessage(`Query failed: ${err.message}`);
    return;
  }
  if (abort.signal.aborted) return;

  await legendReady; // pins are coloured on the heatmap's scale
  if (abort.signal.aborted) return;

  clearMessage();
  current = data;
  queryPoint = { lat, lon };
  updateUrl(lat, lon);
  const pinned = render(data);
  map.setHeatmap({ lat: lat.toFixed(4), lon: lon.toFixed(4), align: p.align, exclude_km: p.exclude });
  if (fit === "never") return;
  await pinned;
  if (!abort.signal.aborted) map.fitResults({ ifHidden: fit === "ifHidden" });
}

// ---------------------------------------------------------------- rendering

/** Show a /similar response in the panel and on the map; resolves once the pins are on the map. */
function render(data) {
  const q = data.query;
  const colors = matchColors(data.matches, heatStops);
  ui.empty.hidden = true;
  ui.legend.hidden = !ui.legend.childElementCount; // with a heatmap to explain (and a legend to show)

  ui.querySummary.innerHTML = `
    <h2><span class="star">★</span> ${escapeHtml(placeLabel(q.place, q.cell))}</h2>
    ${q.place ? `<div class="coords">${fmtCoord(q.cell.lat, q.cell.lon)}</div>` : ""}
    <div class="stats">${statsHtml(q.summary)}</div>`;
  ui.querySummary.hidden = false;

  ui.results.innerHTML = "";
  if (!data.matches.length) {
    ui.results.innerHTML = `<li class="none">No matches with these settings.</li>`;
  }
  for (const m of data.matches) {
    const li = document.createElement("li");
    li.dataset.rank = m.rank;
    li.style.setProperty("--rank-color", colors.get(m.rank)); // same colour as its map pin
    const shift = m.seasons_shifted_6_months
      ? ` <span class="shift" title="Seasons shifted 6 months: its climate matches the query's six months later (e.g. the other hemisphere)">⇄</span>`
      : "";
    li.innerHTML = `
      <span class="rank">${m.rank}</span>
      <span class="label" title="${escapeHtml(placeLabel(m.place, m.cell))}">${escapeHtml(placeLabel(m.place, m.cell))}</span>
      <span class="pct" title="More similar than ${m.more_similar_than_pct_of_land}% of land beyond the exclusion radius">${fmtTop(m.more_similar_than_pct_of_land)}${shift}</span>
      <span class="meta">${fmtDistance(m.km_from_query)} away · <span title="How different the two climates are (0 = identical): the gap between their standardized monthly features">climate difference ${m.distance.toFixed(2)}</span></span>`;
    // Mouse only: a tap would leave the row and its pin highlighted, with no "leave" to clear them.
    li.addEventListener("pointerenter", (e) => e.pointerType === "mouse" && hover(m.rank));
    li.addEventListener("pointerleave", (e) => e.pointerType === "mouse" && hover(null));
    li.addEventListener("click", () => select(m.rank));
    ui.results.append(li);
  }

  map.setQuery(q.cell.lat, q.cell.lon);
  const pinned = map.setMatches(data.matches, colors);
  selectedRank = null;
  ui.compare.hidden = true;
  ui.compare.innerHTML = "";
  return pinned;
}

function showComparison(match) {
  const labels = {
    query: placeLabel(current.query.place, current.query.cell),
    match: placeLabel(match.place, match.cell),
    queryName: shortPlaceName(current.query.place),
    matchName: shortPlaceName(match.place),
  };
  ui.compare.hidden = false; // before rendering: the charts take the panel's width
  renderComparison(ui.compare, current.query, match, labels);
  comparedWidth = ui.compare.clientWidth;
}

// Redraw the charts when the panel's width changes (a phone rotating, a window resized).
let comparedWidth = null;
new ResizeObserver(() => {
  const match = current?.matches.find((m) => m.rank === selectedRank);
  if (match && !ui.compare.hidden && ui.compare.clientWidth !== comparedWidth) showComparison(match);
}).observe(ui.compare);

function hover(rank) {
  map.markMatch("hover", rank);
  for (const li of ui.results.children) li.classList.toggle("hover", Number(li.dataset.rank) === rank);
}

function select(rank) {
  const match = current?.matches.find((m) => m.rank === rank);
  if (!match) return;
  selectedRank = rank;
  map.markMatch("selected", rank);
  for (const li of ui.results.children) {
    const on = Number(li.dataset.rank) === rank;
    li.classList.toggle("selected", on);
    if (on) li.scrollIntoView({ block: "nearest" });
  }
  map.focusMatch(rank);
  showComparison(match);
  ui.compare.scrollIntoView({ block: "nearest", behavior: "smooth" });
}

// ---------------------------------------------------------------- search box

let searchAbort = null;
let searchTimer = null;
let suggestions = [];
let active = -1;

function closeSuggestions() {
  ui.suggestions.hidden = true;
  ui.search.setAttribute("aria-expanded", "false");
  suggestions = [];
  active = -1;
}

function showSuggestions(items) {
  suggestions = items;
  active = items.length ? 0 : -1;
  ui.suggestions.innerHTML = "";
  items.forEach((p, i) => {
    const li = document.createElement("li");
    li.setAttribute("role", "option");
    li.innerHTML = `${escapeHtml(p.label)}<span class="pop">${p.population ? p.population.toLocaleString("en-US") : ""}</span>`;
    // mousedown (not click) so it fires before the input's blur closes the list
    li.addEventListener("mousedown", (e) => {
      e.preventDefault();
      choose(i);
    });
    ui.suggestions.append(li);
  });
  if (!items.length) {
    ui.suggestions.innerHTML = `<li class="none" aria-disabled="true">No places found</li>`;
  }
  ui.suggestions.hidden = false;
  ui.search.setAttribute("aria-expanded", "true");
  highlightActive();
}

function highlightActive() {
  [...ui.suggestions.children].forEach((li, i) => li.setAttribute("aria-selected", String(i === active)));
}

function choose(i) {
  const p = suggestions[i];
  if (!p) return;
  ui.search.value = p.label;
  closeSuggestions();
  runQuery(p.lat, p.lon);
}

async function fetchSuggestions(q) {
  searchAbort?.abort();
  const abort = (searchAbort = new AbortController());
  try {
    const res = await fetch(`/places/search?${new URLSearchParams({ q, limit: 8 })}`, { signal: abort.signal });
    if (!res.ok) throw new Error(res.statusText);
    const items = await res.json();
    if (!abort.signal.aborted && ui.search.value.trim() === q) showSuggestions(items);
  } catch (err) {
    if (err.name !== "AbortError") showMessage(`Place search failed: ${err.message}`);
  }
}

ui.search.addEventListener("input", () => {
  clearTimeout(searchTimer);
  const q = ui.search.value.trim();
  if (q.length < 2) {
    searchAbort?.abort();
    closeSuggestions();
    return;
  }
  searchTimer = setTimeout(() => fetchSuggestions(q), 200);
});

ui.search.addEventListener("keydown", (e) => {
  if (ui.suggestions.hidden) return;
  if (e.key === "ArrowDown" || e.key === "ArrowUp") {
    e.preventDefault();
    if (!suggestions.length) return;
    active = (active + (e.key === "ArrowDown" ? 1 : -1) + suggestions.length) % suggestions.length;
    highlightActive();
    ui.suggestions.children[active].scrollIntoView({ block: "nearest" });
  } else if (e.key === "Enter") {
    e.preventDefault();
    choose(active);
  } else if (e.key === "Escape") {
    closeSuggestions();
  }
});

ui.search.addEventListener("blur", closeSuggestions);

// ---------------------------------------------------------------- my location

ui.locate.addEventListener("click", () => {
  if (!navigator.geolocation) {
    showMessage("This browser can't share its location.");
    return;
  }
  showMessage("Finding your location…", "info");
  navigator.geolocation.getCurrentPosition(
    (pos) => runQuery(pos.coords.latitude, pos.coords.longitude),
    (err) => {
      const hint = window.isSecureContext ? "" : " (browsers only allow this on https or localhost)";
      showMessage(`Couldn't get your location: ${err.message}${hint}`);
    },
    { timeout: 15000 },
  );
});

// ---------------------------------------------------------------- controls

for (const [input, out] of [
  [ui.exclude, ui.excludeOut],
  [ui.suppress, ui.suppressOut],
]) {
  input.addEventListener("input", () => (out.textContent = fmtSliderKm(input.value)));
  input.addEventListener("change", () => queryPoint && runQuery(queryPoint.lat, queryPoint.lon, { fit: "ifHidden" }));
}
ui.align.addEventListener("change", () => queryPoint && runQuery(queryPoint.lat, queryPoint.lon, { fit: "never" }));

/** Show the active units on the toggle and slider readouts. */
function showUnits() {
  for (const b of ui.units.querySelectorAll("button")) b.setAttribute("aria-pressed", String(b.dataset.units === units()));
  ui.excludeOut.textContent = fmtSliderKm(ui.exclude.value);
  ui.suppressOut.textContent = fmtSliderKm(ui.suppress.value);
}

// Switching units re-renders the last response (keeping the selection) without refetching.
ui.units.addEventListener("click", (e) => {
  const value = e.target.closest("button")?.dataset.units;
  if (!value || value === units()) return;
  setUnits(value);
  showUnits();
  if (!current) return;
  const rank = selectedRank;
  const scroll = ui.results.scrollTop;
  render(current);
  ui.results.scrollTop = scroll;
  const match = current.matches.find((m) => m.rank === rank);
  if (match) {
    selectedRank = rank;
    map.markMatch("selected", rank);
    for (const li of ui.results.children) li.classList.toggle("selected", Number(li.dataset.rank) === rank);
    showComparison(match);
  }
});

// ---------------------------------------------------------------- startup

function initFromUrl() {
  const u = new URLSearchParams(location.search);
  const num = (k) => (u.has(k) && u.get(k) !== "" && Number.isFinite(Number(u.get(k))) ? Number(u.get(k)) : null);
  const exclude = num("exclude");
  const suppress = num("suppress");
  if (exclude !== null) ui.exclude.value = exclude;
  if (suppress !== null) ui.suppress.value = suppress;
  showUnits();
  if (u.has("align")) ui.align.checked = u.get("align") !== "calendar";

  const lat = num("lat");
  const lon = num("lon");
  if (lat !== null && lon !== null) runQuery(lat, lon); // eases from the world view to the results
}

// The heatmap's colour stops, which also colour the match pins (null: pins fall back to one colour).
let heatStops = null;
const legendReady = fetch("/heatmap/legend")
  .then((res) => (res.ok ? res.json() : Promise.reject(new Error(res.statusText))))
  .then((legend) => {
    heatStops = legend.stops;
    renderLegend(ui.legend, legend.stops, legend.excluded);
  })
  .catch((err) => console.warn("No heatmap legend:", err.message));

initFromUrl();
