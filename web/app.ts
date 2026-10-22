// Climate Twins UI: wires the search box, controls, map, results list and comparison panel.
// Query state lives in the URL (?lat&lon&exclude&suppress&align); the display units (units.js)
// live only in memory.

import "maplibre-gl/dist/maplibre-gl.css";
import { ClimateMap, matchColors, renderLegend } from "./map.js";
import { escapeHtml, renderComparison, statsHtml } from "./charts.js";
import { fmtDistance, setUnits, units } from "./units.js";
import type { Cell, Legend, LegendStop, Match, Place, Ruler, SearchPlace, SimilarResponse } from "./api.js";

const N_RESULTS = 20;
const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

const ui = {
  search: $<HTMLInputElement>("search"),
  suggestions: $("suggestions"),
  locate: $("locate"),
  exclude: $<HTMLInputElement>("exclude"),
  excludeOut: $("exclude-out"),
  suppress: $<HTMLInputElement>("suppress"),
  suppressOut: $("suppress-out"),
  align: $<HTMLInputElement>("align"),
  adjust: $("adjust"),
  units: $("units"),
  message: $("message"),
  empty: $("empty"),
  querySummary: $("query-summary"),
  results: $("results"),
  compare: $("compare"),
  legend: $("legend"),
};
const resultRows = () => ui.results.children as HTMLCollectionOf<HTMLLIElement>;

let current: SimilarResponse | null = null; // last successful /similar response
let queryPoint: { lat: number; lon: number } | null = null; // {lat, lon} of the last requested query
let selectedRank: number | null = null;
let queryAbort: AbortController | null = null;

const map = new ClimateMap("map", {
  onClick: (lat, lon) => runQuery(lat, lon),
  onMarkerClick: (rank) => select(rank),
  onMarkerHover: (rank) => hover(rank),
  overlays: [ui.legend],
});

// ---------------------------------------------------------------- formatting

function fmtCoord(lat: number, lon: number) {
  const ns = lat >= 0 ? "N" : "S";
  const ew = lon >= 0 ? "E" : "W";
  return `${fmtDegMin(lat)} ${ns}, ${fmtDegMin(lon)} ${ew}`;
}

/** |deg| as degrees and whole minutes, "45°31′": a minute (~2 km) is finer than the ~4 km grid. */
function fmtDegMin(deg: number) {
  const minutes = Math.round(Math.abs(deg) * 60);
  return `${Math.floor(minutes / 60)}°${String(minutes % 60).padStart(2, "0")}′`;
}

/**
 * A Place label (docs/api.md) in the active units, rebuilt from its structured fields. With
 * `in_area` (the point is across a border from the city), an exact-tier city is only "near".
 */
function placeLabel(place: Place | null, cell: Cell) {
  if (!place) return fmtCoord(cell.lat, cell.lon);
  if (place.tier === "region") return `in ${place.base_label}`;
  const where = place.in_area ? `, in ${place.in_area}` : "";
  if (place.tier === "exact") return place.in_area ? `near ${place.base_label}${where}` : place.base_label;
  if (place.tier === "near") return `near ${place.base_label}${where}`;
  return `${fmtDistance(place.km)} ${place.direction} of ${place.base_label}${where}`;
}

/**
 * A place label as two lines, [primary, secondary]: "120 km NW of Boise" / "Idaho, United States".
 * The second line is where the point is (the city's region and country, or `in_area` when the point
 * is across a border from it), or null when there's nothing to add.
 */
function placeLines(place: Place | null, cell: Cell): [string, string | null] {
  if (!place) return [fmtCoord(cell.lat, cell.lon), null];
  const { name, base_label: base, in_area: inArea } = place;
  // base_label is "name, region, country" (or just the name with in_area); keep it whole if it isn't
  const [city, rest] = base.startsWith(`${name}, `) ? [name, base.slice(name.length + 2)] : [base, null];
  const where = inArea ?? rest;
  if (place.tier === "region") return [city, rest];
  if (place.tier === "exact") return [inArea ? `near ${city}` : city, where];
  if (place.tier === "near") return [`near ${city}`, where];
  return [`${fmtDistance(place.km)} ${place.direction} of ${city}`, where];
}

/** A place label's two lines (placeLines) as HTML. */
function placeLinesHtml(place: Place | null, cell: Cell) {
  const [primary, secondary] = placeLines(place, cell);
  return `<span class="primary">${escapeHtml(primary)}</span>${secondary ? `<span class="secondary">${escapeHtml(secondary)}</span>` : ""}`;
}

/**
 * A short name for a place: the city if the point is in or near it; else the point's own region where
 * labels show it, else its own country (from `area`, as the labelling city can be across a border).
 */
function shortPlaceName(place: Place | null) {
  if (!place) return null;
  if (place.tier === "exact" || place.tier === "near") return place.name;
  const { admin1, country } = place.area ?? {};
  return admin1 ?? country ?? null;
}

/** "Portland" -> "Portland's", "Brussels" -> "Brussels'". */
const possessive = (name: string) => (/s$/i.test(name) ? `${name}'` : `${name}'s`);

/** The query's own name, for "Portland's climate", only when it's in or near a city ("97 km NW of
 * Cairns" isn't Cairns); else null, and callers say "your spot". */
function queryName(place: Place | null) {
  return place?.tier === "exact" || place?.tier === "near" ? place.name : null;
}

/** The tab title, as the server's page.py first sets it: "Portland's climate twins" in or near a
 * city, else the place's first line. */
function tabTitle(place: Place | null, cell: Cell) {
  const name = queryName(place);
  return `${name ? `${possessive(name)} climate twins` : placeLines(place, cell)[0]} · Climate Twins`;
}

/**
 * How a match compares with well-known cities near the query (the server's `ruler_comparison`, a
 * city in the response's `ruler`), as {text, title}: "Closer to Portland's climate than Seattle
 * is", with the numbers behind it in `title`. Null when there's nothing to compare with.
 */
function rulerLine(match: Match, ruler: Ruler, queryPlace: Place | null) {
  const cmp = match.ruler_comparison;
  if (!cmp) return null;
  const name = queryName(queryPlace);
  const climate = name ? `${possessive(name)} climate` : "your spot's climate";
  const big = `Big cities: ${ruler.min_population.toLocaleString("en-US")}+ people, ${fmtDistance(ruler.min_km, { step: 10 })}–${fmtDistance(ruler.max_km, { step: 100 })} away.`;
  const own = `Climate difference ${match.distance.toFixed(2)} (0 = identical).`;
  if (cmp.kind === "all") {
    const closest = ruler.cities[0]; // sorted most similar first
    return {
      text: `Closer to ${climate} than any big city within ${fmtDistance(ruler.max_km, { step: 100 })}`,
      title: `${own} The most similar big city, ${closest.label}, is ${closest.distance.toFixed(2)}. ${big}`,
    };
  }
  const { label, distance } = ruler.cities[cmp.city];
  return {
    text: cmp.kind === "closer" ? `Closer to ${climate} than ${label} is` : `About as close to ${climate} as ${label} is`,
    title: `${own} ${label} is ${distance.toFixed(2)}. ${big}`,
  };
}

/** A slider's km value (the input's string), to the nearest 10 mi in US units. */
function fmtSliderKm(km: string) {
  return fmtDistance(Number(km), { step: units() === "us" ? 10 : 1 });
}

// ---------------------------------------------------------------- messages

function showMessage(text: string, kind = "error") {
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

function updateUrl(lat: number, lon: number) {
  const p = params();
  const qs = new URLSearchParams({
    lat: lat.toFixed(4),
    lon: lon.toFixed(4),
    exclude: String(p.exclude),
    suppress: String(p.suppress),
    align: p.align,
  });
  history.replaceState(null, "", `?${qs}`);
}

/**
 * Run a similarity query; only the latest request's response is rendered. Then the map shows the
 * star and all the matches (`fit`): "always" for a new place, "ifHidden" when a slider changes which
 * matches a place has (move only if one is out of view), "never" to leave the camera be.
 */
async function runQuery(lat: number, lon: number, { fit = "always" } = {}) {
  queryAbort?.abort();
  const abort = (queryAbort = new AbortController());
  const p = params();
  const qs = new URLSearchParams({
    lat: lat.toFixed(4),
    lon: lon.toFixed(4),
    n: String(N_RESULTS),
    exclude_km: String(p.exclude),
    suppress_km: String(p.suppress),
    align: p.align,
    detail: "true",
  });
  showMessage("Searching…", "info");
  let res;
  try {
    res = await fetch(`/similar?${qs}`, { signal: abort.signal });
  } catch (err) {
    if ((err as Error).name === "AbortError") return;
    showMessage(`Couldn't reach the server: ${(err as Error).message}`);
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
  let data: SimilarResponse;
  try {
    if (!res.ok) throw new Error((await res.json()).detail ?? res.statusText);
    data = await res.json();
  } catch (err) {
    if (abort.signal.aborted) return;
    showMessage(`Query failed: ${(err as Error).message}`);
    return;
  }
  if (abort.signal.aborted) return;

  await legendReady; // ranks are coloured on the heatmap's scale
  if (abort.signal.aborted) return;

  clearMessage();
  current = data;
  queryPoint = { lat, lon };
  updateUrl(lat, lon);
  const pinned = render(data);
  map.setHeatmap({ lat: lat.toFixed(4), lon: lon.toFixed(4), align: p.align, exclude_km: String(p.exclude) });
  if (fit === "never") return;
  await pinned;
  if (!abort.signal.aborted) map.fitResults({ ifHidden: fit === "ifHidden" });
}

// ---------------------------------------------------------------- rendering

/** Show a /similar response in the panel and on the map; resolves once the pins are on the map. */
function render(data: SimilarResponse) {
  const q = data.query;
  const colors = matchColors(data.matches, heatStops);
  ui.empty.hidden = true;
  ui.legend.hidden = !ui.legend.childElementCount; // with a heatmap to explain (and a legend to show)

  ui.querySummary.innerHTML = `
    <h2 title="${escapeHtml(placeLabel(q.place, q.cell))}"><span class="star">★</span> ${placeLinesHtml(q.place, q.cell)}</h2>
    ${q.place ? `<div class="coords">${fmtCoord(q.cell.lat, q.cell.lon)}</div>` : ""}
    <div class="stats">${statsHtml(q.summary)}</div>
    <h3 class="twins">Its climate twins</h3>`;
  ui.querySummary.hidden = false;
  document.title = tabTitle(q.place, q.cell);

  ui.results.innerHTML = "";
  if (!data.matches.length) {
    ui.results.innerHTML = `<li class="none">No matches with these settings.</li>`;
  }
  for (const m of data.matches) {
    const li = document.createElement("li");
    li.dataset.rank = String(m.rank);
    li.style.setProperty("--rank-color", colors.get(m.rank)!); // the heatmap's colour at its similarity
    const shift = m.seasons_shifted_6_months
      ? ` <span class="shift" title="Seasons shifted 6 months: its climate matches the query's six months later (e.g. the other hemisphere)">⇄</span>`
      : "";
    // The raw climate difference stays in a tooltip: on the comparison line, else on "away".
    const ruler = rulerLine(m, data.ruler, q.place);
    const difference = `Climate difference ${m.distance.toFixed(2)} (0 = identical): the gap between their standardized monthly features.`;
    const away = `<span class="meta"${ruler ? "" : ` title="${escapeHtml(difference)}"`}>${fmtDistance(m.km_from_query)} away</span>`;
    li.innerHTML = `
      <span class="rank">${m.rank}</span>
      <span class="label" title="${escapeHtml(placeLabel(m.place, m.cell))}">${placeLinesHtml(m.place, m.cell)}${away}</span>
      <span class="flags">${shift}</span>
      ${ruler ? `<span class="meta ruler" title="${escapeHtml(ruler.title)}">${escapeHtml(ruler.text)}</span>` : ""}`;
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

function showComparison(match: Match) {
  if (!current) return;
  const labels = {
    query: placeLabel(current.query.place, current.query.cell),
    queryName: shortPlaceName(current.query.place),
    matchName: shortPlaceName(match.place),
    verdictName: queryName(current.query.place) ?? "your spot", // as the results' comparison lines name it
    matchLines: placeLines(match.place, match.cell),
  };
  ui.compare.hidden = false; // before rendering: the charts take the panel's width
  renderComparison(ui.compare, current.query, match, labels);
  comparedWidth = ui.compare.clientWidth;
}

// Redraw the charts when the panel's width changes (a phone rotating, a window resized).
let comparedWidth: number | null = null;
new ResizeObserver(() => {
  const match = current?.matches.find((m) => m.rank === selectedRank);
  if (match && !ui.compare.hidden && ui.compare.clientWidth !== comparedWidth) showComparison(match);
}).observe(ui.compare);

function hover(rank: number | null) {
  map.markMatch("hover", rank);
  for (const li of resultRows()) li.classList.toggle("hover", Number(li.dataset.rank) === rank);
}

function select(rank: number) {
  const match = current?.matches.find((m) => m.rank === rank);
  if (!match) return;
  selectedRank = rank;
  map.markMatch("selected", rank);
  for (const li of resultRows()) {
    const on = Number(li.dataset.rank) === rank;
    li.classList.toggle("selected", on);
    if (on) li.scrollIntoView({ block: "nearest" });
  }
  map.focusMatch(rank);
  showComparison(match);
  ui.compare.scrollIntoView({ block: "nearest", behavior: "smooth" });
}

// ---------------------------------------------------------------- search box

let searchAbort: AbortController | null = null;
let searchTimer: number | undefined;
let suggestions: SearchPlace[] = [];
let active = -1;

function closeSuggestions() {
  ui.suggestions.hidden = true;
  ui.search.setAttribute("aria-expanded", "false");
  suggestions = [];
  active = -1;
}

function showSuggestions(items: SearchPlace[]) {
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

function choose(i: number) {
  const p = suggestions[i];
  if (!p) return;
  ui.search.value = p.label;
  closeSuggestions();
  runQuery(p.lat, p.lon);
}

async function fetchSuggestions(q: string) {
  searchAbort?.abort();
  const abort = (searchAbort = new AbortController());
  try {
    const res = await fetch(`/places/search?${new URLSearchParams({ q, limit: "8" })}`, { signal: abort.signal });
    if (!res.ok) throw new Error(res.statusText);
    const items: SearchPlace[] = await res.json();
    if (!abort.signal.aborted && ui.search.value.trim() === q) showSuggestions(items);
  } catch (err) {
    if ((err as Error).name !== "AbortError") showMessage(`Place search failed: ${(err as Error).message}`);
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

ui.adjust.addEventListener("click", () => {
  ui.adjust.setAttribute("aria-expanded", String(ui.adjust.getAttribute("aria-expanded") !== "true"));
});

for (const [input, out] of [
  [ui.exclude, ui.excludeOut],
  [ui.suppress, ui.suppressOut],
] as const) {
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
  const value = (e.target as Element).closest("button")?.dataset.units;
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
    for (const li of resultRows()) li.classList.toggle("selected", Number(li.dataset.rank) === rank);
    showComparison(match);
  }
});

// ---------------------------------------------------------------- startup

function initFromUrl() {
  const u = new URLSearchParams(location.search);
  const num = (k: string) => (u.has(k) && u.get(k) !== "" && Number.isFinite(Number(u.get(k))) ? Number(u.get(k)) : null);
  const exclude = num("exclude");
  const suppress = num("suppress");
  if (exclude !== null) ui.exclude.value = String(exclude);
  if (suppress !== null) ui.suppress.value = String(suppress);
  showUnits();
  if (u.has("align")) ui.align.checked = u.get("align") !== "calendar";

  const lat = num("lat");
  const lon = num("lon");
  if (lat !== null && lon !== null) runQuery(lat, lon); // eases from the world view to the results
}

// The heatmap's colour stops, which also colour the results' ranks (null: they fall back to one colour).
let heatStops: LegendStop[] | null = null;
const legendReady = fetch("/heatmap/legend")
  .then((res) => (res.ok ? res.json() : Promise.reject(new Error(res.statusText))))
  .then((legend: Legend) => {
    heatStops = legend.stops;
    renderLegend(ui.legend, legend.stops, legend.excluded);
  })
  .catch((err) => console.warn("No heatmap legend:", err.message));

initFromUrl();
