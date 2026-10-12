// Map: basemap, similarity heatmap tiles, query star and numbered match pins.

import { Map, Marker, NavigationControl, Popup } from "https://cdn.jsdelivr.net/npm/maplibre-gl@6.12.0/dist/maplibre-gl.mjs";

const STYLE = "https://tiles.openfreemap.org/styles/positron";
const HEATMAP = "heatmap";
const MATCHES = "matches";
const PINS = "match-pins";
const HALO = "match-halo";
const FONT = ["Noto Sans Bold"]; // served by the style's glyph server
const SELECTED_RING = "#0b0b0b";
const LABELS = "match-labels";

/**
 * Pin colours. A match's colour is the heatmap's colour at its similarity (top % of land), so a pin
 * reads on the same scale as the land around it. #1-5 are filled with it; #6 onwards are white
 * discs ringed (and numbered) in it, so the lesser matches recede.
 */
const FILLED_RANKS = 5;
const FALLBACK_INK = "#540d6e"; // the heatmap's strongest stop, if the legend didn't load

/** The heatmap's colour (opaque) at `topPct`, interpolated in log10(top %) like tiles.py. */
function heatColor(topPct, stops) {
  const s = [...stops].sort((a, b) => b.top_pct - a.top_pct); // 5% (faint) -> 0.01% (strong)
  const t = Math.log10(Math.min(Math.max(topPct, s.at(-1).top_pct), s[0].top_pct));
  let i = 0;
  while (i < s.length - 2 && Math.log10(s[i + 1].top_pct) > t) i++;
  const lp = Math.log10(s[i].top_pct), lq = Math.log10(s[i + 1].top_pct);
  const f = (lp - t) / (lp - lq);
  const rgb = [0, 1, 2].map((k) => Math.round(s[i].rgba[k] + f * (s[i + 1].rgba[k] - s[i].rgba[k])));
  return `#${rgb.map((v) => v.toString(16).padStart(2, "0")).join("")}`;
}

function luminance(hex) {
  const c = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255);
  const [r, g, b] = c.map((v) => (v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4));
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

const INK = "#1f1f1d";

/** WCAG contrast ratio between two #rrggbb colours. */
function contrast(a, b) {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (hi + 0.05) / (lo + 0.05);
}

/** A filled pin's number: white where legible (>= 4.5:1). A white pin's number: its ring colour where legible. */
function numberColor(color, filled) {
  const legible = contrast(color, "#ffffff") >= 4.5;
  if (filled) return legible ? "#fff" : INK;
  return legible ? color : INK;
}

/** rank -> identifying colour for this result set (also used by the sidebar's rank numbers). */
export function matchColors(matches, stops) {
  const color = (m) => (stops ? heatColor(100 - m.more_similar_than_pct_of_land, stops) : FALLBACK_INK);
  return new globalThis.Map(matches.map((m) => [m.rank, color(m)]));
}

// A pin's rank (a merged pin is its best match's).
const RANK = ["get", "rank"];
const FILLED = ["<=", RANK, FILLED_RANKS];
const SORT = ["-", 0, RANK]; // best ranks draw on top

// Pins whose centres are closer than this (px) would overlap, so the worse one merges into the better.
const MERGE_PX = 26;

/** Web Mercator position in [0, 1] world units. */
function mercator(lon, lat) {
  const y = Math.log(Math.tan(Math.PI / 4 + (lat * Math.PI) / 360));
  return [(lon + 180) / 360, 0.5 - y / (2 * Math.PI)];
}

/** Screen distance (px) between two matches at `zoom`, taking the shorter way round the world. */
function pixelDistance(a, b, zoom) {
  const dx = Math.abs(a.xy[0] - b.xy[0]);
  return Math.hypot(Math.min(dx, 1 - dx), a.xy[1] - b.xy[1]) * 512 * 2 ** zoom;
}

/**
 * Merge pins that would overlap at `zoom`: best rank first, each match joins the first (best) pin it
 * would overlap, else becomes a pin itself. A merged pin stays at its best match's real location.
 * Returns [{ lead, members }], members including lead, both in rank order.
 */
function mergePins(points, zoom) {
  const pins = [];
  for (const p of points) {
    const pin = pins.find((q) => pixelDistance(q.lead, p, zoom) < MERGE_PX);
    if (pin) pin.members.push(p);
    else pins.push({ lead: p, members: [p] });
  }
  return pins;
}

function pinFeature({ lead, members }) {
  return {
    type: "Feature",
    geometry: { type: "Point", coordinates: [lead.lon, lead.lat] },
    // ",1,,8,,13," so that `in ",8,"` finds a match whether it is a pin's lead or merged into it
    properties: { rank: lead.rank, ranks: members.map((m) => `,${m.rank},`).join(""), merged: members.length > 1 },
  };
}

/** Pin radius, growing mildly with zoom and when `marked`. */
function pinRadius(marked) {
  const r = (base) => ["+", base, ["case", marked, 2, 0]];
  return ["interpolate", ["linear"], ["zoom"], 1, r(10), 6, r(12)];
}

/** A soft glow around a `marked` (hovered or selected) pin. */
function haloRadius() {
  return ["interpolate", ["linear"], ["zoom"], 1, 19, 6, 21];
}

function haloOpacity(marked) {
  return ["case", marked, 0.4, 0];
}

/** Pins: halo (highlight glow), disc and rank number. */
function matchLayers() {
  return [
    {
      id: HALO,
      type: "circle",
      source: MATCHES,
      layout: { "circle-sort-key": SORT },
      paint: { "circle-radius": haloRadius(), "circle-color": FALLBACK_INK, "circle-opacity": haloOpacity(false) },
    },
    {
      id: PINS,
      type: "circle",
      source: MATCHES,
      layout: { "circle-sort-key": SORT },
      paint: {
        "circle-radius": pinRadius(false),
        "circle-color": FALLBACK_INK,
        "circle-stroke-color": "#fff",
        "circle-stroke-width": 2,
      },
    },
    {
      id: LABELS,
      type: "symbol",
      source: MATCHES,
      layout: {
        "text-field": ["to-string", RANK],
        "text-font": FONT,
        "text-size": ["interpolate", ["linear"], ["zoom"], 1, 11, 6, 13],
        "text-allow-overlap": true,
        "text-padding": 9, // keep basemap labels clear of the pin
        "symbol-sort-key": SORT,
      },
      paint: { "text-color": "#fff" },
    },
  ];
}

export class ClimateMap {
  constructor(container, { onClick, onMarkerClick, onMarkerHover }) {
    this.map = new Map({ container, style: STYLE, center: [0, 20], zoom: 1.3, attributionControl: { compact: true } });
    this.map.addControl(new NavigationControl({ showCompass: false }), "top-left");
    this.map.getCanvas().style.cursor = "crosshair";
    this.map.on("click", (e) => this.handleClick(e, onClick));
    this.map.on("mousemove", (e) => this.handleHover(e));
    this.map.on("movestart", () => this.handleHover(null)); // pins slide out from under a still mouse
    this.map.getCanvasContainer().addEventListener("mouseleave", () => this.handleHover(null));
    this.loaded = new Promise((resolve) => this.map.once("load", resolve));
    this.onMarkerClick = onMarkerClick;
    this.onMarkerHover = onMarkerHover;
    this.queryMarker = null;
    this.marks = { hover: null, selected: null }; // rank highlighted by markMatch
    this.hoverKey = null; // pin under the mouse
    this.tip = new Popup({ closeButton: false, closeOnClick: false, offset: 16, className: "merged-tip" });
  }

  /** Replace the heatmap with tiles for a new query (a fresh source so all tiles refetch). */
  async setHeatmap(params) {
    await this.loaded;
    if (this.map.getLayer(HEATMAP)) this.map.removeLayer(HEATMAP);
    if (this.map.getSource(HEATMAP)) this.map.removeSource(HEATMAP);
    const qs = new URLSearchParams(params);
    this.map.addSource(HEATMAP, {
      type: "raster",
      tiles: [`${location.origin}/tiles/{z}/{x}/{y}.png?${qs}`],
      tileSize: 256,
      maxzoom: 6,
    });
    // Below water so that water, borders, roads and labels draw on top.
    const beforeId = this.map.getLayer("water") ? "water" : undefined;
    this.map.addLayer(
      { id: HEATMAP, type: "raster", source: HEATMAP, paint: { "raster-opacity": 0.8, "raster-resampling": "linear" } },
      beforeId,
    );
  }

  setQuery(lat, lon) {
    this.queryMarker?.remove();
    const el = document.createElement("div");
    el.className = "marker-query";
    el.textContent = "★";
    el.title = "Query";
    this.queryMarker = new Marker({ element: el }).setLngLat([lon, lat]).addTo(this.map);
  }

  /** Show the matches as numbered pins, merged into the best-ranked one where they would overlap. */
  async setMatches(matches, colors) {
    await this.loaded;
    this.points = [...matches]
      .sort((a, b) => a.rank - b.rank)
      .map((m) => ({ rank: m.rank, lon: m.cell.lon, lat: m.cell.lat, xy: mercator(m.cell.lon, m.cell.lat) }));
    this.pinsKey = null;
    if (!this.map.getSource(MATCHES)) this.addMatchLayers();
    this.updatePins();
    this.paintColors(colors);
    this.hideTip();
  }

  /** Re-merge the pins for the current zoom; only touches the source when the grouping changes. */
  updatePins() {
    if (!this.points) return;
    this.pins = mergePins(this.points, this.map.getZoom());
    const key = this.pins.map((p) => p.members.map((m) => m.rank).join(",")).join(";");
    if (key === this.pinsKey) return;
    this.pinsKey = key;
    this.map.getSource(MATCHES).setData({ type: "FeatureCollection", features: this.pins.map(pinFeature) });
  }

  /** Colour pins from `colors` (rank -> colour, from matchColors); a merged pin takes its best rank's. */
  paintColors(colors) {
    const byRank = (f) =>
      colors.size ? ["match", RANK, ...[...colors].flatMap(([rank, c]) => [rank, f(rank, c)]), FALLBACK_INK] : FALLBACK_INK;
    this.ink = byRank((rank, c) => c);
    this.map.setPaintProperty(HALO, "circle-color", this.ink);
    this.map.setPaintProperty(PINS, "circle-color", ["case", FILLED, this.ink, "#fff"]);
    this.map.setPaintProperty(LABELS, "text-color", byRank((rank, c) => numberColor(c, rank <= FILLED_RANKS)));
    this.paintMarks();
  }

  addMatchLayers() {
    this.map.addSource(MATCHES, { type: "geojson", data: { type: "FeatureCollection", features: [] } });
    for (const layer of matchLayers()) this.map.addLayer(layer);
    this.map.on("zoom", () => this.updatePins());
    this.paintMarks();
  }

  /** Highlight ("hover" / "selected") the pin holding one match, its own or merged (or none). */
  markMatch(kind, rank) {
    this.marks[kind] = rank;
    this.paintMarks();
  }

  paintMarks() {
    if (!this.map.getLayer(PINS)) return;
    const { hover, selected } = this.marks;
    const has = (rank) => (rank == null ? false : ["in", `,${rank},`, ["get", "ranks"]]);
    const marked = ["any", has(hover), has(selected)];
    this.map.setPaintProperty(PINS, "circle-radius", pinRadius(marked));
    this.map.setPaintProperty(HALO, "circle-opacity", haloOpacity(marked));
    this.map.setPaintProperty(PINS, "circle-stroke-color", ["case", has(selected), SELECTED_RING, FILLED, "#fff", this.ink ?? FALLBACK_INK]);
    this.map.setPaintProperty(PINS, "circle-stroke-width", ["case", has(selected), 2.5, 2]);
  }

  /** The topmost (best-ranked) pin under a screen point, if any. */
  matchAt(point) {
    if (!this.map.getLayer(PINS)) return null;
    const hits = this.map.queryRenderedFeatures(point, { layers: [PINS] });
    return hits.sort((a, b) => a.properties.rank - b.properties.rank)[0] ?? null;
  }

  handleClick(e, onClick) {
    const feature = this.matchAt(e.point);
    if (!feature) onClick(e.lngLat.lat, e.lngLat.lng);
    else this.onMarkerClick(feature.properties.rank); // a merged pin is its best match's
  }

  /**
   * Fly to a match (from the list or a pin), at least to zoom 4 and far enough that it stands alone:
   * no other match within MERGE_PX, so it isn't merged and anything merged into it splits off.
   */
  focusMatch(rank) {
    const point = this.points?.find((p) => p.rank === rank);
    if (!point) return;
    const zoom = this.map.getZoom();
    const nearest = Math.min(...this.points.filter((p) => p !== point).map((p) => pixelDistance(point, p, zoom)));
    const alone = nearest < MERGE_PX ? zoom + Math.log2(MERGE_PX / Math.max(nearest, 1e-6)) + 0.1 : zoom;
    const target = Math.min(Math.max(zoom, 4, alone), this.map.getMaxZoom());
    this.map.flyTo({ center: [point.lon, point.lat], zoom: target, essential: true });
  }

  handleHover(e) {
    const feature = e && this.matchAt(e.point);
    const key = feature ? `${feature.properties.merged ? "c" : "p"}${feature.properties.rank}` : null;
    this.map.getCanvas().style.cursor = feature ? "pointer" : "crosshair";
    if (key === this.hoverKey) return;
    if (this.hoverKey?.startsWith("p")) this.onMarkerHover(null);
    this.hoverKey = key;
    this.hideTip();
    if (!feature) return;
    if (feature.properties.merged) this.showTip(feature);
    else this.onMarkerHover(feature.properties.rank);
  }

  /** A tooltip listing the matches merged into a pin. */
  showTip(feature) {
    const ranks = feature.properties.ranks.split(",").filter(Boolean).map(Number).sort((a, b) => a - b);
    this.tip
      .setLngLat(feature.geometry.coordinates)
      .setHTML(ranks.map((r) => `#${r}`).join(", "))
      .addTo(this.map);
  }

  hideTip() {
    this.tip.remove();
  }

  flyTo(lat, lon, zoom) {
    this.map.flyTo({ center: [lon, lat], zoom: zoom ?? Math.max(this.map.getZoom(), 4), essential: true });
  }

  jumpTo(lat, lon, zoom) {
    this.map.jumpTo({ center: [lon, lat], zoom });
  }
}

/** Render the heatmap legend from /heatmap/legend stops into `el`. */
export function renderLegend(el, stops, excluded) {
  stops = [...stops].sort((a, b) => b.top_pct - a.top_pct); // 5% (faint) -> 0.01% (strong)
  const rgba = (c) => `rgba(${c[0]}, ${c[1]}, ${c[2]}, ${c.length < 4 ? 1 : c[3] > 1 ? (c[3] / 255).toFixed(3) : c[3]})`;
  // Colors interpolate in log(top_pct), so place the stops on a log scale.
  const lo = Math.log(stops[0].top_pct), hi = Math.log(stops.at(-1).top_pct);
  const pos = (i) => (100 * (lo - Math.log(stops[i].top_pct))) / (lo - hi);
  const gradient = stops.map((s, i) => `${rgba(s.rgba)} ${pos(i)}%`).join(", ");
  el.innerHTML = `
    <div class="title">Most similar land (top % of land beyond the exclusion radius)</div>
    <div class="bar" style="background: linear-gradient(to right, ${gradient})"></div>
    <div class="ticks">${stops.map((s, i) => `<span style="left:${pos(i)}%">${s.top_pct}%</span>`).join("")}</div>
    ${excluded ? `<div class="excluded"><span class="swatch" style="background:${rgba(excluded.rgba)}"></span>${excluded.label}</div>` : ""}
  `;
  el.hidden = false;
}
