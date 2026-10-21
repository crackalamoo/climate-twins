// Map: basemap, similarity heatmap tiles, query star and numbered match pins.

import type { ExpressionSpecification, LayerSpecification } from "@maplibre/maplibre-gl-style-spec";
import {
  type GeoJSONSource, Map, type MapGeoJSONFeature, type MapMouseEvent, Marker, NavigationControl, type PointLike, Popup,
  setWorkerUrl,
} from "maplibre-gl";
import type { Legend, LegendStop, Match } from "./api.js";

// MapLibre looks for its worker beside its own module; the build bundles it beside app.js instead.
setWorkerUrl(new URL("maplibre-gl-worker.js", import.meta.url).href);

// A printed-atlas basemap, set in Libre Caslon, on OpenFreeMap's vector tiles.
const STYLE = "atlas/style.json";
const HEATMAP = "heatmap";
const MATCHES = "matches";
const PINS = "match-pins";
const HALO = "match-halo";
const FONT = ["Libre Caslon Text Bold"]; // from the style's glyphs
const LABELS = "match-labels";

// A match's colour is the heatmap's at its similarity (top % of land), so its rank in the result list
// reads on the same scale as the map.
const FALLBACK_INK = "#540d6e"; // the heatmap's strongest stop, if the legend didn't load

/** The heatmap's colour (opaque) at `topPct`, interpolated in log10(top %) like tiles.py. */
function heatColor(topPct: number, stops: LegendStop[]) {
  const s = [...stops].sort((a, b) => b.top_pct - a.top_pct); // 5% (faint) -> 0.01% (strong)
  const t = Math.log10(Math.min(Math.max(topPct, s.at(-1)!.top_pct), s[0].top_pct));
  let i = 0;
  while (i < s.length - 2 && Math.log10(s[i + 1].top_pct) > t) i++;
  const lp = Math.log10(s[i].top_pct), lq = Math.log10(s[i + 1].top_pct);
  const f = (lp - t) / (lp - lq);
  const rgb = [0, 1, 2].map((k) => Math.round(s[i].rgba[k] + f * (s[i + 1].rgba[k] - s[i].rgba[k])));
  return `#${rgb.map((v) => v.toString(16).padStart(2, "0")).join("")}`;
}

/** rank -> identifying colour for this result set (pins and the sidebar's rank numbers). */
export function matchColors(matches: Match[], stops: LegendStop[] | null) {
  const color = (m: Match) => (stops ? heatColor(100 - m.more_similar_than_pct_of_land, stops) : FALLBACK_INK);
  return new globalThis.Map<number, string>(matches.map((m) => [m.rank, color(m)]));
}

// #1-5 are discs of their rank colour ringed in paper; #6 onwards paper discs ringed in it, so the
// lesser matches recede.
const FILLED_RANKS = 5;
const INK = "#2a2420";
const PAPER = "#f7f1e3";

function luminance(hex: string) {
  const c = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255);
  const [r, g, b] = c.map((v) => (v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4));
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

/** WCAG contrast ratio between two #rrggbb colours. */
function contrast(a: string, b: string) {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (hi + 0.05) / (lo + 0.05);
}

/** A filled pin's number: paper where legible (>= 4.5:1). A paper pin's number: its ring colour where legible. Else ink. */
function numberColor(color: string, filled: boolean) {
  const legible = contrast(color, PAPER) >= 4.5;
  if (filled) return legible ? PAPER : INK;
  return legible ? color : INK;
}

// A pin's rank (a merged pin is its best match's).
const RANK: ExpressionSpecification = ["get", "rank"];
const FILLED: ExpressionSpecification = ["<=", RANK, FILLED_RANKS];
const SORT: ExpressionSpecification = ["-", 0, RANK]; // best ranks draw on top

// Pins whose centres are closer than this (px) would overlap, so the worse one merges into the better.
const MERGE_PX = 26;
// Camera moves to a selected match: a brief pan, or for a long way (more than FAR_SCREENS screen
// widths at the current zoom) a quick fly out and back in, rather than a smear across the map.
const FOCUS_MS = 500;
const FAR_SCREENS = 2;
const FAR_MS = 1000; // a fixed length: flyTo's speed-based ones run 2 s+ on a phone
// Showing a new query with its matches: an ease of the same fixed length, no closer than the data's
// native resolution (~4.6 km cells), and no farther out than one world across (beyond that the map
// only repeats it).
const FIT_MS = 1000;
const FIT_MAX_ZOOM = 5;

interface Point {
  rank: number;
  color: string;
  lon: number;
  lat: number;
  xy: [number, number];
}

interface Pin {
  lead: Point;
  members: Point[];
}

type Padding = Record<"top" | "bottom" | "left" | "right", number>;

interface Options {
  onClick: (lat: number, lon: number) => void;
  onMarkerClick: (rank: number) => void;
  onMarkerHover: (rank: number | null) => void;
  overlays?: HTMLElement[];
}

/** Web Mercator position in [0, 1] world units. */
function mercator(lon: number, lat: number): [number, number] {
  const y = Math.log(Math.tan(Math.PI / 4 + (lat * Math.PI) / 360));
  return [(lon + 180) / 360, 0.5 - y / (2 * Math.PI)];
}

/** Screen distance (px) between two matches at `zoom`, taking the shorter way round the world. */
function pixelDistance(a: Pick<Point, "xy">, b: Pick<Point, "xy">, zoom: number) {
  const dx = Math.abs(a.xy[0] - b.xy[0]);
  return Math.hypot(Math.min(dx, 1 - dx), a.xy[1] - b.xy[1]) * 512 * 2 ** zoom;
}

/**
 * [west, east] of the narrowest span of longitude holding all `lons`: everything but the widest gap
 * between neighbours, which may be the one across the antimeridian or not (then east > 180).
 */
function lonSpan(lons: number[]) {
  const s = [...lons].sort((a, b) => a - b);
  let [west, east, gap] = [s[0], s.at(-1)!, s[0] + 360 - s.at(-1)!];
  for (let i = 1; i < s.length; i++) {
    if (s[i] - s[i - 1] > gap) [west, east, gap] = [s[i], s[i - 1] + 360, s[i] - s[i - 1]];
  }
  return [west, east];
}

/**
 * Merge pins that would overlap at `zoom`: best rank first, each match joins the first (best) pin it
 * would overlap, else becomes a pin itself. A merged pin stays at its best match's real location.
 * Returns [{ lead, members }], members including lead, both in rank order.
 */
function mergePins(points: Point[], zoom: number) {
  const pins: Pin[] = [];
  for (const p of points) {
    const pin = pins.find((q) => pixelDistance(q.lead, p, zoom) < MERGE_PX);
    if (pin) pin.members.push(p);
    else pins.push({ lead: p, members: [p] });
  }
  return pins;
}

function pinFeature({ lead, members }: Pin): GeoJSON.Feature<GeoJSON.Point> {
  return {
    type: "Feature",
    geometry: { type: "Point", coordinates: [lead.lon, lead.lat] },
    // ",1,,8,,13," so that `in ",8,"` finds a match whether it is a pin's lead or merged into it
    properties: {
      rank: lead.rank,
      ranks: members.map((m) => `,${m.rank},`).join(""),
      merged: members.length > 1,
      color: lead.color,
      number: numberColor(lead.color, lead.rank <= FILLED_RANKS),
    },
  };
}

/** Pin radius, growing mildly with zoom and when `marked`. */
function pinRadius(marked: ExpressionSpecification | boolean): ExpressionSpecification {
  const r = (base: number): ExpressionSpecification => ["+", base, ["case", marked, 2, 0]];
  return ["interpolate", ["linear"], ["zoom"], 1, r(10), 6, r(12)];
}

/** A paper ring around a `marked` (hovered or selected) pin, edged in ink (more firmly when selected). */
function haloRadius(): ExpressionSpecification {
  return ["interpolate", ["linear"], ["zoom"], 1, 16, 6, 18];
}

function haloOpacity(marked: ExpressionSpecification | boolean): ExpressionSpecification {
  return ["case", marked, 0.92, 0];
}

/** Pins: halo (highlight ring), disc and rank number. */
function matchLayers(): LayerSpecification[] {
  return [
    {
      id: HALO,
      type: "circle",
      source: MATCHES,
      layout: { "circle-sort-key": SORT },
      paint: {
        "circle-radius": haloRadius(),
        "circle-color": PAPER,
        "circle-opacity": haloOpacity(false),
        "circle-stroke-color": INK,
        "circle-stroke-opacity": 0,
      },
    },
    {
      id: PINS,
      type: "circle",
      source: MATCHES,
      layout: { "circle-sort-key": SORT },
      paint: {
        "circle-radius": pinRadius(false),
        "circle-color": ["case", FILLED, ["get", "color"], PAPER],
        "circle-stroke-color": ["case", FILLED, PAPER, ["get", "color"]],
        "circle-stroke-width": 1.5,
      },
    },
    {
      id: LABELS,
      type: "symbol",
      source: MATCHES,
      layout: {
        "text-field": ["to-string", RANK],
        "text-font": FONT,
        "text-size": ["interpolate", ["linear"], ["zoom"], 1, 12, 6, 14],
        "text-allow-overlap": true,
        "text-padding": 9, // keep basemap labels clear of the pin
        "symbol-sort-key": SORT,
      },
      paint: { "text-color": ["get", "number"] },
    },
  ];
}

export class ClimateMap {
  map: Map;
  loaded: Promise<unknown>;
  onMarkerClick: Options["onMarkerClick"];
  onMarkerHover: Options["onMarkerHover"];
  queryMarker: Marker | null;
  queryPoint: { lat: number; lon: number } | null;
  overlays: HTMLElement[];
  marks: Record<"hover" | "selected", number | null>;
  hoverKey: string | null;
  tip: Popup;
  points?: Point[];
  pins?: Pin[];
  pinsKey?: string | null;

  /** `overlays`: elements of the page laid over the map (the legend), which fitResults keeps pins clear of. */
  constructor(container: string, { onClick, onMarkerClick, onMarkerHover, overlays = [] }: Options) {
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
    this.queryPoint = null; // {lat, lon} of the star
    this.overlays = overlays;
    this.marks = { hover: null, selected: null }; // rank highlighted by markMatch
    this.hoverKey = null; // pin under the mouse
    this.tip = new Popup({ closeButton: false, closeOnClick: false, offset: 16, className: "merged-tip" });
    this.map.setMissingStyleImageResolver((id) => addAtlasIcon(this.map, id));
  }

  /** Replace the heatmap with tiles for a new query (a fresh source so all tiles refetch). */
  async setHeatmap(params: Record<string, string>) {
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

  setQuery(lat: number, lon: number) {
    this.queryMarker?.remove();
    const el = document.createElement("div");
    el.className = "marker-query";
    el.textContent = "★";
    el.title = "Query";
    this.queryMarker = new Marker({ element: el }).setLngLat([lon, lat]).addTo(this.map);
    this.queryPoint = { lat, lon };
  }

  /**
   * Show the matches as numbered pins in their `colors` (rank -> #rrggbb, from matchColors), merged
   * into the best-ranked one where they would overlap.
   */
  async setMatches(matches: Match[], colors: ReadonlyMap<number, string>) {
    // New matches: the old highlights name ranks of the old ones. Cleared before the await, so a
    // caller can mark one of the new matches straight after (the units toggle keeps its selection).
    this.marks = { hover: null, selected: null };
    await this.loaded;
    this.points = [...matches]
      .sort((a, b) => a.rank - b.rank)
      .map((m) => ({
        rank: m.rank,
        color: colors.get(m.rank) ?? FALLBACK_INK,
        lon: m.cell.lon,
        lat: m.cell.lat,
        xy: mercator(m.cell.lon, m.cell.lat),
      }));
    this.pinsKey = null;
    if (!this.map.getSource(MATCHES)) this.addMatchLayers();
    this.updatePins();
    this.paintMarks();
    this.hideTip();
  }

  /** Re-merge the pins for the current zoom; only touches the source when the grouping changes. */
  updatePins() {
    if (!this.points) return;
    this.pins = mergePins(this.points, this.map.getZoom());
    const key = this.pins.map((p) => p.members.map((m) => m.rank).join(",")).join(";");
    if (key === this.pinsKey) return;
    this.pinsKey = key;
    this.map.getSource<GeoJSONSource>(MATCHES)!.setData({ type: "FeatureCollection", features: this.pins.map(pinFeature) });
  }

  addMatchLayers() {
    this.map.addSource(MATCHES, { type: "geojson", data: { type: "FeatureCollection", features: [] } });
    for (const layer of matchLayers()) this.map.addLayer(layer);
    this.map.on("zoom", () => this.updatePins());
    this.paintMarks();
  }

  /** Highlight ("hover" / "selected") the pin holding one match, its own or merged (or none). */
  markMatch(kind: "hover" | "selected", rank: number | null) {
    this.marks[kind] = rank;
    this.paintMarks();
  }

  paintMarks() {
    if (!this.map.getLayer(PINS)) return;
    const { hover, selected } = this.marks;
    const has = (rank: number | null): ExpressionSpecification | false => (rank == null ? false : ["in", `,${rank},`, ["get", "ranks"]]);
    const marked: ExpressionSpecification = ["any", has(hover), has(selected)];
    this.map.setPaintProperty(PINS, "circle-radius", pinRadius(marked));
    this.map.setPaintProperty(HALO, "circle-opacity", haloOpacity(marked));
    this.map.setPaintProperty(HALO, "circle-stroke-opacity", ["case", marked, 1, 0]);
    this.map.setPaintProperty(HALO, "circle-stroke-width", ["case", has(selected), 1.5, 0.75]);
  }

  /** The topmost (best-ranked) pin under a screen point, if any. */
  matchAt(point: PointLike): MapGeoJSONFeature | null {
    if (!this.map.getLayer(PINS)) return null;
    const hits = this.map.queryRenderedFeatures(point, { layers: [PINS] });
    return hits.sort((a, b) => a.properties.rank - b.properties.rank)[0] ?? null;
  }

  handleClick(e: MapMouseEvent, onClick: Options["onClick"]) {
    const feature = this.matchAt(e.point);
    if (!feature) onClick(e.lngLat.lat, e.lngLat.lng);
    else this.onMarkerClick(feature.properties.rank); // a merged pin is its best match's
  }

  /**
   * Bring a match (from the list or a pin) into view on a pin of its own, moving as little as
   * possible: zoom in only as far as it takes to stand alone (no other match within MERGE_PX, so
   * anything merged into it splits off), and leave the camera be if it's already in view and alone.
   */
  focusMatch(rank: number) {
    const point = this.points?.find((p) => p.rank === rank);
    if (!point) return;
    const zoom = this.map.getZoom();
    const nearest = Math.min(...this.points!.filter((p) => p !== point).map((p) => pixelDistance(point, p, zoom)));
    const alone = nearest < MERGE_PX ? zoom + Math.log2(MERGE_PX / Math.max(nearest, 1e-6)) + 0.1 : zoom;
    const target = Math.min(alone, this.map.getMaxZoom());
    if (target === zoom && this.inView(point)) return;
    const camera = { center: [point.lon, point.lat] as [number, number], zoom: target, essential: true };
    const { lng, lat } = this.map.getCenter();
    const far = pixelDistance({ xy: mercator(lng, lat) }, point, zoom) > FAR_SCREENS * this.map.getContainer().clientWidth;
    if (far) this.map.flyTo({ ...camera, duration: FAR_MS });
    else this.map.easeTo({ ...camera, duration: FOCUS_MS });
  }

  /**
   * Whether a point is on screen, `pad` px in from the edges (a pin's width by default), in the
   * copy of the world nearest the centre: the one pins and the star draw in.
   */
  inView({ lon, lat }: { lon: number; lat: number }, pad: Padding = { top: MERGE_PX, bottom: MERGE_PX, left: MERGE_PX, right: MERGE_PX }) {
    const centre = this.map.getCenter().lng;
    const { x, y } = this.map.project([lon + 360 * Math.round((centre - lon) / 360), lat]);
    const { clientWidth: w, clientHeight: h } = this.map.getContainer();
    return x >= pad.left && x <= w - pad.right && y >= pad.top && y <= h - pad.bottom;
  }

  /**
   * Padding (px) that keeps pins a pin's width clear of the map's edges and of whatever sits over it:
   * the zoom buttons, the attribution and `overlays` (the legend). Each overlay is in a corner, so it
   * pads the top or bottom by as far as it reaches in.
   */
  fitPadding(): Padding {
    const box = this.map.getContainer().getBoundingClientRect();
    const pad = { top: 0, bottom: 0, left: 0, right: 0 };
    for (const el of [...this.map.getContainer().querySelectorAll(".maplibregl-ctrl"), ...this.overlays]) {
      const r = el.getBoundingClientRect();
      if (!r.width || r.bottom <= box.top || r.top >= box.bottom || r.right <= box.left || r.left >= box.right) continue;
      if (r.top + r.bottom < box.top + box.bottom) pad.top = Math.max(pad.top, r.bottom - box.top);
      else pad.bottom = Math.max(pad.bottom, box.bottom - r.top);
    }
    // On a short map, never pad away more than half its height.
    const squeeze = Math.min(1, (box.height / 2) / (pad.top + pad.bottom || 1));
    return {
      top: pad.top * squeeze + MERGE_PX,
      bottom: pad.bottom * squeeze + MERGE_PX,
      left: MERGE_PX,
      right: MERGE_PX,
    };
  }

  /**
   * Show the query's star and all its matches: ease to the closest view that holds them clear of the
   * overlays (fitPadding), going the shorter way round the world (lonSpan), between one world across
   * and FIT_MAX_ZOOM. With `ifHidden`, only if one of them is out of that view now.
   */
  async fitResults({ ifHidden = false } = {}) {
    await this.loaded;
    const points = [...(this.points ?? []), ...(this.queryPoint ? [this.queryPoint] : [])];
    if (!points.length) return;
    const padding = this.fitPadding();
    if (ifHidden && points.every((p) => this.inView(p, padding))) return;
    const [west, east] = lonSpan(points.map((p) => p.lon));
    const lats = points.map((p) => p.lat);
    const camera = this.map.cameraForBounds([[west, Math.min(...lats)], [east, Math.max(...lats)]], { padding });
    if (!camera) return; // the padding leaves no room
    const oneWorld = Math.log2(this.map.getContainer().clientWidth / 512); // 512 px: the world's width at zoom 0
    const zoom = Math.min(Math.max(camera.zoom!, oneWorld, this.map.getMinZoom()), FIT_MAX_ZOOM);
    this.map.easeTo({ center: camera.center, zoom, duration: FIT_MS, essential: true });
  }

  handleHover(e: MapMouseEvent | null) {
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
  showTip(feature: MapGeoJSONFeature) {
    const ranks = (feature.properties.ranks as string).split(",").filter(Boolean).map(Number).sort((a, b) => a - b);
    this.tip
      .setLngLat((feature.geometry as GeoJSON.Point).coordinates as [number, number])
      .setHTML(ranks.map((r) => `#${r}`).join(", "))
      .addTo(this.map);
  }

  hideTip() {
    this.tip.remove();
  }
}

/**
 * The basemap's settlement symbols, drawn on demand (it has no sprite): "atlas-dot", a city's ink
 * dot, and "atlas-capital", a capital's ringed dot. Both are edged in paper to stand off the heatmap.
 */
function addAtlasIcon(map: Map, id: string) {
  const r = ({ "atlas-dot": 3.5, "atlas-capital": 5 } as Record<string, number | undefined>)[id]; // outer radius, CSS px
  if (!r || map.hasImage(id)) return;
  const ratio = 2;
  const ctx = Object.assign(document.createElement("canvas"), { width: 2 * r * ratio, height: 2 * r * ratio }).getContext("2d")!;
  ctx.scale(ratio, ratio);
  const disc = (radius: number, color: string) => {
    ctx.beginPath();
    ctx.arc(r, r, radius, 0, 2 * Math.PI);
    ctx.fillStyle = color;
    ctx.fill();
  };
  disc(r, PAPER);
  disc(r - 1, INK);
  if (id === "atlas-capital") {
    disc(r - 2, PAPER);
    disc(r - 3, INK);
  }
  map.addImage(id, ctx.getImageData(0, 0, ctx.canvas.width, ctx.canvas.height), { pixelRatio: ratio });
}

/** Render the heatmap legend from /heatmap/legend stops into `el`; the caller shows it once there is a heatmap. */
export function renderLegend(el: HTMLElement, stops: LegendStop[], excluded: Legend["excluded"]) {
  stops = [...stops].sort((a, b) => b.top_pct - a.top_pct); // 5% (faint) -> 0.01% (strong)
  const rgba = (c: number[]) => `rgba(${c[0]}, ${c[1]}, ${c[2]}, ${c.length < 4 ? 1 : c[3] > 1 ? (c[3] / 255).toFixed(3) : c[3]})`;
  // Colors interpolate in log(top_pct), so place the stops on a log scale.
  const lo = Math.log(stops[0].top_pct), hi = Math.log(stops.at(-1)!.top_pct);
  const pos = (i: number) => (100 * (lo - Math.log(stops[i].top_pct))) / (lo - hi);
  const gradient = stops.map((s, i) => `${rgba(s.rgba)} ${pos(i)}%`).join(", ");
  // A short title: the exclusion radius has its own line.
  el.innerHTML = `
    <div class="title">Most similar · top % of land</div>
    <div class="bar" style="background: linear-gradient(to right, ${gradient})"></div>
    <div class="ticks">${stops.map((s, i) => `<span style="left:${pos(i)}%">${s.top_pct}%</span>`).join("")}</div>
    ${excluded ? `<div class="excluded"><span class="swatch" style="background:${rgba(excluded.rgba)}"></span>${excluded.label}</div>` : ""}
  `;
}
