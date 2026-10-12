// Map: basemap, similarity heatmap tiles, query star and numbered match pins.

import { Map, Marker, NavigationControl } from "https://cdn.jsdelivr.net/npm/maplibre-gl@6.12.0/dist/maplibre-gl.mjs";

const STYLE = "https://tiles.openfreemap.org/styles/positron";
const HEATMAP = "heatmap";

export class ClimateMap {
  constructor(container, { onClick, onMarkerClick, onMarkerHover }) {
    this.map = new Map({ container, style: STYLE, center: [0, 20], zoom: 1.3, attributionControl: { compact: true } });
    this.map.addControl(new NavigationControl({ showCompass: false }), "top-left");
    this.map.getCanvas().style.cursor = "crosshair";
    this.map.on("click", (e) => onClick(e.lngLat.lat, e.lngLat.lng));
    this.loaded = new Promise((resolve) => this.map.once("load", resolve));
    this.onMarkerClick = onMarkerClick;
    this.onMarkerHover = onMarkerHover;
    this.queryMarker = null;
    this.matchMarkers = new globalThis.Map(); // rank -> Marker
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

  setMatches(matches) {
    for (const m of this.matchMarkers.values()) m.remove();
    this.matchMarkers.clear();
    for (const match of matches) {
      const el = document.createElement("div");
      el.className = "marker-match";
      el.textContent = match.rank;
      el.addEventListener("click", (e) => {
        e.stopPropagation(); // don't start a new query from the map click
        this.onMarkerClick(match.rank);
      });
      el.addEventListener("mouseenter", () => this.onMarkerHover(match.rank));
      el.addEventListener("mouseleave", () => this.onMarkerHover(null));
      const marker = new Marker({ element: el }).setLngLat([match.cell.lon, match.cell.lat]).addTo(this.map);
      this.matchMarkers.set(match.rank, marker);
    }
  }

  /** Toggle a CSS class ("hover" / "selected") on exactly one match pin (or none). */
  markMatch(cls, rank) {
    for (const [r, marker] of this.matchMarkers) {
      marker.getElement().classList.toggle(cls, r === rank);
    }
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
