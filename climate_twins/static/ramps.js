// Colour ramps for climate values, shared with crackalamoo/building-earth (frontend/src/lib/globe/
// colormap.ts and precipitationColormap.ts) so the two projects read alike. Both interpolate in RGB.
// Here they are background washes behind the comparison charts.

/** Temperature (°C) stops. The step at 0 °C is deliberate: freezing reads as a visible break. */
export const TEMP_STOPS = [
  [-30, [59, 30, 109]],
  [-15, [11, 30, 109]],
  [0, [30, 136, 229]],
  [0.01, [100, 181, 246]],
  [10, [102, 187, 106]],
  [21, [255, 235, 59]],
  [25, [251, 140, 0]],
  [30, [211, 47, 47]],
  [35, [181, 56, 42]],
  [40, [138, 0, 0]],
];

/** Precipitation (mm/day) stops: dry tan, through greens, to wet deep blue. */
export const PRECIP_STOPS = [
  [0, [210, 200, 180]],
  [1, [180, 210, 170]],
  [3, [100, 190, 120]],
  [6, [40, 150, 100]],
  [10, [30, 100, 140]],
  [15, [20, 50, 120]],
];

export const DAYS_PER_MONTH = 30.44;

function hex(rgb) {
  return `#${rgb.map((v) => Math.round(v).toString(16).padStart(2, "0")).join("")}`;
}

function interpolate(stops, value) {
  if (value <= stops[0][0]) return stops[0][1];
  for (let i = 1; i < stops.length; i++) {
    const [v1, c1] = stops[i];
    if (value <= v1) {
      const [v0, c0] = stops[i - 1];
      const f = (value - v0) / (v1 - v0);
      return c0.map((c, k) => c + f * (c1[k] - c));
    }
  }
  return stops.at(-1)[1];
}

/**
 * Gradient stops [offset 0-1, colour] for `stops` over a vertical span from `top` (offset 0) down to
 * `bottom` (offset 1), both in the stops' units. Keeps the ramp's own stops (and any hard step) inside.
 */
export function gradientStops(stops, top, bottom) {
  const offset = (v) => (top - v) / (top - bottom);
  const inside = stops.filter(([v]) => v < top && v > bottom).reverse(); // top (high) to bottom (low)
  return [
    [0, hex(interpolate(stops, top))],
    ...inside.map(([v, rgb]) => [offset(v), hex(rgb)]),
    [1, hex(interpolate(stops, bottom))],
  ];
}
