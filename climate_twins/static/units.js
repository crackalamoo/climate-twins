// Display units: metric or US. The API is always metric; this only converts for display.
// The choice lives in memory only: each page load starts from the browser locale.

const KM_PER_MI = 1.609344;

/** US units iff the browser locale's region is the US (e.g. "en-US", or "en" which maximizes to it). */
function detect() {
  try {
    return new Intl.Locale(navigator.language).maximize().region === "US" ? "us" : "metric";
  } catch {
    return "metric";
  }
}

let system = detect();

export function units() {
  return system;
}

export function setUnits(value) {
  system = value === "us" ? "us" : "metric";
}

const us = () => system === "us";

// Converters (numbers in, numbers out) for chart data.
export const temp = (c) => (us() ? (c * 9) / 5 + 32 : c);
export const precip = (mm) => (us() ? mm / 25.4 : mm);
export const tempUnit = () => (us() ? "°F" : "°C");
export const precipUnit = () => (us() ? "in" : "mm");

// Formatters (metric numbers in, display strings out).
export function fmtTemp(c) {
  return `${temp(c).toFixed(1)} ${tempUnit()}`;
}

/** A monthly amount: whole mm, or inches to 2 decimals (1 when 10+). */
export function fmtPrecip(mm) {
  if (!us()) return `${Math.round(mm)} mm`;
  const inches = precip(mm);
  return `${inches.toFixed(inches >= 10 ? 1 : 2)} in`;
}

/** An annual total: whole mm, or inches to 1 decimal. */
export function fmtAnnualPrecip(mm) {
  return us() ? `${precip(mm).toFixed(1)} in` : `${Math.round(mm)} mm`;
}

/** A wind speed (m/s in): km/h, matching the toggle's km, or mph, to 1 decimal. */
export function fmtWind(ms) {
  const kmh = ms * 3.6;
  return us() ? `${(kmh / KM_PER_MI).toFixed(1)} mph` : `${kmh.toFixed(1)} km/h`;
}

/** A distance; `step` rounds to the nearest multiple of it (in the display unit). */
export function fmtDistance(km, { step = 1 } = {}) {
  const v = us() ? km / KM_PER_MI : km;
  return `${(Math.round(v / step) * step).toLocaleString("en-US")} ${us() ? "mi" : "km"}`;
}
