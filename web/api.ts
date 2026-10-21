// The API's responses, as the UI uses them (docs/api.md). All metric.

export type Feature = "tmax" | "tmin" | "dewpoint" | "ppt" | "clearness" | "ws";

export interface Cell {
  index: number;
  lat: number;
  lon: number;
}

export interface Summary {
  annual_mean_temp_c: number;
  warmest_month_high_c: number;
  coldest_month_low_c: number;
  annual_precip_mm: number;
  wettest_month_precip_mm: number;
  driest_month_precip_mm: number;
  muggiest_month_dewpoint_c: number;
  mean_solar_radiation_w_m2: number;
  mean_clearness: number;
  mean_wind_m_s: number;
}

/** 12 monthly values (January first) of each variable. */
export type Monthly = Record<Feature | "srad", number[]>;

interface PlaceFields {
  name: string;
  admin1: string | null;
  country: string | null;
  country_code: string;
  base_label: string;
  label: string;
}

/** A /places/search result. */
export interface SearchPlace extends PlaceFields {
  population: number;
  lat: number;
  lon: number;
}

interface LabelFields extends PlaceFields {
  in_area: string | null;
  area: { admin1: string | null; country: string | null } | null;
}

/** A Place label naming a city: in it, near it, or `km` from it in `direction`. */
interface CityLabel extends LabelFields, SearchPlace {
  tier: "exact" | "near" | "offset";
  km: number;
  direction: string | null;
}

/** A Place label naming only the region the point is in. */
interface RegionLabel extends LabelFields {
  tier: "region";
  population: null;
  lat: null;
  lon: null;
  km: null;
  direction: null;
}

export type Place = CityLabel | RegionLabel;

export type RulerComparison = { kind: "all" } | { kind: "about" | "closer"; city: number };

export interface Ruler {
  min_km: number;
  max_km: number;
  min_population: number;
  cities: {
    name: string;
    label: string;
    population: number;
    lat: number;
    lon: number;
    km_from_query: number;
    distance: number;
  }[];
}

/** /similar's `query`, with detail (the UI always asks for it). */
export interface QueryPoint {
  lat: number;
  lon: number;
  cell: Cell;
  snapped_km: number;
  summary: Summary;
  monthly: Monthly;
  place: Place | null;
}

export interface Match {
  rank: number;
  cell: Cell;
  km_from_query: number;
  distance: number;
  more_similar_than_pct_of_land: number;
  seasons_shifted_6_months: boolean;
  difference_share: Record<Feature, number>;
  difference_d2: Record<Feature, number>;
  summary: Summary;
  monthly: Monthly;
  place: Place | null;
  ruler_comparison: RulerComparison | null;
}

export interface SimilarResponse {
  query: QueryPoint;
  params: { n: number; exclude_km: number; suppress_km: number; align: "auto" | "calendar" };
  matches: Match[];
  ruler: Ruler;
  elapsed_ms: number;
}

/** A /heatmap/legend colour stop; rgba's alpha is 0-255. */
export interface LegendStop {
  top_pct: number;
  rgba: number[];
}

export interface Legend {
  stops: LegendStop[];
  excluded: { rgba: number[]; label: string };
}
