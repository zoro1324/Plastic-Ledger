import type {
  RunSummary,
  DetectionFeatureCollection,
  AttributionEntry,
  BacktrackEntry,
  RunMetadata,
  DebrisSummaryRow,
  IngestMetadata,
} from "@/types";

const BASE = "/data/runs/run_001";

async function fetchJson<T>(path: string): Promise<T> {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`Failed to fetch ${path}: ${res.status}`);
  return res.json();
}

export async function loadRunSummary(): Promise<RunSummary> {
  return fetchJson<RunSummary>(`${BASE}/run_summary.json`);
}

export async function loadDetections(): Promise<DetectionFeatureCollection> {
  return fetchJson<DetectionFeatureCollection>(`${BASE}/detections/detections_classified.geojson`);
}

export async function loadFinalReport(): Promise<DetectionFeatureCollection> {
  return fetchJson<DetectionFeatureCollection>(`${BASE}/reports/final_report.geojson`);
}

export async function loadAttribution(): Promise<AttributionEntry[]> {
  return fetchJson<AttributionEntry[]>(`${BASE}/attribution/attribution_report.json`);
}

export async function loadBacktrackSummary(): Promise<BacktrackEntry[]> {
  return fetchJson<BacktrackEntry[]>(`${BASE}/attribution/backtrack_summary.json`);
}

export async function loadRunMetadata(): Promise<RunMetadata> {
  return fetchJson<RunMetadata>(`${BASE}/attribution/run_metadata.json`);
}

export async function loadIngestMetadata(): Promise<IngestMetadata> {
  return fetchJson<IngestMetadata>(`${BASE}/ingest_metadata.json`);
}

export async function loadDebrisSummaryCsv(): Promise<DebrisSummaryRow[]> {
  const res = await fetch(`${BASE}/reports/debris_summary.csv`);
  const text = await res.text();
  const lines = text.trim().split("\n");
  const headers = lines[0].split(",");
  
  return lines.slice(1).map((line) => {
    const values = line.split(",");
    const row: any = {};
    headers.forEach((h, i) => {
      const v = values[i]?.trim() ?? "";
      if (["cluster_id", "lat", "lon", "area_sq_m", "confidence", "attribution_score", "source_lat", "source_lon"].includes(h)) {
        row[h] = v === "" ? 0 : parseFloat(v);
      } else {
        row[h] = v;
      }
    });
    return row as DebrisSummaryRow;
  });
}

// Report file paths
export const REPORT_FILES = {
  pdf: `${BASE}/reports/final_report.pdf`,
  geojson: `${BASE}/reports/final_report.geojson`,
  csv: `${BASE}/reports/debris_summary.csv`,
  backtrackMap: `${BASE}/reports/backtrack_map.html`,
  detectionMap: `${BASE}/reports/detection_map.png`,
  polymerDist: `${BASE}/reports/polymer_distribution.png`,
  rgbMap: `${BASE}/reports/rgb_map.png`,
};
