import type {
  RunSummary,
  DetectionFeatureCollection,
  AttributionEntry,
  BacktrackEntry,
  RunMetadata,
  DebrisSummaryRow,
  IngestMetadata,
} from "@/types";

const getBase = (runId: string) => `/data/runs/${runId}`;

export function getSceneId(summary: RunSummary): string | null {
  const scenePath = summary.outputs.raw_scenes?.[0];
  if (!scenePath) return null;
  return scenePath.split(/[\\/]/).pop() || null;
}

async function fetchJson<T>(path: string): Promise<T> {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`Failed to fetch ${path}: ${res.status}`);
  return res.json();
}

export async function loadRunSummary(runId: string): Promise<RunSummary> {
  return fetchJson<RunSummary>(`${getBase(runId)}/run_summary.json`);
}

export async function loadDetections(runId: string): Promise<DetectionFeatureCollection> {
  const summary = await loadRunSummary(runId);
  const sceneId = getSceneId(summary);
  return fetchJson<DetectionFeatureCollection>(`${getBase(runId)}/detections/${sceneId}/detections_classified.geojson`);
}

export async function loadFinalReport(runId: string): Promise<DetectionFeatureCollection> {
  const summary = await loadRunSummary(runId);
  const sceneId = getSceneId(summary);
  return fetchJson<DetectionFeatureCollection>(`${getBase(runId)}/reports/${sceneId}/final_report.geojson`);
}

export async function loadAttribution(runId: string): Promise<AttributionEntry[]> {
  const summary = await loadRunSummary(runId);
  const sceneId = getSceneId(summary);
  return fetchJson<AttributionEntry[]>(`${getBase(runId)}/attribution/${sceneId}/attribution_report.json`);
}

export async function loadBacktrackSummary(runId: string): Promise<BacktrackEntry[]> {
  const summary = await loadRunSummary(runId);
  const sceneId = getSceneId(summary);
  return fetchJson<BacktrackEntry[]>(`${getBase(runId)}/attribution/${sceneId}/backtrack_summary.json`);
}

export async function loadRunMetadata(runId: string): Promise<RunMetadata> {
  const summary = await loadRunSummary(runId);
  const sceneId = getSceneId(summary);
  return fetchJson<RunMetadata>(`${getBase(runId)}/attribution/${sceneId}/run_metadata.json`);
}

export async function loadIngestMetadata(runId: string): Promise<IngestMetadata> {
  return fetchJson<IngestMetadata>(`${getBase(runId)}/raw/ingest_metadata.json`);
}

export async function loadDebrisSummaryCsv(runId: string): Promise<DebrisSummaryRow[]> {
  const summary = await loadRunSummary(runId);
  const sceneId = getSceneId(summary);
  const res = await fetch(`${getBase(runId)}/reports/${sceneId}/debris_summary.csv`);
  if (!res.ok) throw new Error(`Failed to fetch csv`);
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

// Report file paths generator
export const getReportFiles = (runId: string, sceneId: string) => ({
  pdf: `${getBase(runId)}/reports/${sceneId}/final_report.pdf`,
  geojson: `${getBase(runId)}/reports/${sceneId}/final_report.geojson`,
  csv: `${getBase(runId)}/reports/${sceneId}/debris_summary.csv`,
  backtrackMap: `${getBase(runId)}/reports/${sceneId}/backtrack_map.html`,
  detectionMap: `${getBase(runId)}/reports/${sceneId}/detection_map.png`,
  polymerDist: `${getBase(runId)}/reports/${sceneId}/polymer_distribution.png`,
  rgbMap: `${getBase(runId)}/reports/${sceneId}/rgb_map.png`,
});
