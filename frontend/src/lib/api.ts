import { PipelineRun } from "@/types";

export async function getPipelineRuns(): Promise<PipelineRun[]> {
  const res = await fetch("/api/pipeline/runs/");
  if (!res.ok) {
    throw new Error(`Failed to fetch runs: ${res.statusText}`);
  }
  return res.json();
}

export async function getPipelineRun(id: string): Promise<PipelineRun> {
  const res = await fetch(`/api/pipeline/runs/${id}/`);
  if (!res.ok) {
    throw new Error(`Failed to fetch run ${id}: ${res.statusText}`);
  }
  return res.json();
}

export async function createPipelineRun(data: {
  bbox: string;
  target_date: string;
  cloud_cover?: number;
  backtrack_days?: number;
  max_clusters?: number;
}): Promise<PipelineRun> {
  const res = await fetch("/api/pipeline/runs/", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(data),
  });
  
  if (!res.ok) {
    throw new Error(`Failed to create run: ${res.statusText}`);
  }
  
  return res.json();
}
