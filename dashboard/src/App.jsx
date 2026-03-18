const kpis = [
  {
    label: "Detected Micro-Clusters",
    value: "14,208",
    delta: "+8.2%",
    tone: "text-aurora-cyan",
  },
  {
    label: "Source Confidence",
    value: "93.6%",
    delta: "+2.9%",
    tone: "text-aurora-lime",
  },
  {
    label: "Active Water Bodies",
    value: "27",
    delta: "+4",
    tone: "text-aurora-blue",
  },
  {
    label: "Average Fragment Age",
    value: "41 days",
    delta: "-6 days",
    tone: "text-aurora-amber",
  },
];

const missionModules = [
  {
    title: "Multi-Spectral Segmentation",
    summary:
      "SWIR/NIR fusion separates polymer signatures from biogenic clutter in low-resolution sectors.",
    status: "Operational",
    metric: "IoU 0.82",
    tone: "bg-aurora-cyan/20 text-aurora-cyan",
  },
  {
    title: "Hydrodynamic Reverse-Modeling",
    summary:
      "Physics-informed path reconstruction estimates discharge origin using wind, current, and tide fields.",
    status: "Converged",
    metric: "Path Error 2.3 km",
    tone: "bg-aurora-lime/20 text-aurora-lime",
  },
  {
    title: "Degradation Simulation",
    summary:
      "Aging curves infer polymer weathering stage to estimate fragment age and probable product lineage.",
    status: "Learning",
    metric: "R2 0.89",
    tone: "bg-aurora-amber/20 text-aurora-amber",
  },
  {
    title: "Source Attribution",
    summary:
      "Chemical fingerprint matching ranks likely industrial emitters by spectral compatibility and transport fit.",
    status: "Flagged",
    metric: "Top Source 97%",
    tone: "bg-aurora-coral/20 text-aurora-coral",
  },
];

const topSources = [
  { name: "Delta Polymer Works", probability: 0.97, region: "Delta-03" },
  { name: "Maris Packaging Plant", probability: 0.92, region: "Coastline-11" },
  { name: "Helio Resin Terminal", probability: 0.86, region: "Bay-07" },
  { name: "Orca Industrial Port", probability: 0.78, region: "Channel-04" },
];

const timeline = [
  "Ingested 12 Sentinel-2 strips + 4 drone flights",
  "Detected 482 candidate polymer plumes in SWIR/NIR",
  "Backtracked trajectories against 72-hour current vectors",
  "Ranked 4 industrial source clusters with confidence bands",
];

function App() {
  return (
    <div className="min-h-screen bg-nebula text-slate-100">
      <div className="pointer-events-none fixed inset-0 opacity-60">
        <div className="absolute left-[12%] top-[20%] h-44 w-44 animate-pulse rounded-full bg-aurora-cyan/10 blur-3xl" />
        <div className="absolute bottom-[15%] right-[14%] h-56 w-56 animate-pulse rounded-full bg-aurora-lime/10 blur-3xl [animation-delay:1000ms]" />
      </div>

      <div className="relative mx-auto max-w-7xl px-4 pb-10 pt-6 sm:px-6 lg:px-8">
        <header className="rounded-2xl border border-white/10 bg-cosmos-900/70 p-5 shadow-glow backdrop-blur">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
            <div>
              <p className="font-display text-xs uppercase tracking-[0.3em] text-aurora-cyan">
                Plastic Ledger Mission Console
              </p>
              <h1 className="mt-2 font-display text-2xl text-white sm:text-3xl">
                Autonomous Micro-Plastic Fingerprinting Dashboard
              </h1>
              <p className="mt-2 max-w-3xl text-sm text-slate-300 sm:text-base">
                Detecting polymer signatures, simulating degradation, and tracing pollution backward to likely industrial discharge points.
              </p>
            </div>
            <div className="grid grid-cols-2 gap-3 text-xs sm:text-sm">
              <InfoChip label="Area" value="South China Sea" />
              <InfoChip label="Model" value="MARIDA-v3" />
              <InfoChip label="Last Sync" value="03:14 UTC" />
              <InfoChip label="Status" value="Live Monitoring" />
            </div>
          </div>
        </header>

        <main className="mt-6 grid gap-6 lg:grid-cols-12">
          <section className="lg:col-span-9 space-y-6">
            <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
              {kpis.map((item) => (
                <article
                  key={item.label}
                  className="group rounded-xl border border-white/10 bg-cosmos-900/60 p-4 backdrop-blur transition duration-300 hover:-translate-y-0.5 hover:border-aurora-cyan/40"
                >
                  <p className="text-xs uppercase tracking-[0.16em] text-slate-300">{item.label}</p>
                  <p className="mt-3 font-display text-2xl text-white">{item.value}</p>
                  <p className={`mt-2 text-sm ${item.tone}`}>{item.delta} last cycle</p>
                </article>
              ))}
            </div>

            <section className="rounded-2xl border border-white/10 bg-cosmos-900/65 p-5 shadow-glow backdrop-blur">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <h2 className="font-display text-lg text-white">Mission Modules</h2>
                <p className="text-xs uppercase tracking-[0.2em] text-slate-400">Pipeline Health</p>
              </div>
              <div className="mt-4 grid gap-4 md:grid-cols-2">
                {missionModules.map((module) => (
                  <article
                    key={module.title}
                    className="rounded-xl border border-white/10 bg-cosmos-800/50 p-4"
                  >
                    <div className="flex items-start justify-between gap-2">
                      <h3 className="font-display text-base text-white">{module.title}</h3>
                      <span className={`rounded-full px-3 py-1 text-xs font-semibold ${module.tone}`}>
                        {module.status}
                      </span>
                    </div>
                    <p className="mt-2 text-sm text-slate-300">{module.summary}</p>
                    <p className="mt-3 text-xs uppercase tracking-[0.16em] text-slate-400">{module.metric}</p>
                  </article>
                ))}
              </div>
            </section>

            <section className="rounded-2xl border border-white/10 bg-cosmos-900/65 p-5 shadow-glow backdrop-blur">
              <div className="flex items-center justify-between gap-2">
                <h2 className="font-display text-lg text-white">Hydrodynamic Trace Overlay</h2>
                <span className="rounded-full bg-aurora-blue/15 px-3 py-1 text-xs text-aurora-blue">
                  Backtrack Horizon: 72h
                </span>
              </div>
              <div className="mt-4 overflow-hidden rounded-xl border border-white/10 bg-cosmos-800/60 p-4">
                <div className="h-56 rounded-lg bg-[radial-gradient(circle_at_20%_25%,rgba(94,244,255,0.2),transparent_30%),radial-gradient(circle_at_70%_40%,rgba(183,255,104,0.12),transparent_35%),linear-gradient(180deg,#111a35,#0a1022)] p-4 sm:h-64">
                  <div className="relative h-full w-full rounded-lg border border-white/10">
                    <div className="absolute left-[18%] top-[26%] h-2 w-2 rounded-full bg-aurora-cyan shadow-[0_0_14px_rgba(94,244,255,0.9)]" />
                    <div className="absolute left-[34%] top-[33%] h-2 w-2 rounded-full bg-aurora-lime shadow-[0_0_14px_rgba(183,255,104,0.9)]" />
                    <div className="absolute right-[20%] bottom-[30%] h-2 w-2 rounded-full bg-aurora-amber shadow-[0_0_14px_rgba(255,184,77,0.9)]" />
                    <svg className="absolute inset-0 h-full w-full" viewBox="0 0 100 60" fill="none">
                      <path d="M12 16 C26 22, 30 35, 44 38" stroke="rgba(94,244,255,0.8)" strokeWidth="0.8" strokeDasharray="3 2" />
                      <path d="M27 20 C36 28, 48 32, 61 36" stroke="rgba(183,255,104,0.8)" strokeWidth="0.8" strokeDasharray="3 2" />
                      <path d="M82 43 C74 38, 67 33, 60 27" stroke="rgba(255,184,77,0.8)" strokeWidth="0.8" strokeDasharray="3 2" />
                    </svg>
                    <p className="absolute bottom-2 left-2 text-[10px] uppercase tracking-[0.2em] text-slate-300">
                      Synthetic preview of trace vectors
                    </p>
                  </div>
                </div>
              </div>
            </section>
          </section>

          <aside className="lg:col-span-3 space-y-6">
            <section className="rounded-2xl border border-white/10 bg-cosmos-900/65 p-5 shadow-glow backdrop-blur">
              <h2 className="font-display text-lg text-white">Top Suspected Sources</h2>
              <div className="mt-4 space-y-3">
                {topSources.map((source) => (
                  <div key={source.name} className="rounded-lg border border-white/10 bg-cosmos-800/55 p-3">
                    <div className="flex items-center justify-between gap-2">
                      <p className="text-sm text-white">{source.name}</p>
                      <p className="text-xs text-aurora-cyan">{Math.round(source.probability * 100)}%</p>
                    </div>
                    <p className="mt-1 text-xs text-slate-400">Region: {source.region}</p>
                    <div className="mt-2 h-1.5 rounded-full bg-white/10">
                      <div
                        className="h-1.5 rounded-full bg-gradient-to-r from-aurora-cyan to-aurora-lime"
                        style={{ width: `${source.probability * 100}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </section>

            <section className="rounded-2xl border border-white/10 bg-cosmos-900/65 p-5 shadow-glow backdrop-blur">
              <h2 className="font-display text-lg text-white">Latest Sequence</h2>
              <ol className="mt-4 space-y-3">
                {timeline.map((step, index) => (
                  <li key={step} className="flex gap-3 text-sm text-slate-300">
                    <span className="mt-0.5 flex h-5 w-5 flex-none items-center justify-center rounded-full bg-cosmos-700 text-xs text-aurora-cyan">
                      {index + 1}
                    </span>
                    <span>{step}</span>
                  </li>
                ))}
              </ol>
            </section>
          </aside>
        </main>
      </div>
    </div>
  );
}

function InfoChip({ label, value }) {
  return (
    <div className="rounded-lg border border-white/10 bg-cosmos-800/60 px-3 py-2">
      <p className="text-[10px] uppercase tracking-[0.16em] text-slate-400">{label}</p>
      <p className="mt-1 text-sm text-white">{value}</p>
    </div>
  );
}

export default App;
