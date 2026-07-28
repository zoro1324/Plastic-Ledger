import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import Navbar from '@/components/Navbar';
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  AreaChart,
  Area,
} from 'recharts';
import { analyticsAPI } from '@/lib/api/mockAPI';
import { AnalyticsData } from '@/types';
import {
  ArrowLeft,
  BarChart3,
  TrendingUp,
  Activity,
  ShieldAlert,
  Zap,
  Download,
  Calendar,
  Filter,
} from 'lucide-react';

const AnalyticsPage: React.FC = () => {
  const [data, setData] = useState<AnalyticsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [timeRange, setTimeRange] = useState<'30d' | '6m' | '1y'>('6m');

  useEffect(() => {
    const loadData = async () => {
      const analyticsData = await analyticsAPI();
      setData(analyticsData);
      setLoading(false);
    };
    loadData();
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen bg-[#07172A] flex items-center justify-center text-cyan-400 font-mono">
        <div className="flex items-center space-x-3">
          <Activity className="w-6 h-6 animate-spin" />
          <span>Loading Spatial Analytics & Neural Metrics...</span>
        </div>
      </div>
    );
  }

  if (!data) return null;

  const COLORS = ['#0A84FF', '#00F5D4', '#F59E0B', '#EF4444', '#22C55E', '#4CC9F0'];

  // River Discharge Mock Data for rich visualization
  const riverDischargeData = [
    { river: 'Adyar Estuary', discharge: 420, risk: 'High' },
    { river: 'Kosasthalaiyar River', discharge: 310, risk: 'High' },
    { river: 'Cooum River Outlet', discharge: 580, risk: 'Critical' },
    { river: 'Kovalam Creek', discharge: 190, risk: 'Medium' },
    { river: 'Ennore Creek', discharge: 490, risk: 'High' },
  ];

  const handleExportCSV = () => {
    const csvContent =
      'data:text/csv;charset=utf-8,' +
      'Month,Clusters,Weight_Tons\n' +
      data.monthlyTrend.map((e) => `${e.month},${e.clusters},${e.weight}`).join('\n');
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', 'plastic_ledger_analytics.csv');
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const ChartCard: React.FC<{
    title: string;
    subtitle?: string;
    children: React.ReactNode;
    delay?: number;
  }> = ({ title, subtitle, children, delay = 0 }) => (
    <motion.div
      className="bg-[#07172A]/90 border border-[#0A84FF]/30 backdrop-blur-xl rounded-2xl p-5 shadow-xl hover:border-[#0A84FF]/60 transition-all duration-300"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay }}
    >
      <div className="mb-4">
        <h3 className="text-base font-extrabold text-white font-mono flex items-center justify-between">
          <span>{title}</span>
          <span className="w-2 h-2 rounded-full bg-cyan-400" />
        </h3>
        {subtitle && <p className="text-xs text-gray-400 font-mono mt-0.5">{subtitle}</p>}
      </div>
      {children}
    </motion.div>
  );

  return (
    <div className="min-h-screen bg-[#07172A] text-white font-sans relative">
      {/* Main Top Navigation Bar */}
      <Navbar />

      <div className="max-w-7xl mx-auto px-6 pt-24 pb-16">
        {/* Header Navigation & Back Button */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between mb-8 pb-4 border-b border-white/10 gap-4">
          <div className="flex items-center space-x-4">
            <Link
              to="/dashboard"
              className="flex items-center space-x-2 px-3.5 py-2 rounded-xl bg-white/5 hover:bg-[#0A84FF]/20 text-cyan-300 border border-[#0A84FF]/30 transition-all duration-200 text-xs font-mono font-bold"
            >
              <ArrowLeft className="w-4 h-4" />
              <span>Back to GIS Dashboard</span>
            </Link>

            <div>
              <h1 className="text-2xl font-extrabold text-white tracking-wide font-mono flex items-center gap-2.5">
                <BarChart3 className="w-6 h-6 text-[#0A84FF]" />
                Spatial Analytics & Intelligence Hub
              </h1>
              <p className="text-xs text-gray-400 font-mono">
                AI Hyperspectral Detection Metrics, Source Attribution & Ocean Drift Trends
              </p>
            </div>
          </div>

          {/* Action Bar (Filters & Export) */}
          <div className="flex items-center space-x-3">
            <div className="flex items-center bg-white/5 p-1 rounded-xl border border-white/10 text-xs font-mono">
              <button
                onClick={() => setTimeRange('30d')}
                className={`px-3 py-1 rounded-lg transition-all ${
                  timeRange === '30d' ? 'bg-[#0A84FF] text-white font-bold' : 'text-gray-400 hover:text-white'
                }`}
              >
                30D
              </button>
              <button
                onClick={() => setTimeRange('6m')}
                className={`px-3 py-1 rounded-lg transition-all ${
                  timeRange === '6m' ? 'bg-[#0A84FF] text-white font-bold' : 'text-gray-400 hover:text-white'
                }`}
              >
                6M
              </button>
              <button
                onClick={() => setTimeRange('1y')}
                className={`px-3 py-1 rounded-lg transition-all ${
                  timeRange === '1y' ? 'bg-[#0A84FF] text-white font-bold' : 'text-gray-400 hover:text-white'
                }`}
              >
                1Y
              </button>
            </div>

            <button
              onClick={handleExportCSV}
              className="flex items-center space-x-2 px-3.5 py-2 rounded-xl bg-gradient-to-r from-[#0A84FF] to-cyan-500 hover:from-[#0066CC] hover:to-cyan-400 text-slate-950 font-extrabold text-xs shadow-lg transition-all cursor-pointer"
            >
              <Download className="w-4 h-4" />
              <span>Export Report</span>
            </button>
          </div>
        </div>

        {/* Top KPI Metrics Bar */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          <div className="bg-[#07172A]/90 border border-[#0A84FF]/30 p-4 rounded-2xl backdrop-blur-xl">
            <span className="text-[10px] font-mono text-gray-400 uppercase tracking-wider block">
              Total Detected Waste
            </span>
            <div className="flex items-baseline space-x-2 mt-1">
              <span className="text-2xl font-extrabold text-cyan-300 font-mono">1,842.6</span>
              <span className="text-xs text-cyan-400 font-mono">Metric Tons</span>
            </div>
            <span className="text-[10px] text-emerald-400 font-mono mt-1 block flex items-center gap-1">
              <TrendingUp className="w-3 h-3" /> +14.2% from last month
            </span>
          </div>

          <div className="bg-[#07172A]/90 border border-[#0A84FF]/30 p-4 rounded-2xl backdrop-blur-xl">
            <span className="text-[10px] font-mono text-gray-400 uppercase tracking-wider block">
              Active High-Risk Zones
            </span>
            <div className="flex items-baseline space-x-2 mt-1">
              <span className="text-2xl font-extrabold text-red-400 font-mono">38</span>
              <span className="text-xs text-red-300 font-mono">Hotspots</span>
            </div>
            <span className="text-[10px] text-amber-400 font-mono mt-1 block flex items-center gap-1">
              <ShieldAlert className="w-3 h-3" /> 12 Critical Estuary Outlets
            </span>
          </div>

          <div className="bg-[#07172A]/90 border border-[#0A84FF]/30 p-4 rounded-2xl backdrop-blur-xl">
            <span className="text-[10px] font-mono text-gray-400 uppercase tracking-wider block">
              Detection Model Accuracy
            </span>
            <div className="flex items-baseline space-x-2 mt-1">
              <span className="text-2xl font-extrabold text-emerald-400 font-mono">96.4%</span>
              <span className="text-xs text-emerald-300 font-mono">Precision</span>
            </div>
            <span className="text-[10px] text-cyan-400 font-mono mt-1 block flex items-center gap-1">
              <Zap className="w-3 h-3" /> Hyperspectral ResNet-50
            </span>
          </div>

          <div className="bg-[#07172A]/90 border border-[#0A84FF]/30 p-4 rounded-2xl backdrop-blur-xl">
            <span className="text-[10px] font-mono text-gray-400 uppercase tracking-wider block">
              Source Attribution Rate
            </span>
            <div className="flex items-baseline space-x-2 mt-1">
              <span className="text-2xl font-extrabold text-amber-300 font-mono">89.2%</span>
              <span className="text-xs text-amber-400 font-mono">Matched</span>
            </div>
            <span className="text-[10px] text-gray-400 font-mono mt-1 block">
              Hydro-dynamic drift backtracked
            </span>
          </div>
        </div>

        {/* Analytics Chart Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          {/* Monthly Pollution Trend */}
          <ChartCard
            title="Monthly Plastic Waste Accumulation"
            subtitle="Detected cluster counts vs total estimated plastic mass (Metric Tons)"
          >
            <ResponsiveContainer width="100%" height={280}>
              <AreaChart data={data.monthlyTrend}>
                <defs>
                  <linearGradient id="colorClusters" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#0A84FF" stopOpacity={0.8} />
                    <stop offset="95%" stopColor="#0A84FF" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="colorWeight" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#00F5D4" stopOpacity={0.8} />
                    <stop offset="95%" stopColor="#00F5D4" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#0A84FF20" />
                <XAxis dataKey="month" stroke="#94A3B8" tick={{ fontSize: 11 }} />
                <YAxis stroke="#94A3B8" tick={{ fontSize: 11 }} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#07172A',
                    border: '1px solid #0A84FF60',
                    borderRadius: '12px',
                    fontSize: '12px',
                    fontFamily: 'monospace',
                  }}
                />
                <Legend wrapperStyle={{ fontSize: '11px', fontFamily: 'monospace' }} />
                <Area
                  type="monotone"
                  dataKey="clusters"
                  name="Detected Clusters"
                  stroke="#0A84FF"
                  fillOpacity={1}
                  fill="url(#colorClusters)"
                />
                <Area
                  type="monotone"
                  dataKey="weight"
                  name="Plastic Mass (Tons)"
                  stroke="#00F5D4"
                  fillOpacity={1}
                  fill="url(#colorWeight)"
                />
              </AreaChart>
            </ResponsiveContainer>
          </ChartCard>

          {/* Polymer Distribution */}
          <ChartCard
            title="Polymer Spectral Type Breakdown"
            subtitle="Hyperspectral signature classification ratio (PET, HDPE, LDPE, PP, PS, PVC)"
            delay={0.1}
          >
            <ResponsiveContainer width="100%" height={280}>
              <PieChart>
                <Pie
                  data={data.polymerDistribution}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={95}
                  paddingAngle={5}
                  dataKey="percentage"
                  label={({ polymer, percentage }) => `${polymer}: ${percentage.toFixed(1)}%`}
                >
                  {data.polymerDistribution.map((_, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#07172A',
                    border: '1px solid #0A84FF60',
                    borderRadius: '12px',
                    fontSize: '12px',
                  }}
                />
              </PieChart>
            </ResponsiveContainer>
          </ChartCard>

          {/* High Risk Regions */}
          <ChartCard
            title="High-Risk Coastal & Offshore Zones"
            subtitle="Risk index scores based on plastic density and ecological vulnerability"
            delay={0.2}
          >
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={data.highRiskRegions}>
                <CartesianGrid strokeDasharray="3 3" stroke="#0A84FF20" />
                <XAxis dataKey="region" stroke="#94A3B8" tick={{ fontSize: 11 }} />
                <YAxis stroke="#94A3B8" tick={{ fontSize: 11 }} domain={[0, 100]} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#07172A',
                    border: '1px solid #0A84FF60',
                    borderRadius: '12px',
                    fontSize: '12px',
                  }}
                />
                <Bar dataKey="riskScore" name="Risk Score (0-100)" fill="#EF4444" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </ChartCard>

          {/* Detection Accuracy Trend */}
          <ChartCard
            title="Model Detection Accuracy & SLA"
            subtitle="AI model precision consistency over sequential satellite pass runs"
            delay={0.3}
          >
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={data.detectionAccuracyTrend}>
                <CartesianGrid strokeDasharray="3 3" stroke="#0A84FF20" />
                <XAxis dataKey="date" stroke="#94A3B8" tick={{ fontSize: 10 }} />
                <YAxis stroke="#94A3B8" domain={[0.7, 1]} tick={{ fontSize: 11 }} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#07172A',
                    border: '1px solid #0A84FF60',
                    borderRadius: '12px',
                    fontSize: '12px',
                  }}
                />
                <Line
                  type="monotone"
                  dataKey="accuracy"
                  name="Accuracy Rate"
                  stroke="#22C55E"
                  strokeWidth={2.5}
                  dot={{ fill: '#22C55E', r: 4 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </ChartCard>
        </div>

        {/* Industrial Source Ranking & River Discharge */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Industrial Source Ranking */}
          <ChartCard
            title="Industrial Source Attribution Ranking"
            subtitle="Backtracked pollution contribution by industrial facility"
            delay={0.4}
          >
            <ResponsiveContainer width="100%" height={280}>
              <BarChart
                data={data.sourceRanking}
                layout="vertical"
                margin={{ top: 5, right: 30, left: 140, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#0A84FF20" />
                <XAxis type="number" stroke="#94A3B8" tick={{ fontSize: 11 }} />
                <YAxis dataKey="source" type="category" stroke="#94A3B8" tick={{ fontSize: 11 }} width={140} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#07172A',
                    border: '1px solid #0A84FF60',
                    borderRadius: '12px',
                    fontSize: '12px',
                  }}
                />
                <Bar dataKey="contribution" name="Contribution (%)" fill="#F59E0B" radius={[0, 6, 6, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </ChartCard>

          {/* River Discharge & Estuary Input */}
          <ChartCard
            title="River & Estuary Discharge Rate"
            subtitle="Estimated plastic load input into coastal ocean currents (Metric Tons/Yr)"
            delay={0.5}
          >
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={riverDischargeData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#0A84FF20" />
                <XAxis dataKey="river" stroke="#94A3B8" tick={{ fontSize: 10 }} />
                <YAxis stroke="#94A3B8" tick={{ fontSize: 11 }} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#07172A',
                    border: '1px solid #0A84FF60',
                    borderRadius: '12px',
                    fontSize: '12px',
                  }}
                />
                <Bar dataKey="discharge" name="Discharge (Tons/Yr)" fill="#4CC9F0" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </ChartCard>
        </div>
      </div>
    </div>
  );
};

export default AnalyticsPage;
