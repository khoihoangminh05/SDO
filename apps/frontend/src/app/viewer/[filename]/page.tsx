'use client';

import React, { useEffect, useState } from 'react';
import { useParams, useRouter } from 'next/navigation';
import ImageViewer from '../../../components/ImageViewer';
import { TreeExplorer } from '../../../components/TreeExplorer';
import MetricsPanel from '../../../components/MetricsPanel';
import DagTree from '../../../components/DagTree';
import { exportImageBboxesToSvg } from '../../../utils/exportHelper';
import {
  useExplorationStore,
  getClassColor,
  getApiUrl,
} from '../../../store/explorationStore';
import { FolderTree, ChevronLeft, Layers, Flame, Download } from 'lucide-react';

export default function ViewerPage() {
  const params = useParams();
  const router = useRouter();
  const filename = params?.filename as string;

  const { 
    currentNode, 
    breadcrumbs, 
    initialize, 
    navigateToNode, 
    reset, 
    showHeatmap, 
    toggleHeatmap, 
    explorationTree 
  } = useExplorationStore();

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [activeTab, setActiveTab] = useState<'image' | 'dag'>('image');
  const [imageSize, setImageSize] = useState<{ width: number; height: number } | null>(null);

  // Extract image_id by removing extension
  const imageId = filename
    ? filename.substring(0, filename.lastIndexOf('.'))
    : '';
  const imageUrl = getApiUrl(`/uploads/${filename}`);

  useEffect(() => {
    if (!imageId) return;

    const fetchDetections = async () => {
      try {
        setLoading(true);
        const res = await fetch(getApiUrl('/explore'), {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            image_id: imageId,
            zoom_level: 0,
            global_bbox: [0.0, 0.0, 1.0, 1.0],
          }),
        });

        if (!res.ok) {
          throw new Error(`Failed to fetch detections: ${res.statusText}`);
        }

        const data = await res.json();
        initialize(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error');
      } finally {
        setLoading(false);
      }
    };

    fetchDetections();

    return () => {
      reset();
    };
  }, [imageId, initialize, reset]);

  const detections = currentNode?.detections || [];

  return (
    <main className="min-h-screen bg-slate-900 text-white flex flex-col p-6 font-sans">
      <header className="flex justify-between items-center mb-6 border-b border-slate-800 pb-4">
        <div className="flex items-center gap-4">
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="p-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg border border-slate-700 transition-all duration-150 cursor-pointer flex items-center justify-center"
            title={sidebarOpen ? 'Collapse Tree' : 'Expand Tree'}
          >
            <FolderTree className="w-4 h-4" />
          </button>
          <div>
            <h1 className="text-xl font-bold bg-gradient-to-r from-blue-400 to-indigo-400 bg-clip-text text-transparent">
              Recursive Object Detector
            </h1>
            <p className="text-xs text-slate-500 font-mono mt-0.5">
              Image ID: {imageId}
            </p>
          </div>
        </div>

        <button
          onClick={() => router.push('/')}
          className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 text-sm font-medium rounded-lg transition-colors border border-slate-700 cursor-pointer"
        >
          ← Upload another image
        </button>
      </header>

      {/* Evaluation Metrics Row */}
      <MetricsPanel />

      {/* Main Content Area */}
      <div className="flex-1 flex gap-6 overflow-hidden min-h-[600px]">
        {/* Left Sidebar: Collapsible Tree Explorer */}
        <aside
          className={`bg-slate-950 rounded-xl border border-slate-800 flex flex-col transition-all duration-300 overflow-hidden ${
            sidebarOpen
              ? 'w-64 opacity-100 p-4 border'
              : 'w-0 opacity-0 p-0 border-none pointer-events-none'
          }`}
        >
          <div className="flex items-center justify-between mb-4 pb-2 border-b border-slate-800">
            <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
              <FolderTree className="w-4 h-4 text-blue-400" />
              <span>Tree Explorer</span>
            </h3>
            <button
              onClick={() => setSidebarOpen(false)}
              className="text-slate-500 hover:text-slate-300 p-0.5 rounded cursor-pointer"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>
          </div>

          <div className="flex-1 overflow-y-auto pr-1">
            <TreeExplorer />
          </div>
        </aside>

        {/* Center: Dynamic tabs viewer */}
        <div className="flex-1 flex flex-col bg-slate-950 rounded-xl border border-slate-800 overflow-hidden shadow-2xl relative">
          {/* Sticky Tab switcher & Toolbar */}
          <div className="sticky top-0 left-0 right-0 z-40 bg-slate-900/95 backdrop-blur-md px-4 py-3 border-b border-slate-850 flex justify-between items-center overflow-x-auto whitespace-nowrap scrollbar-thin gap-4">
            <div className="flex items-center gap-2">
              <button
                onClick={() => setActiveTab('image')}
                className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all duration-150 cursor-pointer flex items-center gap-1.5 border ${
                  activeTab === 'image'
                    ? 'bg-blue-600 border-blue-500 text-white shadow-md'
                    : 'bg-slate-900 border-slate-800 text-slate-400 hover:bg-slate-800 hover:text-white'
                }`}
              >
                <Layers className="w-3.5 h-3.5" />
                <span>Spatial Explorer</span>
              </button>
              <button
                onClick={() => setActiveTab('dag')}
                className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all duration-150 cursor-pointer flex items-center gap-1.5 border ${
                  activeTab === 'dag'
                    ? 'bg-blue-600 border-blue-500 text-white shadow-md'
                    : 'bg-slate-900 border-slate-800 text-slate-400 hover:bg-slate-800 hover:text-white'
                }`}
              >
                <FolderTree className="w-3.5 h-3.5" />
                <span>DAG Hierarchy Graph</span>
              </button>
            </div>

            {activeTab === 'image' && (
              <div className="flex items-center gap-4">
                {/* Heatmap Toggle */}
                <label className="flex items-center gap-2 text-xs text-slate-300 cursor-pointer select-none">
                  <input
                    type="checkbox"
                    checked={showHeatmap}
                    onChange={toggleHeatmap}
                    className="w-3.5 h-3.5 rounded text-blue-600 focus:ring-blue-500 bg-slate-850 border-slate-800 cursor-pointer"
                  />
                  <span className="flex items-center gap-1">
                    <Flame className="w-3.5 h-3.5 text-orange-400" />
                    <span>Recursion Heatmap</span>
                  </span>
                </label>

                {/* Export Button */}
                <button
                  onClick={() => exportImageBboxesToSvg(imageUrl, explorationTree, imageSize)}
                  className="px-3 py-1 bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium rounded-lg transition-colors border border-slate-700 cursor-pointer flex items-center gap-1.5"
                  title="Export current view as vector SVG"
                >
                  <Download className="w-3.5 h-3.5" />
                  <span>Export SVG</span>
                </button>
              </div>
            )}
          </div>

          {/* Breadcrumbs (only visible in image explorer mode) */}
          {activeTab === 'image' && breadcrumbs.length > 0 && (
            <nav className="bg-slate-900/40 px-4 py-2 border-b border-slate-850/60 flex items-center gap-1.5 overflow-x-auto whitespace-nowrap scrollbar-thin">
              {breadcrumbs.map((node, idx) => (
                <React.Fragment key={node.node_id}>
                  {idx > 0 && (
                    <span className="text-slate-600 font-mono text-xs">/</span>
                  )}
                  <button
                    onClick={() => navigateToNode(node)}
                    disabled={node.node_id === currentNode?.node_id}
                    className={`px-2 py-0.5 rounded text-[10px] transition-all duration-150 cursor-pointer ${
                      node.node_id === currentNode?.node_id
                        ? 'bg-blue-600/20 text-blue-400 font-semibold border border-blue-500/25 pointer-events-none'
                        : 'text-slate-400 hover:bg-slate-800 hover:text-white border border-transparent'
                    }`}
                  >
                    {idx === 0
                      ? 'Root'
                      : `Lvl ${node.zoom_level} [${node.global_bbox.map((v) => v.toFixed(0)).join(', ')}]`}
                  </button>
                </React.Fragment>
              ))}
            </nav>
          )}

          {/* Display Area */}
          <div className="flex-1 min-h-[550px] relative">
            {activeTab === 'image' ? (
              <ImageViewer imageUrl={imageUrl} onImageSizeLoaded={setImageSize} />
            ) : (
              <DagTree />
            )}
          </div>
        </div>

        {/* Right Sidebar: Detections Panel */}
        <aside className="w-80 bg-slate-950 rounded-xl border border-slate-800 p-4 flex flex-col overflow-hidden">
          <div className="flex items-center justify-between mb-4 pb-2 border-b border-slate-800">
            <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
              Detections
            </h3>
            <span className="text-xs bg-slate-850 text-slate-300 font-mono px-2 py-0.5 rounded-full border border-slate-800">
              Lvl {currentNode?.zoom_level || 0}
            </span>
          </div>

          {loading ? (
            <div className="flex-1 flex items-center justify-center">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
            </div>
          ) : error ? (
            <div className="flex-1 flex flex-col items-center justify-center text-center p-4">
              <span className="text-red-400 text-sm mb-2">
                ⚠️ Connection Error
              </span>
              <p className="text-xs text-slate-400">
                Could not retrieve detections.
              </p>
            </div>
          ) : (
            <div className="flex-1 overflow-y-auto space-y-2 pr-2 scrollbar-thin">
              <p className="text-xs text-slate-500 mb-3 font-semibold font-mono">
                Found {detections.length} objects in this window
              </p>

              {detections.map((det, index) => {
                const colorSchema = getClassColor(det.class_id);
                return (
                  <div
                    key={index}
                    className="p-3 bg-slate-900/60 hover:bg-slate-800/40 rounded-lg border transition-all duration-200 flex flex-col gap-1 border-slate-800/80 hover:border-slate-700/80"
                  >
                    <div className="flex justify-between items-center">
                      <span
                        className={`text-sm font-semibold ${colorSchema.text}`}
                      >
                        {det.class_name}
                      </span>
                      <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-slate-850 text-slate-300 border border-slate-800">
                        {(det.confidence * 100).toFixed(0)}%
                      </span>
                    </div>
                    <div className="text-[10px] text-slate-500 font-mono flex justify-between">
                      <span>BBox:</span>
                      <span>
                        [{det.global_bbox.map((v) => v.toFixed(0)).join(', ')}]
                      </span>
                    </div>
                  </div>
                );
              })}

              {detections.length === 0 && (
                <div className="text-center py-12 text-sm text-slate-500">
                  No objects detected.
                </div>
              )}
            </div>
          )}
        </aside>
      </div>
    </main>
  );
}
