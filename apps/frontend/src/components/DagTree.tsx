'use client';

import React, { useRef, useState, useMemo, useEffect } from 'react';
import { useExplorationStore, getBboxKey } from '../store/explorationStore';
import type { RegionNode, Detection } from '@recursive-object-detector/types';
import { Download, ZoomIn, ZoomOut, Maximize } from 'lucide-react';

interface LayoutNode {
  id: string;
  type: 'root' | 'region' | 'detection' | 'sub_detection';
  label: string;
  subLabel?: string;
  x: number;
  y: number;
  data: any;
  parentId: string | null;
}

interface LayoutLink {
  from: string;
  to: string;
}

interface GraphNode {
  id: string;
  type: 'root' | 'region' | 'detection' | 'sub_detection';
  label: string;
  subLabel?: string;
  data: any;
  children: GraphNode[];
}

export default function DagTree() {
  const { explorationTree, currentNode, selectedDetection, navigateToNode, selectDetection } = useExplorationStore();
  
  const svgRef = useRef<SVGSVGElement>(null);
  
  // Pan and Zoom state
  const [zoom, setZoom] = useState(0.85);
  const [pan, setPan] = useState({ x: 50, y: 30 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });

  // 1. Build uniform graph tree recursively
  const graphTree = useMemo(() => {
    if (!explorationTree) return null;

    const buildGraph = (node: RegionNode): GraphNode => {
      const isRoot = node.zoom_level === 0;
      const nodeName = isRoot ? 'Root Image' : `Region (Lvl ${node.zoom_level})`;
      const nodeBboxStr = `[${node.global_bbox.map(v => Math.round(v)).join(', ')}]`;

      const childrenNodes: GraphNode[] = [];

      // Add detections as child nodes of this RegionNode
      if (node.detections) {
        node.detections.forEach((det, idx) => {
          const detBboxKey = getBboxKey(det.global_bbox);
          const detId = `det_${node.node_id}_${idx}_${detBboxKey}`;
          const subDets: GraphNode[] = [];

          if (det.sub_detections) {
            det.sub_detections.forEach((sub, sIdx) => {
              const subBboxKey = getBboxKey(sub.global_bbox);
              const subId = `sub_${detId}_${sIdx}_${subBboxKey}`;
              
              let subLabel = sub.class_name;
              if (sub.class_name?.toLowerCase().includes('digit_')) {
                subLabel = `Digit ${sub.class_name.split('_')[1]}`;
              }

              subDets.push({
                id: subId,
                type: 'sub_detection',
                label: subLabel,
                subLabel: `Conf: ${Math.round(sub.confidence * 100)}%`,
                data: sub,
                children: []
              });
            });
          }

          childrenNodes.push({
            id: detId,
            type: 'detection',
            label: det.class_name,
            subLabel: `Conf: ${Math.round(det.confidence * 100)}%`,
            data: det,
            children: subDets
          });
        });
      }

      // Add spatial children
      if (node.children) {
        node.children.forEach((child) => {
          childrenNodes.push(buildGraph(child));
        });
      }

      return {
        id: node.node_id,
        type: isRoot ? 'root' : 'region',
        label: nodeName,
        subLabel: nodeBboxStr,
        data: node,
        children: childrenNodes
      };
    };

    return buildGraph(explorationTree);
  }, [explorationTree]);

  // 2. Compute 2D coordinates for nodes & connections
  const { nodes, links, bounds } = useMemo(() => {
    const layoutNodes: LayoutNode[] = [];
    const layoutLinks: LayoutLink[] = [];
    const leafState = { count: 0 };

    if (!graphTree) {
      return { nodes: [], links: [], bounds: { width: 800, height: 600 } };
    }

    const colWidth = 280;
    const rowHeight = 70;

    const computeLayout = (
      node: GraphNode,
      parentId: string | null,
      depth: number
    ): number => {
      const x = depth * colWidth + 60;

      if (node.children.length === 0) {
        const y = leafState.count * rowHeight + 60;
        leafState.count += 1;

        layoutNodes.push({
          id: node.id,
          type: node.type,
          label: node.label,
          subLabel: node.subLabel,
          x,
          y,
          data: node.data,
          parentId
        });

        return y;
      } else {
        const childYCoords: number[] = [];
        node.children.forEach((child) => {
          layoutLinks.push({ from: node.id, to: child.id });
          const cy = computeLayout(child, node.id, depth + 1);
          childYCoords.push(cy);
        });

        const minChildY = Math.min(...childYCoords);
        const maxChildY = Math.max(...childYCoords);
        const y = (minChildY + maxChildY) / 2;

        layoutNodes.push({
          id: node.id,
          type: node.type,
          label: node.label,
          subLabel: node.subLabel,
          x,
          y,
          data: node.data,
          parentId
        });

        return y;
      }
    };

    computeLayout(graphTree, null, 0);

    // Find layout bounds
    let maxX = 100;
    let maxY = 100;
    layoutNodes.forEach((n) => {
      if (n.x > maxX) maxX = n.x;
      if (n.y > maxY) maxY = n.y;
    });

    return {
      nodes: layoutNodes,
      links: layoutLinks,
      bounds: {
        width: maxX + 280,
        height: maxY + 100
      }
    };
  }, [graphTree]);

  // Center view on mount or reset
  const resetView = () => {
    if (bounds.height > 0) {
      setZoom(0.8);
      setPan({ x: 40, y: 20 });
    }
  };

  useEffect(() => {
    resetView();
  }, [bounds.height]);

  // Pan and zoom event handlers
  const handleMouseDown = (e: React.MouseEvent) => {
    // Only drag on left click on canvas background, not on interactive nodes
    const target = e.target as SVGElement;
    if (target.closest('.interactive-node')) return;
    
    setIsDragging(true);
    setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging) return;
    setPan({
      x: e.clientX - dragStart.x,
      y: e.clientY - dragStart.y
    });
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const zoomFactor = 1.1;
    const newZoom = e.deltaY < 0 ? zoom * zoomFactor : zoom / zoomFactor;
    setZoom(Math.max(0.2, Math.min(2.5, newZoom)));
  };

  const zoomIn = () => setZoom(z => Math.min(2.5, z * 1.2));
  const zoomOut = () => setZoom(z => Math.max(0.2, z / 1.2));

  // Determine active highlights
  const isNodeActive = (n: LayoutNode): boolean => {
    if (n.type === 'root' || n.type === 'region') {
      return currentNode?.node_id === n.id;
    }
    if (n.type === 'detection' || n.type === 'sub_detection') {
      return !!(selectedDetection && getBboxKey(selectedDetection.global_bbox) === getBboxKey(n.data.global_bbox));
    }
    return false;
  };

  const handleNodeClick = (n: LayoutNode) => {
    if (n.type === 'root' || n.type === 'region') {
      navigateToNode(n.data as RegionNode);
    } else {
      selectDetection(n.data as Detection);
    }
  };

  // Node Color styles for the DAG UI matching paper requirements
  const getNodeStyles = (type: string, isActive: boolean) => {
    let stroke = '#475569';
    let fill = 'url(#bgGradDefault)';
    let glow = 'none';
    let textTitle = '#f1f5f9';
    let textSub = '#94a3b8';

    if (type === 'root') {
      stroke = isActive ? '#a855f7' : '#6366f1';
      fill = 'url(#bgGradRoot)';
      glow = isActive ? 'url(#glowPurpleActive)' : 'url(#glowPurple)';
    } else if (type === 'region') {
      stroke = isActive ? '#3b82f6' : '#1e3a8a';
      fill = 'url(#bgGradRegion)';
      glow = isActive ? 'url(#glowBlueActive)' : 'none';
    } else if (type === 'detection') {
      stroke = isActive ? '#fbbf24' : '#b45309';
      fill = 'url(#bgGradDet)';
      glow = isActive ? 'url(#glowAmberActive)' : 'none';
    } else if (type === 'sub_detection') {
      stroke = isActive ? '#39ff14' : '#047857';
      fill = 'url(#bgGradSubDet)';
      glow = isActive ? 'url(#glowGreenActive)' : 'none';
      
      // Warm glows for digits/lamps based on classification
      const name = String(type).toLowerCase();
    }

    return { stroke, fill, glow, textTitle, textSub };
  };

  // 3. Export to Standalone SVG for Academic Inclusion
  const handleExport = () => {
    if (!svgRef.current) return;
    
    // Create a deep copy of the SVG DOM
    const originalSvg = svgRef.current;
    const clonedSvg = originalSvg.cloneNode(true) as SVGSVGElement;
    
    // Remove the interactive zoom/pan transformations to fit page boundaries in paper
    // Or set viewport explicitly
    clonedSvg.setAttribute('width', bounds.width.toString());
    clonedSvg.setAttribute('height', bounds.height.toString());
    
    // Find the main group inside cloned SVG and reset zoom transform for printing
    const mainGroup = clonedSvg.querySelector('#dag-main-group');
    if (mainGroup) {
      mainGroup.removeAttribute('transform');
    }
    
    // Inline necessary font/styles
    clonedSvg.setAttribute('style', 'background-color: #020617; font-family: ui-sans-serif, system-ui, sans-serif;');
    
    const svgSerializer = new XMLSerializer();
    let svgString = svgSerializer.serializeToString(clonedSvg);
    
    // Create download blob
    const blob = new Blob([svgString], { type: 'image/svg+xml;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `DAG_Hierarchy_Visualizer_${new Date().toISOString().split('T')[0]}.svg`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="relative w-full h-full flex flex-col bg-slate-950 rounded-xl border border-slate-800 shadow-2xl overflow-hidden min-h-[600px]">
      
      {/* Canvas Toolbars */}
      <div className="absolute top-4 right-4 z-30 flex items-center gap-2">
        <button
          onClick={zoomIn}
          className="p-2 bg-slate-900/80 hover:bg-slate-800 text-slate-300 rounded-lg border border-slate-800 hover:border-slate-700 backdrop-blur-sm transition-all cursor-pointer flex items-center justify-center shadow-lg"
          title="Zoom In"
        >
          <ZoomIn className="w-4 h-4" />
        </button>
        <button
          onClick={zoomOut}
          className="p-2 bg-slate-900/80 hover:bg-slate-800 text-slate-300 rounded-lg border border-slate-800 hover:border-slate-700 backdrop-blur-sm transition-all cursor-pointer flex items-center justify-center shadow-lg"
          title="Zoom Out"
        >
          <ZoomOut className="w-4 h-4" />
        </button>
        <button
          onClick={resetView}
          className="p-2 bg-slate-900/80 hover:bg-slate-800 text-slate-300 rounded-lg border border-slate-800 hover:border-slate-700 backdrop-blur-sm transition-all cursor-pointer flex items-center justify-center shadow-lg"
          title="Recenter"
        >
          <Maximize className="w-4 h-4" />
        </button>
        <button
          onClick={handleExport}
          className="ml-2 px-3 py-2 bg-blue-600 hover:bg-blue-500 text-white text-xs font-semibold rounded-lg shadow-lg flex items-center gap-1.5 transition-all cursor-pointer border border-blue-500/20"
          title="Export Standalone High-Res SVG for paper"
        >
          <Download className="w-3.5 h-3.5" />
          <span>Export SVG</span>
        </button>
      </div>

      <div className="absolute top-4 left-4 z-30 pointer-events-none">
        <div className="bg-slate-900/85 backdrop-blur-sm px-3 py-1.5 rounded-lg border border-slate-850 shadow-md">
          <p className="text-[10px] uppercase font-bold tracking-widest text-indigo-400">Payload Graph</p>
          <h4 className="text-xs font-bold text-slate-200 mt-0.5">Directed Acyclic Tree Representation</h4>
        </div>
      </div>

      {/* SVG Canvas Area */}
      <div 
        className="flex-1 w-full h-full overflow-hidden cursor-grab active:cursor-grabbing"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onWheel={handleWheel}
      >
        <svg
          ref={svgRef}
          className="w-full h-full select-none"
          style={{ backgroundColor: '#020617' }}
        >
          {/* Defs for gradients, filters, shadows */}
          <defs>
            {/* Gradients */}
            <linearGradient id="bgGradDefault" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0%" stopColor="#1e293b" />
              <stop offset="100%" stopColor="#0f172a" />
            </linearGradient>
            <linearGradient id="bgGradRoot" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0%" stopColor="#2e1065" stopOpacity="0.85" />
              <stop offset="100%" stopColor="#1e1b4b" stopOpacity="0.85" />
            </linearGradient>
            <linearGradient id="bgGradRegion" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0%" stopColor="#172554" stopOpacity="0.85" />
              <stop offset="100%" stopColor="#0f172a" stopOpacity="0.85" />
            </linearGradient>
            <linearGradient id="bgGradDet" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0%" stopColor="#451a03" stopOpacity="0.85" />
              <stop offset="100%" stopColor="#1c0d02" stopOpacity="0.85" />
            </linearGradient>
            <linearGradient id="bgGradSubDet" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0%" stopColor="#022c22" stopOpacity="0.85" />
              <stop offset="100%" stopColor="#02140d" stopOpacity="0.85" />
            </linearGradient>

            {/* Glowing active outline shadow filters */}
            <filter id="glowPurple" x="-30%" y="-30%" width="160%" height="160%">
              <feGaussianBlur stdDeviation="3" result="blur" />
              <feComponentTransfer in="blur" result="glow">
                <feFuncA type="linear" slope="0.4" />
              </feComponentTransfer>
              <feMerge>
                <feMergeNode in="glow" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
            <filter id="glowPurpleActive" x="-40%" y="-40%" width="180%" height="180%">
              <feGaussianBlur stdDeviation="8" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
            <filter id="glowBlueActive" x="-40%" y="-40%" width="180%" height="180%">
              <feGaussianBlur stdDeviation="6" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
            <filter id="glowAmberActive" x="-40%" y="-40%" width="180%" height="180%">
              <feGaussianBlur stdDeviation="6" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
            <filter id="glowGreenActive" x="-40%" y="-40%" width="180%" height="180%">
              <feGaussianBlur stdDeviation="6" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>

          {/* Main Transformation Group */}
          <g 
            id="dag-main-group"
            transform={`translate(${pan.x}, ${pan.y}) scale(${zoom})`}
          >
            {/* Draw links first, so nodes render on top */}
            <g id="dag-links-layer">
              {links.map((link, idx) => {
                const parentNode = nodes.find(n => n.id === link.from);
                const childNode = nodes.find(n => n.id === link.to);
                
                if (!parentNode || !childNode) return null;

                // Bezier curve calculations for smooth connection
                const startX = parentNode.x + 220; // output port
                const startY = parentNode.y + 25;
                const endX = childNode.x; // input port
                const endY = childNode.y + 25;
                
                const controlX = (startX + endX) / 2;

                const pathData = `M ${startX} ${startY} C ${controlX} ${startY}, ${controlX} ${endY}, ${endX} ${endY}`;
                
                // Color code link based on parent's type
                let strokeColor = 'rgba(71, 85, 105, 0.4)';
                let glowColor = 'none';

                if (parentNode.type === 'root') {
                  strokeColor = 'rgba(99, 102, 241, 0.4)';
                } else if (parentNode.type === 'region') {
                  strokeColor = 'rgba(59, 130, 246, 0.4)';
                } else if (parentNode.type === 'detection') {
                  strokeColor = 'rgba(245, 158, 11, 0.4)';
                }

                // If currently traversing or active, highlight the connection
                const isConnectionHighlighted = isNodeActive(parentNode) && isNodeActive(childNode);
                if (isConnectionHighlighted) {
                  strokeColor = parentNode.type === 'root' ? '#818cf8' : 
                                parentNode.type === 'region' ? '#60a5fa' : '#fbbf24';
                  glowColor = strokeColor;
                }

                return (
                  <g key={`link_${idx}`}>
                    {/* Shadow glow for highlighted links */}
                    {glowColor !== 'none' && (
                      <path
                        d={pathData}
                        fill="none"
                        stroke={glowColor}
                        strokeWidth="4"
                        strokeOpacity="0.4"
                        className="transition-all duration-300"
                        style={{ filter: 'blur(3px)' }}
                      />
                    )}
                    <path
                      d={pathData}
                      fill="none"
                      stroke={strokeColor}
                      strokeWidth={isConnectionHighlighted ? '2.5' : '1.5'}
                      strokeDasharray={parentNode.type === 'region' ? '4,4' : 'none'}
                      className="transition-all duration-300"
                    />
                  </g>
                );
              })}
            </g>

            {/* Draw Nodes */}
            <g id="dag-nodes-layer">
              {nodes.map((node) => {
                const isActive = isNodeActive(node);
                const { stroke, fill, glow, textTitle, textSub } = getNodeStyles(node.type, isActive);
                
                // Card dimensions
                const cardWidth = 220;
                const cardHeight = 50;

                // Icon drawing helpers
                const renderIcon = () => {
                  if (node.type === 'root') {
                    // Home/Image icon
                    return (
                      <g fill="none" stroke={isActive ? '#a855f7' : '#818cf8'} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                        <rect x="5" y="5" width="14" height="14" rx="2" />
                        <circle cx="9" cy="9" r="2" />
                        <path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21" />
                      </g>
                    );
                  }
                  if (node.type === 'region') {
                    // Folder/Grid icon
                    return (
                      <g fill="none" stroke={isActive ? '#3b82f6' : '#60a5fa'} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M4 20h16a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.93a2 2 0 0 1-1.66-.9l-.82-1.2A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2Z" />
                      </g>
                    );
                  }
                  if (node.type === 'detection') {
                    // Bounding Box target
                    return (
                      <g fill="none" stroke={isActive ? '#fbbf24' : '#f59e0b'} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                        <circle cx="12" cy="12" r="10" />
                        <circle cx="12" cy="12" r="6" />
                        <circle cx="12" cy="12" r="2" />
                      </g>
                    );
                  }
                  // Sub-detection
                  return (
                    <g fill="none" stroke={isActive ? '#39ff14' : '#10b981'} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6" />
                    </g>
                  );
                };

                return (
                  <g
                    key={node.id}
                    transform={`translate(${node.x}, ${node.y})`}
                    className="interactive-node cursor-pointer group"
                    onClick={() => handleNodeClick(node)}
                  >
                    {/* Node Glowing effect under outline */}
                    {glow !== 'none' && (
                      <rect
                        width={cardWidth}
                        height={cardHeight}
                        rx="8"
                        ry="8"
                        fill="none"
                        stroke={stroke}
                        strokeWidth="5"
                        strokeOpacity="0.4"
                        style={{ filter: glow }}
                      />
                    )}

                    {/* Node Box */}
                    <rect
                      width={cardWidth}
                      height={cardHeight}
                      rx="8"
                      ry="8"
                      fill={fill}
                      stroke={stroke}
                      strokeWidth={isActive ? '2' : '1.2'}
                      className="transition-all duration-200 group-hover:stroke-slate-400"
                    />

                    {/* Left Border Accent strip */}
                    <path
                      d={`M 1.5 8 A 6.5 6.5 0 0 1 8 1.5 L 8 48.5 A 6.5 6.5 0 0 1 1.5 42 Z`}
                      fill={stroke}
                    />

                    {/* Icon Container */}
                    <g transform="translate(14, 13)">
                      {/* background circle for icon */}
                      <circle cx="12" cy="12" r="13" fill="#0f172a" stroke="rgba(255,255,255,0.05)" strokeWidth="1" />
                      <g transform="translate(0, 0)">
                        {renderIcon()}
                      </g>
                    </g>

                    {/* Labels */}
                    <text
                      x="50"
                      y="22"
                      fill={textTitle}
                      fontSize="11"
                      fontWeight="600"
                      fontFamily="ui-sans-serif, system-ui, sans-serif"
                    >
                      {node.label}
                    </text>

                    {node.subLabel && (
                      <text
                        x="50"
                        y="36"
                        fill={textSub}
                        fontSize="9"
                        fontFamily="ui-sans-serif, system-ui, sans-serif"
                        className="font-mono text-slate-500"
                      >
                        {node.subLabel}
                      </text>
                    )}

                    {/* active bubble indicator */}
                    {isActive && (
                      <circle
                        cx={cardWidth - 12}
                        cy="25"
                        r="4"
                        fill={stroke}
                        className="animate-pulse"
                      />
                    )}
                  </g>
                );
              })}
            </g>
          </g>
        </svg>
      </div>

      {/* Footer Info */}
      <div className="bg-slate-950 px-4 py-3 border-t border-slate-800 flex items-center justify-between text-[10px] text-slate-500 font-mono">
        <div className="flex gap-4">
          <span className="flex items-center gap-1">
            <span className="w-2.5 h-2.5 rounded bg-indigo-500 border border-indigo-400 inline-block" />
            Root
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2.5 h-2.5 rounded border border-blue-500 border-dashed inline-block" style={{ backgroundColor: '#172554' }} />
            Region Node
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2.5 h-2.5 rounded bg-amber-950 border border-amber-500 inline-block" />
            Detection Node
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2.5 h-2.5 rounded bg-emerald-950 border border-emerald-500 inline-block" />
            Detail Node
          </span>
        </div>
        <div>
          Nodes: {nodes.length} | Links: {links.length}
        </div>
      </div>
    </div>
  );
}
