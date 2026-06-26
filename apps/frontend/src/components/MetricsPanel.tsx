'use client';

import React, { useMemo } from 'react';
import { useExplorationStore } from '../store/explorationStore';
import { Zap, Layers, Target, HelpCircle } from 'lucide-react';
import type { RegionNode } from '@recursive-object-detector/types';

export default function MetricsPanel() {
  const { explorationTree } = useExplorationStore();

  const metrics = useMemo(() => {
    if (!explorationTree) {
      return {
        totalRegions: 0,
        flopsSaved: 0,
        avgConfLvl1: 0,
        avgConfLvl2: 0,
        avgConfLvl3: 0,
      };
    }

    // 1. Count total region nodes processed (each is an inference crop)
    const countRegionNodes = (node: RegionNode): number => {
      let count = 1;
      if (node.children) {
        node.children.forEach((child) => {
          count += countRegionNodes(child);
        });
      }
      return count;
    };

    const totalRegions = countRegionNodes(explorationTree);

    // 2. Compute FLOPs Saved vs Standard Grid Slicing
    // Standard Grid search at Lvl 1: 12 slices, Lvl 2: 48 slices, Lvl 3: 192 slices = 252 theoretical slices
    const theoreticalSlices = 252;
    const flopsSaved = Math.max(
      0,
      Math.min(99.9, (1 - totalRegions / theoreticalSlices) * 100)
    );

    // 3. Calculate Average Confidence per Level
    const levels: { 1: number[]; 2: number[]; 3: number[] } = {
      1: [],
      2: [],
      3: [],
    };

    const collectConfidenceByLevel = (node: RegionNode) => {
      if (node.detections) {
        node.detections.forEach((det) => {
          if (node.zoom_level === 0) {
            // Level 1: Root detections (Intersection / vehicle crops)
            levels[1].push(det.confidence);
          } else if (node.zoom_level === 1) {
            // Level 2: Traffic Light Pole detections
            levels[2].push(det.confidence);
          }

          // Level 3: Nested details (Lamps and easyOCR Digit readouts)
          if (det.sub_detections) {
            det.sub_detections.forEach((sub) => {
              levels[3].push(sub.confidence);
            });
          }
        });
      }

      if (node.children) {
        node.children.forEach((child) => {
          collectConfidenceByLevel(child);
        });
      }
    };

    collectConfidenceByLevel(explorationTree);

    const calculateAverage = (arr: number[]): number => {
      if (arr.length === 0) return 0;
      const sum = arr.reduce((a, b) => a + b, 0);
      return sum / arr.length;
    };

    return {
      totalRegions,
      flopsSaved,
      avgConfLvl1: calculateAverage(levels[1]),
      avgConfLvl2: calculateAverage(levels[2]),
      avgConfLvl3: calculateAverage(levels[3]),
    };
  }, [explorationTree]);

  const stats = [
    {
      label: 'Regions Processed',
      value: metrics.totalRegions.toString(),
      subtext: 'Targeted Inference Windows',
      icon: <Layers className="w-5 h-5 text-blue-400" />,
      color: 'from-blue-500/10 to-indigo-500/5',
      borderColor: 'border-blue-500/20',
      glow: 'shadow-blue-500/5',
    },
    {
      label: 'FLOPs Saved vs Grid Slicing',
      value: `${metrics.flopsSaved.toFixed(1)}%`,
      subtext: 'Quadtree Space Reduction',
      icon: <Zap className="w-5 h-5 text-amber-400" />,
      color: 'from-amber-500/10 to-orange-500/5',
      borderColor: 'border-amber-500/20',
      glow: 'shadow-amber-500/5',
      extra: (
        <div className="w-full bg-slate-800 rounded-full h-1.5 mt-2 overflow-hidden">
          <div
            className="bg-gradient-to-r from-amber-500 to-orange-400 h-1.5 rounded-full transition-all duration-500"
            style={{ width: `${metrics.flopsSaved}%` }}
          />
        </div>
      ),
    },
    {
      label: 'Avg Confidence (Lvl 1)',
      value: metrics.avgConfLvl1 > 0 ? `${(metrics.avgConfLvl1 * 100).toFixed(1)}%` : '-',
      subtext: 'Intersection Context',
      icon: <Target className="w-5 h-5 text-indigo-400" />,
      color: 'from-indigo-500/10 to-purple-500/5',
      borderColor: 'border-indigo-500/20',
      glow: 'shadow-indigo-500/5',
    },
    {
      label: 'Avg Confidence (Lvl 2)',
      value: metrics.avgConfLvl2 > 0 ? `${(metrics.avgConfLvl2 * 100).toFixed(1)}%` : '-',
      subtext: 'Traffic Light Poles',
      icon: <Target className="w-5 h-5 text-cyan-400" />,
      color: 'from-cyan-500/10 to-blue-500/5',
      borderColor: 'border-cyan-500/20',
      glow: 'shadow-cyan-500/5',
    },
    {
      label: 'Avg Confidence (Lvl 3)',
      value: metrics.avgConfLvl3 > 0 ? `${(metrics.avgConfLvl3 * 100).toFixed(1)}%` : '-',
      subtext: 'LED Digits & Lamps',
      icon: <Target className="w-5 h-5 text-emerald-400" />,
      color: 'from-emerald-500/10 to-teal-500/5',
      borderColor: 'border-emerald-500/20',
      glow: 'shadow-emerald-500/5',
    },
  ];

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4 mb-6">
      {stats.map((stat, i) => (
        <div
          key={i}
          className={`bg-slate-900/40 backdrop-blur-md border ${stat.borderColor} ${stat.glow} p-4 rounded-xl flex flex-col justify-between shadow-lg hover:border-slate-700/60 transition-all duration-200 group`}
        >
          <div className="flex items-start justify-between">
            <div>
              <p className="text-[10px] uppercase font-bold tracking-wider text-slate-500 group-hover:text-slate-400 transition-colors">
                {stat.label}
              </p>
              <h3 className="text-2xl font-extrabold text-white mt-1 tracking-tight">
                {stat.value}
              </h3>
            </div>
            <div className={`p-2 bg-slate-950/60 rounded-lg border ${stat.borderColor}`}>
              {stat.icon}
            </div>
          </div>
          <div className="mt-3">
            <span className="text-[10px] font-mono text-slate-500">
              {stat.subtext}
            </span>
            {stat.extra}
          </div>
        </div>
      ))}
    </div>
  );
}
