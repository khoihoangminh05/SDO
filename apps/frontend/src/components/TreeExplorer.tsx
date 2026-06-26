import React from 'react';
import { useExplorationStore, getBboxKey } from '../store/explorationStore';
import type { RegionNode } from '@recursive-object-detector/types';
import {
  ChevronDown,
  ChevronRight,
  Folder,
  FolderOpen,
  Target,
  Eye,
  Layers,
  CircleDot,
} from 'lucide-react';

interface TreeNodeProps {
  node: RegionNode;
  parent: RegionNode | null;
}

interface DetectionNodeProps {
  det: any;
  label: string;
}

const DetectionNode: React.FC<DetectionNodeProps> = ({ det, label }) => {
  const { selectedDetection, selectDetection } = useExplorationStore();
  const [expanded, setExpanded] = React.useState(true);

  const isSelected = selectedDetection && getBboxKey(selectedDetection.global_bbox) === getBboxKey(det.global_bbox);
  const hasSubDetections = det.sub_detections && det.sub_detections.length > 0;

  const handleClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    selectDetection(det);
  };

  const handleToggle = (e: React.MouseEvent) => {
    e.stopPropagation();
    setExpanded(!expanded);
  };

  return (
    <div className="pl-3 select-none">
      <div
        onClick={handleClick}
        className={`flex items-center gap-1.5 py-1 px-2 rounded-lg cursor-pointer transition-colors duration-150 group text-xs ${
          isSelected
            ? 'bg-yellow-600/90 text-white font-medium shadow-md shadow-yellow-500/10'
            : 'text-slate-300 hover:bg-slate-800/40 hover:text-white'
        }`}
      >
        {hasSubDetections ? (
          <button
            onClick={handleToggle}
            className="p-0.5 hover:bg-slate-700/50 rounded transition-colors"
          >
            {expanded ? (
              <ChevronDown className="w-3 h-3 text-slate-400 group-hover:text-slate-200" />
            ) : (
              <ChevronRight className="w-3 h-3 text-slate-400 group-hover:text-slate-200" />
            )}
          </button>
        ) : (
          <span className="w-4" />
        )}

        {hasSubDetections ? (
          <Layers className={`w-3.5 h-3.5 ${isSelected ? 'text-white' : 'text-yellow-400'}`} />
        ) : (
          <CircleDot className={`w-3.5 h-3.5 ${isSelected ? 'text-white' : 'text-cyan-400'}`} />
        )}

        <span className="truncate flex-1">{label}</span>

        {isSelected && (
          <Eye className="w-3 h-3 text-white/80 animate-pulse" />
        )}
      </div>

      {hasSubDetections && expanded && (
        <div className="mt-0.5 border-l border-slate-800 ml-2 pl-1.5 flex flex-col gap-0.5">
          {det.sub_detections.map((sub: any, idx: number) => {
            let subLabel = sub.class_name || `Detail ${idx + 1}`;
            if (sub.class_name?.toLowerCase().includes('digit_')) {
              subLabel = `Digit ${sub.class_name.split('_')[1]}`;
            } else if (sub.class_name?.toLowerCase().includes('light')) {
              subLabel = `${sub.class_name}`;
            }
            return (
              <DetectionNode
                key={`sub-node-${idx}-${getBboxKey(sub.global_bbox)}`}
                det={sub}
                label={subLabel}
              />
            );
          })}
        </div>
      )}
    </div>
  );
};

const TreeNode: React.FC<TreeNodeProps> = ({ node, parent }) => {
  const { currentNode, navigateToNode } = useExplorationStore();
  const [expanded, setExpanded] = React.useState(true);

  const isActive = currentNode?.node_id === node.node_id;

  const label = React.useMemo(() => {
    if (node.zoom_level === 0) return 'Root Image';
    if (parent) {
      const match = parent.detections.find((d) => {
        const cx = (d.global_bbox[0] + d.global_bbox[2]) / 2;
        const cy = (d.global_bbox[1] + d.global_bbox[3]) / 2;
        return (
          cx >= node.global_bbox[0] &&
          cx <= node.global_bbox[2] &&
          cy >= node.global_bbox[1] &&
          cy <= node.global_bbox[3]
        );
      });
      if (match) return `${match.class_name} (Lvl ${node.zoom_level})`;
    }
    return `Region Lvl ${node.zoom_level}`;
  }, [node, parent]);

  const detectionsList = React.useMemo(() => {
    if (!node.detections) return [];
    const counts: Record<string, number> = {};
    return node.detections.map((det) => {
      let rawName = det.class_name || 'Object';
      let formattedName = rawName
        .split(' ')
        .map((w: string) => w.charAt(0).toUpperCase() + w.slice(1))
        .join(' ');

      if (formattedName.toLowerCase() === 'traffic light') {
        formattedName = 'Traffic Light';
      }

      counts[formattedName] = (counts[formattedName] || 0) + 1;
      const label = `${formattedName} ${counts[formattedName]}`;
      return { det, label };
    });
  }, [node.detections]);

  const hasChildren = node.children && node.children.length > 0;
  const hasDetections = node.detections && node.detections.length > 0;
  const hasExpandableContent = hasChildren || hasDetections;

  const handleClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    navigateToNode(node);
  };

  const handleToggle = (e: React.MouseEvent) => {
    e.stopPropagation();
    setExpanded(!expanded);
  };

  return (
    <div className="pl-3 select-none">
      <div
        onClick={handleClick}
        className={`flex items-center gap-1.5 py-1.5 px-2 rounded-lg cursor-pointer transition-colors duration-150 group text-sm ${
          isActive
            ? 'bg-blue-600/90 text-white font-medium shadow-md shadow-blue-500/10'
            : 'text-slate-300 hover:bg-slate-800/60 hover:text-white'
        }`}
      >
        {hasExpandableContent ? (
          <button
            onClick={handleToggle}
            className="p-0.5 hover:bg-slate-700/50 rounded transition-colors"
          >
            {expanded ? (
              <ChevronDown className="w-3.5 h-3.5 text-slate-400 group-hover:text-slate-200" />
            ) : (
              <ChevronRight className="w-3.5 h-3.5 text-slate-400 group-hover:text-slate-200" />
            )}
          </button>
        ) : (
          <span className="w-4.5" />
        )}

        {hasChildren ? (
          expanded ? (
            <FolderOpen
              className={`w-4 h-4 ${isActive ? 'text-white' : 'text-blue-400'}`}
            />
          ) : (
            <Folder
              className={`w-4 h-4 ${isActive ? 'text-white' : 'text-blue-400'}`}
            />
          )
        ) : (
          <Target
            className={`w-4 h-4 ${isActive ? 'text-white' : 'text-teal-400'}`}
          />
        )}

        <span className="truncate flex-1">{label}</span>

        {isActive && (
          <Eye className="w-3.5 h-3.5 text-white/80 animate-pulse" />
        )}
      </div>

      {expanded && hasExpandableContent && (
        <div className="mt-1 border-l border-slate-800 ml-2.5 flex flex-col gap-0.5">
          {/* First render detections belonging to this region */}
          {detectionsList.map(({ det, label }, idx) => (
            <DetectionNode
              key={`det-node-${idx}-${getBboxKey(det.global_bbox)}`}
              det={det}
              label={label}
            />
          ))}

          {/* Then render sub-regions */}
          {node.children?.map((child) => (
            <TreeNode key={child.node_id} node={child} parent={node} />
          ))}
        </div>
      )}
    </div>
  );
};

export const TreeExplorer: React.FC = () => {
  const { explorationTree } = useExplorationStore();

  if (!explorationTree) {
    return (
      <div className="text-center text-xs text-slate-500 py-8">
        No active exploration tree.
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-1 pr-1">
      <TreeNode node={explorationTree} parent={null} />
    </div>
  );
};
