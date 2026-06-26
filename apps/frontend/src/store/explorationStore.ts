import { create } from 'zustand';
import type { RegionNode, BoundingBox, Detection } from '@recursive-object-detector/types';

export const getApiUrl = (path: string): string => {
  if (typeof window !== 'undefined') {
    const hostname = window.location.hostname;
    const targetHost = hostname === 'localhost' ? '127.0.0.1' : hostname;
    return `http://${targetHost}:3000${path}`;
  }
  return `http://127.0.0.1:3000${path}`;
};


interface ExplorationState {
  rootImageId: string;
  explorationTree: RegionNode | null;
  currentNode: RegionNode | null;
  breadcrumbs: RegionNode[];
  loadingBboxes: Record<string, boolean>;
  selectedDetection: Detection | null;
  showHeatmap: boolean;

  initialize: (rootNode: RegionNode) => void;
  exploreRegion: (
    imageId: string,
    zoomLevel: number,
    globalBbox: BoundingBox,
  ) => Promise<void>;
  navigateToNode: (node: RegionNode) => void;
  selectDetection: (detection: Detection | null) => void;
  toggleHeatmap: () => void;
  reset: () => void;
}

export const getBboxKey = (bbox: BoundingBox): string => {
  return `${bbox[0].toFixed(6)}_${bbox[1].toFixed(6)}_${bbox[2].toFixed(6)}_${bbox[3].toFixed(6)}`;
};

export interface ClassColorSchema {
  border: string;
  bg: string;
  hoverBorder: string;
  hoverBg: string;
  text: string;
  labelBg: string;
}

const colors: ClassColorSchema[] = [
  {
    border: 'border-red-500',
    bg: 'bg-red-500/10',
    hoverBorder: 'hover:border-red-400',
    hoverBg: 'hover:bg-red-500/20',
    text: 'text-red-400',
    labelBg: 'bg-red-600',
  },
  {
    border: 'border-blue-500',
    bg: 'bg-blue-500/10',
    hoverBorder: 'hover:border-blue-400',
    hoverBg: 'hover:bg-blue-500/20',
    text: 'text-blue-400',
    labelBg: 'bg-blue-600',
  },
  {
    border: 'border-emerald-500',
    bg: 'bg-emerald-500/10',
    hoverBorder: 'hover:border-emerald-400',
    hoverBg: 'hover:bg-emerald-500/20',
    text: 'text-emerald-400',
    labelBg: 'bg-emerald-600',
  },
  {
    border: 'border-purple-500',
    bg: 'bg-purple-500/10',
    hoverBorder: 'hover:border-purple-400',
    hoverBg: 'hover:bg-purple-500/20',
    text: 'text-purple-400',
    labelBg: 'bg-purple-600',
  },
  {
    border: 'border-amber-500',
    bg: 'bg-amber-500/10',
    hoverBorder: 'hover:border-amber-400',
    hoverBg: 'hover:bg-amber-500/20',
    text: 'text-amber-400',
    labelBg: 'bg-amber-600',
  },
  {
    border: 'border-pink-500',
    bg: 'bg-pink-500/10',
    hoverBorder: 'hover:border-pink-400',
    hoverBg: 'hover:bg-pink-500/20',
    text: 'text-pink-400',
    labelBg: 'bg-pink-600',
  },
  {
    border: 'border-cyan-500',
    bg: 'bg-cyan-500/10',
    hoverBorder: 'hover:border-cyan-400',
    hoverBg: 'hover:bg-cyan-500/20',
    text: 'text-cyan-400',
    labelBg: 'bg-cyan-600',
  },
];

export const getClassColor = (classId: number): ClassColorSchema => {
  return colors[classId % colors.length];
};

// The backend now provides the hierarchical children tree. 
// We just return the node as is to respect the backend's logic.
export const mapDetectionsToChildren = (node: RegionNode): RegionNode => {
  return node;
};

// Recursive helper to find a node by its ID in the tree
const findNodeInTree = (node: RegionNode, nodeId: string): RegionNode | null => {
  if (node.node_id === nodeId) return node;
  if (node.children) {
    for (const child of node.children) {
      const found = findNodeInTree(child, nodeId);
      if (found) return found;
    }
  }
  return null;
};

// Recursive helper to update a specific node in the tree by its global_bbox coordinate match
const updateNodeInTree = (
  node: RegionNode,
  globalBbox: BoundingBox,
  updatedNode: RegionNode,
): RegionNode => {
  if (getBboxKey(node.global_bbox) === getBboxKey(globalBbox)) {
    return {
      ...node,
      ...updatedNode,
      children: updatedNode.children || node.children || [],
    };
  }
  if (node.children) {
    return {
      ...node,
      children: node.children.map((child) =>
        updateNodeInTree(child, globalBbox, updatedNode),
      ),
    };
  }
  return node;
};

// Rebuild path from root node to the node matching targetBbox to maintain breadcrumbs consistency
const findPathToNode = (
  root: RegionNode,
  targetBbox: BoundingBox,
  path: RegionNode[] = [],
): RegionNode[] | null => {
  const currentPath = [...path, root];
  if (getBboxKey(root.global_bbox) === getBboxKey(targetBbox)) {
    return currentPath;
  }
  if (root.children) {
    for (const child of root.children) {
      const result = findPathToNode(child, targetBbox, currentPath);
      if (result) return result;
    }
  }
  return null;
};

export const useExplorationStore = create<ExplorationState>((set, get) => ({
  rootImageId: '',
  explorationTree: null,
  currentNode: null,
  breadcrumbs: [],
  loadingBboxes: {},
  selectedDetection: null,
  showHeatmap: false,

  initialize: (rootNode: RegionNode) => {
    const initializedRoot = mapDetectionsToChildren(rootNode);
    set({
      rootImageId: initializedRoot.image_id,
      explorationTree: initializedRoot,
      currentNode: initializedRoot,
      breadcrumbs: [initializedRoot],
      loadingBboxes: {},
      selectedDetection: null,
      showHeatmap: false,
    });
  },

  exploreRegion: async (
    imageId: string,
    zoomLevel: number,
    globalBbox: BoundingBox,
  ) => {
    const { currentNode } = get();
    if (!currentNode) {
      console.error('No current node to explore from');
      return;
    }

    const bboxKey = getBboxKey(globalBbox);
    set((state) => ({
      loadingBboxes: { ...state.loadingBboxes, [bboxKey]: true },
    }));

    try {
      const response = await fetch(getApiUrl('/explore'), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image_id: imageId,
          zoom_level: zoomLevel,
          global_bbox: globalBbox,
        }),
      });

      if (!response.ok) {
        throw new Error(`Failed to explore region: ${response.statusText}`);
      }

      const newNode: RegionNode = await response.json();
      const initializedNewNode = mapDetectionsToChildren(newNode);

      set((state) => {
        if (!state.explorationTree) {
          return {
            explorationTree: initializedNewNode,
            currentNode: initializedNewNode,
            breadcrumbs: [initializedNewNode],
            rootImageId: imageId,
            selectedDetection: null,
          };
        }

        const updatedTree = updateNodeInTree(
          state.explorationTree,
          globalBbox,
          initializedNewNode,
        );

        const path = findPathToNode(updatedTree, globalBbox);

        return {
          explorationTree: updatedTree,
          currentNode: path ? path[path.length - 1] : initializedNewNode,
          breadcrumbs: path || [...state.breadcrumbs, initializedNewNode],
          selectedDetection: null,
        };
      });
    } catch (error) {
      console.error('exploreRegion error:', error);
      throw error;
    } finally {
      set((state) => ({
        loadingBboxes: { ...state.loadingBboxes, [bboxKey]: false },
      }));
    }
  },

  navigateToNode: (node: RegionNode) => {
    set((state) => {
      if (!state.explorationTree) return {};
      const latestNode = findNodeInTree(state.explorationTree, node.node_id);
      if (!latestNode) return {};

      const index = state.breadcrumbs.findIndex(
        (b) => b.node_id === node.node_id,
      );
      if (index === -1) return {};

      const path = findPathToNode(state.explorationTree, latestNode.global_bbox);

      return {
        currentNode: latestNode,
        breadcrumbs: path || state.breadcrumbs.slice(0, index + 1),
        selectedDetection: null,
      };
    });
  },

  selectDetection: (detection: Detection | null) => {
    set({ selectedDetection: detection });
  },

  toggleHeatmap: () => {
    set((state) => ({ showHeatmap: !state.showHeatmap }));
  },

  reset: () => {
    set({
      rootImageId: '',
      explorationTree: null,
      currentNode: null,
      breadcrumbs: [],
      loadingBboxes: {},
      selectedDetection: null,
      showHeatmap: false,
    });
  },
}));
