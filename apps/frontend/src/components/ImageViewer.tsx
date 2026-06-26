'use client';
/* eslint-disable @typescript-eslint/no-explicit-any */

import React, { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import {
  useExplorationStore,
  getBboxKey,
  getClassColor,
} from '../store/explorationStore';
import type { RegionNode, BoundingBox } from '@recursive-object-detector/types';

interface ImageViewerProps {
  imageUrl: string;
  onImageSizeLoaded?: (size: { width: number; height: number }) => void;
}

const getNodeBboxPixels = (
  node: RegionNode,
  imageSize: { width: number; height: number },
): BoundingBox => {
  if (node.zoom_level === 0) {
    return [0, 0, imageSize.width, imageSize.height];
  }
  return node.global_bbox;
};

interface DetectionBoxProps {
  node: RegionNode;
  parentNode: RegionNode | null;
  imageSize: { width: number; height: number };
  viewer: any;
}

const HeatmapCanvas: React.FC<{
  show: boolean;
  imageSize: { width: number; height: number };
}> = ({ show, imageSize }) => {
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const { explorationTree } = useExplorationStore();

  React.useEffect(() => {
    if (!canvasRef.current || !explorationTree) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    if (!show) return;

    const drawRegionGlow = (node: RegionNode) => {
      const [xMin, yMin, xMax, yMax] = node.global_bbox;
      const w = xMax - xMin;
      const h = yMax - yMin;
      if (w <= 0 || h <= 0) return;

      let color = 'rgba(59, 130, 246, 0.1)';
      let shadowColor = '#3b82f6';
      let blur = 20;

      if (node.zoom_level === 1) {
        color = 'rgba(59, 130, 246, 0.2)';
        shadowColor = '#3b82f6';
        blur = 30;
      } else if (node.zoom_level === 2) {
        color = 'rgba(249, 115, 22, 0.4)';
        shadowColor = '#f97316';
        blur = 45;
      } else if (node.zoom_level >= 3) {
        color = 'rgba(239, 68, 68, 0.6)';
        shadowColor = '#ef4444';
        blur = 60;
      }

      ctx.save();
      ctx.shadowBlur = blur;
      ctx.shadowColor = shadowColor;
      ctx.fillStyle = color;
      
      ctx.beginPath();
      const radius = Math.min(w, h, 20);
      if (radius > 0) {
        ctx.roundRect(xMin, yMin, w, h, radius);
      } else {
        ctx.rect(xMin, yMin, w, h);
      }
      ctx.fill();
      ctx.restore();

      if (node.children) {
        node.children.forEach(drawRegionGlow);
      }
    };

    drawRegionGlow(explorationTree);
  }, [show, imageSize, explorationTree]);

  return (
    <canvas
      ref={canvasRef}
      width={imageSize.width}
      height={imageSize.height}
      className="absolute inset-0 pointer-events-none transition-opacity duration-300"
      style={{
        width: '100%',
        height: '100%',
        opacity: show ? 0.75 : 0,
        zIndex: 5,
      }}
    />
  );
};

const renderAggregatedTitle = (det: any) => {
  if (det.class_name?.toLowerCase() !== 'traffic light') {
    return <span className="text-white font-sans font-bold tracking-wider text-[10.5px] uppercase">{det.class_name} {Math.round(det.confidence * 100)}%</span>;
  }

  let status: string | null = null;
  let timer: string | null = null;

  if (det.sub_detections) {
    det.sub_detections.forEach((sub: any) => {
      const subName = sub.class_name?.toLowerCase() || '';
      if (subName.includes('red')) {
        status = 'RED';
      } else if (subName.includes('yellow')) {
        status = 'YELLOW';
      } else if (subName.includes('green')) {
        status = 'GREEN';
      }

      if (subName.includes('digit_')) {
        timer = subName.split('_')[1];
      } else {
        const match = subName.match(/\d+/);
        if (match) {
          timer = match[0];
        }
      }
    });
  }

  return (
    <span className="flex items-center gap-1.5 font-sans font-bold tracking-wider text-[11px] text-white select-none">
      <span className="text-slate-300">Traffic Light</span>
      {status && (
        <>
          <span className="text-slate-500 font-mono">|</span>
          <span className="text-slate-400">Status:</span>
          <span className={status === 'RED' ? 'text-red-400 font-black' : status === 'YELLOW' ? 'text-amber-400 font-black' : 'text-emerald-400 font-black'}>
            {status}
          </span>
        </>
      )}
      {timer && (
        <>
          <span className="text-slate-500 font-mono">|</span>
          <span className="text-slate-400">Timer:</span>
          <span className="text-cyan-400 font-mono font-black text-[11.5px]">{timer}</span>
        </>
      )}
    </span>
  );
};

const SubDetectionBox: React.FC<{
  sub: any;
  parentBbox: BoundingBox;
  zoomLevel: number;
  viewer: any;
  imageSize: { width: number; height: number };
}> = ({ sub, parentBbox, zoomLevel, viewer, imageSize }) => {
  const { selectedDetection, selectDetection } = useExplorationStore();
  const [isHovered, setIsHovered] = useState(false);

  const subW = parentBbox[2] - parentBbox[0];
  const subH = parentBbox[3] - parentBbox[1];

  const sLeft = subW > 0 ? ((sub.global_bbox[0] - parentBbox[0]) / subW) * 100 : 0;
  const sTop = subH > 0 ? ((sub.global_bbox[1] - parentBbox[1]) / subH) * 100 : 0;
  const sWidth = subW > 0 ? ((sub.global_bbox[2] - sub.global_bbox[0]) / subW) * 100 : 0;
  const sHeight = subH > 0 ? ((sub.global_bbox[3] - sub.global_bbox[1]) / subH) * 100 : 0;

  const isSubSelected = selectedDetection && getBboxKey(selectedDetection.global_bbox) === getBboxKey(sub.global_bbox);
  const isActiveOrHovered = isSubSelected || isHovered;
  
  const showSubLabel = isActiveOrHovered || zoomLevel > 3.0;
  const isSubNearTop = sTop < 5;

  const handleZoom = (bbox: BoundingBox) => {
    if (!viewer) return;
    const [xMin, yMin, xMax, yMax] = bbox;
    const imgWidth = imageSize.width;
    import('openseadragon').then((OpenSeadragonModule) => {
      const OpenSeadragon = OpenSeadragonModule.default;
      const targetRect = new OpenSeadragon.Rect(
        xMin / imgWidth,
        yMin / imgWidth,
        (xMax - xMin) / imgWidth,
        (yMax - yMin) / imgWidth,
      );
      viewer.viewport.fitBounds(targetRect, false);
    });
  };

  const subLabelStyle: React.CSSProperties = {
    transform: `scale(${1 / zoomLevel})`,
    transformOrigin: 'top left',
    position: 'absolute',
    left: '0px',
    top: isSubNearTop ? '0px' : `-${24 / zoomLevel}px`,
    zIndex: 50,
    pointerEvents: 'none',
    whiteSpace: 'nowrap',
  };

  let subLabelText = sub.class_name;
  if (sub.class_name?.toLowerCase().includes('digit_')) {
    subLabelText = `Digit ${sub.class_name.split('_')[1]}`;
  }

  const defaultBorderWidth = 2 / zoomLevel;
  const activeBorderWidth = 3 / zoomLevel;

  return (
    <div
      className="absolute group pointer-events-auto hover:z-30 transition-all cursor-pointer"
      style={{
        left: `${sLeft}%`,
        top: `${sTop}%`,
        width: `${sWidth}%`,
        height: `${sHeight}%`,
        border: `${isActiveOrHovered ? activeBorderWidth : defaultBorderWidth}px solid ${isActiveOrHovered ? '#22D3EE' : 'rgba(255, 255, 255, 0.8)'}`,
        backgroundColor: isActiveOrHovered ? 'rgba(34, 211, 238, 0.05)' : 'rgba(255, 255, 255, 0.01)',
        boxShadow: isActiveOrHovered ? '0 0 10px rgba(34, 211, 238, 0.6)' : 'none',
      }}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      onClick={(e) => {
        e.stopPropagation();
        selectDetection(sub);
        handleZoom(sub.global_bbox);
      }}
    >
      {showSubLabel && (
        <div
          style={subLabelStyle}
          className="bg-slate-900/90 backdrop-blur-[4px] border border-white/10 text-white text-[9.5px] font-semibold tracking-wider uppercase px-2 py-0.5 rounded shadow-lg pointer-events-none"
        >
          {subLabelText} {Math.round(sub.confidence * 100)}%
        </div>
      )}
    </div>
  );
};

const LeafDetection: React.FC<{
  det: any;
  parentBbox: BoundingBox;
  imageSize: { width: number; height: number };
  viewer: any;
  zoomLevel: number;
  siblingDetections: any[];
}> = ({ det, parentBbox, imageSize, viewer, zoomLevel, siblingDetections }) => {
  const { selectedDetection, selectDetection } = useExplorationStore();
  const [isHovered, setIsHovered] = useState(false);

  const pWidth = parentBbox[2] - parentBbox[0];
  const pHeight = parentBbox[3] - parentBbox[1];

  const left = pWidth > 0 ? ((det.global_bbox[0] - parentBbox[0]) / pWidth) * 100 : 0;
  const top = pHeight > 0 ? ((det.global_bbox[1] - parentBbox[1]) / pHeight) * 100 : 0;
  const width = pWidth > 0 ? ((det.global_bbox[2] - det.global_bbox[0]) / pWidth) * 100 : 0;
  const height = pHeight > 0 ? ((det.global_bbox[3] - det.global_bbox[1]) / pHeight) * 100 : 0;

  const hasSubDetections = det.sub_detections && det.sub_detections.length > 0;
  const isSelected = selectedDetection && getBboxKey(selectedDetection.global_bbox) === getBboxKey(det.global_bbox);
  const isChildSelected = selectedDetection && det.sub_detections?.some((sub: any) => getBboxKey(sub.global_bbox) === getBboxKey(selectedDetection.global_bbox));
  const isExpanded = isSelected || isChildSelected;

  const isActiveOrHovered = isExpanded || isSelected || isHovered;

  const handleZoom = (bbox: BoundingBox) => {
    if (!viewer) return;
    const [xMin, yMin, xMax, yMax] = bbox;
    const imgWidth = imageSize.width;
    import('openseadragon').then((OpenSeadragonModule) => {
      const OpenSeadragon = OpenSeadragonModule.default;
      const targetRect = new OpenSeadragon.Rect(
        xMin / imgWidth,
        yMin / imgWidth,
        (xMax - xMin) / imgWidth,
        (yMax - yMin) / imgWidth,
      );
      viewer.viewport.fitBounds(targetRect, false);
    });
  };

  const handleParentClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    selectDetection(det);
    handleZoom(det.global_bbox);
  };

  const showLabel = true;
  const isNearTop = top < 5;

  // Scaled Label Tag Styles
  const labelStyle: React.CSSProperties = {
    transform: `scale(${1 / zoomLevel})`,
    transformOrigin: 'top left',
    position: 'absolute',
    left: '0px',
    top: isNearTop ? '0px' : `-${28 / zoomLevel}px`,
    zIndex: 50,
    pointerEvents: 'none',
    whiteSpace: 'nowrap',
  };

  const activeBorderWidth = 4 / zoomLevel;
  const defaultBorderWidth = 2 / zoomLevel;

  if (hasSubDetections) {
    return (
      <div
        className="absolute group pointer-events-auto z-10 hover:z-20 transition-all cursor-pointer"
        style={{
          left: `${left}%`,
          top: `${top}%`,
          width: `${width}%`,
          height: `${height}%`,
          border: `${isActiveOrHovered ? activeBorderWidth : defaultBorderWidth}px solid ${isActiveOrHovered ? '#FDE047' : 'rgba(255, 255, 255, 0.8)'}`,
          backgroundColor: isActiveOrHovered ? 'rgba(253, 224, 71, 0.05)' : 'rgba(255, 255, 255, 0.01)',
          boxShadow: isActiveOrHovered ? '0 0 12px rgba(253, 224, 71, 0.6)' : 'none',
        }}
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
        onClick={handleParentClick}
      >
        {showLabel && (
          <div
            style={labelStyle}
            className="bg-slate-900/90 backdrop-blur-[4px] border border-white/10 px-2.5 py-1 rounded-md shadow-xl pointer-events-none"
          >
            {renderAggregatedTitle(det)}
          </div>
        )}

        {isExpanded && det.sub_detections.map((sub: any, index: number) => (
          <SubDetectionBox
            key={`sub_${index}`}
            sub={sub}
            parentBbox={det.global_bbox}
            zoomLevel={zoomLevel}
            viewer={viewer}
            imageSize={imageSize}
          />
        ))}
      </div>
    );
  }

  return (
    <div
      className="absolute group pointer-events-auto z-0 hover:z-20 cursor-pointer"
      style={{
        left: `${left}%`,
        top: `${top}%`,
        width: `${width}%`,
        height: `${height}%`,
        border: `${isActiveOrHovered ? activeBorderWidth : defaultBorderWidth}px solid ${isActiveOrHovered ? '#FDE047' : 'rgba(255, 255, 255, 0.8)'}`,
        backgroundColor: isActiveOrHovered ? 'rgba(253, 224, 71, 0.05)' : 'rgba(255, 255, 255, 0.01)',
        boxShadow: isActiveOrHovered ? '0 0 12px rgba(253, 224, 71, 0.6)' : 'none',
      }}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      onClick={(e) => {
        e.stopPropagation();
        selectDetection(det);
        handleZoom(det.global_bbox);
      }}
    >
      {showLabel && (
        <div
          style={labelStyle}
          className="bg-slate-900/90 backdrop-blur-[4px] border border-white/10 text-white text-[10px] font-semibold tracking-wider uppercase px-2 py-0.5 rounded shadow-lg pointer-events-none"
        >
          {det.class_name} {Math.round(det.confidence * 100)}%
        </div>
      )}
    </div>
  );
};

interface DetectionBoxProps {
  node: RegionNode;
  parentNode: RegionNode | null;
  imageSize: { width: number; height: number };
  viewer: any;
  zoomLevel: number;
}

const DetectionBox: React.FC<DetectionBoxProps> = ({
  node,
  parentNode,
  imageSize,
  viewer,
  zoomLevel,
}) => {
  const { exploreRegion, loadingBboxes, navigateToNode, currentNode } = useExplorationStore();
  const [isHovered, setIsHovered] = useState(false);

  const hasChildren = node.children && node.children.length > 0;
  const isRoot = node.zoom_level === 0;

  const parentBbox = parentNode
    ? getNodeBboxPixels(parentNode, imageSize)
    : [0, 0, imageSize.width, imageSize.height];
  const nodeBbox = getNodeBboxPixels(node, imageSize);

  // If root node, render the full container that fills the viewer
  if (isRoot) {
    return (
      <div
        className="absolute inset-0 pointer-events-none"
        style={{ left: 0, top: 0, width: '100%', height: '100%' }}
      >
        {node.detections?.map((det, i) => (
          <LeafDetection 
            key={`det_${i}`} 
            det={det} 
            parentBbox={nodeBbox} 
            imageSize={imageSize} 
            viewer={viewer} 
            zoomLevel={zoomLevel}
            siblingDetections={node.detections}
          />
        ))}
        {node.children?.map((child) => (
          <DetectionBox
            key={child.node_id}
            node={child}
            parentNode={node}
            imageSize={imageSize}
            viewer={viewer}
            zoomLevel={zoomLevel}
          />
        ))}
      </div>
    );
  }

  // Calculate percentage coordinates relative to the parent bounding box
  const pWidth = parentBbox[2] - parentBbox[0];
  const pHeight = parentBbox[3] - parentBbox[1];

  const left = pWidth > 0 ? ((nodeBbox[0] - parentBbox[0]) / pWidth) * 100 : 0;
  const top = pHeight > 0 ? ((nodeBbox[1] - parentBbox[1]) / pHeight) * 100 : 0;
  const width = pWidth > 0 ? ((nodeBbox[2] - nodeBbox[0]) / pWidth) * 100 : 0;
  const height = pHeight > 0 ? ((nodeBbox[3] - nodeBbox[1]) / pHeight) * 100 : 0;

  const bboxKey = getBboxKey(node.global_bbox);
  const isLoading = loadingBboxes[bboxKey];

  // Click handler
  const handleClick = async (e: React.MouseEvent) => {
    e.stopPropagation();

    // Zoom in immediately to this node's bounds
    const [xMin, yMin, xMax, yMax] = node.global_bbox;
    const imgWidth = imageSize.width;
    import('openseadragon').then((OpenSeadragonModule) => {
      const OpenSeadragon = OpenSeadragonModule.default;
      const targetRect = new OpenSeadragon.Rect(
        xMin / imgWidth,
        yMin / imgWidth,
        (xMax - xMin) / imgWidth,
        (yMax - yMin) / imgWidth,
      );
      viewer.viewport.fitBounds(targetRect, false);
    });

    if (hasChildren) {
      navigateToNode(node);
    } else {
      try {
        await exploreRegion(
          node.image_id,
          node.zoom_level,
          node.global_bbox,
        );
      } catch (err) {
        console.error('Failed to explore region:', err);
      }
    }
  };

  const isActive = currentNode?.node_id === node.node_id;
  const isActiveOrHovered = isActive || isHovered;

  const activeBorderWidth = 4 / zoomLevel;
  const defaultBorderWidth = 2 / zoomLevel;

  if (isLoading) {
    return (
      <div
        className="bg-yellow-500/10 absolute cursor-wait animate-pulse transition-all duration-200 pointer-events-none z-10"
        style={{
          left: `${left}%`,
          top: `${top}%`,
          width: `${width}%`,
          height: `${height}%`,
          border: `${activeBorderWidth}px solid #fbbf24`,
        }}
      >
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="w-4 h-4 border-2 border-yellow-500 border-t-transparent rounded-full animate-spin"></div>
        </div>
      </div>
    );
  }

  return (
    <div
      onClick={handleClick}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      className="bg-slate-500/5 absolute group cursor-pointer hover:bg-slate-500/10 transition-all duration-200 z-10 pointer-events-auto"
      style={{
        left: `${left}%`,
        top: `${top}%`,
        width: `${width}%`,
        height: `${height}%`,
        border: `${isActiveOrHovered ? activeBorderWidth : defaultBorderWidth}px solid ${isActiveOrHovered ? '#22D3EE' : 'rgba(255, 255, 255, 0.8)'}`,
        boxShadow: isActiveOrHovered ? '0 0 12px rgba(34, 211, 238, 0.6)' : 'none',
      }}
    >
      <div 
        style={{
          transform: `scale(${1 / zoomLevel})`,
          transformOrigin: 'top left',
          position: 'absolute',
          left: '0px',
          top: top < 5 ? '0px' : `-${24 / zoomLevel}px`,
          zIndex: 50,
          pointerEvents: 'none',
          whiteSpace: 'nowrap',
        }}
        className="bg-slate-900/90 backdrop-blur-[4px] border border-white/10 text-white text-[9.5px] font-semibold tracking-wider uppercase px-2 py-0.5 rounded shadow-lg opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none"
      >
        Region Lvl {node.zoom_level}
      </div>

      {node.detections?.map((det, i) => (
        <LeafDetection 
          key={`det_${i}`} 
          det={det} 
          parentBbox={nodeBbox} 
          imageSize={imageSize} 
          viewer={viewer} 
          zoomLevel={zoomLevel}
          siblingDetections={node.detections}
        />
      ))}
      
      {node.children?.map((child) => (
        <DetectionBox
          key={child.node_id}
          node={child}
          parentNode={node}
          imageSize={imageSize}
          viewer={viewer}
          zoomLevel={zoomLevel}
        />
      ))}
    </div>
  );
};

interface DetectionTreeProps {
  viewer: any;
  imageSize: { width: number; height: number };
}

const DetectionTree: React.FC<DetectionTreeProps> = ({ viewer, imageSize }) => {
  const { explorationTree, showHeatmap } = useExplorationStore();
  const [zoomLevel, setZoomLevel] = useState(1);

  useEffect(() => {
    if (!viewer) return;
    const updateZoom = () => {
      setZoomLevel(viewer.viewport.getZoom());
    };
    viewer.addHandler('animation', updateZoom);
    viewer.addHandler('zoom', updateZoom);
    updateZoom();
    return () => {
      viewer.removeHandler('animation', updateZoom);
      viewer.removeHandler('zoom', updateZoom);
    };
  }, [viewer]);

  if (!explorationTree) return null;

  return (
    <>
      <HeatmapCanvas show={showHeatmap} imageSize={imageSize} />
      <DetectionBox
        node={explorationTree}
        parentNode={null}
        imageSize={imageSize}
        viewer={viewer}
        zoomLevel={zoomLevel}
      />
    </>
  );
};

export default function ImageViewer({ imageUrl, onImageSizeLoaded }: ImageViewerProps) {
  const viewerRef = useRef<HTMLDivElement>(null);
  const [viewer, setViewer] = useState<any>(null);
  const [imageSize, setImageSize] = useState<{
    width: number;
    height: number;
  } | null>(null);

  const { currentNode, explorationTree, loadingBboxes, selectedDetection, showHeatmap } = useExplorationStore();
  const reactRootRef = useRef<any>(null);

  // Initialize OpenSeadragon
  useEffect(() => {
    if (!viewerRef.current) return;

    let osdInstance: any = null;

    import('openseadragon').then((OpenSeadragonModule) => {
      const OpenSeadragon = OpenSeadragonModule.default;

      osdInstance = OpenSeadragon({
        element: viewerRef.current!,
        prefixUrl:
          'https://cdnjs.cloudflare.com/ajax/libs/openseadragon/5.0.1/images/',
        tileSources: {
          type: 'image',
          url: imageUrl,
        },
        showNavigationControl: true,
        showNavigator: true,
        navigatorPosition: 'BOTTOM_RIGHT',
        constrainDuringPan: true,
        visibilityRatio: 1.0,
        defaultZoomLevel: 0,
        minZoomLevel: 0.5,
        maxZoomLevel: 10,
      });

      osdInstance.addHandler('open', () => {
        const tiledImage = osdInstance.world.getItemAt(0);
        if (tiledImage) {
          const contentSize = tiledImage.getContentSize();
          const size = { width: contentSize.x, height: contentSize.y };
          setImageSize(size);
          if (onImageSizeLoaded) {
            onImageSizeLoaded(size);
          }
        }
      });

      setViewer(osdInstance);
    });

    return () => {
      if (osdInstance) {
        osdInstance.destroy();
      }
    };
  }, [imageUrl]);

  // Sync zoom/pan with currentNode changes (e.g. from breadcrumbs)
  useEffect(() => {
    if (!viewer || !imageSize || !currentNode) return;

    import('openseadragon').then((OpenSeadragonModule) => {
      const OpenSeadragon = OpenSeadragonModule.default;

      let targetRect: any;
      if (currentNode.zoom_level === 0) {
        // Root level: zoom to fit full image
        targetRect = new OpenSeadragon.Rect(
          0,
          0,
          1.0,
          imageSize.height / imageSize.width,
        );
      } else {
        // Nested levels: zoom to region global bbox
        const [xMin, yMin, xMax, yMax] = currentNode.global_bbox;
        const imgWidth = imageSize.width;
        targetRect = new OpenSeadragon.Rect(
          xMin / imgWidth,
          yMin / imgWidth,
          (xMax - xMin) / imgWidth,
          (yMax - yMin) / imgWidth,
        );
      }

      viewer.viewport.fitBounds(targetRect, false);
    });
  }, [viewer, imageSize, currentNode]);

  // Sync zoom/pan with selectedDetection changes
  useEffect(() => {
    if (!viewer || !imageSize || !selectedDetection) return;

    import('openseadragon').then((OpenSeadragonModule) => {
      const OpenSeadragon = OpenSeadragonModule.default;

      const [xMin, yMin, xMax, yMax] = selectedDetection.global_bbox;
      const imgWidth = imageSize.width;
      const targetRect = new OpenSeadragon.Rect(
        xMin / imgWidth,
        yMin / imgWidth,
        (xMax - xMin) / imgWidth,
        (yMax - yMin) / imgWidth,
      );

      viewer.viewport.fitBounds(targetRect, false);
    });
  }, [viewer, imageSize, selectedDetection]);

  // Sync React-managed overlays with OpenSeadragon
  useEffect(() => {
    if (!viewer || !imageSize) return;

    let overlayEl = document.getElementById('react-overlay-root');
    const root = reactRootRef.current;

    if (!overlayEl) {
      overlayEl = document.createElement('div');
      overlayEl.id = 'react-overlay-root';
      overlayEl.style.width = '100%';
      overlayEl.style.height = '100%';
      overlayEl.style.position = 'relative';
      overlayEl.style.pointerEvents = 'none';

      import('openseadragon').then((OpenSeadragonModule) => {
        const OpenSeadragon = OpenSeadragonModule.default;
        const rect = new OpenSeadragon.Rect(
          0,
          0,
          1.0,
          imageSize.height / imageSize.width,
        );

        viewer.addOverlay({
          element: overlayEl!,
          location: rect,
        });

        const rootInstance = createRoot(overlayEl!);
        reactRootRef.current = rootInstance;
        rootInstance.render(
          <DetectionTree imageSize={imageSize} viewer={viewer} />,
        );
      });
    } else {
      let activeRoot = root;
      if (!activeRoot) {
        activeRoot = createRoot(overlayEl);
        reactRootRef.current = activeRoot;
      }
      activeRoot.render(<DetectionTree imageSize={imageSize} viewer={viewer} />);
    }
  }, [viewer, imageSize, explorationTree, loadingBboxes, selectedDetection, showHeatmap]);

  // Clean up React root on unmount
  useEffect(() => {
    return () => {
      if (reactRootRef.current) {
        reactRootRef.current.unmount();
        reactRootRef.current = null;
      }
    };
  }, []);

  return (
    <div className="relative w-full h-full bg-slate-950 flex flex-col items-center justify-center overflow-hidden rounded-xl border border-slate-800 shadow-2xl">
      <div
        ref={viewerRef}
        className="w-full h-full"
        style={{ minHeight: '600px' }}
      />
    </div>
  );
}
