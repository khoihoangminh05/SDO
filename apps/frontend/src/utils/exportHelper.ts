import type { RegionNode, Detection } from '@recursive-object-detector/types';

export const exportImageBboxesToSvg = (
  imageUrl: string,
  explorationTree: RegionNode | null,
  imageSize: { width: number; height: number } | null
) => {
  if (!explorationTree || !imageSize) {
    console.error('Cannot export: Tree or image size is missing');
    return;
  }

  const { width, height } = imageSize;
  const elements: string[] = [];

  const hexColors = [
    '#3b82f6', // Blue
    '#f59e0b', // Amber
    '#a855f7', // Purple
    '#ec4899', // Pink
    '#10b981', // Emerald
    '#ef4444', // Red
    '#06b6d4', // Cyan
  ];

  const getSubColor = (sub: Detection) => {
    const isDigit = sub.class_id === 94 || sub.class_name?.toLowerCase().includes('digit');
    if (isDigit) return '#22d3ee'; // Cyan (LED Digits)
    if (sub.class_name?.toLowerCase().includes('red')) return '#f87171'; // Red
    if (sub.class_name?.toLowerCase().includes('yellow')) return '#fbbf24'; // Yellow
    if (sub.class_name?.toLowerCase().includes('green')) return '#34d399'; // Green
    return '#22d3ee'; // Cyan
  };

  const drawNode = (node: RegionNode) => {
    // 1. Draw RegionNode boundaries for Levels 1+ (crisp dashed Cyan border)
    if (node.zoom_level > 0) {
      const [xMin, yMin, xMax, yMax] = node.global_bbox;
      const w = xMax - xMin;
      const h = yMax - yMin;
      
      elements.push(`
  <!-- Region Lvl ${node.zoom_level} -->
  <rect x="${xMin}" y="${yMin}" width="${w}" height="${h}" fill="none" stroke="#22d3ee" stroke-width="4" stroke-dasharray="8,8" />
  <rect x="${xMin}" y="${yMin - 24}" width="120" height="24" fill="#0f172a" rx="4" stroke="#ffffff" stroke-opacity="0.1" stroke-width="1" />
  <text x="${xMin + 8}" y="${yMin - 8}" fill="#22d3ee" font-size="11" font-weight="bold" font-family="ui-sans-serif, system-ui, sans-serif">Region Lvl ${node.zoom_level}</text>`);
    }

    // 2. Draw object detections belonging to this region
    if (node.detections) {
      node.detections.forEach((det) => {
        const [xMin, yMin, xMax, yMax] = det.global_bbox;
        const w = xMax - xMin;
        const h = yMax - yMin;
        const hasSub = det.sub_detections && det.sub_detections.length > 0;

        if (hasSub) {
          // Parent Detection (Vibrant Cyberpunk Yellow box, e.g. Traffic Light Pole)
          let status = '';
          let timer = '';
          let statusColor = '#34d399'; // default green

          det.sub_detections!.forEach((sub) => {
            const subName = sub.class_name?.toLowerCase() || '';
            if (subName.includes('red')) {
              status = 'RED';
              statusColor = '#f87171';
            } else if (subName.includes('yellow')) {
              status = 'YELLOW';
              statusColor = '#fbbf24';
            } else if (subName.includes('green')) {
              status = 'GREEN';
              statusColor = '#34d399';
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

          // Calculate label content length for badge width sizing
          let textLength = 13; // "Traffic Light"
          if (status) textLength += 11 + status.length; // " | Status: RED"
          if (timer) textLength += 10 + timer.length;   // " | Timer: 15"
          const labelWidth = Math.max(120, textLength * 7 + 20);

          elements.push(`
  <!-- Parent Object: ${det.class_name} -->
  <rect x="${xMin}" y="${yMin}" width="${w}" height="${h}" fill="none" stroke="#fde047" stroke-width="5" />
  <rect x="${xMin}" y="${yMin - 26}" width="${labelWidth}" height="26" fill="#0f172a" rx="4" stroke="#ffffff" stroke-opacity="0.1" stroke-width="1" />
  <text x="${xMin + 10}" y="${yMin - 9}" fill="#ffffff" font-size="11" font-weight="bold" font-family="ui-sans-serif, system-ui, sans-serif">
    <tspan fill="#cbd5e1">Traffic Light</tspan>${status ? `<tspan fill="#64748b"> | </tspan><tspan fill="#94a3b8">Status: </tspan><tspan fill="${statusColor}">${status}</tspan>` : ''}${timer ? `<tspan fill="#64748b"> | </tspan><tspan fill="#94a3b8">Timer: </tspan><tspan fill="#22d3ee">${timer}</tspan>` : ''}
  </text>`);

          // Nested Sub-Detections (e.g. countdown digits and neon colored lamps)
          det.sub_detections!.forEach((sub) => {
            const [sxMin, syMin, sxMax, syMax] = sub.global_bbox;
            const sw = sxMax - sxMin;
            const sh = syMax - syMin;
            const color = getSubColor(sub);

            let subLabelText = sub.class_name;
            if (sub.class_name?.toLowerCase().includes('digit_')) {
              subLabelText = `Digit ${sub.class_name.split('_')[1]}`;
            }

            const labelStr = `${subLabelText} (${Math.round(sub.confidence * 100)}%)`;
            const subLabelWidth = labelStr.length * 6 + 12;

            elements.push(`
  <!-- Sub-Detection Detail: ${sub.class_name} -->
  <rect x="${sxMin}" y="${syMin}" width="${sw}" height="${sh}" fill="none" stroke="${color}" stroke-width="2" />
  <rect x="${sxMin}" y="${syMin - 18}" width="${subLabelWidth}" height="18" fill="#0f172a" rx="2" stroke="#ffffff" stroke-opacity="0.1" stroke-width="1" />
  <text x="${sxMin + 6}" y="${syMin - 5}" fill="${color}" font-size="9.5" font-weight="bold" font-family="ui-sans-serif, system-ui, sans-serif">${labelStr}</text>`);
          });
        } else {
          // Normal leaf detections (e.g. cars, buses, bikes) styled in Slate-900 badges
          const color = hexColors[det.class_id % hexColors.length] || hexColors[0];
          const labelStr = `${det.class_name} (${Math.round(det.confidence * 100)}%)`;
          const labelWidth = labelStr.length * 7 + 16;

          elements.push(`
  <!-- Leaf Object: ${det.class_name} -->
  <rect x="${xMin}" y="${yMin}" width="${w}" height="${h}" fill="none" stroke="${color}" stroke-width="3" />
  <rect x="${xMin}" y="${yMin - 24}" width="${labelWidth}" height="24" fill="#0f172a" rx="4" stroke="#ffffff" stroke-opacity="0.1" stroke-width="1" />
  <text x="${xMin + 8}" y="${yMin - 8}" fill="${color}" font-size="11" font-weight="bold" font-family="ui-sans-serif, system-ui, sans-serif">${labelStr}</text>`);
        }
      });
    }

    // Recurse children
    if (node.children) {
      node.children.forEach(drawNode);
    }
  };

  drawNode(explorationTree);

  // XML serialization
  const svgContent = `<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
  <defs>
    <style>
      text {
        font-family: ui-sans-serif, system-ui, -apple-system, sans-serif;
      }
    </style>
  </defs>

  <!-- Background Image -->
  <image href="${imageUrl}" x="0" y="0" width="${width}" height="${height}" />
  
  <!-- Vector Overlays -->
  ${elements.join('\n')}
</svg>
`;

  // Create XML blob and download
  const blob = new Blob([svgContent], { type: 'image/svg+xml;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `Image_Detections_${new Date().toISOString().split('T')[0]}.svg`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
};
