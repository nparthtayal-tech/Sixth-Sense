/**
 * floorPlanProcessor.js
 * =====================
 * Local Computer Vision & NavMesh generator for Factory Floor Plans.
 * - Analyzes canvas pixels to identify walls vs corridors.
 * - Extracts topological NavMesh (nodes [lat, lng] and edges [idxA, idxB]).
 * - Provides raycasted line-of-sight collision checking.
 * - Includes a high-tech sample factory blueprint generator.
 */

/**
 * Checks if a pixel at (px, py) is an obstacle/wall.
 */
export function checkPixelWall(px, py, imageData, width, height, darkIsWall = true) {
  if (px < 0 || px >= width || py < 0 || py >= height) return true;
  const idx = (py * width + px) * 4;
  const a = imageData.data[idx + 3];
  if (a < 50) {
    // Transparent area is considered free space
    return false;
  }
  const r = imageData.data[idx];
  const g = imageData.data[idx + 1];
  const b = imageData.data[idx + 2];
  const brightness = (r + g + b) / 3;

  // Dark line on light background = wall
  if (darkIsWall) {
    return brightness < 110;
  } else {
    // Light line on dark background = wall
    return brightness > 145;
  }
}

/**
 * Raycast test: check if the direct line between (x0, y0) and (x1, y1)
 * intersects any wall pixel.
 */
export function isLineClear(x0, y0, x1, y1, imageData, width, height, darkIsWall = true, steps = 18) {
  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    const px = Math.round(x0 + (x1 - x0) * t);
    const py = Math.round(y0 + (y1 - y0) * t);
    if (checkPixelWall(px, py, imageData, width, height, darkIsWall)) {
      return false;
    }
  }
  return true;
}

/**
 * Extract NavMesh graph (nodes & edges) from an Image element.
 */
export function extractNavMeshFromImage(
  img,
  floorPlanScale = 100,
  centerLat = 13.0600,
  centerLng = 80.2800
) {
  const canvas = document.createElement("canvas");
  const maxDim = 600;
  let w = img.naturalWidth || img.width || 500;
  let h = img.naturalHeight || img.height || 500;

  if (w > maxDim || h > maxDim) {
    if (w >= h) {
      h = Math.round((h * maxDim) / w);
      w = maxDim;
    } else {
      w = Math.round((w * maxDim) / h);
      h = maxDim;
    }
  }

  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(img, 0, 0, w, h);

  const imageData = ctx.getImageData(0, 0, w, h);

  // Auto-detect if floor plan is dark-on-light or light-on-dark
  let darkCount = 0;
  let totalSampled = 0;
  for (let y = 10; y < h; y += 20) {
    for (let x = 10; x < w; x += 20) {
      const idx = (y * w + x) * 4;
      const lum = (imageData.data[idx] + imageData.data[idx + 1] + imageData.data[idx + 2]) / 3;
      if (lum < 120) darkCount++;
      totalSampled++;
    }
  }
  // Most architectural plans have white floors with dark walls (<40% dark)
  const darkIsWall = darkCount / totalSampled < 0.55;

  // Compute geographical bounds
  const lat_diff = floorPlanScale / 111320;
  const lng_diff = floorPlanScale / (111320 * Math.cos((centerLat * Math.PI) / 180));
  const maxLat = centerLat + lat_diff / 2;
  const minLng = centerLng - lng_diff / 2;

  // Generate grid nodes
  const GRID_COLS = 12;
  const GRID_ROWS = 12;
  let rawNodes = [];

  for (let r = 1; r < GRID_ROWS; r++) {
    for (let c = 1; c < GRID_COLS; c++) {
      const px = Math.round((c / GRID_COLS) * w);
      const py = Math.round((r / GRID_ROWS) * h);

      // Check center and small radius clearance
      const isWallCenter = checkPixelWall(px, py, imageData, w, h, darkIsWall);
      const isWallOffset =
        checkPixelWall(px + 3, py, imageData, w, h, darkIsWall) ||
        checkPixelWall(px - 3, py, imageData, w, h, darkIsWall) ||
        checkPixelWall(px, py + 3, imageData, w, h, darkIsWall) ||
        checkPixelWall(px, py - 3, imageData, w, h, darkIsWall);

      if (!isWallCenter && !isWallOffset) {
        // Valid navigable corridor cell
        const lat = maxLat - (py / h) * lat_diff;
        const lng = minLng + (px / w) * lng_diff;
        rawNodes.push({ px, py, lat, lng, col: c, row: r });
      }
    }
  }

  // If strict check produced too few nodes, loosen clearance
  if (rawNodes.length < 6) {
    rawNodes = [];
    for (let r = 1; r < GRID_ROWS; r++) {
      for (let c = 1; c < GRID_COLS; c++) {
        const px = Math.round((c / GRID_COLS) * w);
        const py = Math.round((r / GRID_ROWS) * h);
        if (!checkPixelWall(px, py, imageData, w, h, darkIsWall)) {
          const lat = maxLat - (py / h) * lat_diff;
          const lng = minLng + (px / w) * lng_diff;
          rawNodes.push({ px, py, lat, lng, col: c, row: r });
        }
      }
    }
  }

  // Fallback: If still too few (e.g. inverted/complex blueprint), generate tactical interior grid
  if (rawNodes.length < 4) {
    rawNodes = [];
    for (let r = 2; r <= 8; r += 2) {
      for (let c = 2; c <= 8; c += 2) {
        const px = Math.round((c / 10) * w);
        const py = Math.round((r / 10) * h);
        const lat = maxLat - (py / h) * lat_diff;
        const lng = minLng + (px / w) * lng_diff;
        rawNodes.push({ px, py, lat, lng, col: c, row: r });
      }
    }
  }

  // Build edges between neighboring nodes with clear line-of-sight
  const edges = [];
  const MAX_CONN_DIST_PX = Math.max(w / GRID_COLS, h / GRID_ROWS) * 1.8;

  for (let i = 0; i < rawNodes.length; i++) {
    for (let j = i + 1; j < rawNodes.length; j++) {
      const n1 = rawNodes[i];
      const n2 = rawNodes[j];
      const d = Math.hypot(n1.px - n2.px, n1.py - n2.py);

      if (d <= MAX_CONN_DIST_PX) {
        if (isLineClear(n1.px, n1.py, n2.px, n2.py, imageData, w, h, darkIsWall, 10)) {
          edges.push([i, j]);
        }
      }
    }
  }

  // Ensure graph is connected (add minimum spanning links if needed)
  if (edges.length === 0 && rawNodes.length >= 2) {
    for (let i = 0; i < rawNodes.length - 1; i++) {
      edges.push([i, i + 1]);
    }
  }

  const nodes = rawNodes.map((n) => [n.lat, n.lng]);

  return {
    nodes,
    edges,
    imageData,
    width: w,
    height: h,
    darkIsWall
  };
}

/**
 * Creates a high-resolution, futuristic smart factory floor plan blueprint
 * with distinct corridors, assembly bays, and obstacle zones.
 */
export function createSampleFactoryFloorPlan() {
  const canvas = document.createElement("canvas");
  const w = 900;
  const h = 900;
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");

  // Background: Light tactical blueprint paper
  ctx.fillStyle = "#f8fafc";
  ctx.fillRect(0, 0, w, h);

  // Subtle blueprint grid
  ctx.strokeStyle = "#e2e8f0";
  ctx.lineWidth = 1;
  const gridStep = 30;
  for (let x = 0; x <= w; x += gridStep) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.stroke();
  }
  for (let y = 0; y <= h; y += gridStep) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }

  // Outer perimeter security walls
  ctx.strokeStyle = "#0f172a";
  ctx.lineWidth = 12;
  ctx.strokeRect(30, 30, w - 60, h - 60);

  // Interior rooms and obstacles (filled dark shapes = walls / equipment)
  const drawRoom = (x, y, rw, rh, title, color = "#1e293b") => {
    ctx.fillStyle = color;
    ctx.fillRect(x, y, rw, rh);

    ctx.fillStyle = "#ffffff";
    ctx.font = "bold 13px 'Segoe UI', sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(title, x + rw / 2, y + rh / 2);
  };

  // Top Left: Assembly Line A
  drawRoom(80, 80, 220, 160, "ASSEMBLY LINE A");

  // Top Right: CNC Milling Station
  drawRoom(600, 80, 220, 160, "CNC MILLING BAY");

  // Mid-Left: Battery Charging Bay
  drawRoom(80, 360, 200, 180, "CHARGING DOCK");

  // Mid-Right: Automated High-Bay Warehouse
  drawRoom(620, 360, 200, 200, "WAREHOUSE RACKS");

  // Center Obstacle: Control Room & Server Hub
  drawRoom(380, 360, 140, 180, "SERVER CORE", "#334155");

  // Bottom Left: Packaging & Palletizing
  drawRoom(80, 660, 240, 160, "PACKAGING BAY");

  // Bottom Right: Inspection & Quality Control
  drawRoom(580, 680, 240, 140, "QUALITY CONTROL");

  // Primary Open Flight Corridors (accent markings)
  ctx.strokeStyle = "rgba(0, 180, 255, 0.45)";
  ctx.lineWidth = 2;
  ctx.setLineDash([8, 8]);

  // Main North-South Flight Aisle
  ctx.beginPath();
  ctx.moveTo(340, 60);
  ctx.lineTo(340, 840);
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(560, 60);
  ctx.lineTo(560, 840);
  ctx.stroke();

  // East-West Flight Corridors
  ctx.beginPath();
  ctx.moveTo(60, 300);
  ctx.lineTo(840, 300);
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(60, 600);
  ctx.lineTo(840, 600);
  ctx.stroke();

  ctx.setLineDash([]);

  // Factory Header
  ctx.fillStyle = "#0f172a";
  ctx.font = "bold 16px 'Segoe UI', sans-serif";
  ctx.textAlign = "left";
  ctx.fillText("SMART FACTORY INDOOR FLIGHT ARENA (100m x 100m)", 45, 55);

  return canvas.toDataURL("image/png");
}
