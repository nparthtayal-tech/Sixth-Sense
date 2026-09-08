export function findPathAStar(startX, startY, endX, endY, imageData, width, height) {
  // Grid resolution: check every Nth pixel to speed up A* significantly
  const STEP = 5; 

  // Ensure start/end are within bounds
  if (startX < 0 || startX >= width || startY < 0 || startY >= height) return null;
  if (endX < 0 || endX >= width || endY < 0 || endY >= height) return null;

  const startGridX = Math.floor(startX / STEP);
  const startGridY = Math.floor(startY / STEP);
  const endGridX = Math.floor(endX / STEP);
  const endGridY = Math.floor(endY / STEP);

  const gridW = Math.ceil(width / STEP);
  const gridH = Math.ceil(height / STEP);

  // Helper to check if a grid cell is a wall (check center pixel of the step)
  const isWall = (gx, gy) => {
    const px = Math.min(gx * STEP + Math.floor(STEP/2), width - 1);
    const py = Math.min(gy * STEP + Math.floor(STEP/2), height - 1);
    const idx = (py * width + px) * 4;
    const a = imageData[idx + 3];
    if (a < 50) return false;
    const r = imageData[idx];
    const g = imageData[idx + 1];
    const b = imageData[idx + 2];
    if ((r + g + b) / 3 < 100) return true; // Dark = wall
    return false;
  };

  if (isWall(startGridX, startGridY) || isWall(endGridX, endGridY)) {
    console.warn("Start or End is inside a wall!");
    return null;
  }

  const heuristic = (x, y) => Math.abs(x - endGridX) + Math.abs(y - endGridY);
  
  const openSet = new Set();
  const openSetList = []; // Simple priority queue
  
  const nodeId = (x, y) => `${x},${y}`;
  
  const startId = nodeId(startGridX, startGridY);
  openSet.add(startId);
  openSetList.push({ x: startGridX, y: startGridY, f: heuristic(startGridX, startGridY) });

  const cameFrom = new Map();
  const gScore = new Map();
  gScore.set(startId, 0);

  const fScore = new Map();
  fScore.set(startId, heuristic(startGridX, startGridY));

  let loops = 0;
  const MAX_LOOPS = 20000; // prevent browser hang

  while (openSetList.length > 0) {
    loops++;
    if (loops > MAX_LOOPS) {
       console.warn("A* Timeout!");
       return null;
    }

    // Sort to get lowest f score
    openSetList.sort((a, b) => a.f - b.f);
    const current = openSetList.shift();
    const currId = nodeId(current.x, current.y);
    openSet.delete(currId);

    if (current.x === endGridX && current.y === endGridY) {
      // Reconstruct path
      const path = [];
      let temp = currId;
      while (cameFrom.has(temp)) {
        const [x, y] = temp.split(',').map(Number);
        path.push({ x: x * STEP, y: y * STEP });
        temp = cameFrom.get(temp);
      }
      path.push({ x: startX, y: startY });
      return path.reverse();
    }

    const neighbors = [
      { x: current.x, y: current.y - 1 },
      { x: current.x, y: current.y + 1 },
      { x: current.x - 1, y: current.y },
      { x: current.x + 1, y: current.y },
      // Diagonals
      { x: current.x - 1, y: current.y - 1 },
      { x: current.x + 1, y: current.y - 1 },
      { x: current.x - 1, y: current.y + 1 },
      { x: current.x + 1, y: current.y + 1 }
    ];

    for (const neighbor of neighbors) {
      if (neighbor.x < 0 || neighbor.x >= gridW || neighbor.y < 0 || neighbor.y >= gridH) continue;
      if (isWall(neighbor.x, neighbor.y)) continue;

      const nId = nodeId(neighbor.x, neighbor.y);
      const isDiagonal = (neighbor.x !== current.x) && (neighbor.y !== current.y);
      const tentativeG = gScore.get(currId) + (isDiagonal ? 1.414 : 1);

      if (!gScore.has(nId) || tentativeG < gScore.get(nId)) {
        cameFrom.set(nId, currId);
        gScore.set(nId, tentativeG);
        const f = tentativeG + heuristic(neighbor.x, neighbor.y);
        fScore.set(nId, f);

        if (!openSet.has(nId)) {
          openSet.add(nId);
          openSetList.push({ x: neighbor.x, y: neighbor.y, f: f });
        }
      }
    }
  }

  return null; // No path found
}
