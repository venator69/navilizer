/** 16×16 grid: 0 free, 1 wall */
export const GRID_SIZE = 16;

export function idx(r, c) {
  return r * GRID_SIZE + c;
}

export function fromIdx(i) {
  return { r: Math.floor(i / GRID_SIZE), c: i % GRID_SIZE };
}

export function cloneGrid(g) {
  return g.slice();
}

/** BFS from start to goal on 4-neighbors; walls block. */
export function isReachable(grid, start, goal) {
  const [sr, sc] = start;
  const [gr, gc] = goal;
  if (grid[idx(sr, sc)] === 1 || grid[idx(gr, gc)] === 1) return false;
  const seen = new Uint8Array(GRID_SIZE * GRID_SIZE);
  const q = [[sr, sc]];
  seen[idx(sr, sc)] = 1;
  const dirs = [
    [0, 1],
    [0, -1],
    [1, 0],
    [-1, 0],
  ];
  while (q.length) {
    const [r, c] = q.shift();
    if (r === gr && c === gc) return true;
    for (const [dr, dc] of dirs) {
      const nr = r + dr,
        nc = c + dc;
      if (nr < 0 || nr >= GRID_SIZE || nc < 0 || nc >= GRID_SIZE) continue;
      const i = idx(nr, nc);
      if (seen[i] || grid[i] === 1) continue;
      seen[i] = 1;
      q.push([nr, nc]);
    }
  }
  return false;
}

/**
 * Random terrain: ~22% walls, guaranteed path from start to goal.
 */
export function generateTerrain(wallProb = 0.22) {
  let grid;
  let start, goal;
  let attempts = 0;
  do {
    grid = new Uint8Array(GRID_SIZE * GRID_SIZE);
    for (let i = 0; i < grid.length; i++) {
      if (Math.random() < wallProb) grid[i] = 1;
    }
    start = [0, 0];
    goal = [GRID_SIZE - 1, GRID_SIZE - 1];
    grid[idx(...start)] = 0;
    grid[idx(...goal)] = 0;
    attempts++;
  } while (!isReachable(grid, start, goal) && attempts < 200);

  if (!isReachable(grid, start, goal)) {
    grid.fill(0);
    grid[idx(...start)] = 0;
    grid[idx(...goal)] = 0;
  }
  return { grid, start, goal };
}

export function neighbors4(r, c) {
  const out = [];
  if (c + 1 < GRID_SIZE) out.push([r, c + 1]);
  if (c - 1 >= 0) out.push([r, c - 1]);
  if (r + 1 < GRID_SIZE) out.push([r + 1, c]);
  if (r - 1 >= 0) out.push([r - 1, c]);
  return out;
}

export function manhattan(a, b) {
  return Math.abs(a[0] - b[0]) + Math.abs(a[1] - b[1]);
}
