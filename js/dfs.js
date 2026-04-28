import { GRID_SIZE, idx, neighbors4 } from "./grid.js";

function buildNodesEdges(grid) {
  const nodes = [];
  const edges = [];
  for (let r = 0; r < GRID_SIZE; r++) {
    for (let c = 0; c < GRID_SIZE; c++) {
      const i = idx(r, c);
      if (grid[i] === 1) continue;
      nodes.push({ r, c, i });
      for (const [nr, nc] of neighbors4(r, c)) {
        const j = idx(nr, nc);
        if (grid[j] === 1) continue;
        if (j > i) edges.push({ a: i, b: j });
      }
    }
  }
  return { nodes, edges };
}

/**
 * Iterative DFS — first path found; neighbor push order reversed for stable depth-first walk.
 * Path length is not guaranteed optimal.
 */
export function dfs(grid, start, goal) {
  const [sr, sc] = start;
  const [gr, gc] = goal;
  const si = idx(sr, sc),
    gi = idx(gr, gc);
  const nCells = GRID_SIZE * GRID_SIZE;
  const prev = new Int32Array(nCells);
  prev.fill(-1);
  const visited = new Uint8Array(nCells);
  const stack = [si];
  visited[si] = 1;
  const order = [];
  let found = false;

  while (stack.length) {
    const u = stack.pop();
    order.push(u);
    if (u === gi) {
      found = true;
      break;
    }
    const r = Math.floor(u / GRID_SIZE),
      c = u % GRID_SIZE;
    const nbrs = neighbors4(r, c);
    for (let k = nbrs.length - 1; k >= 0; k--) {
      const [nr, nc] = nbrs[k];
      const v = idx(nr, nc);
      if (grid[v] === 1 || visited[v]) continue;
      visited[v] = 1;
      prev[v] = u;
      stack.push(v);
    }
  }

  const { nodes, edges } = buildNodesEdges(grid);
  if (!found) {
    return { path: [], dist: Infinity, order, edges, nodes };
  }

  const path = [];
  let cur = gi;
  while (cur !== -1) {
    path.push([Math.floor(cur / GRID_SIZE), cur % GRID_SIZE]);
    cur = prev[cur];
  }
  path.reverse();

  return {
    path,
    dist: path.length - 1,
    order,
    edges,
    nodes,
  };
}
