import { GRID_SIZE, idx, neighbors4, manhattan } from "./grid.js";

function makeHeap() {
  const h = [];
  const swap = (i, j) => {
    const t = h[i];
    h[i] = h[j];
    h[j] = t;
  };
  const up = (i) => {
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (h[i].f >= h[p].f) break;
      swap(i, p);
      i = p;
    }
  };
  const down = (i) => {
    const n = h.length;
    for (;;) {
      let m = i,
        l = i * 2 + 1,
        r = l + 1;
      if (l < n && h[l].f < h[m].f) m = l;
      if (r < n && h[r].f < h[m].f) m = r;
      if (m === i) break;
      swap(i, m);
      i = m;
    }
  };
  return {
    push(x) {
      h.push(x);
      up(h.length - 1);
    },
    pop() {
      if (!h.length) return null;
      const top = h[0];
      const last = h.pop();
      if (h.length) {
        h[0] = last;
        down(0);
      }
      return top;
    },
    get size() {
      return h.length;
    },
  };
}

/**
 * A* with Manhattan heuristic, unit cost.
 */
export function astar(grid, start, goal) {
  const [sr, sc] = start;
  const [gr, gc] = goal;
  const si = idx(sr, sc),
    gi = idx(gr, gc);
  const nCells = GRID_SIZE * GRID_SIZE;
  const gScore = new Float32Array(nCells);
  gScore.fill(Infinity);
  const prev = new Int32Array(nCells);
  prev.fill(-1);
  const closed = new Uint8Array(nCells);

  const h0 = manhattan(start, goal);
  const heap = makeHeap();
  gScore[si] = 0;
  heap.push({ i: si, f: h0 });
  const order = [];

  while (heap.size) {
    const { i: u } = heap.pop();
    if (closed[u]) continue;
    closed[u] = 1;
    order.push(u);
    if (u === gi) break;

    const r = Math.floor(u / GRID_SIZE),
      c = u % GRID_SIZE;
    for (const [nr, nc] of neighbors4(r, c)) {
      const v = idx(nr, nc);
      if (grid[v] === 1) continue;
      const tentative = gScore[u] + 1;
      if (tentative < gScore[v]) {
        prev[v] = u;
        gScore[v] = tentative;
        const f = tentative + manhattan([nr, nc], goal);
        heap.push({ i: v, f });
      }
    }
  }

  const path = [];
  if (gScore[gi] === Infinity) {
    const edges = [];
    const nodes = [];
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
    return { path: [], dist: Infinity, order, edges, nodes };
  }

  let cur = gi;
  while (cur !== -1) {
    path.push([Math.floor(cur / GRID_SIZE), cur % GRID_SIZE]);
    cur = prev[cur];
  }
  path.reverse();

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

  return {
    path,
    dist: gScore[gi],
    order,
    edges,
    nodes,
  };
}
