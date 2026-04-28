import { GRID_SIZE, idx, neighbors4, manhattan } from "./grid.js";

const ACTIONS = [
  [0, 1],
  [1, 0],
  [0, -1],
  [-1, 0],
]; // E, S, W, N

function argmaxQ(Q, s) {
  let best = 0;
  let v = Q[s * 4];
  for (let a = 1; a < 4; a++) {
    const q = Q[s * 4 + a];
    if (q > v) {
      v = q;
      best = a;
    }
  }
  return best;
}

function maxQ(Q, s) {
  let v = Q[s * 4];
  for (let a = 1; a < 4; a++) v = Math.max(v, Q[s * 4 + a]);
  return v;
}

/**
 * One episode: epsilon-greedy from start until goal or maxSteps.
 * Returns { steps, path (cells visited in order), reachedGoal, Q snapshot (copy) }
 */
export function runEpisode(grid, start, goal, Q, epsilon, alpha, gamma, maxSteps = 400) {
  const Qcopy = new Float32Array(Q);
  let [r, c] = start;
  const [gr, gc] = goal;
  const path = [[r, c]];
  let steps = 0;
  let reached = false;

  for (let t = 0; t < maxSteps; t++) {
    steps++;
    const s = idx(r, c);
    if (r === gr && c === gc) {
      reached = true;
      break;
    }

    let a;
    if (Math.random() < epsilon) {
      a = (Math.random() * 4) | 0;
    } else {
      a = argmaxQ(Qcopy, s);
    }

    const [dr, dc] = ACTIONS[a];
    let nr = r + dr,
      nc = c + dc;
    let hitWall = false;
    if (nr < 0 || nr >= GRID_SIZE || nc < 0 || nc >= GRID_SIZE) {
      hitWall = true;
      nr = r;
      nc = c;
    } else if (grid[idx(nr, nc)] === 1) {
      hitWall = true;
      nr = r;
      nc = c;
    }

    const s2 = idx(nr, nc);
    let reward = -0.2;
    if (hitWall) reward = -2;
    if (nr === gr && nc === gc) reward = 100;

    const qsa = Qcopy[s * 4 + a];
    const nextMax = maxQ(Qcopy, s2);
    Qcopy[s * 4 + a] = qsa + alpha * (reward + gamma * nextMax - qsa);

    r = nr;
    c = nc;
    path.push([r, c]);
    if (r === gr && c === gc) {
      reached = true;
      break;
    }
  }

  return { steps, path, reachedGoal: reached, Q: Qcopy };
}

/**
 * Greedy path from start using current Q (for path length after training).
 */
export function greedyPathFromQ(grid, start, goal, Q, maxSteps = 256) {
  const [gr, gc] = goal;
  let [r, c] = start;
  const path = [[r, c]];
  const seen = new Uint8Array(GRID_SIZE * GRID_SIZE);
  seen[idx(r, c)] = 1;

  for (let t = 0; t < maxSteps; t++) {
    if (r === gr && c === gc) {
      return { path, dist: path.length - 1, ok: true };
    }
    const s = idx(r, c);
    const a = argmaxQ(Q, s);
    const [dr, dc] = ACTIONS[a];
    let nr = r + dr,
      nc = c + dc;
    if (nr < 0 || nr >= GRID_SIZE || nc < 0 || nc >= GRID_SIZE) continue;
    if (grid[idx(nr, nc)] === 1) continue;
    const i2 = idx(nr, nc);
    if (seen[i2]) {
      return { path, dist: path.length - 1, ok: false };
    }
    seen[i2] = 1;
    r = nr;
    c = nc;
    path.push([r, c]);
  }
  return { path, dist: path.length - 1, ok: r === gr && c === gc };
}

export function initQ(nStates = GRID_SIZE * GRID_SIZE) {
  return new Float32Array(nStates * 4);
}

export { ACTIONS, maxQ, argmaxQ, manhattan };
