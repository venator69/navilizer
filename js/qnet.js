import { GRID_SIZE, idx, neighbors4 } from "./grid.js";

const ACTIONS = [
  [0, 1],
  [1, 0],
  [0, -1],
  [-1, 0],
];

function randnScale(scale) {
  return (Math.random() * 2 - 1) * scale;
}

/** Tiny 2-layer net: 6 inputs -> H hidden -> 4 Q. Inputs: r,c,gr,gc normalized + bias. */
export class QNetwork {
  constructor(hidden = 32) {
    this.H = hidden;
    this.W1 = new Float32Array(6 * hidden);
    this.b1 = new Float32Array(hidden);
    this.W2 = new Float32Array(hidden * 4);
    this.b2 = new Float32Array(4);
    const s = Math.sqrt(2 / 6);
    for (let i = 0; i < this.W1.length; i++) this.W1[i] = randnScale(s);
    for (let i = 0; i < this.b1.length; i++) this.b1[i] = 0;
    for (let i = 0; i < this.W2.length; i++) this.W2[i] = randnScale(Math.sqrt(2 / hidden));
    for (let i = 0; i < 4; i++) this.b2[i] = 0;
  }

  forward(x, outQ, outH = null) {
    const h = outH || new Float32Array(this.H);
    for (let j = 0; j < this.H; j++) {
      let s = this.b1[j];
      for (let i = 0; i < 6; i++) s += x[i] * this.W1[i * this.H + j];
      h[j] = Math.max(0, s);
    }
    for (let k = 0; k < 4; k++) {
      let s = this.b2[k];
      for (let j = 0; j < this.H; j++) s += h[j] * this.W2[j * 4 + k];
      outQ[k] = s;
    }
    return h;
  }

  argmaxQ(x) {
    const q = new Float32Array(4);
    this.forward(x, q);
    let best = 0;
    for (let a = 1; a < 4; a++) if (q[a] > q[best]) best = a;
    return best;
  }
}

function stateVec(r, c, gr, gc) {
  const inv = 1 / (GRID_SIZE - 1 || 1);
  return new Float32Array([1, r * inv, c * inv, gr * inv, gc * inv, 1]);
}

function stepEnv(grid, r, c, a, goal) {
  const [gr, gc] = goal;
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
  let reward = -0.2;
  if (hitWall) reward = -2;
  if (nr === gr && nc === gc) reward = 100;
  return { nr, nc, reward };
}

/**
 * One training episode with backprop (one-step TD target).
 */
export function trainEpisodeQNet(net, grid, start, goal, opts) {
  const {
    epsilon = 0.3,
    alpha = 0.02,
    gamma = 0.95,
    maxSteps = 400,
  } = opts;
  let [r, c] = start;
  const path = [[r, c]];
  let totalLoss = 0;
  let steps = 0;
  const x = new Float32Array(6);
  const h = new Float32Array(net.H);
  const hPre = new Float32Array(net.H);
  const q = new Float32Array(4);
  const qNext = new Float32Array(4);
  const gradH = new Float32Array(net.H);
  const gradW2 = new Float32Array(net.W2.length);
  const gradB2 = new Float32Array(4);
  const gradW1 = new Float32Array(net.W1.length);
  const gradB1 = new Float32Array(net.H);

  for (let t = 0; t < maxSteps; t++) {
    steps++;
    const [gr, gc] = goal;
    if (r === gr && c === gc) break;

    const sv = stateVec(r, c, gr, gc);
    for (let i = 0; i < 6; i++) x[i] = sv[i];

    for (let j = 0; j < net.H; j++) {
      let s = net.b1[j];
      for (let i = 0; i < 6; i++) s += x[i] * net.W1[i * net.H + j];
      hPre[j] = s;
      h[j] = Math.max(0, s);
    }
    for (let k = 0; k < 4; k++) {
      let s = net.b2[k];
      for (let j = 0; j < net.H; j++) s += h[j] * net.W2[j * 4 + k];
      q[k] = s;
    }

    let a;
    if (Math.random() < epsilon) {
      a = (Math.random() * 4) | 0;
    } else {
      a = 0;
      for (let k = 1; k < 4; k++) if (q[k] > q[a]) a = k;
    }

    const { nr, nc, reward } = stepEnv(grid, r, c, a, goal);
    const sv2 = stateVec(nr, nc, gr, gc);
    for (let k = 0; k < 4; k++) {
      let s2 = net.b2[k];
      for (let j = 0; j < net.H; j++) {
        let s = net.b1[j];
        for (let i = 0; i < 6; i++) s += sv2[i] * net.W1[i * net.H + j];
        const hj = Math.max(0, s);
        s2 += hj * net.W2[j * 4 + k];
      }
      qNext[k] = s2;
    }
    let maxNext = qNext[0];
    for (let k = 1; k < 4; k++) maxNext = Math.max(maxNext, qNext[k]);
    const target = reward + gamma * maxNext;
    const tdErr = target - q[a];
    totalLoss += tdErr * tdErr;

    gradB2.fill(0);
    gradW2.fill(0);
    gradH.fill(0);
    gradB1.fill(0);
    gradW1.fill(0);

    for (let k = 0; k < 4; k++) {
      const d = k === a ? -tdErr : 0;
      gradB2[k] = d;
      for (let j = 0; j < net.H; j++) {
        gradW2[j * 4 + k] = d * h[j];
        gradH[j] += d * net.W2[j * 4 + k];
      }
    }

    for (let j = 0; j < net.H; j++) {
      const relu = hPre[j] > 0 ? 1 : 0;
      const gh = gradH[j] * relu;
      gradB1[j] = gh;
      for (let i = 0; i < 6; i++) {
        gradW1[i * net.H + j] = gh * x[i];
      }
    }

    for (let i = 0; i < net.W1.length; i++) net.W1[i] -= alpha * gradW1[i];
    for (let j = 0; j < net.H; j++) net.b1[j] -= alpha * gradB1[j];
    for (let i = 0; i < net.W2.length; i++) net.W2[i] -= alpha * gradW2[i];
    for (let k = 0; k < 4; k++) net.b2[k] -= alpha * gradB2[k];

    r = nr;
    c = nc;
    path.push([r, c]);
    if (r === gr && c === gc) break;
  }

  const reached = r === goal[0] && c === goal[1];
  return { steps, path, reachedGoal: reached, loss: totalLoss / Math.max(1, steps) };
}

/** Approximate max Q per cell for heatmap (sample actions from cell). */
export function cellMaxQGrid(net, grid, goal) {
  const [gr, gc] = goal;
  const vals = new Float32Array(GRID_SIZE * GRID_SIZE);
  const x = new Float32Array(6);
  const q = new Float32Array(4);
  for (let r = 0; r < GRID_SIZE; r++) {
    for (let c = 0; c < GRID_SIZE; c++) {
      const i = idx(r, c);
      if (grid[i] === 1) {
        vals[i] = NaN;
        continue;
      }
      const sv = stateVec(r, c, gr, gc);
      for (let k = 0; k < 6; k++) x[k] = sv[k];
      net.forward(x, q);
      let m = q[0];
      for (let a = 1; a < 4; a++) m = Math.max(m, q[a]);
      vals[i] = m;
    }
  }
  return vals;
}

export function greedyPathFromQNet(net, grid, start, goal, maxSteps = 256) {
  const [gr, gc] = goal;
  let [r, c] = start;
  const path = [[r, c]];
  const seen = new Uint8Array(GRID_SIZE * GRID_SIZE);
  seen[idx(r, c)] = 1;
  const x = stateVec(r, c, gr, gc);

  for (let t = 0; t < maxSteps; t++) {
    if (r === gr && c === gc) {
      return { path, dist: path.length - 1, ok: true };
    }
    for (let k = 0; k < 6; k++) x[k] = stateVec(r, c, gr, gc)[k];
    const a = net.argmaxQ(x);
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

export { ACTIONS };
