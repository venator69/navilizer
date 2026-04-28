import { GRID_SIZE, generateTerrain, idx, manhattan } from "./grid.js";
import { bfs } from "./bfs.js";
import { dfs } from "./dfs.js";
import { astar } from "./astar.js";
import {
  initQ,
  runEpisode,
  greedyPathFromQ,
  maxQ,
} from "./qlearning.js";
import {
  QNetwork,
  collectEpisodeQNet,
  trainOneStepQNet,
  greedyPathFromQNet,
  cellMaxQGrid,
} from "./qnet.js";

const CELL = 32;
const CANVAS_SIZE = GRID_SIZE * CELL;

const algoDesc = {
  bfs: `<strong>Breadth-first search (BFS)</strong> explores the grid layer by layer using a queue. On this unit-cost grid, the first time the goal is dequeued yields a <em>shortest</em> path (minimum number of moves).`,
  dfs: `<strong>Depth-first search (DFS)</strong> follows one branch as far as possible using a stack, then backtracks. The <em>first</em> path found to the goal is not guaranteed to be shortest—compare path length with BFS or A*.`,
  astar: `<strong>A*</strong> combines best-first search with a heuristic: it prioritizes cells that look closer to the goal. Here we use Manhattan distance <em>h</em> (shown in each cell on the world grid), which is admissible on a 4-connected grid, so the first time the goal is expanded, the path is still optimal—often with fewer explored nodes than BFS.`,
  qlearning: `<strong>Tabular Q-learning</strong> learns action values <em>Q(s,a)</em> by trial and error. <strong>Local best</strong> shows the shortest successful episode so far, measured in environment <em>steps</em> (<code>out.steps</code> when the goal is reached)—it only improves when a new episode hits the goal in fewer steps than any previous one.`,
  qnet: `<strong>Deep Q-network</strong>: Phase 1 fills a replay buffer (no weight updates). <strong>Local best</strong> is the minimum rollout step count among acquisition episodes that reach the goal. With acquisition viz on, rollouts animate and the log prints buffer tails; with it off, phase 1 stays off-canvas. Phase 2 runs TD replay; heatmaps and the greedy path appear after training.`,
};

const els = {
  algo: document.getElementById("algo"),
  desc: document.getElementById("algo-desc"),
  pathLen: document.getElementById("path-len"),
  explored: document.getElementById("explored-count"),
  episodeLog: document.getElementById("episode-log"),
  btnNew: document.getElementById("btn-new"),
  btnRun: document.getElementById("btn-run"),
  btnStop: document.getElementById("btn-stop"),
  qEpisodesWrap: document.getElementById("q-episodes-wrap"),
  qEpisodesInput: document.getElementById("q-episodes-input"),
  qPathModeWrap: document.getElementById("q-path-mode-wrap"),
  qPathFinalOnly: document.getElementById("q-path-final-only"),
  dqVizAcqWrap: document.getElementById("dq-viz-acq-wrap"),
  dqVizAcquisition: document.getElementById("dq-viz-acquisition"),
  localBest: document.getElementById("local-best"),
  graphSection: document.getElementById("graph-section"),
  qvizSection: document.getElementById("qviz-section"),
};

const gridCanvas = document.getElementById("grid-canvas");
const graphCanvas = document.getElementById("graph-canvas");
const qvizCanvas = document.getElementById("qviz-canvas");
const gctx = gridCanvas.getContext("2d");
const gxctx = graphCanvas.getContext("2d");
const qvx = qvizCanvas.getContext("2d");

gridCanvas.width = graphCanvas.width = qvizCanvas.width = CANVAS_SIZE;
gridCanvas.height = graphCanvas.height = qvizCanvas.height = CANVAS_SIZE;

let state = {
  grid: null,
  start: null,
  goal: null,
  Q: initQ(),
  qnet: new QNetwork(32),
  running: false,
  stopRequested: false,
};

function setPathLength(text) {
  els.pathLen.textContent = text;
}

function setExplored(text) {
  els.explored.textContent = text;
}

function setLocalBest(text) {
  els.localBest.textContent = text;
}

function goalCellIndex(goal) {
  return idx(goal[0], goal[1]);
}

function updateDesc() {
  const a = els.algo.value;
  els.desc.innerHTML = algoDesc[a] || "";
  const showGraph = a === "bfs" || a === "dfs" || a === "astar";
  const showQviz = a === "qlearning" || a === "qnet";
  els.graphSection.style.display = showGraph ? "block" : "none";
  els.qvizSection.style.display = showQviz ? "block" : "none";
  els.qEpisodesWrap.style.display = showQviz ? "inline-flex" : "none";
  els.qPathModeWrap.style.display = a === "qlearning" ? "inline-flex" : "none";
  els.dqVizAcqWrap.style.display = a === "qnet" ? "inline-flex" : "none";
}

/**
 * @param {{ heuristicGoal?: [number, number] | null }} [opts]
 */
function drawGridBase(opts = {}) {
  const { grid, start, goal } = state;
  const heuristicGoal = opts.heuristicGoal ?? null;
  gctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  for (let r = 0; r < GRID_SIZE; r++) {
    for (let c = 0; c < GRID_SIZE; c++) {
      const i = idx(r, c);
      const x = c * CELL,
        y = r * CELL;
      if (grid[i] === 1) {
        gctx.fillStyle = "#484f58";
      } else {
        gctx.fillStyle = "#21262d";
      }
      gctx.fillRect(x, y, CELL, CELL);
      gctx.strokeStyle = "#30363d";
      gctx.strokeRect(x + 0.5, y + 0.5, CELL - 1, CELL - 1);
    }
  }
  if (heuristicGoal) {
    drawManhattanHeuristicLabels(grid, heuristicGoal);
  }
  const [sr, sc] = start;
  const [gr, gc] = goal;
  gctx.fillStyle = "#58a6ff";
  gctx.fillRect(sc * CELL + 2, sr * CELL + 2, CELL - 4, CELL - 4);
  gctx.fillStyle = "#3fb950";
  gctx.fillRect(gc * CELL + 2, gr * CELL + 2, CELL - 4, CELL - 4);
}

/** Manhattan distance to goal (A* heuristic h) in each walkable cell. */
function drawManhattanHeuristicLabels(grid, goal) {
  const [gr, gc] = goal;
  const fontPx = Math.min(16, Math.max(10, Math.floor(CELL * 0.34)));
  gctx.save();
  gctx.font = `600 ${fontPx}px ui-monospace, monospace`;
  gctx.textAlign = "center";
  gctx.textBaseline = "middle";
  for (let r = 0; r < GRID_SIZE; r++) {
    for (let c = 0; c < GRID_SIZE; c++) {
      const i = idx(r, c);
      if (grid[i] === 1) continue;
      const h = manhattan([r, c], goal);
      const cx = c * CELL + CELL / 2;
      const cy = r * CELL + CELL / 2;
      gctx.strokeStyle = "rgba(15, 20, 25, 0.55)";
      gctx.lineWidth = 3;
      gctx.strokeText(String(h), cx, cy);
      gctx.fillStyle =
        h === 0 ? "#0f1419" : r === gr && c === gc ? "#0f1419" : "#e6edf3";
      gctx.fillText(String(h), cx, cy);
    }
  }
  gctx.restore();
}

function drawPath(path, color, alpha = 0.4) {
  if (!path || path.length < 2) return;
  gctx.fillStyle = color;
  for (let i = 1; i < path.length - 1; i++) {
    const [r, c] = path[i];
    gctx.globalAlpha = alpha;
    gctx.fillRect(c * CELL + 4, r * CELL + 4, CELL - 8, CELL - 8);
  }
  gctx.globalAlpha = 1;
}

function drawForklift(r, c) {
  const x = c * CELL,
    y = r * CELL;
  gctx.fillStyle = "#f0883e";
  gctx.fillRect(x + 3, y + 5, CELL - 6, CELL - 8);
  gctx.fillStyle = "#c97a35";
  gctx.fillRect(x + 4, y + CELL - 6, CELL - 8, 3);
}

function drawQHeatOnGrid(Qvals, grid) {
  let mn = Infinity,
    mx = -Infinity;
  for (let i = 0; i < Qvals.length; i++) {
    if (grid[i] === 1 || Number.isNaN(Qvals[i])) continue;
    mn = Math.min(mn, Qvals[i]);
    mx = Math.max(mx, Qvals[i]);
  }
  if (mn === Infinity) return;
  const span = mx - mn || 1;
  for (let r = 0; r < GRID_SIZE; r++) {
    for (let c = 0; c < GRID_SIZE; c++) {
      const i = idx(r, c);
      if (grid[i] === 1) continue;
      const t = (Qvals[i] - mn) / span;
      const x = c * CELL,
        y = r * CELL;
      gctx.fillStyle = `hsla(${200 + t * 100}, 70%, ${35 + t * 25}%, 0.65)`;
      gctx.fillRect(x + 1, y + 1, CELL - 2, CELL - 2);
    }
  }
}

function drawGraphStructure(nodes, edges) {
  gxctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  gxctx.strokeStyle = "#30363d";
  gxctx.lineWidth = 1;
  for (const e of edges) {
    const ar = Math.floor(e.a / GRID_SIZE),
      ac = e.a % GRID_SIZE;
    const br = Math.floor(e.b / GRID_SIZE),
      bc = e.b % GRID_SIZE;
    gxctx.beginPath();
    gxctx.moveTo(ac * CELL + CELL / 2, ar * CELL + CELL / 2);
    gxctx.lineTo(bc * CELL + CELL / 2, br * CELL + CELL / 2);
    gxctx.stroke();
  }
  gxctx.fillStyle = "#8b949e";
  for (const n of nodes) {
    gxctx.beginPath();
    gxctx.arc(
      n.c * CELL + CELL / 2,
      n.r * CELL + CELL / 2,
      Math.max(2.5, CELL * 0.12),
      0,
      Math.PI * 2
    );
    gxctx.fill();
  }
}

/**
 * @param {{ allTraveledGreen?: boolean }} [vizOpts]
 */
function drawGraphExploration(order, upto, pathSet, vizOpts = {}) {
  const { grid } = state;
  const allGreen = vizOpts.allTraveledGreen === true;
  for (let k = 0; k < upto && k < order.length; k++) {
    const u = order[k];
    if (grid[u] === 1) continue;
    const r = Math.floor(u / GRID_SIZE),
      c = u % GRID_SIZE;
    if (allGreen) {
      gxctx.fillStyle = "rgba(63, 185, 80, 0.5)";
    } else {
      gxctx.fillStyle = pathSet.has(u)
        ? "rgba(63, 185, 80, 0.35)"
        : "rgba(88, 166, 255, 0.2)";
    }
    gxctx.beginPath();
    gxctx.arc(
      c * CELL + CELL / 2,
      r * CELL + CELL / 2,
      Math.max(4, CELL * 0.18),
      0,
      Math.PI * 2
    );
    gxctx.fill();
  }
}

function drawQvizHeat(Qvals, grid) {
  qvx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  let mn = Infinity,
    mx = -Infinity;
  for (let i = 0; i < Qvals.length; i++) {
    if (grid[i] === 1 || Number.isNaN(Qvals[i])) continue;
    mn = Math.min(mn, Qvals[i]);
    mx = Math.max(mx, Qvals[i]);
  }
  if (mn === Infinity) return;
  const span = mx - mn || 1;
  for (let r = 0; r < GRID_SIZE; r++) {
    for (let c = 0; c < GRID_SIZE; c++) {
      const i = idx(r, c);
      if (grid[i] === 1) continue;
      const t = (Qvals[i] - mn) / span;
      const x = c * CELL,
        y = r * CELL;
      qvx.fillStyle = `hsla(${260 - t * 80}, 65%, ${40 + t * 20}%, 0.9)`;
      qvx.fillRect(x, y, CELL, CELL);
    }
  }
}

/**
 * @param {{ heuristicGoal?: [number, number] | null }} [opts]
 */
async function animatePath(path, speedMs = 45, opts = {}) {
  const hGoal = opts.heuristicGoal ?? null;
  for (let i = 0; i < path.length; i++) {
    if (state.stopRequested) return;
    drawGridBase({ heuristicGoal: hGoal });
    drawPath(path, "rgba(88, 166, 255, 0.5)", 0.5);
    const [r, c] = path[i];
    drawForklift(r, c);
    await new Promise((resolve) => setTimeout(resolve, speedMs));
  }
}

/** Animate rollout path on top of current Q heatmap (Q-learning / Q-network). */
async function animateEpisodePathWithQ(path, Qmax, grid, speedMs = 28) {
  for (let i = 0; i < path.length; i++) {
    if (state.stopRequested) return;
    drawGridBase();
    drawQHeatOnGrid(Qmax, grid);
    drawPath(path, "rgba(88, 166, 255, 0.45)", 0.45);
    const [r, c] = path[i];
    drawForklift(r, c);
    await new Promise((resolve) => setTimeout(resolve, speedMs));
  }
}

/** Data-acquisition rollout on empty map (no Q heatmap). */
async function animateAcquisitionRollout(path, speedMs = 16) {
  if (!path || path.length < 2) return;
  for (let i = 0; i < path.length; i++) {
    if (state.stopRequested) return;
    drawGridBase();
    drawPath(path, "rgba(241, 143, 61, 0.48)", 0.48);
    const [r, c] = path[i];
    drawForklift(r, c);
    await new Promise((resolve) => setTimeout(resolve, speedMs));
  }
}

const ACTION_NAMES_NET = ["E", "S", "W", "N"];

function cellFromQStateVec(xv) {
  const inv = GRID_SIZE - 1 || 1;
  const r = Math.min(
    GRID_SIZE - 1,
    Math.max(0, Math.round(xv[1] * inv))
  );
  const c = Math.min(
    GRID_SIZE - 1,
    Math.max(0, Math.round(xv[2] * inv))
  );
  return [r, c];
}

function appendBufferSnippetToLog(buffer, episodeDone) {
  const maxTail = Math.min(buffer.length, 80);
  const start = buffer.length - maxTail;
  const lines = [
    `—— Replay buffer after ep ${episodeDone} (size ${buffer.length}) — showing last ${maxTail} transitions —`,
  ];
  for (let i = start; i < buffer.length; i++) {
    const t = buffer[i];
    const [r1, c1] = cellFromQStateVec(t.x);
    const [r2, c2] = cellFromQStateVec(t.x2);
    const an = ACTION_NAMES_NET[t.a] ?? String(t.a);
    lines.push(
      `  #${i} s=(${r1},${c1}) a=${an} r=${t.reward.toFixed(2)} s'=(${r2},${c2}) done=${t.done ? 1 : 0}`
    );
  }
  lines.push("");
  return lines.join("\n");
}

/**
 * @param {{
 *   worldHeuristicGoal?: [number, number] | null;
 *   allTraveledGreen?: boolean;
 *   onProgress?: (expandedVizPrefix: number) => void;
 * }} [opts]
 */
async function animateGraph(order, nodes, edges, finalPath, opts = {}) {
  const pathSet = new Set();
  for (const [r, c] of finalPath) pathSet.add(idx(r, c));
  const hGoal = opts.worldHeuristicGoal ?? null;
  const exploreViz = { allTraveledGreen: opts.allTraveledGreen === true };
  const onProgress = opts.onProgress;
  if (hGoal) {
    drawGridBase({ heuristicGoal: hGoal });
    drawForklift(state.start[0], state.start[1]);
  }
  drawGraphStructure(nodes, edges);
  const step = Math.max(1, Math.floor(order.length / 120));
  for (let u = 0; u <= order.length; u += step) {
    if (state.stopRequested) return;
    drawGraphStructure(nodes, edges);
    drawGraphExploration(order, u, pathSet, exploreViz);
    if (typeof onProgress === "function") onProgress(Math.min(u, order.length));
    await new Promise((resolve) => setTimeout(resolve, 8));
  }
  drawGraphStructure(nodes, edges);
  drawGraphExploration(order, order.length, pathSet, exploreViz);
  if (typeof onProgress === "function") onProgress(order.length);
}

function newMap() {
  const t = generateTerrain(0.22);
  state.grid = t.grid;
  state.start = t.start;
  state.goal = t.goal;
  state.Q = initQ();
  state.qnet = new QNetwork(32);
  drawGridBase();
  drawForklift(state.start[0], state.start[1]);
  gxctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  qvx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  setPathLength("—");
  setExplored("—");
  setLocalBest("—");
  els.episodeLog.textContent = "";
}

async function runAlgorithm() {
  if (state.running) return;
  state.running = true;
  state.stopRequested = false;
  els.btnRun.disabled = true;
  els.btnNew.disabled = true;

  const algo = els.algo.value;
  const { grid, start, goal } = state;

  try {
    if (algo === "bfs") {
      const res = bfs(grid, start, goal);
      const gi = goalCellIndex(goal);
      const firstGk = res.order.findIndex((id) => id === gi);
      if (res.path.length === 0) {
        setPathLength("No path");
        setExplored(`Nodes expanded: ${res.order.length}`);
        setLocalBest("—");
      } else {
        setPathLength(String(res.path.length - 1));
        setLocalBest("—");
        await animateGraph(res.order, res.nodes, res.edges, res.path, {
          allTraveledGreen: true,
          onProgress(upto) {
            setExplored(
              `Iteration (viz steps): ${upto}/${res.order.length} · total order: ${res.order.length}`
            );
            setLocalBest(
              firstGk >= 0 && upto > firstGk ? String(res.dist) : "—"
            );
          },
        });
        setExplored(`Nodes expanded: ${res.order.length}`);
        setLocalBest(String(res.dist));
        await animatePath(res.path);
      }
    } else if (algo === "dfs") {
      const res = dfs(grid, start, goal);
      const gi = goalCellIndex(goal);
      const firstGk = res.order.findIndex((id) => id === gi);
      if (res.path.length === 0) {
        setPathLength("No path");
        setExplored(`Nodes expanded: ${res.order.length}`);
        setLocalBest("—");
      } else {
        setPathLength(String(res.path.length - 1));
        setLocalBest("—");
        await animateGraph(res.order, res.nodes, res.edges, res.path, {
          onProgress(upto) {
            setExplored(
              `Iteration (viz steps): ${upto}/${res.order.length} · total order: ${res.order.length}`
            );
            setLocalBest(
              firstGk >= 0 && upto > firstGk ? String(res.dist) : "—"
            );
          },
        });
        setExplored(`Nodes expanded: ${res.order.length}`);
        setLocalBest(String(res.dist));
        await animatePath(res.path);
      }
    } else if (algo === "astar") {
      const res = astar(grid, start, goal);
      const gi = goalCellIndex(goal);
      const firstGk = res.order.findIndex((id) => id === gi);
      if (res.path.length === 0) {
        setPathLength("No path");
        setExplored(`Nodes expanded: ${res.order.length}`);
        setLocalBest("—");
      } else {
        setPathLength(String(res.path.length - 1));
        setLocalBest("—");
        await animateGraph(res.order, res.nodes, res.edges, res.path, {
          worldHeuristicGoal: goal,
          onProgress(upto) {
            setExplored(
              `Iteration (viz steps): ${upto}/${res.order.length} · total order: ${res.order.length}`
            );
            setLocalBest(
              firstGk >= 0 && upto > firstGk ? String(res.dist) : "—"
            );
          },
        });
        setExplored(`Nodes expanded: ${res.order.length}`);
        setLocalBest(String(res.dist));
        await animatePath(res.path, 40, { heuristicGoal: goal });
      }
    } else if (algo === "qlearning") {
      const episodes = Math.min(
        200,
        Math.max(1, parseInt(els.qEpisodesInput.value, 10) || 40)
      );
      let epsilon = 0.35;
      const alpha = 0.15;
      const gamma = 0.95;
      const pathFinalEpisodeOnly = els.qPathFinalOnly.checked === true;
      els.episodeLog.textContent = "";
      let localBestEpisodeSteps = Infinity;
      for (let ep = 1; ep <= episodes; ep++) {
        if (state.stopRequested) break;
        epsilon = Math.max(0.05, epsilon * 0.97);
        const out = runEpisode(grid, start, goal, state.Q, epsilon, alpha, gamma);
        state.Q = out.Q;
        if (out.reachedGoal && out.steps < localBestEpisodeSteps) {
          localBestEpisodeSteps = out.steps;
        }
        const Qmax = new Float32Array(GRID_SIZE * GRID_SIZE);
        for (let s = 0; s < GRID_SIZE * GRID_SIZE; s++) {
          Qmax[s] = maxQ(state.Q, s);
        }
        drawGridBase();
        drawQHeatOnGrid(Qmax, grid);
        drawForklift(start[0], start[1]);
        drawQvizHeat(Qmax, grid);
        const greedy = greedyPathFromQ(grid, start, goal, state.Q);
        const pl = greedy.ok ? greedy.dist : "∞ (policy loop)";
        setPathLength(String(pl));
        setExplored(
          `Episode ${ep}/${episodes} · steps ${out.steps} · reached ${out.reachedGoal ? "yes" : "no"}`
        );
        setLocalBest(
          localBestEpisodeSteps < Infinity
            ? String(localBestEpisodeSteps)
            : "—"
        );
        els.episodeLog.textContent += `ep ${ep}: steps=${out.steps} goal=${out.reachedGoal ? 1 : 0} ε=${epsilon.toFixed(3)} pathLen(greedy)=${greedy.ok ? greedy.dist : "—"}\n`;
        els.episodeLog.scrollTop = els.episodeLog.scrollHeight;
        const shouldAnimatePathEachGoal =
          out.reachedGoal && out.path.length > 1 && !pathFinalEpisodeOnly;
        const shouldAnimatePathFinal =
          out.reachedGoal &&
          out.path.length > 1 &&
          pathFinalEpisodeOnly &&
          ep === episodes;
        if (shouldAnimatePathEachGoal || shouldAnimatePathFinal) {
          await animateEpisodePathWithQ(out.path, Qmax, grid, 26);
          drawGridBase();
          drawQHeatOnGrid(Qmax, grid);
          drawForklift(start[0], start[1]);
          drawQvizHeat(Qmax, grid);
        } else {
          await new Promise((resolve) => setTimeout(resolve, 40));
        }
      }
      const final = greedyPathFromQ(grid, start, goal, state.Q);
      if (final.ok && final.path.length) {
        setPathLength(String(final.dist));
        if (localBestEpisodeSteps < Infinity) {
          setLocalBest(String(localBestEpisodeSteps));
        }
        await animatePath(final.path);
      }
    } else if (algo === "qnet") {
      const episodes = Math.min(
        200,
        Math.max(1, parseInt(els.qEpisodesInput.value, 10) || 40)
      );
      let epsilon = 0.4;
      const vizAcquisition = els.dqVizAcquisition.checked === true;
      els.episodeLog.textContent = "";
      const buffer = [];
      let localBestEpisodeSteps = Infinity;

      if (!vizAcquisition) {
        qvx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
      }
      drawGridBase();
      drawForklift(start[0], start[1]);

      els.episodeLog.textContent +=
        "--- Phase 1: data acquisition (no weight updates) ---\n";

      for (let ep = 1; ep <= episodes; ep++) {
        if (state.stopRequested) break;
        epsilon = Math.max(0.12, epsilon * 0.97);
        const out = collectEpisodeQNet(state.qnet, grid, start, goal, {
          epsilon,
          maxSteps: 400,
        });
        for (const t of out.transitions) buffer.push(t);
        if (out.reachedGoal && out.steps < localBestEpisodeSteps) {
          localBestEpisodeSteps = out.steps;
        }
        setExplored(
          `Acquisition · ep ${ep}/${episodes} · steps ${out.steps} · buffer ${buffer.length}`
        );
        setPathLength("—");
        setLocalBest(
          localBestEpisodeSteps < Infinity
            ? String(localBestEpisodeSteps)
            : "—"
        );
        if (vizAcquisition) {
          els.episodeLog.textContent += `acq ep ${ep}/${episodes}: steps=${out.steps} batch+=${out.transitions.length} reached=${out.reachedGoal ? "yes" : "no"}\n`;
          els.episodeLog.textContent += appendBufferSnippetToLog(buffer, ep);
          els.episodeLog.scrollTop = els.episodeLog.scrollHeight;
          if (out.path.length > 1) {
            await animateAcquisitionRollout(out.path, 16);
          }
          drawGridBase();
          drawForklift(start[0], start[1]);
          qvx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
        } else {
          els.episodeLog.textContent += `acq ep ${ep}: steps=${out.steps} buf+=${out.transitions.length} total=${buffer.length}\n`;
          els.episodeLog.scrollTop = els.episodeLog.scrollHeight;
        }
        await new Promise((resolve) => setTimeout(resolve, vizAcquisition ? 25 : 1));
      }

      if (vizAcquisition && buffer.length) {
        els.episodeLog.textContent += `\n—— End of phase 1: replay buffer holds ${buffer.length} transitions ——\n`;
        els.episodeLog.scrollTop = els.episodeLog.scrollHeight;
      }

      const gradSteps = Math.max(2500, buffer.length * 8);
      if (buffer.length === 0) {
        setExplored("No transitions collected — skipping training.");
        els.episodeLog.textContent +=
          "--- No data; run more episodes or check map. ---\n";
      } else {
        els.episodeLog.textContent += `\n--- Phase 2: training (${gradSteps} grad steps on buffer=${buffer.length}) ---\n`;
        let gradDone = 0;
        const trainChunk = 400;
        while (gradDone < gradSteps && !state.stopRequested) {
          const end = Math.min(gradDone + trainChunk, gradSteps);
          for (; gradDone < end; gradDone++) {
            const trans = buffer[(Math.random() * buffer.length) | 0];
            trainOneStepQNet(state.qnet, trans, 0.02, 0.95);
          }
          setExplored(
            `Training · grad steps ${Math.min(gradDone, gradSteps)}/${gradSteps} · buffer ${buffer.length}`
          );
          setLocalBest(
            localBestEpisodeSteps < Infinity
              ? String(localBestEpisodeSteps)
              : "—"
          );
          await new Promise((resolve) => setTimeout(resolve, 0));
        }
        els.episodeLog.textContent += "--- Training finished. ---\n";
      }

      if (buffer.length && !vizAcquisition) {
        els.episodeLog.textContent +=
          "\n--- Visualization (post-training): Q heatmap + greedy forklift path — acquisition phase had no canvas updates ---\n";
        els.episodeLog.scrollTop = els.episodeLog.scrollHeight;
      } else if (buffer.length && vizAcquisition) {
        els.episodeLog.textContent +=
          "\n--- Visualization (post-training): trained Q heatmap + greedy path ---\n";
        els.episodeLog.scrollTop = els.episodeLog.scrollHeight;
      }

      const Qfinal = cellMaxQGrid(state.qnet, grid, goal);
      drawGridBase();
      drawQHeatOnGrid(Qfinal, grid);
      drawForklift(start[0], start[1]);
      drawQvizHeat(Qfinal, grid);

      const final = greedyPathFromQNet(state.qnet, grid, start, goal);
      if (final.ok && final.path.length) {
        setPathLength(String(final.dist));
        setLocalBest(
          localBestEpisodeSteps < Infinity
            ? String(localBestEpisodeSteps)
            : "—"
        );
        setExplored(
          buffer.length ? `Training done · greedy path edges: ${final.dist}` : "—"
        );
        await animatePath(final.path);
      } else if (final.path.length) {
        setPathLength("∞ (policy loop)");
        setExplored(`Training done · greedy did not solve goal`);
        setLocalBest(
          localBestEpisodeSteps < Infinity
            ? String(localBestEpisodeSteps)
            : "—"
        );
      }
    }
  } finally {
    state.running = false;
    els.btnRun.disabled = false;
    els.btnNew.disabled = false;
  }
}

els.algo.addEventListener("change", updateDesc);
els.btnNew.addEventListener("click", newMap);
els.btnRun.addEventListener("click", runAlgorithm);
els.btnStop.addEventListener("click", () => {
  state.stopRequested = true;
});

updateDesc();
newMap();
