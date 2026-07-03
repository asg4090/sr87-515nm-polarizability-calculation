const app = {
  data: null,
  paramValues: [],
};

const $ = (id) => document.getElementById(id);
const fmt = (value, digits = 3) => Number.isFinite(value) ? value.toFixed(digits) : "--";
const RESONANCES_3P2_NM = [496.2263, 496.7944, 497.1668];
const SELECTED_ROOT_TARGET_NM = 515.2;
const MODELS = ["current_sos", "sr1pol_projected"];
const MODEL_LABELS = {
  current_sos: "Current SOS",
  sr1pol_projected: "Sr1Pol projected",
};
const MODEL_COLORS = {
  current_sos: "#2455c3",
  sr1pol_projected: "#12805c",
};

function unique(values) {
  return [...new Set(values)].sort((a, b) => String(a).localeCompare(String(b), undefined, { numeric: true }));
}

function rationalValue(text) {
  if (String(text).includes("/")) {
    const [a, b] = String(text).split("/").map(Number);
    return a / b;
  }
  return Number(text);
}

function interpolation(xs, ys, x) {
  if (!Number.isFinite(x) || !xs.length || x < xs[0] || x > xs[xs.length - 1]) return NaN;
  const exact = Math.round((x - xs[0]) / (xs[1] - xs[0]));
  if (Math.abs(xs[exact] - x) < 1e-10) return ys[exact];
  let lo = 0;
  let hi = xs.length - 1;
  while (hi - lo > 1) {
    const mid = Math.floor((lo + hi) / 2);
    if (xs[mid] <= x) lo = mid;
    else hi = mid;
  }
  const t = (x - xs[lo]) / (xs[hi] - xs[lo]);
  return ys[lo] + t * (ys[hi] - ys[lo]);
}

function populateSelect(el, values, preferred) {
  const oldValue = preferred ?? el.value;
  el.innerHTML = "";
  for (const value of values) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    el.appendChild(option);
  }
  if (values.includes(oldValue)) el.value = oldValue;
}

function stateOptions(state) {
  return app.data.levels[state].F;
}

function currentParameter() {
  return app.paramValues[Number($("parameter").value)];
}

function selectedModel() {
  return $("sourceModel").value;
}

function updateParameterGrid() {
  const family = $("polarizationFamily").value;
  const rows = app.data.magicComparison.filter((row) => row.polarization_family === family);
  app.paramValues = unique(rows.map((row) => Number(row.parameter_deg))).map(Number);
  const old = Number($("parameterValue").value);
  $("parameter").min = 0;
  $("parameter").max = Math.max(app.paramValues.length - 1, 0);
  $("parameter").step = 1;
  let index = 0;
  if (Number.isFinite(old)) {
    const nearest = app.paramValues.reduce((best, value, i) => {
      const distance = Math.abs(value - old);
      return distance < best.distance ? { i, distance } : best;
    }, { i: 0, distance: Infinity });
    index = nearest.i;
  } else if (family === "linear_xz") {
    index = app.paramValues.findIndex((v) => Math.abs(v - 54.7356) < 1e-6);
    if (index < 0) index = 0;
  } else {
    index = app.paramValues.findIndex((v) => Math.abs(v) < 1e-6);
  }
  $("parameter").value = String(Math.max(index, 0));
  populateSelect($("parameterValue"), app.paramValues.map((v) => String(v)), String(app.paramValues[Math.max(index, 0)]));
  $("parameterLabel").textContent = family === "linear_xz" ? "theta index" : "gamma index";
}

function updateFAndMfOptions() {
  const state = $("excitedState").value;
  const fValues = stateOptions(state).map((row) => row.F);
  populateSelect($("excitedF"), fValues, $("excitedF").value || "9/2");
  updateTransitionMfOptions();
}

function mfValuesFor(state, f) {
  return stateOptions(state).find((row) => row.F === f)?.mF.map((row) => row.mF) ?? [];
}

function transitionPairs(deltaMf, state = $("excitedState").value, f = $("excitedF").value) {
  const excitedMfs = new Set(mfValuesFor(state, f));
  return mfValuesFor("5s2 1S0", "9/2")
    .map((groundMf) => {
      const excitedValue = rationalValue(groundMf) + deltaMf;
      const excitedMf = mfValuesFor(state, f).find((value) => Math.abs(rationalValue(value) - excitedValue) < 1e-9);
      return excitedMf && excitedMfs.has(excitedMf) ? { groundMf, excitedMf } : null;
    })
    .filter(Boolean);
}

function updateTransitionMfOptions() {
  const deltaMf = Number($("deltaMf").value);
  const pairs = transitionPairs(deltaMf);
  const oldGround = $("groundMf").value || "1/2";
  const groundValues = pairs.map((pair) => pair.groundMf);
  populateSelect($("groundMf"), groundValues, oldGround);
  updateExcitedMfFromGround();
}

function updateExcitedMfFromGround() {
  const deltaMf = Number($("deltaMf").value);
  const pairs = transitionPairs(deltaMf);
  const groundMf = $("groundMf").value;
  const pair = pairs.find((item) => item.groundMf === groundMf) ?? pairs[0];
  populateSelect($("excitedMf"), pair ? [pair.excitedMf] : [], pair?.excitedMf);
}

function getWeights(state, f, mf) {
  const fRow = stateOptions(state).find((row) => row.F === f);
  const mfRow = fRow?.mF.find((row) => row.mF === mf);
  return mfRow?.weights ?? [{ mJ: 0, weight: 1 }];
}

function polarizationFactors(condition = selectedCondition()) {
  const family = condition.polarizationFamily;
  const deg = condition.parameterDeg;
  if (family === "linear_xz") {
    const theta = deg * Math.PI / 180;
    return {
      circ: 0,
      tensor: (3 * Math.cos(theta) ** 2 - 1) / 2,
      label: `theta=${fmt(deg, deg % 1 ? 4 : 0)}°`,
    };
  }
  const gamma = deg * Math.PI / 180;
  const circ = Math.sin(2 * gamma);
  return {
    circ,
    tensor: -0.5,
    label: `gamma=${fmt(deg, deg % 1 ? 3 : 0)}°`,
  };
}

function alphaByMJ(model, state, mj, condition = selectedCondition()) {
  const comps = app.data.models[model].components[state];
  const j = app.data.levels[state].J;
  const factors = polarizationFactors(condition);
  const alpha0 = comps.alpha0;
  if (j === 0) return alpha0;
  const tensorState = (3 * mj * mj - j * (j + 1)) / (j * (2 * j - 1));
  return alpha0.map((a0, i) => (
    a0
    + comps.alpha1[i] * factors.circ * (mj / j)
    + comps.alpha2[i] * factors.tensor * tensorState
  ));
}

function hyperfineAlpha(model, state, f, mf, condition = selectedCondition()) {
  const wavelengths = app.data.wavelengths_nm;
  const total = new Array(wavelengths.length).fill(0);
  for (const { mJ, weight } of getWeights(state, f, mf)) {
    const values = alphaByMJ(model, state, Number(mJ), condition);
    for (let i = 0; i < total.length; i += 1) total[i] += weight * values[i];
  }
  return total;
}

function deltaAlphaSeries(model, condition = selectedCondition()) {
  const ground = hyperfineAlpha(model, "5s2 1S0", "9/2", condition.groundMf, condition);
  const excited = hyperfineAlpha(model, condition.excitedState, condition.excitedF, condition.excitedMf, condition);
  return excited.map((value, i) => value - ground[i]);
}

function selectedCondition() {
  return {
    polarizationFamily: $("polarizationFamily").value,
    parameterDeg: currentParameter(),
    deltaMf: Number($("deltaMf").value),
    groundMf: $("groundMf").value,
    excitedState: $("excitedState").value,
    excitedF: $("excitedF").value,
    excitedMf: $("excitedMf").value,
  };
}

function resonanceRisk(excitedState, rootNm, leftNm, rightNm) {
  if (excitedState !== "5s5p 3P2") return "clean";
  for (const resonance of RESONANCES_3P2_NM) {
    if (Math.abs(rootNm - resonance) <= 0.30 || (Math.min(leftNm, rightNm) <= resonance && resonance <= Math.max(leftNm, rightNm))) {
      return "near_3P2_resonance";
    }
  }
  return "clean";
}

function zeroCrossings(delta, condition) {
  const xs = app.data.wavelengths_nm;
  const rows = [];
  for (let i = 0; i < xs.length - 1; i += 1) {
    const x0 = xs[i];
    const x1 = xs[i + 1];
    const y0 = delta[i];
    const y1 = delta[i + 1];
    if (!Number.isFinite(y0) || !Number.isFinite(y1)) continue;
    let root = NaN;
    if (y0 === 0) root = x0;
    else if (y0 * y1 > 0 || y1 === y0) continue;
    else root = x0 - y0 * (x1 - x0) / (y1 - y0);
    rows.push({
      crossing_index: rows.length + 1,
      magic_wavelength_nm: root,
      left_wavelength_nm: x0,
      right_wavelength_nm: x1,
      left_delta_alpha_au: y0,
      right_delta_alpha_au: y1,
      slope_au_per_nm: (y1 - y0) / (x1 - x0),
      risk_flag: resonanceRisk(condition.excitedState, root, x0, x1),
    });
  }
  return rows;
}

function selectedRoot(roots) {
  const clean = roots.filter((row) => row.risk_flag === "clean");
  const pool = clean.length ? clean : roots;
  if (!pool.length) return null;
  return pool.reduce((best, row) => {
    const distance = Math.abs(row.magic_wavelength_nm - SELECTED_ROOT_TARGET_NM);
    return !best || distance < best.distance ? { row, distance } : best;
  }, null).row;
}

function rootResult(model, condition = selectedCondition()) {
  const delta = deltaAlphaSeries(model, condition);
  const roots = zeroCrossings(delta, condition);
  const selected = selectedRoot(roots);
  return { model, delta, roots, selected };
}

function resizeCanvas(canvas) {
  const ratio = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(320, Math.floor(rect.width * ratio));
  const height = Math.max(220, Math.floor((rect.width * 0.42) * ratio));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  return ratio;
}

function drawAxes(ctx, width, height, xMin, xMax, yMin, yMax, pad) {
  ctx.strokeStyle = "#cbd5e1";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.l, pad.t);
  ctx.lineTo(pad.l, height - pad.b);
  ctx.lineTo(width - pad.r, height - pad.b);
  ctx.stroke();

  ctx.fillStyle = "#64748b";
  ctx.font = "12px system-ui";
  ctx.textAlign = "center";
  for (let i = 0; i <= 5; i += 1) {
    const x = xMin + (xMax - xMin) * i / 5;
    const px = pad.l + (x - xMin) / (xMax - xMin) * (width - pad.l - pad.r);
    ctx.fillText(x.toFixed(1), px, height - 14);
  }
  ctx.textAlign = "right";
  for (let i = 0; i <= 5; i += 1) {
    const y = yMin + (yMax - yMin) * i / 5;
    const py = height - pad.b - (y - yMin) / (yMax - yMin) * (height - pad.t - pad.b);
    ctx.fillText(y.toFixed(0), pad.l - 8, py + 4);
    ctx.strokeStyle = "#eef2f7";
    ctx.beginPath();
    ctx.moveTo(pad.l, py);
    ctx.lineTo(width - pad.r, py);
    ctx.stroke();
  }
}

function drawDeltaChart(result) {
  const canvas = $("deltaChart");
  const ratio = resizeCanvas(canvas);
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  const pad = { l: 70 * ratio, r: 22 * ratio, t: 26 * ratio, b: 48 * ratio };
  const xs = app.data.wavelengths_nm;
  const values = result.delta;
  const ys = values.filter(Number.isFinite);
  let yMin = Math.min(...ys);
  let yMax = Math.max(...ys);
  if (yMin === yMax) { yMin -= 1; yMax += 1; }
  const margin = Math.max((yMax - yMin) * 0.08, 10);
  yMin -= margin;
  yMax += margin;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, width, height);
  const xMin = xs[0];
  const xMax = xs[xs.length - 1];
  const sx = (x) => pad.l + (x - xMin) / (xMax - xMin) * (width - pad.l - pad.r);
  const sy = (y) => height - pad.b - (y - yMin) / (yMax - yMin) * (height - pad.t - pad.b);
  drawAxes(ctx, width, height, xMin, xMax, yMin, yMax, pad);

  if (yMin < 0 && yMax > 0) {
    ctx.strokeStyle = "#334155";
    ctx.setLineDash([5 * ratio, 5 * ratio]);
    ctx.beginPath();
    ctx.moveTo(pad.l, sy(0));
    ctx.lineTo(width - pad.r, sy(0));
    ctx.stroke();
    ctx.setLineDash([]);
  }

  function drawLine(values, color) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.2 * ratio;
    ctx.beginPath();
    values.forEach((value, i) => {
      const x = sx(xs[i]);
      const y = sy(value);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }
  drawLine(values, MODEL_COLORS[result.model] ?? "#2455c3");

  const roots = [
    { value: result.selected?.magic_wavelength_nm, color: MODEL_COLORS[result.model] ?? "#2455c3" },
  ];
  for (const root of roots) {
    if (!Number.isFinite(root.value)) continue;
    const x = sx(root.value);
    ctx.strokeStyle = root.color;
    ctx.lineWidth = 1.4 * ratio;
    ctx.setLineDash([3 * ratio, 4 * ratio]);
    ctx.beginPath();
    ctx.moveTo(x, pad.t);
    ctx.lineTo(x, height - pad.b);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  const readout = Number($("wavelength").value);
  if (Number.isFinite(readout) && readout >= xMin && readout <= xMax) {
    const x = sx(readout);
    ctx.strokeStyle = "#b42318";
    ctx.lineWidth = 1.2 * ratio;
    ctx.beginPath();
    ctx.moveTo(x, pad.t);
    ctx.lineTo(x, height - pad.b);
    ctx.stroke();
  }

  ctx.fillStyle = "#64748b";
  ctx.textAlign = "left";
  ctx.font = `${12 * ratio}px system-ui`;
  ctx.fillText("wavelength (nm)", pad.l, height - 14 * ratio);
  ctx.save();
  ctx.translate(18 * ratio, height / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("Delta alpha (a.u.)", 0, 0);
  ctx.restore();
}

function riskClass(flag) {
  return flag === "clean" ? "clean" : "risk";
}

function renderCards(result) {
  const wavelength = Number($("wavelength").value);
  const delta = interpolation(app.data.wavelengths_nm, result.delta, wavelength);
  const root = result.selected;
  const cleanCount = result.roots.filter((row) => row.risk_flag === "clean").length;
  $("selectedMagic").textContent = root ? `${fmt(root.magic_wavelength_nm, 6)} nm` : "--";
  $("selectedMagicNote").innerHTML = root
    ? `<span class="${riskClass(root.risk_flag)}">${root.risk_flag}</span>, selected from ${cleanCount} clean roots and ${result.roots.length} total roots`
    : "no crossing";
  $("deltaAtLambda").textContent = Number.isFinite(delta) ? `${fmt(delta, 4)} a.u.` : "--";
  $("deltaAtLambdaNote").textContent = Number.isFinite(delta)
    ? `${MODEL_LABELS[result.model]} at ${fmt(wavelength, 3)} nm`
    : "Wavelength outside the data window; no extrapolation";
  $("sourceCard").textContent = MODEL_LABELS[result.model] ?? result.model;
  $("sourceCardNote").textContent = result.model === "current_sos"
    ? "Project SOS dataset"
    : "Sr1Pol electronic grid with Sr-87 projection";
}

function renderRootsTable(result) {
  const roots = result.roots;
  const cleanCount = roots.filter((row) => row.risk_flag === "clean").length;
  const flaggedCount = roots.length - cleanCount;
  $("rootsSummary").textContent = roots.length === 1
    ? "1 root found; details are optional"
    : `${roots.length} roots found (${cleanCount} clean, ${flaggedCount} flagged)`;
  if (!roots.length) {
    $("rootsSummary").textContent = "No roots found";
    $("rootsTable").innerHTML = "<p class='notes'>No roots for the current condition.</p>";
    return;
  }
  $("rootsTable").innerHTML = `<table><thead><tr>
    <th>#</th><th>magic nm</th><th>risk</th><th>bracket nm</th><th>slope au/nm</th>
  </tr></thead><tbody>${roots.map((row) => `<tr>
    <td>${row.crossing_index}</td>
    <td>${fmt(row.magic_wavelength_nm, 6)}</td>
    <td class="${riskClass(row.risk_flag)}">${row.risk_flag}</td>
    <td>${fmt(row.left_wavelength_nm, 3)}-${fmt(row.right_wavelength_nm, 3)}</td>
    <td>${fmt(row.slope_au_per_nm, 3)}</td>
  </tr>`).join("")}</tbody></table>`;
}

function renderCandidateTable() {
  const condition = selectedCondition();
  const model = selectedModel();
  const cleanOnly = $("cleanOnly").checked;
  const rows = [];
  for (const fRow of stateOptions(condition.excitedState)) {
    for (const pair of transitionPairs(condition.deltaMf, condition.excitedState, fRow.F)) {
      const rowCondition = { ...condition, excitedF: fRow.F, groundMf: pair.groundMf, excitedMf: pair.excitedMf };
      const result = rootResult(model, rowCondition);
      const root = result.selected;
      if (cleanOnly && (!root || root.risk_flag !== "clean")) continue;
      rows.push({
        excitedF: fRow.F,
        groundMf: pair.groundMf,
        excitedMf: pair.excitedMf,
        root,
        rootsTotal: result.roots.length,
      });
    }
  }
  const limitedRows = rows
    .sort((a, b) => {
      const da = (a.excitedF === condition.excitedF ? 0 : 10) + (a.groundMf === condition.groundMf ? 0 : 3) + Math.abs(rationalValue(a.groundMf));
      const db = (b.excitedF === condition.excitedF ? 0 : 10) + (b.groundMf === condition.groundMf ? 0 : 3) + Math.abs(rationalValue(b.groundMf));
      return da - db;
    })
    .slice(0, 80);
  $("candidateTable").innerHTML = `<table><thead><tr>
    <th>F</th><th>transition</th><th>magic nm</th><th>roots</th><th>risk</th>
  </tr></thead><tbody>${limitedRows.map((row) => {
    const active = row.excitedF === condition.excitedF && row.groundMf === condition.groundMf && row.excitedMf === condition.excitedMf;
    return `<tr class="${active ? "activeRow" : ""}">
      <td>${row.excitedF}</td>
      <td>${row.groundMf} -> ${row.excitedMf}</td>
      <td>${row.root ? fmt(row.root.magic_wavelength_nm, 6) : "--"}</td>
      <td>${row.rootsTotal}</td>
      <td class="${riskClass(row.root?.risk_flag)}">${row.root?.risk_flag ?? "no root"}</td>
    </tr>`;
  }).join("")}</tbody></table>`;
}

function render() {
  if (!app.data) return;
  $("parameterValue").value = String(currentParameter());
  const wavelength = Number($("wavelength").value);
  const min = app.data.metadata.wl_min_nm;
  const max = app.data.metadata.wl_max_nm;
  if (wavelength < min || wavelength > max) {
    $("dataStatus").innerHTML = `<span class="warnText">Wavelength outside ${min}-${max} nm; no extrapolation</span>`;
  } else {
    const condition = selectedCondition();
    $("dataStatus").textContent = `${MODEL_LABELS[selectedModel()]}, ${$("polarizationFamily").value}, ${polarizationFactors().label}, Delta mF=${condition.deltaMf}`;
  }
  const result = rootResult(selectedModel());
  renderCards(result);
  drawDeltaChart(result);
  renderRootsTable(result);
  renderCandidateTable();
  $("curveLegend").textContent = MODEL_LABELS[selectedModel()];
  $("curveLegendColor").style.background = MODEL_COLORS[selectedModel()] ?? "#2455c3";
}

function init() {
  populateSelect($("excitedState"), app.data.excitedStates, "5s5p 3P1");
  updateParameterGrid();
  updateFAndMfOptions();
  $("gridMeta").textContent = `${app.data.metadata.wl_min_nm}-${app.data.metadata.wl_max_nm} nm, step ${app.data.metadata.wl_step_nm} nm`;
  $("dataStatus").textContent = "Data loaded";

  $("sourceModel").addEventListener("change", render);
  $("excitedState").addEventListener("change", () => { updateFAndMfOptions(); render(); });
  $("excitedF").addEventListener("change", () => { updateTransitionMfOptions(); render(); });
  $("deltaMf").addEventListener("change", () => { updateTransitionMfOptions(); render(); });
  $("groundMf").addEventListener("change", () => { updateExcitedMfFromGround(); render(); });
  $("excitedMf").addEventListener("change", render);
  $("polarizationFamily").addEventListener("change", () => { updateParameterGrid(); render(); });
  $("parameter").addEventListener("input", render);
  $("parameterValue").addEventListener("change", () => {
    const value = Number($("parameterValue").value);
    const idx = app.paramValues.findIndex((v) => Math.abs(v - value) < 1e-8);
    if (idx >= 0) $("parameter").value = String(idx);
    render();
  });
  $("wavelength").addEventListener("input", render);
  $("cleanOnly").addEventListener("change", render);
  window.addEventListener("resize", render);
  render();
}

fetch("data/magic_web_data.json")
  .then((response) => {
    if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
    return response.json();
  })
  .then((payload) => {
    app.data = payload;
    init();
  })
  .catch((error) => {
    $("dataStatus").innerHTML = `<span class="warnText">Failed to load data: ${error}</span>`;
  });
