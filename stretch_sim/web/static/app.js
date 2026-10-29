'use strict';
/*
 * Stretch Sim web viewer. Gets the static world once, then a stream of
 * {state, events} from /api/stream, and draws the map, the robot panels and the log.
 * The map is SVG in centimetres (world metres * U), y flipped so north is up.
 */

const U = 100;                       // SVG units per metre
const SVGNS = 'http://www.w3.org/2000/svg';
const KINDS = ['command', 'result', 'step', 'state', 'violation', 'info', 'bt', 'belief'];
const MAX_ROWS = 2500;

const $ = (sel) => document.querySelector(sel);
const fmt = (v, d = 2) => (v == null ? '—' : Number(v).toFixed(d));
const deg = (rad) => rad * 180 / Math.PI;

let world = null;
let state = null;
let H = 0;                            // world height, metres
const sx = (x) => x * U;
const sy = (y) => (H - y) * U;

/* ------------------------------------------------------------------ API */
async function sendCommand(cmd) {
  try {
    const res = await fetch('/api/command', {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(cmd),
    });
    const body = await res.json();
    if (!body.ok) toast(body.message, 'error');
    return body;
  } catch (err) {
    toast(`Command failed: ${err}`, 'error');
    return {ok: false};
  }
}

function toast(message, kind = '') {
  const el = document.createElement('div');
  el.className = `toast ${kind}`;
  el.textContent = message;
  $('#toasts').appendChild(el);
  setTimeout(() => el.remove(), 4000);
  while ($('#toasts').children.length > 4) $('#toasts').firstChild.remove();
}

/* ------------------------------------------------------------------ SVG helpers */
function svg(tag, attrs = {}, parent = null) {
  const el = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  if (parent) parent.appendChild(el);
  return el;
}

function rectAttrs(r) {
  return {x: sx(r.cx - r.w / 2), y: sy(r.cy + r.d / 2), width: r.w * U, height: r.d * U};
}

function points(list) {
  return list.map(([x, y]) => `${sx(x).toFixed(1)},${sy(y).toFixed(1)}`).join(' ');
}

/* ------------------------------------------------------------------ Map */
const map = {
  el: null, layers: {}, view: {x: 0, y: 0, w: 1, h: 1}, objects: new Map(), robot: null,
  drag: null, pan: null, hover: null, selected: null, missed: new Set(),
};

function buildMap() {
  const el = $('#map');
  map.el = el;
  el.innerHTML = '';
  for (const name of ['floor', 'static', 'path', 'robot', 'objects', 'overlay']) {
    map.layers[name] = svg('g', {class: `layer-${name}`}, el);
  }
  const L = map.layers;
  svg('rect', {class: 'floor', x: 0, y: 0, width: world.width * U, height: world.height * U}, L.floor);
  // The camera cannot see through the outer walls.
  const clip = svg('clipPath', {id: 'room-clip'}, svg('defs', {}, el));
  svg('rect', {x: 0, y: 0, width: world.width * U, height: world.height * U}, clip);

  for (const o of world.obstacles) {
    const f = svg('rect', {class: 'furniture', ...rectAttrs(o.rect), rx: 4}, L.static);
    if (o.color) { f.setAttribute('fill', o.color); f.style.fill = o.color; f.style.fillOpacity = 'var(--surface-opacity)'; }
    const t = svg('text', {class: 'furniture-label', x: sx(o.rect.cx), y: sy(o.rect.cy)}, L.static);
    t.textContent = o.name;
  }

  map.locations = {};
  for (const loc of world.locations) {
    const g = svg('g', {class: 'location', 'data-location': loc.name}, L.static);
    const r = loc.rect;
    if (loc.kind === 'person') {
      svg('circle', {class: 'surface', cx: sx(r.cx), cy: sy(r.cy), r: Math.min(r.w, r.d) / 2 * U, fill: loc.color}, g);
      const face = svg('text', {class: 'person-face', x: sx(r.cx), y: sy(r.cy)}, g);
      face.textContent = '☺';
      const label = svg('text', {class: 'location-label', x: sx(r.cx), y: sy(r.cy + r.d / 2) - 8}, g);
      label.textContent = loc.name;
    } else {
      svg('rect', {class: 'surface', ...rectAttrs(r), rx: 3, fill: loc.color}, g);
      // Centred: objects fill the zone from the edge the robot parks at, so the middle stays clear.
      const label = svg('text', {class: 'location-label', x: sx(r.cx), y: sy(r.cy) - 2}, g);
      label.textContent = loc.name;
    }
    const count = svg('text', {class: 'location-count', x: sx(r.cx), y: loc.kind === 'person' ? sy(r.cy + r.d / 2) - 24 : sy(r.cy) + 13}, g);
    const btLabel = svg('text', {class: 'bt-target-label', x: sx(r.cx), y: sy(r.cy - r.d / 2) - 8}, L.overlay);
    map.locations[loc.name] = {g, count, btLabel};
    g.addEventListener('click', (e) => {
      if (map.suppressClick) return;
      e.stopPropagation();
      sendCommand({cmd: 'goto', location: loc.name});
    });

    // Where the robot parks for this location
    const a = loc.approach;
    const b = world.robot.base_size * U;
    const ap = svg('g', {class: 'approach-g', transform: `translate(${sx(a.x)},${sy(a.y)}) rotate(${-deg(a.yaw)})`}, L.static);
    svg('rect', {class: 'approach', x: -b / 2, y: -b / 2, width: b, height: b, rx: 4}, ap);
    svg('line', {class: 'approach', x1: 0, y1: 0, x2: b * 0.7, y2: 0}, ap);
  }

  for (const [a, b] of world.walls) {
    svg('line', {class: 'wall', x1: sx(a[0]), y1: sy(a[1]), x2: sx(b[0]), y2: sy(b[1])}, L.static);
  }

  map.path = svg('polyline', {class: 'path-line', points: ''}, L.path);
  map.goal = svg('circle', {class: 'path-goal', r: 5, cx: -100, cy: -100}, L.path);
  buildRobot();
  map.objects.clear();

  fitView();
  bindMapEvents();
}

function buildRobot() {
  const L = map.layers.robot;
  const b = world.robot.base_size * U;
  const g = svg('g', {class: 'robot'}, L);
  const arm = svg('rect', {class: 'robot-arm', x: -3, y: 0, width: 6, height: 0}, g);
  const fingers = svg('g', {}, g);
  const f1 = svg('line', {class: 'robot-finger'}, fingers);
  const f2 = svg('line', {class: 'robot-finger'}, fingers);
  svg('rect', {class: 'robot-body', x: -b / 2, y: -b / 2, width: b, height: b, rx: 6}, g);
  svg('line', {class: 'robot-front', x1: b / 2 - 4, y1: -b / 3, x2: b / 2 - 4, y2: b / 3}, g);
  const gazeG = svg('g', {}, g);
  svg('line', {class: 'robot-gaze', x1: 0, y1: 0, x2: 26, y2: 0}, gazeG);
  svg('circle', {class: 'robot-head', cx: 0, cy: 0, r: 6}, g);
  map.robot = {g, arm, f1, f2, gazeG};
}

function updateRobot(r) {
  const R = map.robot;
  R.g.setAttribute('transform', `translate(${sx(r.x).toFixed(1)},${sy(r.y).toFixed(1)}) rotate(${(-deg(r.yaw)).toFixed(2)})`);
  // The arm extends to the robot's right, which is +y in the flipped SVG frame.
  const reach = (world.robot.arm_min_reach + r.arm) * U;
  R.arm.setAttribute('height', reach.toFixed(1));
  const half = 4 + r.aperture / 2 * U;
  R.f1.setAttribute('x1', -half); R.f1.setAttribute('y1', reach); R.f1.setAttribute('x2', -half); R.f1.setAttribute('y2', reach + 7);
  R.f2.setAttribute('x1', half); R.f2.setAttribute('y1', reach); R.f2.setAttribute('x2', half); R.f2.setAttribute('y2', reach + 7);
  R.gazeG.setAttribute('transform', `rotate(${(-deg(r.pan)).toFixed(2)})`);
}

function abbreviations(names) {
  const out = {};
  const used = new Set();
  for (const n of [...names].sort()) {
    const words = n.split(' ');
    let abbr = words.length === 1 ? words[0].slice(0, 3) : words[0].slice(0, 2) + words.slice(1).map((w) => w[0]).join('');
    let len = abbr.length;
    while (used.has(abbr)) abbr = n.replace(/ /g, '').slice(0, ++len);
    used.add(abbr);
    out[n] = abbr;
  }
  return out;
}

function updateObjects(objects) {
  if (!map.abbr || map.objects.size !== objects.length) map.abbr = abbreviations(objects.map((o) => o.name));
  const seen = new Set();
  for (const o of objects) {
    seen.add(o.name);
    let e = map.objects.get(o.name);
    if (!e) {
      const g = svg('g', {class: `obj ${o.category === 'Tools' ? 'tool' : 'ingredient'}`, 'data-name': o.name}, map.layers.objects);
      const r = Math.max(o.size / 2, 0.035) * U;
      const ring = svg('circle', {class: 'det-ring', r: r + 3.5, cx: 0, cy: 0}, g);
      const missRing = svg('circle', {class: 'miss-ring', r: r + 3.5, cx: 0, cy: 0}, g);
      const sel = svg('circle', {class: 'sel-ring', r: r + 6, cx: 0, cy: 0}, g);
      if (o.category === 'Tools') svg('rect', {class: 'shape', x: -r, y: -r, width: 2 * r, height: 2 * r, rx: 1.5}, g);
      else svg('circle', {class: 'shape', r, cx: 0, cy: 0}, g);
      const label = svg('text', {class: 'obj-label', x: 0, y: r + 8}, g);
      label.textContent = map.abbr[o.name];
      g.addEventListener('pointerenter', () => { map.hover = o.name; });
      g.addEventListener('pointerleave', () => { if (map.hover === o.name) map.hover = null; hideTooltip(); });
      g.addEventListener('pointerdown', (ev) => startObjectDrag(ev, o.name));
      e = {g, ring, missRing, sel, data: o};
      map.objects.set(o.name, e);
    }
    e.data = o;
    e.g.setAttribute('transform', `translate(${sx(o.x).toFixed(1)},${sy(o.y).toFixed(1)})`);
    e.ring.style.display = o.detected ? '' : 'none';
    e.missRing.style.display = map.missed.has(o.name) ? '' : 'none';
    e.sel.style.display = map.selected === o.name ? '' : 'none';
  }
  for (const [name, e] of map.objects) {
    if (!seen.has(name)) { e.g.remove(); map.objects.delete(name); }
  }
}

function updateMap(s) {
  const scanned = s.scan ? s.scan.surface : null;
  const missed = new Set(s.scan ? s.scan.missed : []);
  for (const [name, {g}] of Object.entries(map.locations)) g.classList.toggle('scanned', name === scanned);
  map.missed = missed;
  map.path.setAttribute('points', s.path.length > 1 ? points(s.path) : '');
  if (s.path.length > 1) {
    const [gx, gy] = s.path[s.path.length - 1];
    map.goal.setAttribute('cx', sx(gx)); map.goal.setAttribute('cy', sy(gy));
  } else {
    map.goal.setAttribute('cx', -100); map.goal.setAttribute('cy', -100);
  }
  updateRobot(s.robot);
  updateObjects(s.objects);

  const target = s.beliefs && s.beliefs.target_object ? s.beliefs : null;
  for (const [name, {g, btLabel}] of Object.entries(map.locations)) {
    const on = !!target && target.target_location === name;
    g.classList.toggle('bt-target', on);
    btLabel.textContent = on ? `looking for ${target.target_object}` : '';
  }

  const counts = {};
  for (const o of s.objects) counts[o.location] = (counts[o.location] || 0) + 1;
  for (const [name, {count}] of Object.entries(map.locations)) {
    const n = counts[name] || 0;
    count.textContent = n ? `${n} object${n === 1 ? '' : 's'}` : '';
  }

  if ($('#opt-follow-robot').checked && !map.pan) centreOn(s.robot.x, s.robot.y);
  if (map.hover) showTooltip(map.hover);
}

/* View: zoom and pan by changing the viewBox */
function applyView() {
  const v = map.view;
  map.el.setAttribute('viewBox', `${v.x} ${v.y} ${v.w} ${v.h}`);
}

function fitView() {
  const rect = map.el.getBoundingClientRect();
  const pad = 30;
  const W = world.width * U, Hh = world.height * U;
  const aspect = rect.width / Math.max(rect.height, 1) || W / Hh;
  let w = W + 2 * pad, h = Hh + 2 * pad;
  if (w / h < aspect) w = h * aspect; else h = w / aspect;
  map.view = {x: W / 2 - w / 2, y: Hh / 2 - h / 2, w, h};
  applyView();
}

function centreOn(x, y) {
  map.view.x = sx(x) - map.view.w / 2;
  map.view.y = sy(y) - map.view.h / 2;
  applyView();
}

function clientToSvg(ev) {
  const rect = map.el.getBoundingClientRect();
  const v = map.view;
  return {x: v.x + (ev.clientX - rect.left) / rect.width * v.w, y: v.y + (ev.clientY - rect.top) / rect.height * v.h};
}

function clientToWorld(ev) {
  const p = clientToSvg(ev);
  return {x: p.x / U, y: H - p.y / U};
}

function locationAt(wx, wy, margin = 0.1) {
  for (const loc of world.locations) {
    const r = loc.rect;
    if (Math.abs(wx - r.cx) <= r.w / 2 + margin && Math.abs(wy - r.cy) <= r.d / 2 + margin) return loc.name;
  }
  return null;
}

function bindMapEvents() {
  const el = map.el;
  el.addEventListener('wheel', (ev) => {
    ev.preventDefault();
    const p = clientToSvg(ev);
    const k = Math.exp(ev.deltaY * 0.0015);
    const v = map.view;
    const w = Math.min(Math.max(v.w * k, 80), world.width * U * 4);
    const scale = w / v.w;
    map.view = {x: p.x - (p.x - v.x) * scale, y: p.y - (p.y - v.y) * scale, w, h: v.h * scale};
    applyView();
  }, {passive: false});

  el.addEventListener('pointerdown', (ev) => {
    if (map.drag || ev.button !== 0) return;
    // Capture only once this turns into a drag, so a plain click still reaches the location under it.
    map.pan = {cx: ev.clientX, cy: ev.clientY, view: {...map.view}, moved: false, id: ev.pointerId};
  });
  el.addEventListener('pointermove', (ev) => {
    if (map.drag) return moveObjectDrag(ev);
    if (map.pan) {
      const rect = el.getBoundingClientRect();
      const dx = (ev.clientX - map.pan.cx) / rect.width * map.pan.view.w;
      const dy = (ev.clientY - map.pan.cy) / rect.height * map.pan.view.h;
      if (!map.pan.moved && Math.abs(ev.clientX - map.pan.cx) + Math.abs(ev.clientY - map.pan.cy) > 4) {
        map.pan.moved = true;
        el.classList.add('panning');
        el.setPointerCapture(map.pan.id);
      }
      map.view = {...map.pan.view, x: map.pan.view.x - dx, y: map.pan.view.y - dy};
      applyView();
    }
    if (map.hover) positionTooltip(ev);
  });
  const endPan = (ev) => {
    if (map.drag) return endObjectDrag(ev);
    if (map.pan) {
      // A drag that moved should not also count as a click on a location.
      map.suppressClick = map.pan.moved;
      setTimeout(() => { map.suppressClick = false; }, 0);
      map.pan = null;
      el.classList.remove('panning');
    }
  };
  el.addEventListener('pointerup', endPan);
  el.addEventListener('pointercancel', endPan);

  // Drop inventory chips onto a location
  el.addEventListener('dragover', (ev) => {
    ev.preventDefault();
    const w = clientToWorld(ev);
    highlightDrop(locationAt(w.x, w.y));
  });
  el.addEventListener('dragleave', () => highlightDrop(null));
  el.addEventListener('drop', (ev) => {
    ev.preventDefault();
    highlightDrop(null);
    const name = ev.dataTransfer.getData('text/plain');
    const w = clientToWorld(ev);
    const target = locationAt(w.x, w.y);
    if (name && target) sendCommand({cmd: 'move_object', name, location: target});
  });
}

function highlightDrop(name) {
  for (const [n, {g}] of Object.entries(map.locations)) g.classList.toggle('drop-target', n === name);
}

function startObjectDrag(ev, name) {
  const e = map.objects.get(name);
  if (!e || e.data.location === 'Robot' || ev.button !== 0) return;
  ev.stopPropagation();
  map.drag = {name, ghost: svg('circle', {class: 'drag-ghost', r: 7, cx: -100, cy: -100}, map.layers.overlay), moved: false};
  map.el.setPointerCapture(ev.pointerId);
  select(name);
}

function moveObjectDrag(ev) {
  const p = clientToSvg(ev);
  map.drag.moved = true;
  map.drag.ghost.setAttribute('cx', p.x);
  map.drag.ghost.setAttribute('cy', p.y);
  const w = clientToWorld(ev);
  highlightDrop(locationAt(w.x, w.y));
}

function endObjectDrag(ev) {
  const {name, ghost, moved} = map.drag;
  ghost.remove();
  map.drag = null;
  highlightDrop(null);
  map.suppressClick = true;
  setTimeout(() => { map.suppressClick = false; }, 0);
  if (!moved) return;
  const w = clientToWorld(ev);
  const target = locationAt(w.x, w.y);
  const current = map.objects.get(name)?.data.location;
  if (target && target !== current) sendCommand({cmd: 'move_object', name, location: target});
}

function select(name) {
  map.selected = name;
  const e = map.objects.get(name);
  if (!e) return;
  const pulse = svg('circle', {class: 'pulse', cx: sx(e.data.x), cy: sy(e.data.y), r: 5}, map.layers.overlay);
  setTimeout(() => pulse.remove(), 2100);
}

/* Tooltip */
function showTooltip(name) {
  const e = map.objects.get(name);
  if (!e) return;
  const o = e.data;
  const tip = $('#tooltip');
  tip.innerHTML = '';
  const title = document.createElement('b');
  title.textContent = o.name;
  tip.appendChild(title);
  const rows = [['Category', o.category], ['Where', o.location], ['Detector label', o.label],
                ['Position', `${fmt(o.x)}, ${fmt(o.y)}, z ${fmt(o.z)}`], ['Just detected', o.detected ? 'yes' : 'no']];
  for (const [k, v] of rows) {
    const row = document.createElement('div');
    row.className = 'row';
    const a = document.createElement('span'); a.textContent = k;
    const b = document.createElement('span'); b.textContent = v;
    row.append(a, b);
    tip.appendChild(row);
  }
  tip.classList.remove('hidden');
}

function positionTooltip(ev) {
  const tip = $('#tooltip');
  const wrap = $('#map-wrap').getBoundingClientRect();
  let x = ev.clientX - wrap.left + 14, y = ev.clientY - wrap.top + 14;
  if (x + tip.offsetWidth > wrap.width) x -= tip.offsetWidth + 28;
  if (y + tip.offsetHeight > wrap.height) y -= tip.offsetHeight + 28;
  tip.style.left = `${x}px`;
  tip.style.top = `${y}px`;
}

function hideTooltip() { $('#tooltip').classList.add('hidden'); }

/* ------------------------------------------------------------------ Panels */
function updatePanels(s) {
  const r = s.robot;
  $('#clock').textContent = `${s.time.toFixed(1)} s`;
  $('#scenario-chip').textContent = `${s.scenario || 'scenario'} · seed ${s.seed ?? '—'}`;
  const pause = $('#btn-pause');
  if (pause.dataset.paused !== String(s.paused)) {
    pause.dataset.paused = String(s.paused);
    pause.innerHTML = s.paused
      ? '<svg viewBox="0 0 16 16" width="14" height="14"><path d="M4 2.5v11l9-5.5z" fill="currentColor"/></svg>'
      : '<svg viewBox="0 0 16 16" width="14" height="14"><rect x="3.5" y="2.5" width="3" height="11" rx="1" fill="currentColor"/><rect x="9.5" y="2.5" width="3" height="11" rx="1" fill="currentColor"/></svg>';
    pause.title = s.paused ? 'Resume (space)' : 'Pause (space)';
  }
  for (const b of document.querySelectorAll('#speed-seg button')) b.classList.toggle('active', Number(b.dataset.speed) === s.speed);

  const v = $('#violations-badge');
  v.classList.toggle('hidden', !s.violations.length);
  v.textContent = `${s.violations.length} violation${s.violations.length === 1 ? '' : 's'}`;

  const held = $('#robot-held');
  held.textContent = r.held || 'nothing';
  held.classList.toggle('empty', !r.held);

  updateBeliefs(s);
  updateSuccess(s);
  updateRos(s.ros);
  updateBt(s.bt);

  $('#script-name').textContent = s.script ? `script: ${s.script}` : '';
  const ops = $('#ops');
  ops.innerHTML = '';
  if (!s.operations.length) {
    const idle = document.createElement('div');
    idle.className = 'idle';
    idle.textContent = 'Idle';
    ops.appendChild(idle);
  }
  for (const op of s.operations) {
    const el = document.createElement('div');
    el.className = 'op';
    el.innerHTML = `<div class="op-top"><span><span class="op-id"></span><span class="op-name"></span></span><span class="mono op-el"></span></div>
      <div class="op-phase"></div><div class="bar"><i></i></div>`;
    el.querySelector('.op-id').textContent = `#${op.id}`;
    el.querySelector('.op-name').textContent = op.name;
    el.querySelector('.op-el').textContent = `${op.elapsed.toFixed(1)} s`;
    el.querySelector('.op-phase').textContent = op.phase || op.status;
    el.querySelector('.bar i').style.width = `${((op.progress ?? 0) * 100).toFixed(0)}%`;
    el.addEventListener('click', () => selectOp(op.id));
    ops.appendChild(el);
  }

  updateInventory(s.objects);
}

let inventoryKey = '';
function updateInventory(objects) {
  const key = objects.map((o) => `${o.name}@${o.location}`).sort().join('|');
  if (key === inventoryKey) return;
  inventoryKey = key;

  const byLoc = {};
  for (const o of objects) (byLoc[o.location] ||= []).push(o);
  const order = [...world.locations.map((l) => l.name), 'Robot'];
  const inv = $('#inventory');
  inv.innerHTML = '';
  for (const loc of order) {
    const list = (byLoc[loc] || []).sort((a, b) => a.name.localeCompare(b.name));
    if (loc === 'Robot' && !list.length) continue;
    const block = document.createElement('div');
    const head = document.createElement('div');
    head.className = 'inv-head';
    head.innerHTML = '<span></span><span></span>';
    head.children[0].textContent = loc === 'Robot' ? 'In the gripper' : loc;
    head.children[1].textContent = list.length;
    const chips = document.createElement('div');
    chips.className = 'inv-chips';
    if (!list.length) {
      const e = document.createElement('span');
      e.className = 'inv-empty';
      e.textContent = 'empty';
      chips.appendChild(e);
    }
    for (const o of list) {
      const chip = document.createElement('span');
      chip.className = `inv-chip ${o.category === 'Tools' ? 'tool' : ''}`;
      chip.draggable = loc !== 'Robot';
      chip.title = loc !== 'Robot' ? 'Drag onto a location on the map to move it' : '';
      chip.innerHTML = '<i></i>';
      chip.appendChild(document.createTextNode(o.name));
      chip.addEventListener('dragstart', (ev) => ev.dataTransfer.setData('text/plain', o.name));
      chip.addEventListener('click', () => {
        select(o.name);
        if (loc !== 'Robot') $('#move-object').value = o.name;
      });
      chips.appendChild(chip);
    }
    block.append(head, chips);
    inv.appendChild(block);
  }

  const moveSel = $('#move-object');
  const moving = moveSel.value;
  moveSel.innerHTML = '';
  for (const o of [...objects].sort((a, b) => a.name.localeCompare(b.name))) {
    if (o.location === 'Robot') continue;     // in the gripper
    const opt = document.createElement('option');
    opt.value = o.name;
    opt.textContent = `${o.name} (${o.location})`;
    moveSel.appendChild(opt);
  }
  if ([...moveSel.options].some((o) => o.value === moving)) moveSel.value = moving;

  const sel = $('#demo-object');
  const current = sel.value;
  sel.innerHTML = '<option value="">Random object</option>';
  const surfaces = new Set(world.locations.filter((l) => l.kind === 'surface').map((l) => l.name));
  for (const o of [...objects].sort((a, b) => a.name.localeCompare(b.name))) {
    if (!surfaces.has(o.location)) continue;
    const opt = document.createElement('option');
    opt.value = o.name;
    opt.textContent = `${o.name} (${o.location})`;
    sel.appendChild(opt);
  }
  sel.value = [...sel.options].some((o) => o.value === current) ? current : '';
}

let beliefsKey = '';
function updateBeliefs(s) {
  const b = s.beliefs;
  $('#beliefs-empty').classList.toggle('hidden', !!b);
  $('#beliefs-body').classList.toggle('hidden', !b);
  if (!b) { beliefsKey = ''; return; }

  const truth = {};
  for (const o of s.objects) truth[o.name] = o.location;
  const requiredOnly = $('#beliefs-required').checked && b.required.length;
  const key = JSON.stringify([b, truth, requiredOnly, s.robot.held]);
  if (key === beliefsKey) return;
  beliefsKey = key;

  const esc = (t) => String(t ?? '—').replace(/[&<>"]/g, (c) => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
  const believedHeld = b.holding_object || null;
  const heldMismatch = believedHeld !== (s.robot.held || null);
  $('#belief-summary').innerHTML = `
    <dt>Tree</dt><dd>${esc(b.tree)}${b.recipe ? ` · ${esc(b.recipe)}` : ''}</dd>
    <dt>Doing</dt><dd>${esc(b.tip)}</dd>
    <dt>Fetching</dt><dd>${b.target_object ? `${esc(b.target_object)} → ${esc(b.target_location)}` : '—'}</dd>
    <dt>Thinks it holds</dt><dd class="${heldMismatch ? 'mismatch' : ''}" title="${heldMismatch ? 'Differs from what the robot is really carrying' : ''}">${esc(believedHeld)}${heldMismatch ? ` (really: ${esc(s.robot.held || 'nothing')})` : ''}</dd>`;

  const names = Object.keys(b.probabilities)
    .filter((o) => !requiredOnly || b.required.includes(o))
    .sort((x, y) => (x === b.target_object ? -1 : y === b.target_object ? 1 : x.localeCompare(y)));
  let html = '<thead><tr><th>Object</th>' + b.locations.map((l) => `<th>${esc(l)}</th>`).join('') + '<th title="Human can reach it">✓</th></tr></thead><tbody>';
  for (const o of names) {
    const isTarget = o === b.target_object;
    const carried = truth[o] === 'Robot';
    html += `<tr class="${isTarget ? 'target' : ''}"><td class="name">${esc(o)}${carried ? ' <span class="carried">(carried)</span>' : ''}</td>`;
    for (const loc of b.locations) {
      const p = b.probabilities[o][loc] ?? 0;
      const cls = ['p', truth[o] === loc ? 'truth' : '', isTarget && loc === b.target_location ? 'target-cell' : ''].join(' ');
      const bg = `background: color-mix(in srgb, var(--accent) ${(p * 75).toFixed(0)}%, transparent)`;
      html += `<td class="${cls}" style="${bg}" title="${esc(o)} at ${esc(loc)}: ${(p * 100).toFixed(1)}%${truth[o] === loc ? ' (really here)' : ''}">${p >= 0.005 ? Math.round(p * 100) : ''}</td>`;
    }
    html += `<td class="reach">${b.human_reachable[o] ? '✓' : ''}</td></tr>`;
  }
  $('#belief-table').innerHTML = html + '</tbody>';
}

function updateSuccess(s) {
  const slider = $('#success-rate');
  if (document.activeElement !== slider) slider.value = Math.round(s.success_rate * 100);
  const probs = s.failure_probabilities || {};
  const overrides = Object.entries(probs).filter(([, p]) => Math.abs(p - (1 - s.success_rate)) > 1e-6);
  $('#success-value').textContent = `${Math.round(s.success_rate * 100)}%` +
    (overrides.length ? ` · overrides: ${overrides.map(([a, p]) => `${a} fails ${Math.round(p * 100)}%`).join(', ')}` : '');
  slider.title = Object.entries(probs).map(([a, p]) => `${a}: fails ${Math.round(p * 100)}%`).join('\n');
}

// The ROS side: the sim playing the robot, the real robot (the map mirrors it), or none.
// It can change at run time (Connect / Disconnect), so it comes with every state.
let rosKey = '';
function updateRos(ros) {
  ros = ros || {};
  const key = JSON.stringify(ros);
  if (key === rosKey) return;
  rosKey = key;
  const note = $('#bt-unavailable'), status = $('#ros-status'), link = $('#btn-robot-link');
  const places = $('#robot-places'), domain = $('#robot-domain');
  const showList = (text, items) => {
    note.innerHTML = '';
    note.append(text);
    const list = document.createElement('code');
    list.textContent = items.join('; ');
    note.appendChild(list);
    note.classList.remove('hidden');
  };
  note.classList.add('hidden');
  status.title = '';
  if (ros.robot) {
    const r = ros.robot;
    status.textContent = `Real robot (ROS domain ${r.domain_id}): the map mirrors what it does`;
    status.title = [`Places: ${r.places}`, ...Object.entries(r.poses).map(([loc, p]) =>
      `${loc}: (${p.x.toFixed(2)}, ${p.y.toFixed(2)}, yaw ${p.yaw.toFixed(2)})`)].join('\n');
    if (r.missing.length) showList('No place on the robot for (the BT cannot navigate there): ', r.missing);
  } else if (ros.available) {
    status.textContent = `Sim plays the robot · ROS interface: ${ros.offered.length} offered` +
      (ros.missing.length ? `, ${ros.missing.length} missing` : '');
    status.title = [...ros.offered.map((o) => `✓ ${o}`), ...ros.missing.map((m) => `✗ ${m}`)].join('\n');
    if (ros.missing.length) showList('Not offered (the BT will fail when it needs them): ', ros.missing);
  } else {
    status.textContent = 'No ROS: behaviour trees cannot run';
    note.textContent = ros.reason ? `ROS is off (${ros.reason}). Connect to the robot, or start the simulator with --ros.` :
      'ROS is off. Connect to the robot, or start the simulator with --ros.';
    note.classList.remove('hidden');
  }
  const connected = ros.mode === 'robot';
  if (connected && ros.runstopped) status.textContent += ' · RUNSTOPPED';
  $('#btn-release-runstop').classList.toggle('hidden', !(connected && ros.runstopped));
  $('#btn-bt-stop').title = connected
    ? "Stop the tree and the robot: cancel its goals and halt its motion. A grasp, handover or scan already running on the robot carries on; Cancel everything runstops it."
    : 'Stop the tree and what it set going';
  $('#btn-cancel').title = connected
    ? 'Runstop the real robot and stop the tree. The robot will not move until the runstop is released.'
    : 'Cancel every running operation';
  link.textContent = connected ? 'Disconnect' : 'Connect to robot';
  link.classList.toggle('danger-outline', connected);
  link.title = connected ? 'Stop the running tree and let the sim play the robot again'
    : 'Stop the running tree and drive the real robot instead; the map then mirrors it';
  places.disabled = domain.disabled = connected;
  const defaults = ros.connect || {};
  if (document.activeElement !== places && (connected || !places.value)) places.value = (ros.robot || defaults).places || '';
  if (document.activeElement !== domain && (connected || !domain.value)) domain.value = (ros.robot || defaults).domain_id ?? '';
}

function updateBt(bt) {
  const pill = $('#bt-pill');
  const start = $('#btn-bt-start');
  const stop = $('#btn-bt-stop');
  const ros = (state && state.ros) || world.ros || {};
  const available = ros.available;
  if (!bt) {
    pill.textContent = available ? 'idle' : 'unavailable';
    pill.className = 'pill';
  } else if (bt.running) {
    pill.textContent = `running ${bt.tree}`;
    pill.className = 'pill running';
  } else if (bt.returncode != null) {
    pill.textContent = `exited (${bt.returncode})`;
    pill.className = `pill ${bt.returncode === 0 ? 'done' : 'error'}`;
  } else {
    pill.textContent = 'idle';
    pill.className = 'pill';
  }
  start.disabled = !available || !bt || bt.running || !world.bt_trees.length;
  // With the real robot, Stop also stops the robot, so it stays available without a tree.
  stop.disabled = !ros.robot && (!bt || !bt.running);
  $('#btn-reset-memory').disabled = !bt || !bt.running;
  // While a tree runs, the form shows what is running.
  const trees = $('#bt-tree'), recipes = $('#bt-recipe');
  if (bt && bt.running) {
    trees.value = bt.tree;
    recipes.value = bt.recipe || '';
  }
  trees.disabled = recipes.disabled = !available || !!(bt && bt.running);
}

/* ------------------------------------------------------------------ Log */
const log = {events: [], bySeq: new Map(), kinds: new Set(KINDS), query: '', selected: null, counts: {}};

function eventText(e) {
  return `${e.source} ${e.message} ${JSON.stringify(e.data)}`.toLowerCase();
}

function visible(e) {
  return log.kinds.has(e.kind) && (!log.query || eventText(e).includes(log.query));
}

function rowFor(e) {
  const tr = document.createElement('tr');
  tr.className = `k-${e.kind}`;
  if (e.kind === 'result' && e.data && e.data.status && e.data.status !== 'succeeded') tr.classList.add(`res-${e.data.status}`);
  tr.dataset.seq = e.seq;
  if (e.op_id != null) tr.dataset.op = e.op_id;
  const cells = [`${e.time.toFixed(2)}`, e.kind, e.op_id != null ? `#${e.op_id}` : '', e.source, e.message];
  const classes = ['t', 'kind', 'op', 'src', 'msg'];
  cells.forEach((text, i) => {
    const td = document.createElement('td');
    td.className = classes[i];
    if (i === 1) {
      const b = document.createElement('span');
      b.className = 'kind-badge';
      b.textContent = text;
      td.appendChild(b);
    } else {
      td.textContent = text;
    }
    tr.appendChild(td);
  });
  if (log.selected && log.selected.op_id != null && log.selected.op_id === e.op_id) tr.classList.add('same-op');
  return tr;
}

function addEvents(events) {
  if (!events.length) return;
  const tbody = $('#log-rows');
  const frag = document.createDocumentFragment();
  for (const e of events) {
    if (log.bySeq.has(e.seq)) continue;
    log.events.push(e);
    log.bySeq.set(e.seq, e);
    log.counts[e.kind] = (log.counts[e.kind] || 0) + 1;
    if (visible(e)) frag.appendChild(rowFor(e));
    if (e.kind === 'violation') toast(`Violation: ${e.message}`, 'error');
    else if (e.kind === 'result' && e.data && e.data.status === 'failed') toast(`${e.source} failed: ${e.message.replace(/^failed: /, '')}`, 'error');
  }
  tbody.appendChild(frag);
  while (tbody.children.length > MAX_ROWS) tbody.firstChild.remove();
  updateKindCounts();
  if ($('#log-follow').checked) {
    const wrap = $('#log-wrap');
    wrap.scrollTop = wrap.scrollHeight;
  }
}

function rebuildLog() {
  const tbody = $('#log-rows');
  tbody.innerHTML = '';
  const frag = document.createDocumentFragment();
  const shown = log.events.filter(visible).slice(-MAX_ROWS);
  for (const e of shown) frag.appendChild(rowFor(e));
  tbody.appendChild(frag);
  markSelection();
  if ($('#log-follow').checked) $('#log-wrap').scrollTop = $('#log-wrap').scrollHeight;
}

function buildKindFilters() {
  const box = $('#kind-filters');
  box.innerHTML = '';
  for (const k of KINDS) {
    const b = document.createElement('button');
    b.className = 'kind-chip on';
    b.dataset.kind = k;
    b.innerHTML = `<i style="background: var(--k-${k})"></i><span></span><span class="count">0</span>`;
    b.children[1].textContent = k;
    b.title = 'Click to show or hide; double-click to show only this kind';
    b.addEventListener('click', () => {
      if (log.kinds.has(k)) log.kinds.delete(k); else log.kinds.add(k);
      b.classList.toggle('on', log.kinds.has(k));
      rebuildLog();
    });
    b.addEventListener('dblclick', () => {
      log.kinds = new Set([k]);
      for (const c of box.children) c.classList.toggle('on', c.dataset.kind === k);
      rebuildLog();
    });
    box.appendChild(b);
  }
}

function updateKindCounts() {
  for (const b of document.querySelectorAll('.kind-chip')) b.querySelector('.count').textContent = log.counts[b.dataset.kind] || 0;
}

function markSelection() {
  const sel = log.selected;
  for (const tr of $('#log-rows').children) {
    tr.classList.toggle('selected', !!sel && Number(tr.dataset.seq) === sel.seq);
    tr.classList.toggle('same-op', !!sel && sel.op_id != null && tr.dataset.op === String(sel.op_id));
  }
}

function jsonHtml(value) {
  const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  return esc(JSON.stringify(value, null, 2))
    .replace(/("(?:\\.|[^"\\])*")(\s*:)?/g, (m, str, colon) => colon ? `<span class="k">${str}</span>${colon}` : `<span class="s">${str}</span>`)
    .replace(/\b(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\b/g, '<span class="n">$1</span>')
    .replace(/\b(true|false|null)\b/g, '<span class="b">$1</span>');
}

function selectEvent(e) {
  log.selected = e;
  $('#log-follow').checked = false;
  markSelection();
  const d = $('#log-detail');
  d.innerHTML = '';
  const title = document.createElement('div');
  title.className = 'detail-title';
  title.textContent = `${e.source}: ${e.message}`;
  const meta = document.createElement('div');
  meta.className = 'detail-meta';
  meta.textContent = `${e.kind.toUpperCase()} · t=${e.time.toFixed(2)} s · seq ${e.seq}${e.op_id != null ? ` · operation #${e.op_id}` : ''}`;
  d.append(title, meta);
  if (e.data && Object.keys(e.data).length) {
    const pre = document.createElement('pre');
    pre.className = 'json';
    pre.innerHTML = jsonHtml(e.data);
    d.appendChild(pre);
  }
  if (e.op_id != null) {
    const related = log.events.filter((o) => o.op_id === e.op_id);
    const head = document.createElement('div');
    head.className = 'detail-section';
    head.textContent = `Operation #${e.op_id} timeline`;
    const ul = document.createElement('ul');
    ul.className = 'timeline';
    for (const o of related) {
      const li = document.createElement('li');
      li.className = o.seq === e.seq ? 'current' : '';
      li.innerHTML = '<span class="t"></span><span class="kind-badge"></span><span></span>';
      li.children[0].textContent = `${o.time.toFixed(2)}s`;
      li.children[1].textContent = o.kind;
      li.children[1].style.color = `var(--k-${o.kind})`;
      li.children[2].textContent = o.message;
      li.addEventListener('click', () => selectEvent(o));
      ul.appendChild(li);
    }
    d.append(head, ul);
  }
}

function selectOp(id) {
  const first = log.events.find((e) => e.op_id === id);
  if (first) selectEvent(first);
}

/* ------------------------------------------------------------------ Controls */
function buildControls() {
  $('#world-chip').textContent = world.name;
  const goto = $('#goto-buttons');
  goto.innerHTML = '';
  world.locations.forEach((loc, i) => {
    const b = document.createElement('button');
    b.className = 'btn';
    b.textContent = loc.name;
    if (i < 9) {
      const k = document.createElement('kbd');
      k.textContent = i + 1;
      b.appendChild(k);
    }
    b.addEventListener('click', () => sendCommand({cmd: 'goto', location: loc.name}));
    goto.appendChild(b);
  });

  const seg = $('#speed-seg');
  seg.innerHTML = '';
  for (const sp of world.speeds.filter((x) => x >= 0.5 && x <= 16)) {
    const b = document.createElement('button');
    b.dataset.speed = sp;
    b.textContent = `${sp}×`;
    b.addEventListener('click', () => sendCommand({cmd: 'speed', speed: sp}));
    seg.appendChild(b);
  }

  for (const b of document.querySelectorAll('[data-cmd]')) {
    b.addEventListener('click', () => sendCommand(JSON.parse(b.dataset.cmd)));
  }
  $('#btn-demo').addEventListener('click', demo);
  $('#btn-shuffle').addEventListener('click', shuffle);
  $('#btn-move').addEventListener('click', async () => {
    const name = $('#move-object').value, location = $('#move-location').value;
    if (!name || !location) return;
    const res = await sendCommand({cmd: 'move_object', name, location});
    if (res.ok) toast(`Moved ${name} to ${location}`, 'ok');
  });
  const locSelect = $('#move-location');
  for (const loc of world.locations) {
    const o = document.createElement('option');
    o.value = o.textContent = loc.name;
    locSelect.appendChild(o);
  }
  $('#btn-reset-memory').addEventListener('click', async () => {
    const button = $('#btn-reset-memory');
    button.disabled = true;
    const res = await sendCommand({cmd: 'reset_memory'});
    button.disabled = false;
    if (res.ok) toast('Robot memory reset to the priors', 'ok');
  });
  $('#success-rate').addEventListener('change', (e) => sendCommand({cmd: 'success_rate', rate: Number(e.target.value) / 100}));
  $('#success-rate').addEventListener('input', (e) => { $('#success-value').textContent = `${e.target.value}%`; });
  buildBtControls();
  $('#btn-pause').addEventListener('click', () => sendCommand({cmd: 'pause'}));
  $('#btn-respawn').addEventListener('click', () => sendCommand({cmd: 'respawn'}));
  $('#btn-fit').addEventListener('click', fitView);
  $('#btn-theme').addEventListener('click', toggleTheme);
  $('#violations-badge').addEventListener('click', () => {
    log.kinds = new Set(['violation']);
    for (const c of document.querySelectorAll('.kind-chip')) c.classList.toggle('on', c.dataset.kind === 'violation');
    rebuildLog();
  });
  $('#beliefs-required').addEventListener('change', () => { beliefsKey = ''; if (state) updateBeliefs(state); });
  $('#opt-labels').addEventListener('change', (e) => map.el.classList.toggle('no-labels', !e.target.checked));
  $('#opt-approach').addEventListener('change', (e) => map.el.classList.toggle('no-approach', !e.target.checked));

  let searchTimer = null;
  $('#log-search').addEventListener('input', (e) => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => { log.query = e.target.value.trim().toLowerCase(); rebuildLog(); }, 150);
  });
  $('#log-rows').addEventListener('click', (ev) => {
    const tr = ev.target.closest('tr');
    if (tr) selectEvent(log.bySeq.get(Number(tr.dataset.seq)));
  });
  $('#log-wrap').addEventListener('wheel', (ev) => { if (ev.deltaY < 0) $('#log-follow').checked = false; });

  document.addEventListener('keydown', onKey);
  window.addEventListener('resize', () => fitView());
}

function buildBtControls() {
  const trees = $('#bt-tree');
  trees.innerHTML = '';
  for (const t of world.bt_trees) {
    const o = document.createElement('option');
    o.value = o.textContent = t;
    trees.appendChild(o);
  }
  if (world.bt_trees.includes('main_demo_loop')) trees.value = 'main_demo_loop';
  const recipes = $('#bt-recipe');
  recipes.innerHTML = '<option value="">(none: the tree sets it)</option>';
  for (const r of world.recipes) {
    const o = document.createElement('option');
    o.value = o.textContent = r;
    recipes.appendChild(o);
  }
  if (world.recipes.length) recipes.value = world.recipes[0];   // main_demo_loop needs one
  const link = $('#btn-robot-link');
  link.addEventListener('click', async () => {
    const ros = (state && state.ros) || world.ros || {};
    link.disabled = true;
    link.textContent = ros.mode === 'robot' ? 'Disconnecting…' : 'Connecting…';
    const res = ros.mode === 'robot'
      ? await sendCommand({cmd: 'disconnect'})
      : await sendCommand({cmd: 'connect', places: $('#robot-places').value, domain_id: $('#robot-domain').value});
    if (res.ok) toast(res.message);
    link.disabled = false;
    rosKey = '';      // redraw the link controls even if the state has not changed yet
    if (state) updateRos(state.ros);
  });
  $('#btn-release-runstop').addEventListener('click', async () => {
    const res = await sendCommand({cmd: 'release_runstop'});
    if (res.ok) toast(res.message);
  });
  updateRos(world.ros);
  $('#btn-bt-start').addEventListener('click', () => sendCommand({cmd: 'bt_start', tree: trees.value, recipe: recipes.value || null}));
  $('#btn-bt-stop').addEventListener('click', () => sendCommand({cmd: 'bt_stop'}));
}

async function shuffle() {
  const count = Math.max(1, Math.min(20, Number($('#shuffle-count').value) || 3));
  const res = await sendCommand({cmd: 'shuffle', count});
  if (res.ok) toast(`Shuffled: ${res.message.replace(/^moved /, '')}`, 'ok');
}

function demo() {
  const object = $('#demo-object').value;
  sendCommand(object ? {cmd: 'demo', object} : {cmd: 'demo'});
}

function onKey(ev) {
  if (ev.target.closest('input, select, textarea') || ev.ctrlKey || ev.metaKey || ev.altKey) return;
  const n = Number(ev.key);
  if (n >= 1 && n <= 9 && world.locations[n - 1]) return sendCommand({cmd: 'goto', location: world.locations[n - 1].name});
  const speeds = world.speeds;
  const actions = {
    s: {cmd: 'detect'},
    g: {cmd: 'grasp_best'}, o: {cmd: 'handover'}, m: {cmd: 'mode'}, c: {cmd: 'cancel'},
    t: {cmd: 'tour'}, r: {cmd: 'respawn'},
  };
  if (actions[ev.key]) return sendCommand(actions[ev.key]);
  if (ev.key === 'd') return demo();
  if (ev.key === 'x') return shuffle();
  if (ev.key === ' ') { ev.preventDefault(); return sendCommand({cmd: 'pause'}); }
  if (ev.key === '0') return fitView();
  if (ev.key === 'l') { $('#opt-labels').click(); return; }
  if ((ev.key === '+' || ev.key === '=' || ev.key === '-') && state) {
    const i = speeds.indexOf(state.speed);
    const j = Math.max(0, Math.min(speeds.length - 1, (i < 0 ? speeds.indexOf(1) : i) + (ev.key === '-' ? -1 : 1)));
    return sendCommand({cmd: 'speed', speed: speeds[j]});
  }
}

/* Theme */
function toggleTheme() {
  const root = document.documentElement;
  const dark = root.dataset.theme ? root.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
  root.dataset.theme = dark ? 'light' : 'dark';
  try { localStorage.setItem('stretch-sim-theme', root.dataset.theme); } catch (e) { /* storage unavailable */ }
}

function restoreTheme() {
  try {
    const t = localStorage.getItem('stretch-sim-theme');
    if (t) document.documentElement.dataset.theme = t;
  } catch (e) { /* storage unavailable */ }
}

/* ------------------------------------------------------------------ Stream */
let lastSeq = 0;
let source = null;

function connect() {
  if (source) source.close();
  source = new EventSource(`/api/stream?since=${lastSeq}`);
  source.onopen = () => { $('#conn').className = 'conn live'; $('#conn').title = 'Connected'; };
  source.onmessage = (msg) => {
    const data = JSON.parse(msg.data);
    if (data.events.length) {
      lastSeq = data.events[data.events.length - 1].seq;
      addEvents(data.events);
    }
    state = data.state;
    updateMap(state);
    updatePanels(state);
  };
  source.onerror = () => {
    $('#conn').className = 'conn down';
    $('#conn').title = 'Disconnected; retrying';
    source.close();
    setTimeout(connect, 1500);
  };
}

async function main() {
  restoreTheme();
  world = await (await fetch('/api/world')).json();
  H = world.height;
  buildKindFilters();
  buildControls();
  buildMap();
  const events = await (await fetch('/api/events?since=0')).json();
  // History is loaded without toasts; only new problems should pop up.
  for (const e of events) {
    log.events.push(e); log.bySeq.set(e.seq, e);
    log.counts[e.kind] = (log.counts[e.kind] || 0) + 1;
  }
  lastSeq = events.length ? events[events.length - 1].seq : 0;
  rebuildLog();
  updateKindCounts();
  connect();
}

main();
