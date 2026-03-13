const fs = require('fs');

const SVG_WIDTH = 960;
const SVG_HEIGHT = 640;
const ORIGIN_X = 360;
const ORIGIN_Y = 154;
const ISO_X = 54;
const ISO_Y = 27;
const ISO_Z = 34;
const STROKE = '#10202B';
const TOTAL_DURATION = 18.0;
const BRICK_SIZE = 0.40;
const BRICK_HEIGHT = 0.70;
const BRICK_COLOR = '#63F2F7';
const FRONT_OFFSET = 1.20;
const CYCLE_DURATION = TOTAL_DURATION / 3.0;
const ROBOT_LOCAL_PIVOT = [0.80, 0.80, 0.0];
const DIRECTIONS = ['left', 'right'];

function mulberry32(seed) {
  let t = seed >>> 0;
  return () => {
    t += 0x6D2B79F5;
    let value = Math.imul(t ^ (t >>> 15), t | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function rawIsoPoint(x, y, z) {
  return {
    x: (x - y) * ISO_X,
    y: (x + y) * ISO_Y - z * ISO_Z,
  };
}

const LOCAL_PIVOT_POINT = rawIsoPoint(...ROBOT_LOCAL_PIVOT);

function worldIsoPoint(x, y, z) {
  const base = rawIsoPoint(x, y, z);
  return { x: ORIGIN_X + base.x, y: ORIGIN_Y + base.y };
}

function localIsoPoint(x, y, z) {
  const base = rawIsoPoint(x, y, z);
  return { x: base.x - LOCAL_PIVOT_POINT.x, y: base.y - LOCAL_PIVOT_POINT.y };
}

function fmt(value) {
  const text = value.toFixed(2).replace(/\.?0+$/, '');
  return text === '-0' ? '0' : text;
}

function pointsToSvg(points) {
  return points.map((point) => `${fmt(point.x)},${fmt(point.y)}`).join(' ');
}

function clamp(value, minimum = 0, maximum = 255) {
  return Math.max(minimum, Math.min(maximum, Math.round(value)));
}

function hexToRgb(value) {
  const source = value.replace('#', '');
  return [
    parseInt(source.slice(0, 2), 16),
    parseInt(source.slice(2, 4), 16),
    parseInt(source.slice(4, 6), 16),
  ];
}

function rgbToHex(rgb) {
  return `#${rgb.map((channel) => clamp(channel).toString(16).padStart(2, '0')).join('').toUpperCase()}`;
}

function mix(color, target, amount) {
  const sourceRgb = hexToRgb(color);
  const targetRgb = hexToRgb(target);
  return rgbToHex(sourceRgb.map((source, index) => source + (targetRgb[index] - source) * amount));
}

function shade(color, amount) {
  return amount >= 0 ? mix(color, '#FFFFFF', amount) : mix(color, '#000000', -amount);
}

function polygon(points, fill, stroke = STROKE, opacity = 1) {
  return `<polygon points="${pointsToSvg(points)}" fill="${fill}" stroke="${stroke}" stroke-width="1.6" opacity="${fmt(opacity)}" stroke-linejoin="round" />`;
}

function ellipse(cx, cy, rx, ry, fill, opacity = 1, stroke = null) {
  const strokePart = stroke ? ` stroke="${stroke}" stroke-width="1.0"` : '';
  return `<ellipse cx="${fmt(cx)}" cy="${fmt(cy)}" rx="${fmt(rx)}" ry="${fmt(ry)}" fill="${fill}" opacity="${fmt(opacity)}"${strokePart} />`;
}

function renderBox(box, project) {
  const p000 = project(box.x, box.y, box.z);
  const p100 = project(box.x + box.dx, box.y, box.z);
  const p010 = project(box.x, box.y + box.dy, box.z);
  const p110 = project(box.x + box.dx, box.y + box.dy, box.z);
  const p001 = project(box.x, box.y, box.z + box.dz);
  const p101 = project(box.x + box.dx, box.y, box.z + box.dz);
  const p011 = project(box.x, box.y + box.dy, box.z + box.dz);
  const p111 = project(box.x + box.dx, box.y + box.dy, box.z + box.dz);

  const nearY = polygon([p011, p111, p110, p010], shade(box.color, -0.16), box.stroke);
  const nearX = polygon([p101, p111, p110, p100], shade(box.color, -0.06), box.stroke);
  const top = polygon([p001, p101, p111, p011], shade(box.color, 0.18), box.stroke);
  return [nearY, nearX, top].join('\n');
}

function brickBox(x, y, z) {
  return { x, y, z, dx: BRICK_SIZE, dy: BRICK_SIZE, dz: BRICK_HEIGHT, color: BRICK_COLOR, stroke: STROKE };
}

function renderFloorTile(x, y, fill, opacity) {
  return polygon([
    worldIsoPoint(x, y, 0),
    worldIsoPoint(x + 1, y, 0),
    worldIsoPoint(x + 1, y + 1, 0),
    worldIsoPoint(x, y + 1, 0),
  ], fill, 'none', opacity);
}

function renderBackground(palette) {
  return `
<defs>
  <linearGradient id="bgGradient" x1="0%" y1="0%" x2="100%" y2="100%">
    <stop offset="0%" stop-color="${palette.sky_top}" />
    <stop offset="100%" stop-color="${palette.sky_bottom}" />
  </linearGradient>
  <filter id="softShadow" x="-35%" y="-35%" width="170%" height="170%">
    <feDropShadow dx="0" dy="8" stdDeviation="10" flood-color="#102030" flood-opacity="0.20" />
  </filter>
</defs>
<rect width="100%" height="100%" fill="url(#bgGradient)" />
`;
}

function renderFloor(rng, palette, stackBase, pileBase) {
  const tiles = [];
  for (let x = 1; x < 6; x += 1) {
    for (let y = 1; y < 6; y += 1) {
      const highlight = rng() < 0.20;
      const fill = shade(palette.floor, highlight ? 0.03 : -0.02);
      const opacity = highlight ? 0.92 : 0.74;
      tiles.push(renderFloorTile(x, y, fill, opacity));
    }
  }

  const margin = 0.12;
  const stackGuide = polygon([
    worldIsoPoint(stackBase[0] - margin, stackBase[1] - margin, 0),
    worldIsoPoint(stackBase[0] + BRICK_SIZE + margin, stackBase[1] - margin, 0),
    worldIsoPoint(stackBase[0] + BRICK_SIZE + margin, stackBase[1] + BRICK_SIZE + margin, 0),
    worldIsoPoint(stackBase[0] - margin, stackBase[1] + BRICK_SIZE + margin, 0),
  ], palette.guide, 'none', 0.20);
  const pickGuide = polygon([
    worldIsoPoint(pileBase[0] - margin, pileBase[1] - margin, 0),
    worldIsoPoint(pileBase[0] + BRICK_SIZE + margin, pileBase[1] - margin, 0),
    worldIsoPoint(pileBase[0] + BRICK_SIZE + margin, pileBase[1] + BRICK_SIZE + margin, 0),
    worldIsoPoint(pileBase[0] - margin, pileBase[1] + BRICK_SIZE + margin, 0),
  ], palette.guide, 'none', 0.14);
  return [...tiles, stackGuide, pickGuide].join('\n');
}

function renderTrackRollersVertical(sideX, palette) {
  return '';
}

function renderTrackRollersHorizontal(sideY, palette) {
  return '';
}

function renderVerticalRobot(carrying, palette) {
  return renderHorizontalRobot(carrying, palette);
}

function renderHorizontalRobot(carrying, palette) {
  const parts = [];

  const boxes = [
    { x: 0.04, y: 0.10, z: 0.0, dx: 1.48, dy: 0.24, dz: 0.18, color: palette.tread, stroke: STROKE },
    { x: 0.04, y: 1.26, z: 0.0, dx: 1.48, dy: 0.24, dz: 0.18, color: palette.tread, stroke: STROKE },
    { x: 0.20, y: 0.24, z: 0.12, dx: 1.04, dy: 1.02, dz: 0.40, color: palette.body, stroke: STROKE },
    { x: 1.00, y: 0.58, z: 0.12, dx: 0.22, dy: 0.34, dz: 3.48, color: palette.mast, stroke: STROKE },
  ];

  for (const item of boxes) {
    parts.push(renderBox(item, localIsoPoint));
  }

  if (carrying) {
    parts.push(renderBox(brickBox(1.18, 0.58, 1.18), localIsoPoint));
  }

  return parts.join('\n');
}

function renderRobotSprite(direction, carrying, palette) {
  if (direction === 'right') return renderHorizontalRobot(carrying, palette);
  return '<g transform="scale(-1 1)">' + renderHorizontalRobot(carrying, palette) + '</g>';
}

function screenDelta(startWorld, endWorld) {
  const start = worldIsoPoint(...startWorld);
  const end = worldIsoPoint(...endWorld);
  return [end.x - start.x, end.y - start.y];
}

function appendFrame(frames, keyTime, value) {
  if (frames.length && Math.abs(frames[frames.length - 1][0] - keyTime) < 0.0001) {
    frames[frames.length - 1] = [keyTime, value];
  } else {
    frames.push([keyTime, value]);
  }
}

function translateKeyframes(frames) {
  const keyTimes = frames.map((frame) => fmt(frame[0])).join(';');
  const values = frames.map((frame) => `${fmt(frame[1][0])} ${fmt(frame[1][1])}`).join(';');
  return [keyTimes, values];
}

function scalarKeyframes(frames) {
  const keyTimes = frames.map((frame) => fmt(frame[0])).join(';');
  const values = frames.map((frame) => fmt(frame[1])).join(';');
  return [keyTimes, values];
}

function buildRobotTimeline(home, pick, place) {
  const cycleStates = [
    [0.0, 'left', false, home],
    [0.7, 'right', false, home],
    [1.7, 'right', false, pick],
    [2.3, 'right', true, pick],
    [3.2, 'right', true, home],
    [3.9, 'left', true, home],
    [4.9, 'left', true, place],
    [5.5, 'left', false, place],
    [6.0, 'left', false, home],
  ];

  const timeline = [];
  for (let cycleIndex = 0; cycleIndex < 3; cycleIndex += 1) {
    const offset = cycleIndex * CYCLE_DURATION;
    for (const [phaseTime, direction, carrying, position] of cycleStates) {
      const normalized = Math.min(offset + phaseTime, TOTAL_DURATION) / TOTAL_DURATION;
      const entry = [normalized, direction, carrying, screenDelta(home, position)];
      if (timeline.length && Math.abs(timeline[timeline.length - 1][0] - normalized) < 0.0001) {
        timeline[timeline.length - 1] = entry;
      } else {
        timeline.push(entry);
      }
    }
  }

  const finalEntry = [1.0, 'left', false, timeline[timeline.length - 1][3]];
  if (timeline.length && Math.abs(timeline[timeline.length - 1][0] - 1.0) < 0.0001) {
    timeline[timeline.length - 1] = finalEntry;
  } else {
    timeline.push(finalEntry);
  }
  return timeline;
}

function buildRobotTranslation(timeline) {
  return translateKeyframes(timeline.map(([timeValue, _direction, _carrying, delta]) => [timeValue, delta]));
}

function buildStateVisibility(timeline, direction, carrying) {
  const frames = [];
  for (const [timeValue, currentDirection, currentCarrying] of timeline) {
    const opacity = currentDirection === direction && currentCarrying === carrying ? 1.0 : 0.0;
    appendFrame(frames, timeValue, opacity);
  }
  return scalarKeyframes(frames);
}

function buildBinaryVisibility(events) {
  const frames = [];
  for (const [absoluteTime, value] of events) {
    appendFrame(frames, absoluteTime / TOTAL_DURATION, value);
  }
  appendFrame(frames, 1.0, frames[frames.length - 1][1]);
  return scalarKeyframes(frames);
}

function createScene(seedValue) {
  const rng = mulberry32(seedValue);

  const palette = {
    sky_top: '#F5EEDB',
    sky_bottom: '#D9E8EE',
    floor: '#D7DED6',
    guide: '#7FDCE3',
    body: '#F0A43E',
    mast: '#465A65',
    tread: '#324651',
    roller: '#7C909A',
    carriage: '#556A77',
    fork: '#B3C1C8',
  };

  const homeRobot = [2.20, 1.35, 0.0];
  const pileBase = [4.60, 1.55, 0.0];
  const stackBase = [2.40, 3.85, 0.0];

  const pickRobot = [pileBase[0] - FRONT_OFFSET, pileBase[1], 0.0];
  const placeRobot = [stackBase[0], stackBase[1] - FRONT_OFFSET, 0.0];

  const pilePositions = [
    [pileBase[0], pileBase[1], 0.0],
    [pileBase[0], pileBase[1], BRICK_HEIGHT],
    [pileBase[0], pileBase[1], BRICK_HEIGHT * 2],
  ];
  const stackPositions = [
    [stackBase[0], stackBase[1], 0.0],
    [stackBase[0], stackBase[1], BRICK_HEIGHT],
    [stackBase[0], stackBase[1], BRICK_HEIGHT * 2],
    [stackBase[0], stackBase[1], BRICK_HEIGHT * 3],
  ];

  const floor = renderFloor(rng, palette, stackBase, pileBase);
  const background = renderBackground(palette);
  const baseBrick = renderBox(brickBox(...stackPositions[0]), worldIsoPoint);

  const robotTimeline = buildRobotTimeline(homeRobot, pickRobot, placeRobot);
  const [robotTranslateTimes, robotTranslateValues] = buildRobotTranslation(robotTimeline);

  const spriteGroups = [];
  for (const direction of DIRECTIONS) {
    for (const carrying of [false, true]) {
      const [opacityTimes, opacityValues] = buildStateVisibility(robotTimeline, direction, carrying);
      spriteGroups.push(
        `<g opacity="0">\n${renderRobotSprite(direction, carrying, palette)}\n<animate attributeName="opacity" dur="${fmt(TOTAL_DURATION)}s" repeatCount="indefinite" calcMode="discrete" keyTimes="${opacityTimes}" values="${opacityValues}" />\n</g>`
      );
    }
  }

  const robotGroup = `<g id="robotTranslate">\n<animateTransform attributeName="transform" type="translate" dur="${fmt(TOTAL_DURATION)}s" repeatCount="indefinite" keyTimes="${robotTranslateTimes}" values="${robotTranslateValues}" calcMode="linear" />\n${spriteGroups.join('\n')}\n</g>`;

  const sourceGroups = [];
  for (let pileIndex = 0; pileIndex < pilePositions.length; pileIndex += 1) {
    const source = pilePositions[pileIndex];
    const cycleIndex = 2 - pileIndex;
    const pickTime = cycleIndex * CYCLE_DURATION + 2.3;
    const [sourceTimes, sourceValues] = buildBinaryVisibility([
      [0.0, 1.0],
      [pickTime, 1.0],
      [pickTime, 0.0],
    ]);
    sourceGroups.push(
      `<g filter="url(#softShadow)" opacity="1">\n${renderBox(brickBox(...source), worldIsoPoint)}\n<animate attributeName="opacity" dur="${fmt(TOTAL_DURATION)}s" repeatCount="indefinite" calcMode="discrete" keyTimes="${sourceTimes}" values="${sourceValues}" />\n</g>`
    );
  }

  const targetGroups = [];
  for (let cycleIndex = 0; cycleIndex < 3; cycleIndex += 1) {
    const target = stackPositions[cycleIndex + 1];
    const placeTime = cycleIndex * CYCLE_DURATION + 5.5;
    const [targetTimes, targetValues] = buildBinaryVisibility([
      [0.0, 0.0],
      [placeTime, 0.0],
      [placeTime, 1.0],
    ]);
    targetGroups.push(
      `<g filter="url(#softShadow)" opacity="0">\n${renderBox(brickBox(...target), worldIsoPoint)}\n<animate attributeName="opacity" dur="${fmt(TOTAL_DURATION)}s" repeatCount="indefinite" calcMode="discrete" keyTimes="${targetTimes}" values="${targetValues}" />\n</g>`
    );
  }

  const robotHomeScreen = worldIsoPoint(...homeRobot);
  return [
    `<svg xmlns="http://www.w3.org/2000/svg" width="${SVG_WIDTH}" height="${SVG_HEIGHT}" viewBox="0 0 ${SVG_WIDTH} ${SVG_HEIGHT}">`,
    background,
    '<g id="scene">',
    floor,
    `<g transform="translate(${fmt(robotHomeScreen.x)} ${fmt(robotHomeScreen.y)})">`,
    robotGroup,
    '</g>',
    baseBrick,
    ...sourceGroups,
    ...targetGroups,
    '</g>',
    '</svg>',
  ].join('\n');
}

const seedValue = 489401;
const svg = createScene(seedValue);
fs.writeFileSync('C:\\dev\\Leia layout\\forklift_bricks_isometric.svg', svg, 'utf8');
console.log(`Wrote C:\\dev\\Leia layout\\forklift_bricks_isometric.svg with seed ${seedValue}`);








