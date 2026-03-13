(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    root.LeiaRobotScene = factory();
  }
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  const DEFAULTS = {
    width: 720,
    height: 420,
    originX: 300,
    originY: 108,
    isoX: 46,
    isoY: 23,
    isoZ: 30,
    stroke: '#10202B',
    brickSize: 0.4,
    brickHeight: 0.23333333333333334,
    brickColor: '#63F2F7',
    maxStackBricks: 50,
    robotPivot: { x: 0.8, y: 0.8, z: 0.0 },
    boardMin: 1,
    boardMax: 5,
    bounds: { minX: 1.7, maxX: 4.5, minY: 1.6, maxY: 3.9 },
    palette: {
      skyTop: '#F5EEDB',
      skyBottom: '#D9E8EE',
      floor: '#D7DED6',
      guide: '#7FDCE3',
      body: '#F0A43E',
      mast: '#465A65',
      tread: '#324651'
    }
  };

  function createLeiaRobotScene(target, options) {
    const config = buildConfig(options || {});
    const seedState = options && options.state ? options.state : buildDefaultState(config);
    const state = normaliseState(seedState, config);
    let mountNode = resolveTarget(target);

    function draw() {
      syncCarriedBrick(state, config);
      const svg = renderSceneMarkup(state, config);
      if (mountNode) {
        mountNode.innerHTML = svg;
      }
      return svg;
    }

    const api = {
      addBrick(brick) {
        const nextBrick = normaliseBrick(brick || {}, state, config);
        state.bricks.push(nextBrick);
        draw();
        return nextBrick.id;
      },
      addBrickToStack(position, patch) {
        if (!hasStackPosition(position) || stackCountAt(state, position) >= config.maxStackBricks) {
          return null;
        }
        const nextBrick = normaliseBrick(Object.assign({}, patch || {}, {
          x: position.x,
          y: position.y,
          z: nextStackZAt(state, position, config)
        }), state, config);
        state.bricks.push(nextBrick);
        draw();
        return nextBrick.id;
      },
      getState() {
        return deepClone(state);
      },
      mount(nextTarget) {
        mountNode = resolveTarget(nextTarget);
        draw();
        return api;
      },
      moveRobot(distance) {
        const delta = Number(distance) || 0;
        const sign = state.robot.facing === 'right' ? 1 : -1;
        state.robot.x = clamp(state.robot.x + delta * sign, config.bounds.minX, config.bounds.maxX);
        draw();
        return api;
      },
      moveRobotBackward(step) {
        return api.moveRobot(-Math.abs(step == null ? 0.28 : step));
      },
      moveRobotForward(step) {
        return api.moveRobot(Math.abs(step == null ? 0.28 : step));
      },
      removeBrick(id) {
        const index = state.bricks.findIndex((brick) => brick.id === id);
        if (index === -1) {
          return false;
        }
        state.bricks.splice(index, 1);
        if (state.robot.carriedBrickId === id) {
          state.robot.carriedBrickId = null;
        }
        draw();
        return true;
      },
      removeTopBrick(position) {
        const brick = topBrickAt(state, position);
        if (!brick) {
          return null;
        }
        state.bricks = state.bricks.filter((item) => item.id !== brick.id);
        if (state.robot.carriedBrickId === brick.id) {
          state.robot.carriedBrickId = null;
        }
        draw();
        return brick.id;
      },
      render() {
        draw();
        return api;
      },
      replaceBricks(bricks) {
        state.bricks = (Array.isArray(bricks) ? bricks : []).map((brick) => normaliseBrick(brick, state, config));
        if (!state.bricks.some((brick) => brick.id === state.robot.carriedBrickId)) {
          state.robot.carriedBrickId = null;
        }
        draw();
        return api;
      },
      rotateRobot(nextFacing) {
        if (nextFacing === 'left' || nextFacing === 'right') {
          state.robot.facing = nextFacing;
        } else {
          state.robot.facing = state.robot.facing === 'right' ? 'left' : 'right';
        }
        draw();
        return api;
      },
      setRobotPose(patch) {
        const next = patch || {};
        if (typeof next.x === 'number') {
          state.robot.x = clamp(next.x, config.bounds.minX, config.bounds.maxX);
        }
        if (typeof next.y === 'number') {
          state.robot.y = clamp(next.y, config.bounds.minY, config.bounds.maxY);
        }
        if (next.facing === 'left' || next.facing === 'right') {
          state.robot.facing = next.facing;
        }
        draw();
        return api;
      },
      setState(nextState) {
        const merged = normaliseState(nextState || {}, config, state);
        state.robot = merged.robot;
        state.bricks = merged.bricks;
        state.nextBrickId = merged.nextBrickId;
        draw();
        return api;
      },
      stackCount(position) {
        return stackCountAt(state, position);
      },
      stickBrick(id) {
        const brick = state.bricks.find((item) => item.id === id);
        if (!brick) {
          return false;
        }
        state.robot.carriedBrickId = id;
        draw();
        return true;
      },
      stickTopBrick(position) {
        const brick = topBrickAt(state, position);
        if (!brick) {
          return null;
        }
        state.robot.carriedBrickId = brick.id;
        draw();
        return brick.id;
      },
      toSVG() {
        return draw();
      },
      unstickBrick(position) {
        if (!state.robot.carriedBrickId) {
          return null;
        }
        const brick = carriedBrick(state);
        if (!brick) {
          state.robot.carriedBrickId = null;
          draw();
          return null;
        }
        if (position && typeof position === 'object') {
          if (typeof position.x === 'number') brick.x = position.x;
          if (typeof position.y === 'number') brick.y = position.y;
          if (typeof position.z === 'number') brick.z = Math.max(0, position.z);
        }
        state.robot.carriedBrickId = null;
        draw();
        return brick.id;
      },
      updateBrick(id, patch) {
        const brick = state.bricks.find((item) => item.id === id);
        if (!brick) {
          return false;
        }
        const next = patch || {};
        if (typeof next.x === 'number') brick.x = next.x;
        if (typeof next.y === 'number') brick.y = next.y;
        if (typeof next.z === 'number') brick.z = Math.max(0, next.z);
        draw();
        return true;
      }
    };

    draw();
    return api;
  }

  function buildConfig(options) {
    const palette = Object.assign({}, DEFAULTS.palette, options.palette || {});
    const bounds = Object.assign({}, DEFAULTS.bounds, options.bounds || {});
    return Object.assign({}, DEFAULTS, options, { palette, bounds });
  }

  function buildDefaultState(config) {
    const rightStack = { x: 4.0, y: 2.2, z: 0.0 };
    const frontStack = { x: 2.15, y: 3.25, z: 0.0 };
    return {
      robot: { x: 2.65, y: 1.95, facing: 'right', carriedBrickId: null },
      bricks: [
        { id: 'front-0', x: frontStack.x, y: frontStack.y, z: 0.0 },
        { id: 'right-0', x: rightStack.x, y: rightStack.y, z: 0.0 },
        { id: 'right-1', x: rightStack.x, y: rightStack.y, z: config.brickHeight },
        { id: 'right-2', x: rightStack.x, y: rightStack.y, z: config.brickHeight * 2 }
      ],
      nextBrickId: 1
    };
  }

  function carriedBrick(state) {
    if (!state.robot.carriedBrickId) {
      return null;
    }
    return state.bricks.find((brick) => brick.id === state.robot.carriedBrickId) || null;
  }

  function carryPosition(robot) {
    return {
      x: robot.facing === 'right' ? robot.x + 0.52 : robot.x - 0.18,
      y: robot.y + 0.03,
      z: 0
    };
  }

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function deepClone(value) {
    return JSON.parse(JSON.stringify(value));
  }

  function getStackBricks(state, position) {
    if (!hasStackPosition(position)) {
      return [];
    }
    return state.bricks
      .filter((brick) => brick.id !== state.robot.carriedBrickId)
      .filter((brick) => Math.abs(brick.x - position.x) < 0.001 && Math.abs(brick.y - position.y) < 0.001)
      .sort((a, b) => b.z - a.z);
  }

  function hasStackPosition(position) {
    return !!position && typeof position.x === 'number' && typeof position.y === 'number';
  }

  function nextStackZAt(state, position, config) {
    const top = topBrickAt(state, position);
    return top ? top.z + config.brickHeight : 0;
  }

  function normaliseBrick(source, state, config) {
    const brick = source || {};
    const nextBrickId = typeof state.nextBrickId === 'number' ? state.nextBrickId : 1;
    const id = brick.id || 'brick-' + nextBrickId;
    state.nextBrickId = brick.id ? nextBrickId : nextBrickId + 1;
    return {
      id,
      x: typeof brick.x === 'number' ? brick.x : 2.2,
      y: typeof brick.y === 'number' ? brick.y : 3.2,
      z: typeof brick.z === 'number' ? Math.max(0, brick.z) : 0
    };
  }

  function normaliseState(input, config, fallback) {
    const seed = fallback ? deepClone(fallback) : buildDefaultState(config);
    const source = input || {};
    const robotSource = source.robot || seed.robot;
    const robot = {
      x: typeof robotSource.x === 'number' ? clamp(robotSource.x, config.bounds.minX, config.bounds.maxX) : seed.robot.x,
      y: typeof robotSource.y === 'number' ? clamp(robotSource.y, config.bounds.minY, config.bounds.maxY) : seed.robot.y,
      facing: robotSource.facing === 'left' ? 'left' : 'right',
      carriedBrickId: robotSource.carriedBrickId || null
    };
    const nextState = {
      robot,
      bricks: [],
      nextBrickId: typeof source.nextBrickId === 'number' ? source.nextBrickId : seed.nextBrickId
    };
    const brickSource = Array.isArray(source.bricks) ? source.bricks : seed.bricks;
    nextState.bricks = brickSource.map((brick) => normaliseBrick(brick, nextState, config));
    if (!nextState.bricks.some((brick) => brick.id === nextState.robot.carriedBrickId)) {
      nextState.robot.carriedBrickId = null;
    }
    syncCarriedBrick(nextState, config);
    return nextState;
  }

  function resolveTarget(target) {
    if (!target) {
      return null;
    }
    if (typeof target === 'string') {
      return typeof document !== 'undefined' ? document.querySelector(target) : null;
    }
    return target;
  }

  function renderLeiaRobotScene(inputState, options) {
    const config = buildConfig(options || {});
    const state = normaliseState(inputState || buildDefaultState(config), config);
    return renderSceneMarkup(state, config);
  }

  function renderSceneMarkup(state, config) {
    const robotScreen = worldIsoPoint(state.robot.x, state.robot.y, 0, config);
    const visibleBricks = state.bricks
      .filter((brick) => brick.id !== state.robot.carriedBrickId)
      .sort((a, b) => (a.z - b.z) || ((a.x + a.y) - (b.x + b.y)));

    return [
      `<svg xmlns="http://www.w3.org/2000/svg" width="${config.width}" height="${config.height}" viewBox="0 0 ${config.width} ${config.height}">`,
      renderBackground(config),
      '<g id="scene">',
      renderFloor(config),
      `<g transform="translate(${fmt(robotScreen.x)} ${fmt(robotScreen.y)})">`,
      renderRobotSprite(state.robot.facing, !!state.robot.carriedBrickId, config),
      '</g>',
      ...visibleBricks.map((brick) => `<g filter="url(#softShadow)">${renderBox(brickBox(brick.x, brick.y, brick.z, config), worldIsoPoint, config)}</g>`),
      '</g>',
      '</svg>'
    ].join('\n');
  }

  function renderBackground(config) {
    return [
      '<defs>',
      `  <linearGradient id="bgGradient" x1="0%" y1="0%" x2="100%" y2="100%">`,
      `    <stop offset="0%" stop-color="${config.palette.skyTop}" />`,
      `    <stop offset="100%" stop-color="${config.palette.skyBottom}" />`,
      '  </linearGradient>',
      '  <filter id="softShadow" x="-35%" y="-35%" width="170%" height="170%">',
      '    <feDropShadow dx="0" dy="6" stdDeviation="6" flood-color="#102030" flood-opacity="0.16" />',
      '  </filter>',
      '</defs>',
      '<rect width="100%" height="100%" fill="url(#bgGradient)" />'
    ].join('\n');
  }

  function renderFloor(config) {
    const tiles = [];
    for (let x = config.boardMin; x < config.boardMax; x += 1) {
      for (let y = config.boardMin; y < config.boardMax; y += 1) {
        const highlight = (x + y) % 2 === 0;
        const fill = shade(config.palette.floor, highlight ? 0.02 : -0.02);
        const opacity = highlight ? 0.88 : 0.76;
        tiles.push(renderFloorTile(x, y, fill, opacity, config));
      }
    }
    return tiles.join('\n');
  }

  function renderFloorTile(x, y, fill, opacity, config) {
    return polygon([
      worldIsoPoint(x, y, 0, config),
      worldIsoPoint(x + 1, y, 0, config),
      worldIsoPoint(x + 1, y + 1, 0, config),
      worldIsoPoint(x, y + 1, 0, config)
    ], fill, 'none', opacity, config);
  }

  function renderRobotSprite(facing, carrying, config) {
    const body = renderRobot(carrying, config);
    if (facing === 'right') {
      return body;
    }
    return '<g transform="scale(-1 1)">' + body + '</g>';
  }

  function renderRobot(carrying, config) {
    const boxes = [
      { x: 0.04, y: 0.12, z: 0.0, dx: 1.44, dy: 0.24, dz: 0.18, color: config.palette.tread, stroke: config.stroke },
      { x: 0.04, y: 1.24, z: 0.0, dx: 1.44, dy: 0.24, dz: 0.18, color: config.palette.tread, stroke: config.stroke },
      { x: 0.18, y: 0.26, z: 0.12, dx: 1.02, dy: 1.08, dz: 0.42, color: config.palette.body, stroke: config.stroke },
      { x: 0.94, y: 0.58, z: 0.12, dx: 0.26, dy: 0.36, dz: 3.36, color: config.palette.mast, stroke: config.stroke }
    ];
    const parts = boxes.map((box) => renderBox(box, localIsoPoint, config));
    if (carrying) {
      parts.push(renderBox(brickBox(1.12, 0.58, 1.1, config), localIsoPoint, config));
    }
    return parts.join('\n');
  }

  function brickBox(x, y, z, config) {
    return { x, y, z, dx: config.brickSize, dy: config.brickSize, dz: config.brickHeight, color: config.brickColor, stroke: config.stroke };
  }

  function renderBox(box, projector, config) {
    const p000 = projector(box.x, box.y, box.z, config);
    const p100 = projector(box.x + box.dx, box.y, box.z, config);
    const p010 = projector(box.x, box.y + box.dy, box.z, config);
    const p110 = projector(box.x + box.dx, box.y + box.dy, box.z, config);
    const p001 = projector(box.x, box.y, box.z + box.dz, config);
    const p101 = projector(box.x + box.dx, box.y, box.z + box.dz, config);
    const p011 = projector(box.x, box.y + box.dy, box.z + box.dz, config);
    const p111 = projector(box.x + box.dx, box.y + box.dy, box.z + box.dz, config);

    return [
      polygon([p011, p111, p110, p010], shade(box.color, -0.16), box.stroke, 1, config),
      polygon([p101, p111, p110, p100], shade(box.color, -0.06), box.stroke, 1, config),
      polygon([p001, p101, p111, p011], shade(box.color, 0.18), box.stroke, 1, config)
    ].join('\n');
  }

  function polygon(points, fill, stroke, opacity, config) {
    return '<polygon points="' + pointsToSvg(points) + '" fill="' + fill + '" stroke="' + (stroke || config.stroke) + '" stroke-width="1.4" opacity="' + fmt(opacity == null ? 1 : opacity) + '" stroke-linejoin="round" />';
  }

  function pointsToSvg(points) {
    return points.map((point) => fmt(point.x) + ',' + fmt(point.y)).join(' ');
  }

  function stackCountAt(state, position) {
    return getStackBricks(state, position).length;
  }

  function syncCarriedBrick(state, config) {
    const brick = carriedBrick(state);
    if (!brick) {
      return;
    }
    const target = carryPosition(state.robot, config);
    brick.x = target.x;
    brick.y = target.y;
    brick.z = target.z;
  }

  function topBrickAt(state, position) {
    return getStackBricks(state, position)[0] || null;
  }

  function worldIsoPoint(x, y, z, config) {
    const base = rawIsoPoint(x, y, z, config);
    return { x: config.originX + base.x, y: config.originY + base.y };
  }

  function localIsoPoint(x, y, z, config) {
    const base = rawIsoPoint(x, y, z, config);
    const pivot = rawIsoPoint(config.robotPivot.x, config.robotPivot.y, config.robotPivot.z, config);
    return { x: base.x - pivot.x, y: base.y - pivot.y };
  }

  function rawIsoPoint(x, y, z, config) {
    return {
      x: (x - y) * config.isoX,
      y: (x + y) * config.isoY - z * config.isoZ
    };
  }

  function fmt(value) {
    return Number(value).toFixed(2).replace(/\.?0+$/, '');
  }

  function shade(color, amount) {
    return mix(color, amount >= 0 ? '#FFFFFF' : '#000000', Math.abs(amount));
  }

  function mix(color, target, amount) {
    const source = hexToRgb(color);
    const goal = hexToRgb(target);
    return rgbToHex(source.map((channel, index) => channel + (goal[index] - channel) * amount));
  }

  function hexToRgb(value) {
    const source = value.replace('#', '');
    return [
      parseInt(source.slice(0, 2), 16),
      parseInt(source.slice(2, 4), 16),
      parseInt(source.slice(4, 6), 16)
    ];
  }

  function rgbToHex(rgb) {
    return '#' + rgb.map((channel) => Math.round(clamp(channel, 0, 255)).toString(16).padStart(2, '0')).join('').toUpperCase();
  }

  return {
    createLeiaRobotScene,
    renderLeiaRobotScene
  };
});

