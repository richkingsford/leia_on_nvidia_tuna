from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence


SVG_WIDTH = 960
SVG_HEIGHT = 640
ORIGIN_X = 360
ORIGIN_Y = 154
ISO_X = 54
ISO_Y = 27
ISO_Z = 34
STROKE = "#10202B"
TOTAL_DURATION = 18.0
BRICK_SIZE = 0.40
BRICK_HEIGHT = 0.70
BRICK_COLOR = "#63F2F7"
FRONT_OFFSET = 1.20
CYCLE_DURATION = TOTAL_DURATION / 3.0
ROBOT_LOCAL_PIVOT = (0.80, 0.80, 0.0)
DIRECTIONS = ("left", "right")


@dataclass(frozen=True)
class Point:
    x: float
    y: float


@dataclass(frozen=True)
class Box:
    x: float
    y: float
    z: float
    dx: float
    dy: float
    dz: float
    color: str
    stroke: str = STROKE


Projector = Callable[[float, float, float], Point]
TimelineEntry = tuple[float, str, bool, tuple[float, float]]


def raw_iso_point(x: float, y: float, z: float) -> Point:
    return Point(
        (x - y) * ISO_X,
        (x + y) * ISO_Y - z * ISO_Z,
    )


LOCAL_PIVOT_POINT = raw_iso_point(*ROBOT_LOCAL_PIVOT)


def world_iso_point(x: float, y: float, z: float) -> Point:
    base = raw_iso_point(x, y, z)
    return Point(ORIGIN_X + base.x, ORIGIN_Y + base.y)


def local_iso_point(x: float, y: float, z: float) -> Point:
    base = raw_iso_point(x, y, z)
    return Point(base.x - LOCAL_PIVOT_POINT.x, base.y - LOCAL_PIVOT_POINT.y)


def fmt(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def points_to_svg(points: Sequence[Point]) -> str:
    return " ".join(f"{fmt(point.x)},{fmt(point.y)}" for point in points)


def clamp(value: float, minimum: float = 0.0, maximum: float = 255.0) -> int:
    return int(max(minimum, min(maximum, round(value))))


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    source = value.lstrip("#")
    return int(source[0:2], 16), int(source[2:4], 16), int(source[4:6], 16)


def rgb_to_hex(rgb: Sequence[int]) -> str:
    return "#" + "".join(f"{clamp(channel):02X}" for channel in rgb)


def mix(color: str, target: str, amount: float) -> str:
    source_rgb = hex_to_rgb(color)
    target_rgb = hex_to_rgb(target)
    blended = [
        source + (target_value - source) * amount
        for source, target_value in zip(source_rgb, target_rgb)
    ]
    return rgb_to_hex(blended)


def shade(color: str, amount: float) -> str:
    if amount >= 0:
        return mix(color, "#FFFFFF", amount)
    return mix(color, "#000000", -amount)


def polygon(points: Sequence[Point], fill: str, stroke: str = STROKE, opacity: float = 1.0) -> str:
    return (
        f'<polygon points="{points_to_svg(points)}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="1.6" opacity="{fmt(opacity)}" '
        'stroke-linejoin="round" />'
    )


def ellipse(
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    fill: str,
    opacity: float = 1.0,
    stroke: Optional[str] = None,
) -> str:
    stroke_part = f' stroke="{stroke}" stroke-width="1.0"' if stroke else ""
    return (
        f'<ellipse cx="{fmt(cx)}" cy="{fmt(cy)}" rx="{fmt(rx)}" ry="{fmt(ry)}" '
        f'fill="{fill}" opacity="{fmt(opacity)}"{stroke_part} />'
    )


def render_box(box: Box, project: Projector) -> str:
    p000 = project(box.x, box.y, box.z)
    p100 = project(box.x + box.dx, box.y, box.z)
    p010 = project(box.x, box.y + box.dy, box.z)
    p110 = project(box.x + box.dx, box.y + box.dy, box.z)
    p001 = project(box.x, box.y, box.z + box.dz)
    p101 = project(box.x + box.dx, box.y, box.z + box.dz)
    p011 = project(box.x, box.y + box.dy, box.z + box.dz)
    p111 = project(box.x + box.dx, box.y + box.dy, box.z + box.dz)

    near_y = polygon([p011, p111, p110, p010], shade(box.color, -0.16), box.stroke)
    near_x = polygon([p101, p111, p110, p100], shade(box.color, -0.06), box.stroke)
    top = polygon([p001, p101, p111, p011], shade(box.color, 0.18), box.stroke)
    return "\n".join([near_y, near_x, top])


def brick_box(x: float, y: float, z: float) -> Box:
    return Box(x, y, z, BRICK_SIZE, BRICK_SIZE, BRICK_HEIGHT, BRICK_COLOR)


def render_floor_tile(x: float, y: float, fill: str, opacity: float) -> str:
    return polygon(
        [
            world_iso_point(x, y, 0),
            world_iso_point(x + 1, y, 0),
            world_iso_point(x + 1, y + 1, 0),
            world_iso_point(x, y + 1, 0),
        ],
        fill=fill,
        stroke="none",
        opacity=opacity,
    )


def render_background(palette: dict[str, str]) -> str:
    return f"""
<defs>
  <linearGradient id="bgGradient" x1="0%" y1="0%" x2="100%" y2="100%">
    <stop offset="0%" stop-color="{palette['sky_top']}" />
    <stop offset="100%" stop-color="{palette['sky_bottom']}" />
  </linearGradient>
  <filter id="softShadow" x="-35%" y="-35%" width="170%" height="170%">
    <feDropShadow dx="0" dy="8" stdDeviation="10" flood-color="#102030" flood-opacity="0.20" />
  </filter>
</defs>
<rect width="100%" height="100%" fill="url(#bgGradient)" />
"""


def render_floor(
    rng: random.Random,
    palette: dict[str, str],
    stack_base: tuple[float, float, float],
    pile_base: tuple[float, float, float],
) -> str:
    tiles: list[str] = []
    for x in range(1, 6):
        for y in range(1, 6):
            highlight = rng.random() < 0.20
            fill = shade(palette["floor"], 0.03 if highlight else -0.02)
            opacity = 0.92 if highlight else 0.74
            tiles.append(render_floor_tile(x, y, fill, opacity))

    margin = 0.12
    stack_guide = polygon(
        [
            world_iso_point(stack_base[0] - margin, stack_base[1] - margin, 0),
            world_iso_point(stack_base[0] + BRICK_SIZE + margin, stack_base[1] - margin, 0),
            world_iso_point(stack_base[0] + BRICK_SIZE + margin, stack_base[1] + BRICK_SIZE + margin, 0),
            world_iso_point(stack_base[0] - margin, stack_base[1] + BRICK_SIZE + margin, 0),
        ],
        fill=palette["guide"],
        stroke="none",
        opacity=0.20,
    )
    pick_guide = polygon(
        [
            world_iso_point(pile_base[0] - margin, pile_base[1] - margin, 0),
            world_iso_point(pile_base[0] + BRICK_SIZE + margin, pile_base[1] - margin, 0),
            world_iso_point(pile_base[0] + BRICK_SIZE + margin, pile_base[1] + BRICK_SIZE + margin, 0),
            world_iso_point(pile_base[0] - margin, pile_base[1] + BRICK_SIZE + margin, 0),
        ],
        fill=palette["guide"],
        stroke="none",
        opacity=0.14,
    )
    return "\n".join(tiles + [stack_guide, pick_guide])


def render_track_rollers_vertical(side_x: float, palette: dict[str, str]) -> str:
    return ""


def render_track_rollers_horizontal(side_y: float, palette: dict[str, str]) -> str:
    return ""


def render_vertical_robot(carrying: bool, palette: dict[str, str]) -> str:
    return render_horizontal_robot(carrying, palette)


def render_horizontal_robot(carrying: bool, palette: dict[str, str]) -> str:
    parts: list[str] = []

    boxes = [
        Box(0.04, 0.10, 0.0, 1.48, 0.24, 0.18, palette["tread"]),
        Box(0.04, 1.26, 0.0, 1.48, 0.24, 0.18, palette["tread"]),
        Box(0.20, 0.24, 0.12, 1.04, 1.02, 0.40, palette["body"]),
        Box(1.00, 0.58, 0.12, 0.22, 0.34, 3.48, palette["mast"]),
    ]

    for item in boxes:
        parts.append(render_box(item, local_iso_point))

    if carrying:
        parts.append(render_box(brick_box(1.18, 0.58, 1.18), local_iso_point))

    return "\n".join(parts)


def render_robot_sprite(direction: str, carrying: bool, palette: dict[str, str]) -> str:
    if direction == "right":
        return render_horizontal_robot(carrying, palette)
    return f'<g transform="scale(-1 1)">{render_horizontal_robot(carrying, palette)}</g>'


def screen_delta(start_world: tuple[float, float, float], end_world: tuple[float, float, float]) -> tuple[float, float]:
    start = world_iso_point(*start_world)
    end = world_iso_point(*end_world)
    return end.x - start.x, end.y - start.y


def append_frame(frames: list[tuple[float, object]], key_time: float, value: object) -> None:
    if frames and abs(frames[-1][0] - key_time) < 0.0001:
        frames[-1] = (key_time, value)
    else:
        frames.append((key_time, value))


def translate_keyframes(frames: Sequence[tuple[float, tuple[float, float]]]) -> tuple[str, str]:
    key_times = ";".join(fmt(frame[0]) for frame in frames)
    values = ";".join(f"{fmt(frame[1][0])} {fmt(frame[1][1])}" for frame in frames)
    return key_times, values


def scalar_keyframes(frames: Sequence[tuple[float, float]]) -> tuple[str, str]:
    key_times = ";".join(fmt(frame[0]) for frame in frames)
    values = ";".join(fmt(frame[1]) for frame in frames)
    return key_times, values


def build_robot_timeline(
    home: tuple[float, float, float],
    pick: tuple[float, float, float],
    place: tuple[float, float, float],
) -> list[TimelineEntry]:
    cycle_states = [
        (0.0, "left", False, home),
        (0.7, "right", False, home),
        (1.7, "right", False, pick),
        (2.3, "right", True, pick),
        (3.2, "right", True, home),
        (3.9, "left", True, home),
        (4.9, "left", True, place),
        (5.5, "left", False, place),
        (6.0, "left", False, home),
    ]

    timeline: list[TimelineEntry] = []
    for cycle_index in range(3):
        offset = cycle_index * CYCLE_DURATION
        for phase_time, direction, carrying, position in cycle_states:
            normalized = min(offset + phase_time, TOTAL_DURATION) / TOTAL_DURATION
            entry = (normalized, direction, carrying, screen_delta(home, position))
            if timeline and abs(timeline[-1][0] - normalized) < 0.0001:
                timeline[-1] = entry
            else:
                timeline.append(entry)

    final_entry = (1.0, "left", False, timeline[-1][3])
    if timeline and abs(timeline[-1][0] - 1.0) < 0.0001:
        timeline[-1] = final_entry
    else:
        timeline.append(final_entry)
    return timeline


def build_robot_translation(timeline: Sequence[TimelineEntry]) -> tuple[str, str]:
    frames = [(time_value, delta) for time_value, _direction, _carrying, delta in timeline]
    return translate_keyframes(frames)


def build_state_visibility(timeline: Sequence[TimelineEntry], direction: str, carrying: bool) -> tuple[str, str]:
    frames: list[tuple[float, float]] = []
    for time_value, current_direction, current_carrying, _delta in timeline:
        opacity = 1.0 if (current_direction == direction and current_carrying == carrying) else 0.0
        append_frame(frames, time_value, opacity)
    return scalar_keyframes(frames)


def build_binary_visibility(events: Sequence[tuple[float, float]]) -> tuple[str, str]:
    frames: list[tuple[float, float]] = []
    for absolute_time, value in events:
        append_frame(frames, absolute_time / TOTAL_DURATION, value)
    append_frame(frames, 1.0, frames[-1][1])
    return scalar_keyframes(frames)


def create_scene(seed_value: int) -> str:
    rng = random.Random(seed_value)

    palette = {
        "sky_top": "#F5EEDB",
        "sky_bottom": "#D9E8EE",
        "floor": "#D7DED6",
        "guide": "#7FDCE3",
        "body": "#F0A43E",
        "mast": "#465A65",
        "tread": "#324651",
        "roller": "#7C909A",
        "carriage": "#556A77",
        "fork": "#B3C1C8",
    }

    home_robot = (2.20, 1.35, 0.0)
    pile_base = (4.60, 1.55, 0.0)
    stack_base = (2.40, 3.85, 0.0)

    pick_robot = (pile_base[0] - FRONT_OFFSET, pile_base[1], 0.0)
    place_robot = (stack_base[0], stack_base[1] - FRONT_OFFSET, 0.0)

    pile_positions = [
        (pile_base[0], pile_base[1], 0.0),
        (pile_base[0], pile_base[1], BRICK_HEIGHT),
        (pile_base[0], pile_base[1], BRICK_HEIGHT * 2),
    ]
    stack_positions = [
        (stack_base[0], stack_base[1], 0.0),
        (stack_base[0], stack_base[1], BRICK_HEIGHT),
        (stack_base[0], stack_base[1], BRICK_HEIGHT * 2),
        (stack_base[0], stack_base[1], BRICK_HEIGHT * 3),
    ]

    floor = render_floor(rng, palette, stack_base, pile_base)
    background = render_background(palette)

    base_brick = render_box(brick_box(*stack_positions[0]), world_iso_point)

    robot_timeline = build_robot_timeline(home_robot, pick_robot, place_robot)
    robot_translate_times, robot_translate_values = build_robot_translation(robot_timeline)

    sprite_groups: list[str] = []
    for direction in DIRECTIONS:
        for carrying in (False, True):
            opacity_times, opacity_values = build_state_visibility(robot_timeline, direction, carrying)
            sprite_groups.append(
                '<g opacity="0">'
                f'\n{render_robot_sprite(direction, carrying, palette)}'
                '\n<animate attributeName="opacity" '
                f'dur="{fmt(TOTAL_DURATION)}s" repeatCount="indefinite" '
                'calcMode="discrete" '
                f'keyTimes="{opacity_times}" values="{opacity_values}" />'
                '\n</g>'
            )

    robot_group = (
        '<g id="robotTranslate">'
        '\n<animateTransform attributeName="transform" type="translate" '
        f'dur="{fmt(TOTAL_DURATION)}s" repeatCount="indefinite" '
        f'keyTimes="{robot_translate_times}" values="{robot_translate_values}" calcMode="linear" />'
        f'\n{chr(10).join(sprite_groups)}'
        '\n</g>'
    )

    source_groups: list[str] = []
    for pile_index, source in enumerate(pile_positions):
        cycle_index = 2 - pile_index
        pick_time = cycle_index * CYCLE_DURATION + 2.3
        source_times, source_values = build_binary_visibility(
            [
                (0.0, 1.0),
                (pick_time, 1.0),
                (pick_time, 0.0),
            ]
        )
        source_groups.append(
            '<g filter="url(#softShadow)" opacity="1">'
            f'\n{render_box(brick_box(*source), world_iso_point)}'
            '\n<animate attributeName="opacity" '
            f'dur="{fmt(TOTAL_DURATION)}s" repeatCount="indefinite" '
            'calcMode="discrete" '
            f'keyTimes="{source_times}" values="{source_values}" />'
            '\n</g>'
        )

    target_groups: list[str] = []
    for cycle_index in range(3):
        target = stack_positions[cycle_index + 1]
        place_time = cycle_index * CYCLE_DURATION + 5.5
        target_times, target_values = build_binary_visibility(
            [
                (0.0, 0.0),
                (place_time, 0.0),
                (place_time, 1.0),
            ]
        )
        target_groups.append(
            '<g filter="url(#softShadow)" opacity="0">'
            f'\n{render_box(brick_box(*target), world_iso_point)}'
            '\n<animate attributeName="opacity" '
            f'dur="{fmt(TOTAL_DURATION)}s" repeatCount="indefinite" '
            'calcMode="discrete" '
            f'keyTimes="{target_times}" values="{target_values}" />'
            '\n</g>'
        )

    robot_home_screen = world_iso_point(*home_robot)
    scene = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_WIDTH}" height="{SVG_HEIGHT}" viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}">',
        background,
        '<g id="scene">',
        floor,
        f'<g transform="translate({fmt(robot_home_screen.x)} {fmt(robot_home_screen.y)})">',
        robot_group,
        '</g>',
        base_brick,
        *source_groups,
        *target_groups,
        '</g>',
        '</svg>',
    ]
    return "\n".join(scene)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an animated isometric SVG of a treaded mini forklift stacking cyan bricks."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("forklift_bricks_isometric.svg"),
        help="Path to the SVG file to write.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for floor variation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_value = args.seed if args.seed is not None else random.SystemRandom().randint(1000, 999999)
    svg = create_scene(seed_value)
    args.output.write_text(svg, encoding="utf-8")
    print(f"Wrote {args.output} with seed {seed_value}")


if __name__ == "__main__":
    main()








