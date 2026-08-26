#!/usr/bin/env python3
"""Render the value-aware multi-scale graph concept as a standalone PNG."""

from __future__ import annotations

import math
import random
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont


SCALE = 2
WIDTH, HEIGHT = 1920, 1080
OUT = Path(__file__).resolve().parent / "assets" / "upbg_value_aware_multiscale_graph.png"

FONT_REGULAR = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"

C = {
    "bg": "#F7FAFC",
    "white": "#FFFFFF",
    "ink": "#20262D",
    "muted": "#5D6B78",
    "line": "#C9D6E2",
    "panel": "#EEF4F8",
    "map_unknown": "#DCE3E9",
    "map_free": "#FAFCFD",
    "wall": "#344554",
    "obstacle": "#8795A1",
    "blue": "#0F6FC6",
    "blue2": "#65A6DE",
    "blue_pale": "#E5F2FC",
    "teal": "#11C99A",
    "teal_dark": "#008C71",
    "teal_pale": "#E4F8F2",
    "orange": "#F28C00",
    "orange_pale": "#FFF1D9",
    "lime": "#A5C83F",
    "red": "#E84A4A",
    "grey_node": "#9EACB8",
    "grey_edge": "#AFC1D0",
}


def sp(value: float) -> int:
    return int(round(value * SCALE))


canvas = Image.new("RGB", (WIDTH * SCALE, HEIGHT * SCALE), C["bg"])
draw = ImageDraw.Draw(canvas)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REGULAR, sp(size))


def box(rect, radius=12, fill=None, outline=None, width=1):
    draw.rounded_rectangle(tuple(sp(v) for v in rect), radius=sp(radius), fill=fill, outline=outline, width=sp(width))


def line(points, fill, width=2, joint="curve"):
    draw.line([(sp(x), sp(y)) for x, y in points], fill=fill, width=sp(width), joint=joint)


def ellipse(rect, fill=None, outline=None, width=1):
    draw.ellipse(tuple(sp(v) for v in rect), fill=fill, outline=outline, width=sp(width))


def polygon(points, fill):
    draw.polygon([(sp(x), sp(y)) for x, y in points], fill=fill)


def text(x, y, value, size, color=C["ink"], bold=False, anchor="la", spacing=4):
    draw.text((sp(x), sp(y)), value, font=font(size, bold), fill=color, anchor=anchor, spacing=sp(spacing))


def centered(x, y, value, size, color=C["ink"], bold=False):
    text(x, y, value, size, color, bold, anchor="mm")


def arrow(points, color, width=5, head=13):
    line(points, color, width)
    x1, y1 = points[-2]
    x2, y2 = points[-1]
    angle = math.atan2(y2 - y1, x2 - x1)
    a = math.radians(28)
    p1 = (x2 - head * math.cos(angle - a), y2 - head * math.sin(angle - a))
    p2 = (x2 - head * math.cos(angle + a), y2 - head * math.sin(angle + a))
    polygon([(x2, y2), p1, p2], color)


def badge(x, y, label, color, pale, width=126):
    box((x, y, x + width, y + 34), 17, pale, None)
    ellipse((x + 12, y + 11, x + 24, y + 23), color)
    text(x + 34, y + 17, label, 14, color, True, anchor="lm")


def draw_robot(x, y, heading=-25):
    ellipse((x - 17, y - 17, x + 17, y + 17), C["white"], C["blue"], 4)
    ellipse((x - 8, y - 8, x + 8, y + 8), C["blue"])
    angle = math.radians(heading)
    tip = (x + 27 * math.cos(angle), y + 27 * math.sin(angle))
    left = (x + 8 * math.cos(angle + 2.4), y + 8 * math.sin(angle + 2.4))
    right = (x + 8 * math.cos(angle - 2.4), y + 8 * math.sin(angle - 2.4))
    polygon([tip, left, right], C["blue"])


def local_floorplan(x, y, w, h):
    mask = Image.new("L", (sp(w), sp(h)), 0)
    md = ImageDraw.Draw(mask)
    rooms = [
        (0.04, 0.05, 0.27, 0.37),
        (0.24, 0.18, 0.75, 0.32),
        (0.68, 0.04, 0.96, 0.31),
        (0.38, 0.28, 0.58, 0.69),
        (0.04, 0.53, 0.40, 0.72),
        (0.54, 0.51, 0.94, 0.73),
        (0.43, 0.66, 0.57, 0.90),
        (0.28, 0.84, 0.72, 0.98),
    ]
    for x1, y1, x2, y2 in rooms:
        md.rounded_rectangle(
            (sp(x1 * w), sp(y1 * h), sp(x2 * w), sp(y2 * h)),
            radius=sp(8),
            fill=255,
        )

    # Soft shadow and a single continuous wall contour make the map read as one space.
    shadow = mask.filter(ImageFilter.GaussianBlur(sp(8)))
    shadow_layer = Image.new("RGB", mask.size, "#9CAAB5")
    canvas.paste(shadow_layer, (sp(x + 4), sp(y + 7)), shadow)
    dilated = mask.filter(ImageFilter.MaxFilter(21))
    wall_ring = ImageChops.subtract(dilated, mask)
    wall_layer = Image.new("RGB", mask.size, C["wall"])
    canvas.paste(wall_layer, (sp(x), sp(y)), wall_ring)

    floor = Image.new("RGB", mask.size, C["map_free"])
    fd = ImageDraw.Draw(floor)
    for gx in range(sp(18), sp(w), sp(36)):
        fd.line((gx, 0, gx, sp(h)), fill="#EDF2F5", width=sp(1))
    for gy in range(sp(18), sp(h), sp(36)):
        fd.line((0, gy, sp(w), gy), fill="#EDF2F5", width=sp(1))
    random.seed(12)
    for _ in range(420):
        px = random.randrange(sp(w))
        py = random.randrange(sp(h))
        if mask.getpixel((px, py)):
            fd.ellipse((px, py, px + sp(1.2), py + sp(1.2)), fill="#DDE7ED")
    canvas.paste(floor, (sp(x), sp(y)), mask)

    def p(nx, ny):
        return (x + nx * w, y + ny * h)

    # Occupied islands and furniture-like obstacles.
    obstacles = [
        (0.10, 0.11, 0.16, 0.23),
        (0.18, 0.25, 0.24, 0.33),
        (0.76, 0.09, 0.82, 0.20),
        (0.86, 0.19, 0.93, 0.27),
        (0.42, 0.38, 0.48, 0.49),
        (0.48, 0.53, 0.55, 0.62),
        (0.12, 0.58, 0.22, 0.67),
        (0.72, 0.57, 0.79, 0.68),
        (0.37, 0.88, 0.46, 0.95),
        (0.57, 0.87, 0.66, 0.95),
    ]
    for x1, y1, x2, y2 in obstacles:
        box((x + x1 * w, y + y1 * h, x + x2 * w, y + y2 * h), 5, C["obstacle"], C["wall"], 2)

    # Visible frontier boundaries: orange indicates where known free space meets unknown space.
    for a, b in [((0.04, 0.57), (0.04, 0.69)), ((0.94, 0.55), (0.94, 0.69)), ((0.72, 0.98), (0.58, 0.98))]:
        ax, ay = p(*a)
        bx, by = p(*b)
        segments = 6
        for i in range(segments):
            t1 = i / segments
            t2 = min(1, t1 + 0.09)
            line([(ax + (bx - ax) * t1, ay + (by - ay) * t1), (ax + (bx - ax) * t2, ay + (by - ay) * t2)], C["orange"], 5)

    return mask, p


def sample_inside(mask, px, py, samples):
    for sx, sy in samples:
        if mask.getpixel((sp(px * sx), sp(py * sy))) == 0:
            return False
    return True


def draw_dense_graph(map_x, map_y, map_w, map_h, mask):
    spacing = 45
    points = {}
    for gy, py in enumerate(range(23, int(map_h) - 15, spacing)):
        for gx, px in enumerate(range(23, int(map_w) - 15, spacing)):
            if mask.getpixel((sp(px), sp(py))) > 0:
                points[(gx, gy)] = (map_x + px, map_y + py)

    for (gx, gy), a in points.items():
        for delta in [(1, 0), (0, 1), (1, 1)]:
            key = (gx + delta[0], gy + delta[1])
            if key in points:
                b = points[key]
                # Keep only short collision-free links.
                mx = ((a[0] + b[0]) / 2 - map_x) / map_w
                my = ((a[1] + b[1]) / 2 - map_y) / map_h
                if mask.getpixel((sp(mx * map_w), sp(my * map_h))) > 0:
                    line([a, b], C["grey_edge"], 1.5)

    for idx, (_, pnt) in enumerate(points.items()):
        color = C["blue2"] if idx % 5 else C["orange"]
        r = 4.8 if color == C["blue2"] else 5.8
        ellipse((pnt[0] - r, pnt[1] - r, pnt[0] + r, pnt[1] + r), color, C["white"], 1)

    robot = (map_x + 0.49 * map_w, map_y + 0.64 * map_h)
    # The long route needs many graph hops in the single-scale representation.
    route = [
        robot,
        (map_x + 0.49 * map_w, map_y + 0.27 * map_h),
        (map_x + 0.72 * map_w, map_y + 0.25 * map_h),
        (map_x + 0.84 * map_w, map_y + 0.16 * map_h),
    ]
    arrow(route, C["red"], 4, 11)
    draw_robot(*robot)
    box((map_x + 430, map_y + 18, map_x + 650, map_y + 72), 10, "#FFF4F4", C["red"], 1)
    text(map_x + 448, map_y + 34, "远端信息需跨 12–16 跳", 15, C["red"], True)
    return robot


def draw_region_node(cx, cy, color, title, subtitle):
    ellipse((cx - 49, cy - 49, cx + 49, cy + 49), C["white"], color, 6)
    ellipse((cx - 38, cy - 38, cx + 38, cy + 38), color)
    centered(cx, cy - 8, title, 18, C["white"], True)
    centered(cx, cy + 15, subtitle, 11, C["white"], True)


def draw_multiscale_graph(map_x, map_y, map_w, map_h, mask, p):
    robot = p(0.49, 0.64)
    junction = p(0.49, 0.58)
    region_a = p(0.15, 0.19)
    region_b = p(0.83, 0.17)
    region_c = p(0.84, 0.62)
    region_d = p(0.50, 0.90)

    # Region-level edges follow the major corridor skeleton and compress many remote nodes.
    region_paths = [
        ([junction, p(0.49, 0.27), p(0.26, 0.25), region_a], C["blue"]),
        ([junction, p(0.49, 0.26), p(0.72, 0.25), region_b], C["lime"]),
        ([junction, p(0.56, 0.62), region_c], C["orange"]),
        ([junction, p(0.50, 0.76), region_d], C["teal"]),
    ]
    for pts, color in region_paths:
        line(pts, C["white"], 10)
        line(pts, color, 5)

    # A magnified local decision zone retains actionable graph detail.
    local_r = 90
    ellipse((robot[0] - local_r, robot[1] - local_r, robot[0] + local_r, robot[1] + local_r), "#EAF7FC", C["blue2"], 3)
    local_nodes = {
        "r": robot,
        "u1": p(0.49, 0.57),
        "u2": p(0.49, 0.49),
        "l1": p(0.41, 0.62),
        "l2": p(0.34, 0.62),
        "r1": p(0.57, 0.62),
        "r2": p(0.65, 0.62),
        "d1": p(0.49, 0.72),
        "d2": p(0.49, 0.80),
    }
    for a, b in [("r", "u1"), ("u1", "u2"), ("r", "l1"), ("l1", "l2"), ("r", "r1"), ("r1", "r2"), ("r", "d1"), ("d1", "d2")]:
        line([local_nodes[a], local_nodes[b]], C["blue2"], 3)

    for key, pnt in local_nodes.items():
        if key == "r":
            continue
        if key in {"l2", "r2"}:
            color, radius = C["orange"], 9
        elif key == "u1":
            color, radius = C["teal"], 9
        else:
            color, radius = C["blue2"], 7
        ellipse((pnt[0] - radius, pnt[1] - radius, pnt[0] + radius, pnt[1] + radius), color, C["white"], 2)

    # The final policy output is still one feasible neighboring waypoint.
    chosen = local_nodes["r1"]
    ellipse((chosen[0] - 18, chosen[1] - 18, chosen[0] + 18, chosen[1] + 18), None, C["blue"], 4)
    arrow([robot, chosen], C["blue"], 5, 12)
    draw_robot(*robot)

    draw_region_node(*region_a, C["blue"], "区域 A", "高潜力")
    draw_region_node(*region_b, C["lime"], "区域 B", "较可靠")
    draw_region_node(*region_c, C["orange"], "区域 C", "待确认")
    draw_region_node(*region_d, C["teal"], "区域 D", "连通骨架")

    # Labels point to the two scales without covering the map.
    box((map_x + 18, map_y + 20, map_x + 240, map_y + 70), 10, C["blue_pale"], C["blue"], 1)
    text(map_x + 34, map_y + 45, "远端：1 个区域 = 1 个方向", 14, C["blue"], True, anchor="lm")
    box((map_x + 410, map_y + 340, map_x + 690, map_y + 390), 10, C["orange_pale"], C["orange"], 1)
    text(map_x + 426, map_y + 365, "近端：frontier / 路口 / 可行邻居", 14, C["orange"], True, anchor="lm")

    # Pointer-style callout for the selected action.
    tx, ty = chosen[0] + 30, chosen[1] - 42
    box((tx, ty, tx + 125, ty + 40), 10, C["white"], C["blue"], 2)
    text(tx + 14, ty + 20, "下一航点", 14, C["blue"], True, anchor="lm")
    line([(tx, ty + 34), (chosen[0] + 12, chosen[1] - 9)], C["blue"], 2)


# Top rule and title block.
draw.rectangle((0, 0, WIDTH * SCALE, sp(10)), fill=C["blue"])
text(70, 50, "IMPROVEMENT 3  ·  GRAPH WAVELET + DIFFPOOL", 15, C["blue"], True)
text(70, 88, "图小波 + 层级池化：有限节点预算应该保留什么？", 34, C["ink"], True)
text(70, 142, "Random-Walk Graph Wavelet 分离低频区域趋势与高频局部变化；DiffPool / Top-K 重组节点。", 19, C["muted"])
line([(70, 184), (1850, 184)], C["line"], 2)

# Comparison panels.
box((58, 216, 824, 844), 16, C["white"], C["line"], 2)
box((1096, 216, 1862, 844), 16, C["white"], C["line"], 2)

badge(84, 240, "改进前", C["red"], "#FFF0F0", 112)
text(214, 257, "单尺度稀疏图", 23, C["ink"], True, anchor="lm")
text(84, 296, "所有节点同一粒度；大图上仍有大量重复节点。", 16, C["muted"])

badge(1122, 240, "改进后", C["teal_dark"], C["teal_pale"], 112)
text(1252, 257, "Graph Wavelet 多尺度图", 23, C["ink"], True, anchor="lm")
text(1122, 296, "远端负责“去哪一片”，近端负责“下一步怎么走”。", 16, C["muted"])

left_map = (84, 338, 714, 438)
right_map = (1122, 338, 714, 438)
box((left_map[0], left_map[1], left_map[0] + left_map[2], left_map[1] + left_map[3]), 12, C["map_unknown"], C["line"], 1)
box((right_map[0], right_map[1], right_map[0] + right_map[2], right_map[1] + right_map[3]), 12, C["map_unknown"], C["line"], 1)

left_mask, _ = local_floorplan(*left_map)
draw_dense_graph(*left_map, left_mask)
right_mask, right_p = local_floorplan(*right_map)
draw_multiscale_graph(*right_map, right_mask, right_p)

# Transformation logic in the center.
centered(960, 278, "Graph Wavelet + DiffPool", 16, C["blue"], True)
badge(884, 330, "低频区域趋势", C["blue"], C["blue_pale"], 152)
badge(884, 380, "高频结构细节", C["orange"], C["orange_pale"], 152)
badge(884, 430, "A* 连通骨架", C["teal_dark"], C["teal_pale"], 152)
box((876, 504, 1018, 576), 18, C["blue"], None)
polygon([(1012, 490), (1068, 540), (1012, 590)], C["blue"])
centered(946, 540, "GWT + Pool", 17, C["white"], True)
centered(960, 628, "低频 → DiffPool 区域 token", 13, C["muted"])
centered(960, 654, "高频 → Top-K 细节节点", 13, C["muted"])

# Footer: the final decision sequence.
box((70, 882, 1850, 1018), 14, C["blue_pale"], C["blue2"], 2)
text(100, 912, "策略执行顺序", 15, C["blue"], True)
centered(340, 960, "低频区域层选方向", 21, C["blue"], True)
arrow([(500, 960), (650, 960)], C["blue2"], 4, 13)
centered(820, 960, "高频局部层展开细节", 21, C["teal_dark"], True)
arrow([(995, 960), (1145, 960)], C["blue2"], 4, 13)
centered(1334, 960, "Pointer Actor 选择下一航点", 21, C["orange"], True)
text(1580, 950, "同样预算下：", 15, C["muted"], True)
text(1580, 978, "方向不丢 · 动作可行 · 通路不断", 15, C["ink"], True)

OUT.parent.mkdir(parents=True, exist_ok=True)
canvas.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS).save(OUT, "PNG", optimize=True)
print(OUT)
