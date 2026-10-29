'''Renders the scene to a PNG without a display, e.g. for CI artefacts or bug reports.'''
from pathlib import Path
from typing import List, Optional

from PIL import Image, ImageColor, ImageDraw, ImageFont

from stretch_sim.events import EVENT_COLORS
from stretch_sim.scene import Shape, build_scene, inventory_lines, status_lines
from stretch_sim.simulator import KinematicSimulator


class Transform:
    def __init__(self, world_w: float, world_h: float, scale: float = 110.0, margin: int = 30):
        self.scale, self.margin, self.world_h = scale, margin, world_h
        self.width = int(world_w * scale) + 2 * margin
        self.height = int(world_h * scale) + 2 * margin

    def __call__(self, x: float, y: float):
        return (self.margin + x * self.scale, self.margin + (self.world_h - y) * self.scale)

    def to_world(self, px: float, py: float):
        return ((px - self.margin) / self.scale, self.world_h - (py - self.margin) / self.scale)


_ANCHORS = {"center": "mm", "n": "mt", "s": "mb", "w": "lm", "e": "rm"}


def _font(size: int, bold: bool = False):
    for name in (("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"),):
        try:
            return ImageFont.truetype(name, size + 3)
        except OSError:
            pass
    return ImageFont.load_default(size=size + 3)


def _dashed(draw, points, fill, width, dash=8.0, gap=6.0):
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        length = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        t = 0.0
        while t < length:
            t1 = min(t + dash, length)
            draw.line([(x0 + (x1 - x0) * t / length, y0 + (y1 - y0) * t / length),
                       (x0 + (x1 - x0) * t1 / length, y0 + (y1 - y0) * t1 / length)], fill=fill, width=width)
            t += dash + gap


def draw_shapes(image: Image.Image, shapes: List[Shape], tf: Transform) -> None:
    draw = ImageDraw.Draw(image)
    for s in shapes:
        pts = [tf(x, y) for x, y in s.points]
        width = int(round(s.width))
        if s.kind == "poly":
            if s.alpha < 1.0:
                overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
                r, g, b = ImageColor.getrgb(s.fill)
                ImageDraw.Draw(overlay).polygon(pts, fill=(r, g, b, int(255 * s.alpha)))
                image.alpha_composite(overlay)
                draw = ImageDraw.Draw(image)
                if s.outline:
                    draw.line(pts + [pts[0]], fill=s.outline, width=width)
            elif s.dash:
                if s.fill:
                    draw.polygon(pts, fill=s.fill)
                _dashed(draw, pts + [pts[0]], s.outline, width)
            else:
                draw.polygon(pts, fill=s.fill, outline=s.outline, width=width)
        elif s.kind == "line":
            if s.dash:
                _dashed(draw, pts, s.outline, width)
            else:
                draw.line(pts, fill=s.outline, width=width, joint="curve")
        elif s.kind == "circle":
            (cx, cy), r = pts[0], s.radius * tf.scale
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=s.fill, outline=s.outline, width=width)
        elif s.kind == "text":
            draw.text(pts[0], s.text, fill=s.fill or "#000000", font=_font(s.size, s.bold), anchor=_ANCHORS[s.anchor])


def render_png(sim: KinematicSimulator, path: Path, scale: float = 110.0, show_labels: bool = True,
               panel_width: int = 230, log_lines: int = 14) -> Path:
    snap = sim.snapshot()
    tf = Transform(sim.world.width, sim.world.height, scale)
    line_h = 15
    log_h = 16 + line_h * (2 + log_lines)
    image = Image.new("RGBA", (tf.width + panel_width, tf.height + log_h), "#ffffff")
    draw_shapes(image, build_scene(sim.world, snap, sim.geom, sim.camera, show_labels), tf)

    draw = ImageDraw.Draw(image)
    y = 12
    for text, heading in inventory_lines(sim.world, snap):
        draw.text((tf.width + 8, y), text, fill="#222222" if heading else "#555555", font=_font(9, heading))
        y += 15 if heading else 13

    y = tf.height
    for text in status_lines(snap, 1.0, False):
        draw.text((12, y), text, fill="#222222", font=_font(8))
        y += line_h
    for event in sim.log.events[-log_lines:]:
        draw.text((12, y), event.format(), fill=EVENT_COLORS.get(event.kind, "#333333"), font=_font(8))
        y += line_h

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path)
    return Path(path)

