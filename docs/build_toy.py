"""Toy diagram of how Skanda-0 reads an image, for the README.

  python3 docs/build_toy.py   → docs/img/skanda-0-toy.svg
Blue = frozen, orange = fine-tuned, green = new, pink = the [SEG] path.
"""

BLUE, ORANGE, GREEN, PINK, INK, MUTED = "#3b82f6", "#f59e0b", "#10b981", "#ec4899", "#1f2937", "#6b7280"
FILL = {BLUE: "#eff6ff", ORANGE: "#fffbeb", GREEN: "#ecfdf5", PINK: "#fdf2f8", MUTED: "#f9fafb"}


def box(x, y, w, h, colour, title, lines, tag=None, num=None):
    s = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="16" fill="{FILL[colour]}" stroke="{colour}" stroke-width="2.5"/>']
    if num is not None:
        s.append(f'<circle cx="{x + 22}" cy="{y + 24}" r="14" fill="{colour}"/>'
                 f'<text x="{x + 22}" y="{y + 29}" text-anchor="middle" font-size="15" font-weight="700" fill="#fff">{num}</text>')
    s.append(f'<text x="{x + (44 if num is not None else 16)}" y="{y + 30}" font-size="18" font-weight="700" fill="{INK}">{title}</text>')
    for i, t in enumerate(lines):
        s.append(f'<text x="{x + 16}" y="{y + 58 + 21 * i}" font-size="14" fill="{INK}">{t}</text>')
    if tag:
        tw = 9 * len(tag) + 16
        s.append(f'<rect x="{x + w - tw - 10}" y="{y + h - 32}" width="{tw}" height="22" rx="11" fill="{colour}"/>'
                 f'<text x="{x + w - tw / 2 - 10}" y="{y + h - 16}" text-anchor="middle" font-size="12" font-weight="700" '
                 f'fill="#fff">{tag}</text>')
    return "".join(s)


def arrow(x1, y1, x2, y2, colour=MUTED, label=None, dash=False):
    d = ' stroke-dasharray="7 6"' if dash else ""
    s = f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{colour}" stroke-width="3"{d} marker-end="url(#a{colour[1:]})"/>'
    if label:
        s += (f'<text x="{(x1 + x2) / 2}" y="{(y1 + y2) / 2 - 9}" text-anchor="middle" font-size="13" font-style="italic" '
              f'fill="{colour}">{label}</text>')
    return s


def xray(x, y, w, h, outline=False):
    """A cartoon chest film: two lungs, a heart; optionally a pink outline + box over the right lower zone."""
    s = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="#111827"/>',
         f'<ellipse cx="{x + w * .32}" cy="{y + h * .5}" rx="{w * .2}" ry="{h * .36}" fill="#4b5563"/>',
         f'<ellipse cx="{x + w * .68}" cy="{y + h * .5}" rx="{w * .2}" ry="{h * .36}" fill="#4b5563"/>',
         f'<ellipse cx="{x + w * .56}" cy="{y + h * .66}" rx="{w * .14}" ry="{h * .16}" fill="#9ca3af"/>',
         f'<ellipse cx="{x + w * .30}" cy="{y + h * .70}" rx="{w * .09}" ry="{h * .09}" fill="#d1d5db" opacity=".75"/>']
    if outline:
        cx, cy, rx, ry = x + w * .30, y + h * .70, w * .10, h * .10
        s.append(f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" fill="{PINK}" fill-opacity=".35" stroke="{PINK}" stroke-width="2.5"/>')
        s.append(f'<rect x="{cx - rx - 4}" y="{cy - ry - 4}" width="{2 * rx + 8}" height="{2 * ry + 8}" fill="none" '
                 f'stroke="#facc15" stroke-width="2" stroke-dasharray="5 4"/>')
    s.append(f'<text x="{x + w * .14}" y="{y + h - 8}" font-size="11" fill="#9ca3af">R</text>'
             f'<text x="{x + w * .82}" y="{y + h - 8}" font-size="11" fill="#9ca3af">L</text>')
    return "".join(s)


def skanda0():
    W, H = 1240, 1000
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Inter, Segoe UI, Helvetica, Arial, sans-serif">',
         "<defs>" + "".join(f'<marker id="a{c[1:]}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
                            f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>'
                            for c in (BLUE, ORANGE, GREEN, PINK, MUTED)) + "</defs>",
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
         f'<text x="40" y="52" font-size="28" font-weight="800" fill="{INK}">How Skanda-0 reads an image: the toy version</text>',
         f'<text x="40" y="80" font-size="15" fill="{MUTED}">One model writes the report; every finding it writes carries a [SEG] token '
         f'that becomes that finding\'s outline. X-ray, CT and MRI alike.</text>']
    # row 1: image → eyes → brain → report card
    s.append(xray(40, 120, 170, 200))
    s.append(f'<text x="125" y="345" text-anchor="middle" font-size="14" font-weight="600" fill="{INK}">1 · the image</text>')
    s.append(arrow(215, 220, 262, 220))
    s.append(box(265, 125, 250, 190, ORANGE, "The eyes", ["Qwen3-VL vision encoder", "cuts the image into patches", "and turns them into", "image tokens"], "LoRA · LAST 11", 2))
    s.append(arrow(518, 220, 565, 220))
    s.append(box(568, 125, 270, 190, ORANGE, "The brain", ["Qwen3-VL 30B language model", "(mixture of experts, 3B active)", "reads the image tokens and", "writes the report, word by word"], "LoRA", 3))
    s.append(arrow(841, 220, 888, 220))
    card = ['{ "normal": false,', '  "findings": [{', '    "title": "Effusion, right",', '    "evidence": "...",',
            '    "location": "right lower zone",', '    "mask": <tspan fill="#ec4899" font-weight="800">[SEG]</tspan> }],',
            '  "impression": "..." }']
    s.append(f'<rect x="891" y="110" width="310" height="225" rx="16" fill="#f9fafb" stroke="{INK}" stroke-width="2"/>'
             f'<circle cx="913" cy="134" r="14" fill="{INK}"/><text x="913" y="139" text-anchor="middle" font-size="15" '
             f'font-weight="700" fill="#fff">4</text><text x="935" y="140" font-size="18" font-weight="700" fill="{INK}">The report</text>')
    for i, t in enumerate(card):
        s.append(f'<text x="906" y="{170 + 22 * i}" font-size="13.5" font-family="ui-monospace, Menlo, monospace" xml:space="preserve" fill="{INK}">{t}</text>')
    # [SEG] drops to row 2
    s.append(f'<path d="M1080,306 C1080,380 760,360 735,445" fill="none" stroke="{PINK}" stroke-width="3.5" marker-end="url(#a{PINK[1:]})"/>')
    s.append(f'<text x="930" y="385" font-size="14" font-weight="700" fill="{PINK}">the brain\'s "thought" at the [SEG] word</text>')
    s.append(f'<text x="930" y="405" font-size="13" fill="{PINK}">(its last hidden state, 2048 numbers)</text>')
    # score head: the brain's state just before it replies
    s.append(f'<path d="M828,318 C872,360 862,525 886,540" fill="none" stroke="{GREEN}" stroke-width="3" stroke-dasharray="7 6" marker-end="url(#a{GREEN[1:]})"/>')
    s.append(box(891, 455, 310, 185, GREEN, "The score head", ["one linear layer on the brain's", "state just before it replies:", "p(abnormal) + 13 X-ray classes", "→ hint line and normal gate"], "NEW", 8))
    # row 2: translator → outline pen → result
    s.append(box(568, 450, 270, 170, GREEN, "The translator", ["[SEG] projector, a small MLP", "2048 → 2048 → 256 numbers:", "turns the thought into an", "outline request for SAM 2"], "NEW", 5))
    s.append(arrow(565, 535, 518, 535, PINK))
    s.append(box(265, 430, 250, 210, ORANGE, "The outline pen", ["SAM 2.1 (from Sa2VA)", "", "", "mask decoder draws the", "outline + object score"], "TRAINED", 6))
    s.append(f'<rect x="281" y="500" width="218" height="34" rx="8" fill="{FILL[BLUE]}" stroke="{BLUE}" stroke-width="2"/>'
             f'<text x="390" y="522" text-anchor="middle" font-size="13" fill="{INK}">its own eyes: image encoder <tspan font-weight="700" fill="{BLUE}">frozen</tspan></text>')
    s.append(f'<path d="M125,352 C125,420 210,470 262,500" fill="none" stroke="{MUTED}" stroke-width="3" stroke-dasharray="7 6" marker-end="url(#a{MUTED[1:]})"/>')
    s.append(f'<text x="128" y="410" text-anchor="end" font-size="13" font-style="italic" fill="{MUTED}">same image,</text>'
             f'<text x="128" y="428" text-anchor="end" font-size="13" font-style="italic" fill="{MUTED}">at 1024²</text>')
    s.append(arrow(262, 535, 215, 590, PINK))
    s.append(xray(40, 520, 170, 200, outline=True))
    s.append(f'<text x="125" y="745" text-anchor="middle" font-size="14" font-weight="600" fill="{INK}">7 · one outline per finding</text>')
    s.append(f'<text x="125" y="765" text-anchor="middle" font-size="12" fill="{MUTED}">box = the outline\'s extent</text>')
    # how it learns
    y = 800
    s.append(f'<text x="40" y="{y}" font-size="20" font-weight="800" fill="{INK}">How it learns</text>')
    stages = [("Stage 0", "connector", "teach the translator + pen:", '"outline the heart" → outline', GREEN),
              ("Stage 1", "SFT, 2D", "write reports with one [SEG]", "per finding; X-ray, CT, MRI", ORANGE),
              ("Stage 2", "3D volumes", "one outline across slices,", "40% 2D replay", ORANGE),
              ("Stage 3", "X-ray polish", "X-ray reports at 0.3× lr,", "15% 3D replay", ORANGE)]
    for i, (a, b, c, d, col) in enumerate(stages):
        x = 40 + i * 296
        s.append(f'<rect x="{x}" y="{y + 18}" width="280" height="118" rx="14" fill="{FILL[col]}" stroke="{col}" stroke-width="2"/>'
                 f'<text x="{x + 14}" y="{y + 46}" font-size="16" font-weight="800" fill="{INK}">{a} <tspan font-weight="500" fill="{MUTED}">· {b}</tspan></text>'
                 f'<text x="{x + 14}" y="{y + 74}" font-size="13.5" fill="{INK}">{c}</text>'
                 f'<text x="{x + 14}" y="{y + 95}" font-size="13.5" fill="{INK}">{d}</text>')
        if i < 3:
            s.append(arrow(x + 281, y + 77, x + 294, y + 77))
    s.append(f'<text x="40" y="{y + 165}" font-size="13" fill="{MUTED}">Loss = report text (cross-entropy, JSON boilerplate × 0.3) + 2 × outline BCE + 0.5 × outline Dice '
             f'+ 0.2 × object-score BCE + 0.5 × score-head BCE.</text>')
    s.append(f'<text x="40" y="{y + 187}" font-size="13" fill="{MUTED}">Blue = frozen · orange = fine-tuned · green = new · pink = the [SEG] path. '
             f'A stage is kept only if its check on validation data is not worse than the best so far.</text>')
    s.append("</svg>")
    return "\n".join(s)


if __name__ == "__main__":
    import os
    os.makedirs("docs/img", exist_ok=True)
    open("docs/img/skanda-0-toy.svg", "w").write(skanda0())
    print("docs/img/skanda-0-toy.svg")
