"""Skanda-0 against MedGemma 27B and MedGemma + MAIRA-2 on held-out test cases: one grouped bar chart per modality.

  python3 docs/build_benchmark.py     → docs/img/benchmark.svg
Numbers are the ones in README.md (held-out test cases, read 1 Nov 2026); MAIRA-2 reads chest X-rays only.
"""
import os

INK, MUTED, GRID, AXIS = "#111827", "#6b7280", "#eef0f3", "#d1d5db"
METRICS = ["Abnormal flagged", "Normals called normal", "Right finding type", "Box precision", "Box recall"]
MODELS = ["MedGemma 27B", "MedGemma + MAIRA-2", "Skanda-0"]
COLOURS = {"MedGemma 27B": "#9ca3af", "MedGemma + MAIRA-2": "#d1d5db", "Skanda-0": "#52cafc"}
# value per metric, None = not measured (MAIRA-2 has no CT/MRI read and gives no boxes in the fused mode)
DATA = {
    ("Chest X-ray", "111 ChestX-Det test films"): {"MedGemma 27B": [90, 72, 41, 6, 4],
                                                   "MedGemma + MAIRA-2": [91, 72, 47, None, None],
                                                   "Skanda-0": [84, 69, 40.5, 25.5, 10]},
    ("CT", "340 held-out slices and volumes"): {"MedGemma 27B": [97, 6, 49, 1, 3], "MedGemma + MAIRA-2": None,
                                                "Skanda-0": [82, 91, 72, 22, 16]},
    ("MRI", "240 held-out slices and volumes"): {"MedGemma 27B": [99, 2, 42, 6, 8], "MedGemma + MAIRA-2": None,
                                                 "Skanda-0": [89, 94, 89, 70, 65]},
}


def panel(x, y, w, h, title, sub, series):
    s = [f'<text x="{x}" y="{y + 18}" font-size="20" font-weight="700" fill="{INK}">{title}</text>',
         f'<text x="{x}" y="{y + 42}" font-size="15" fill="{MUTED}">{sub}</text>']
    top, bottom, left = y + 70, y + h - 40, x + 40
    for v in range(0, 101, 25):
        yy = bottom - (bottom - top) * v / 100
        s.append(f'<line x1="{left}" y1="{yy}" x2="{x + w}" y2="{yy}" stroke="{GRID if v else AXIS}"/>'
                 f'<text x="{left - 8}" y="{yy + 4}" text-anchor="end" font-size="13" fill="{MUTED}">{v}</text>')
    gw = (x + w - left) / len(METRICS)
    present = [m for m in MODELS if series.get(m)]
    bw = min(46, (gw - 30) / len(present))
    for j, metric in enumerate(METRICS):
        gx = left + gw * j + (gw - bw * len(present)) / 2
        for i, m in enumerate(present):
            v = series[m][j]
            bx = gx + i * bw
            if v is None:
                s.append(f'<text x="{bx + bw / 2}" y="{bottom - 6}" text-anchor="middle" font-size="12" fill="{MUTED}">–</text>')
                continue
            bh = (bottom - top) * v / 100
            s.append(f'<rect x="{bx + 2}" y="{bottom - bh}" width="{bw - 4}" height="{max(bh, 1.5)}" rx="3" fill="{COLOURS[m]}"/>'
                     f'<text x="{bx + bw / 2}" y="{bottom - bh - 6}" text-anchor="middle" font-size="12.5" '
                     f'font-weight="{700 if m == "Skanda-0" else 500}" fill="{INK}">{v:g}</text>')
        s.append(f'<text x="{left + gw * (j + .5)}" y="{bottom + 22}" text-anchor="middle" font-size="14" fill="{INK}">{metric}</text>')
    return "".join(s)


def main():
    W, ph = 1200, 330
    H = 120 + ph * len(DATA) + 40
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Inter, Segoe UI, Helvetica, Arial, sans-serif">',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
         f'<text x="24" y="44" font-size="26" font-weight="800" fill="{INK}">Skanda-0 vs MedGemma on held-out test cases</text>',
         f'<text x="24" y="70" font-size="15" fill="{MUTED}">% · higher is better · no test patient was seen in training</text>']
    lx = 24
    for m in MODELS:
        s.append(f'<rect x="{lx}" y="86" width="16" height="16" rx="3" fill="{COLOURS[m]}"/>'
                 f'<text x="{lx + 22}" y="99" font-size="14" fill="{INK}">{m}</text>')
        lx += 30 + 8 * len(m) + 40
    for k, ((title, sub), series) in enumerate(DATA.items()):
        s.append(panel(24, 120 + k * ph, W - 48, ph - 10, title, sub, series))
    s.append(f'<text x="24" y="{H - 14}" font-size="12.5" fill="{MUTED}">MAIRA-2 reads chest X-rays only; – = not measured. '
             f'Box precision / recall: X-ray, box centre inside a radiologist box of the same class; CT/MRI, box IoU ≥ 0.3.</text>')
    s.append("</svg>")
    os.makedirs("docs/img", exist_ok=True)
    open("docs/img/benchmark.svg", "w").write("\n".join(s))
    print("docs/img/benchmark.svg")


if __name__ == "__main__":
    main()
