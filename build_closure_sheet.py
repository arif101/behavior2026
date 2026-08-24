"""Contact sheets for the demo-closure film: one per view, caption bar per tile."""
import glob
import json
import sys

from PIL import Image, ImageDraw

OUT = "/root/demo_closure_film"
demo = int(sys.argv[1]) if len(sys.argv) > 1 else 30
num = json.load(open(f"{OUT}/numerics_d{demo}.json"))
rows = {r["frame"]: r for r in num["rows"]}
TW = 340

for view in ("A", "B"):
    paths = sorted(glob.glob(f"{OUT}/f*_{view}.png"))
    tiles = []
    for p in paths:
        t = int(p.split("/f")[-1][:4])
        r = rows.get(t, {})
        im = Image.open(p)
        im = im.resize((TW, int(im.height * TW / im.width)))
        cap = Image.new("RGB", (TW, 46), (12, 12, 16))
        d = ImageDraw.Draw(cap)
        d.text((6, 4), f"f{t} ({r.get('rel', '?'):+d})  surf {r.get('min_surf_gap', '?')}m"
                       f"  grip {r.get('grip_q', '?')}", fill=(235, 235, 235))
        d.text((6, 24), f"contact={r.get('contact')}  track={r.get('tracking')}"
                        f"  radio_z {r.get('radio_z')}",
               fill=(120, 235, 140) if r.get("contact") else (235, 170, 120))
        canvas = Image.new("RGB", (TW, im.height + 46), (0, 0, 0))
        canvas.paste(im, (0, 0)); canvas.paste(cap, (0, im.height))
        tiles.append(canvas)
    cols = 6
    nrows = (len(tiles) + cols - 1) // cols
    th_ = tiles[0].height
    sheet = Image.new("RGB", (cols * TW + (cols - 1) * 4, nrows * th_ + (nrows - 1) * 4),
                      (24, 24, 28))
    for i, tl in enumerate(tiles):
        sheet.paste(tl, ((i % cols) * (TW + 4), (i // cols) * (th_ + 4)))
    out = f"{OUT}/sheet_d{demo}_{view}.png"
    sheet.save(out)
    print("SHEET", out, len(tiles), "tiles")
