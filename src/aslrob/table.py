"""결과 표 출력(RF geom/norm vs CNN plain/aug, 사진 부분집합). python -m aslrob.table"""
import json

from . import config as C

r = json.loads((C.RESULTS / "rf_results_group.json").read_text())["summary"]
a = json.loads((C.RESULTS / "cnn_aug_group.json").read_text())["summary"]
p = json.loads((C.RESULTS / "cnn_plain_group.json").read_text())["summary"]
cl = {"geom": r["geom|rf|clean"]["photo"]["e2e"], "norm": r["norm|rf|clean"]["photo"]["e2e"],
      "plain": p["clean"]["photo"], "aug": a["clean"]["photo"]}
print("cond | rf_geom e2e (drop) | rf_geom det_only | det_rate | rf_norm e2e | cnn_plain (drop) | cnn_aug (drop, sd)")
for c in a:
    g = r[f"geom|rf|{c}"]["photo"]
    n = r[f"norm|rf|{c}"]["photo"]
    print(f"{c} | {g['e2e']:.3f} ({100*(cl['geom']-g['e2e']):.1f}) | {g['det_only'] or float('nan'):.3f} | {g['det_rate']:.3f} | "
          f"{n['e2e']:.3f} | {p[c]['photo']:.3f} ({100*(cl['plain']-p[c]['photo']):.1f}) | "
          f"{a[c]['photo']:.3f} ({100*(cl['aug']-a[c]['photo']):.1f}, sd {a[c]['photo_sd']:.3f})")
