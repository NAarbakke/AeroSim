"""Bake the F-16 glTF used by the terrain-flight pages into ``aerosim/plots/data/f16/f16.glb``.

    python examples/bake_f16_model.py path/to/sketchfab_export   # folder with scene.gltf + scene.bin

Source: "F-16 Fighter Jet" by iedalton, CC BY 4.0,
https://sketchfab.com/3d-models/f-16-fighter-jet-d84491f443384ee488593cc6f0f0839e
(the Sketchfab glTF export, 4.2k triangles, eight plain-colour materials, no textures).

The export nests a glTF orientation node, an FBX correction and a scale of about 60 over the
meshes. This flattens all of it into the vertices and writes one binary glTF (GLB): nose
along +x, up +y, right wing along +z (three.js / glTF frame), uniformly scaled to the
F-16's real overall length of 15.06 m, origin about 8.5 m behind the nose tip (near the
centre of gravity) on the fuselage centre line. Materials and the attribution are kept.
The model is stylised: at the true length its wing spans 13.7 m (real: 9.96 m with tip
missiles).
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "aerosim" / "plots" / "data" / "f16" / "f16.glb"
LENGTH = 15.06          # m, overall, F-16C
CG_BEHIND_NOSE = 8.5    # m
CENTRE_LINE_Y = 228.0   # model units, fuselage centre line in the export's world frame


def _local(n: dict) -> np.ndarray:
    if "matrix" in n:
        return np.array(n["matrix"], float).reshape(4, 4).T
    M = np.eye(4)
    if "rotation" in n:
        x, y, z, w = n["rotation"]
        M[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    S = np.diag([*n.get("scale", [1, 1, 1]), 1.0])
    T = np.eye(4)
    T[:3, 3] = n.get("translation", [0, 0, 0])
    return T @ M @ S


def bake(src: Path, out: Path = OUT) -> Path:
    g = json.loads((src / "scene.gltf").read_text(encoding="utf-8"))
    buf = (src / g["buffers"][0]["uri"]).read_bytes()

    def acc(i):
        a = g["accessors"][i]
        bv = g["bufferViews"][a["bufferView"]]
        n = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[a["type"]]
        dt = np.dtype({5126: "<f4", 5125: "<u4", 5123: "<u2", 5121: "u1"}[a["componentType"]])
        off = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
        stride = bv.get("byteStride") or n * dt.itemsize
        arr = np.ndarray((a["count"], n), dt, buf, off, (stride, dt.itemsize)).copy()
        return arr if n > 1 else arr[:, 0]

    prims = []

    def walk(i, P):
        n = g["nodes"][i]
        W = P @ _local(n)
        if "mesh" in n:
            for p in g["meshes"][n["mesh"]]["primitives"]:
                if p.get("mode", 4) != 4:
                    raise ValueError("only triangle lists are supported")
                v = acc(p["attributes"]["POSITION"]).astype(float)
                nr = acc(p["attributes"]["NORMAL"]).astype(float)
                idx = acc(p["indices"]) if "indices" in p else np.arange(len(v))
                v = (np.c_[v, np.ones(len(v))] @ W.T)[:, :3]
                nr = nr @ np.linalg.inv(W[:3, :3])         # inverse transpose, row vectors
                nr /= np.linalg.norm(nr, axis=1, keepdims=True)
                if np.linalg.det(W[:3, :3]) < 0:          # mirrored: keep the winding outward
                    idx = idx.reshape(-1, 3)[:, ::-1].ravel()
                prims.append((p["material"], v, nr, idx))
        for c in n.get("children", []):
            walk(c, W)

    for r in g["scenes"][g.get("scene", 0)]["nodes"]:
        walk(r, np.eye(4))
    allv = np.vstack([v for _, v, _, _ in prims])
    tip = allv[:, 0].max()
    k = LENGTH / (tip - allv[:, 0].min())
    origin = np.array([tip - CG_BEHIND_NOSE / k, CENTRE_LINE_Y, 0.0])

    blob, views, accessors, out_prims = bytearray(), [], [], []

    def add(arr: np.ndarray, target: int, comp: int, typ: str, minmax: bool = False) -> int:
        while len(blob) % 4:
            blob.append(0)
        views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": arr.nbytes, "target": target})
        blob.extend(arr.tobytes())
        a = {"bufferView": len(views) - 1, "componentType": comp, "count": len(arr), "type": typ}
        if minmax:
            a["min"], a["max"] = arr.min(0).tolist(), arr.max(0).tolist()
        accessors.append(a)
        return len(accessors) - 1

    for mat, v, nr, idx in prims:
        pos = ((v - origin) * k).astype("<f4")
        ia = idx.astype("<u2") if idx.max() < 65536 else idx.astype("<u4")
        out_prims.append({"attributes": {"POSITION": add(pos, 34962, 5126, "VEC3", True),
                                         "NORMAL": add(nr.astype("<f4"), 34962, 5126, "VEC3")},
                          "indices": add(ia, 34963, 5123 if ia.dtype == np.uint16 else 5125, "SCALAR"),
                          "material": mat})
    extras = dict(g["asset"].get("extras", {}))
    extras["modified"] = ("AeroSim: node transforms flattened, scaled to 15.06 m overall length, "
                          "nose +x, up +y, origin near the centre of gravity")
    doc = {"asset": {"version": "2.0", "generator": "AeroSim bake_f16_model.py", "extras": extras},
           "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"name": "F-16", "mesh": 0}],
           "meshes": [{"name": "F-16", "primitives": out_prims}],
           "materials": g["materials"], "accessors": accessors, "bufferViews": views,
           "buffers": [{"byteLength": len(blob)}]}
    js = json.dumps(doc, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    blob += b"\0" * (-len(blob) % 4)
    glb = (struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(blob))
           + struct.pack("<II", len(js), 0x4E4F534A) + js + struct.pack("<II", len(blob), 0x004E4942) + blob)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(glb)
    span = (allv[:, 2].max() - allv[:, 2].min()) * k
    print(f"wrote {out} ({len(glb) / 1e3:.0f} kB): length {LENGTH} m, span {span:.1f} m, "
          f"{sum(len(i) for *_, i in prims) // 3} triangles")
    return out


if __name__ == "__main__":
    bake(Path(sys.argv[1]))
