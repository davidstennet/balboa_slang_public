import json
from pathlib import Path
import numpy as np
import trimesh

scene_dir = Path("scenes/hw2")
mesh_dir = scene_dir / "clone_trooper"
mesh_dir.mkdir(parents=True, exist_ok=True)

parts = []

WHITE = [0.91, 0.91, 0.93]
OFFWHITE = [0.84, 0.85, 0.88]
LIGHTGRAY = [0.60, 0.62, 0.66]
BLACK = [0.02, 0.02, 0.03]
DARK = [0.10, 0.10, 0.11]


def save_part(name, mesh, color):
    filename = mesh_dir / f"{name}.json"
    data = {
        "vertices": np.asarray(mesh.vertices).tolist(),
        "faces": np.asarray(mesh.faces).tolist(),
        "vertex_colors": [color for _ in range(len(mesh.vertices))]
    }
    with open(filename, "w") as f:
        json.dump(data, f)
    parts.append((name, color))


def apply_rotations(mesh, rx=0, ry=0, rz=0):
    if rx != 0:
        R = trimesh.transformations.rotation_matrix(np.radians(rx), [1, 0, 0])
        mesh.apply_transform(R)
    if ry != 0:
        R = trimesh.transformations.rotation_matrix(np.radians(ry), [0, 1, 0])
        mesh.apply_transform(R)
    if rz != 0:
        R = trimesh.transformations.rotation_matrix(np.radians(rz), [0, 0, 1])
        mesh.apply_transform(R)


def add_box(name, center, size, color, rx=0, ry=0, rz=0):
    mesh = trimesh.creation.box(extents=size)
    apply_rotations(mesh, rx, ry, rz)
    mesh.apply_translation(center)
    save_part(name, mesh, color)


def add_ellipsoid(name, center, scale, color, rx=0, ry=0, rz=0, subdivisions=3):
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=1.0)
    mesh.apply_scale(scale)
    apply_rotations(mesh, rx, ry, rz)
    mesh.apply_translation(center)
    save_part(name, mesh, color)


# =========================================================
# Main overall mass
# =========================================================

# Dome
add_ellipsoid(
    "dome",
    center=[0.0, 0.20, -0.02],
    scale=[1.02, 1.06, 0.90],
    color=WHITE,
    subdivisions=4
)

# Lower rear shell to give that full side profile
add_ellipsoid(
    "rear_shell",
    center=[0.0, -0.22, -0.18],
    scale=[1.02, 0.78, 0.90],
    color=OFFWHITE,
    subdivisions=3
)

# =========================================================
# Top fin / mohawk
# =========================================================

# Main fin body, pitched forward then sweeping backward
add_box(
    "fin_main",
    center=[0.0, 1.10, 0.18],
    size=[0.18, 0.26, 1.18],
    color=WHITE,
    rx=0,
    ry=0,
    rz=0
)

# Back extension to emulate long swept fin
add_box(
    "fin_back",
    center=[0.0, 1.12, -0.34],
    size=[0.18, 0.22, 0.82],
    color=WHITE,
    rx=-18
)

# =========================================================
# Visor / brow
# =========================================================

# Gray side/temple band seen in side refs
add_box(
    "temple_band",
    center=[0.0, 0.20, 0.53],
    size=[1.84, 0.22, 0.46],
    color=LIGHTGRAY
)

# Black brow band
add_box(
    "brow_band",
    center=[0.0, 0.12, 0.84],
    size=[1.78, 0.14, 0.10],
    color=BLACK
)

# Black horizontal visor
add_box(
    "visor_top",
    center=[0.0, -0.02, 1.02],
    size=[1.38, 0.13, 0.07],
    color=BLACK
)

# Black vertical visor
add_box(
    "visor_stem",
    center=[0.0, -0.36, 1.05],
    size=[0.26, 0.70, 0.07],
    color=BLACK
)

# =========================================================
# Face / cheek structure
# =========================================================

# Left cheek
add_box(
    "left_cheek",
    center=[-0.50, -0.28, 0.83],
    size=[0.58, 0.88, 0.18],
    color=WHITE,
    rz=-33
)

# Right cheek
add_box(
    "right_cheek",
    center=[0.50, -0.28, 0.83],
    size=[0.58, 0.88, 0.18],
    color=WHITE,
    rz=33
)

# Left cheek under-support / side plane
add_box(
    "left_side_face",
    center=[-0.78, -0.44, 0.42],
    size=[0.46, 0.76, 0.60],
    color=OFFWHITE,
    rz=-10
)

# Right cheek under-support / side plane
add_box(
    "right_side_face",
    center=[0.78, -0.44, 0.42],
    size=[0.46, 0.76, 0.60],
    color=OFFWHITE,
    rz=10
)

# Nose / mouth center plate
add_box(
    "mouth_plate",
    center=[0.0, -0.90, 0.76],
    size=[0.96, 0.42, 0.25],
    color=WHITE
)

# Chin
add_box(
    "chin",
    center=[0.0, -1.12, 0.61],
    size=[0.28, 0.16, 0.16],
    color=BLACK
)

# Mouth vent slits
vent_xs = [-0.26, -0.16, -0.06, 0.06, 0.16, 0.26]
for i, x in enumerate(vent_xs):
    add_box(
        f"mouth_vent_{i}",
        center=[x, -0.83, 0.92],
        size=[0.045, 0.14, 0.035],
        color=BLACK,
        rz=0
    )

# Lower cheek diagonal accents
add_box(
    "left_cheek_line",
    center=[-0.26, -0.90, 0.80],
    size=[0.40, 0.035, 0.035],
    color=LIGHTGRAY,
    rz=-54
)

add_box(
    "right_cheek_line",
    center=[0.26, -0.90, 0.80],
    size=[0.40, 0.035, 0.035],
    color=LIGHTGRAY,
    rz=54
)

# =========================================================
# Side ear / hinge assemblies
# =========================================================

# Large rectangular side arm
add_box(
    "left_ear_arm",
    center=[-0.98, -0.22, 0.18],
    size=[0.18, 0.88, 0.40],
    color=OFFWHITE
)

add_box(
    "right_ear_arm",
    center=[0.98, -0.22, 0.18],
    size=[0.18, 0.88, 0.40],
    color=OFFWHITE
)

# Circular hinge discs
add_ellipsoid(
    "left_ear_disc",
    center=[-0.98, 0.03, 0.42],
    scale=[0.22, 0.22, 0.08],
    color=OFFWHITE,
    subdivisions=3
)

add_ellipsoid(
    "right_ear_disc",
    center=[0.98, 0.03, 0.42],
    scale=[0.22, 0.22, 0.08],
    color=OFFWHITE,
    subdivisions=3
)

# Small circular inset
add_ellipsoid(
    "left_ear_disc_inner",
    center=[-0.98, 0.03, 0.46],
    scale=[0.08, 0.08, 0.03],
    color=LIGHTGRAY,
    subdivisions=2
)

add_ellipsoid(
    "right_ear_disc_inner",
    center=[0.98, 0.03, 0.46],
    scale=[0.08, 0.08, 0.03],
    color=LIGHTGRAY,
    subdivisions=2
)

# Small side nub on one side like the refs
add_box(
    "right_rear_nub",
    center=[1.10, -0.02, -0.04],
    size=[0.14, 0.24, 0.22],
    color=OFFWHITE
)

add_box(
    "left_rear_nub",
    center=[-1.10, -0.02, -0.04],
    size=[0.14, 0.24, 0.22],
    color=OFFWHITE
)

# =========================================================
# Rear / lower side details
# =========================================================

# Back panel strip
add_box(
    "rear_panel",
    center=[0.0, -0.28, -0.58],
    size=[1.36, 0.86, 0.18],
    color=OFFWHITE
)

# Rear vent bars, left
for i in range(5):
    add_box(
        f"left_rear_vent_{i}",
        center=[-0.52 + i * 0.08, -0.86, -0.73],
        size=[0.035, 0.18, 0.05],
        color=DARK
    )

# Rear vent bars, right
for i in range(5):
    add_box(
        f"right_rear_vent_{i}",
        center=[0.20 + i * 0.08, -0.86, -0.73],
        size=[0.035, 0.18, 0.05],
        color=DARK
    )

# Small rectangular emboss near upper side like the reference
add_box(
    "left_upper_mark",
    center=[-0.56, 0.53, 0.42],
    size=[0.16, 0.05, 0.05],
    color=OFFWHITE
)

add_box(
    "right_upper_mark",
    center=[0.56, 0.53, 0.42],
    size=[0.16, 0.05, 0.05],
    color=OFFWHITE
)

# =========================================================
# Scene output
# =========================================================

keyframes = [
    {
        "time": 0.0,
        "transform": [
            {"scale": [1.0, 1.0, 1.0]},
            {"rotate_x": [0.0]},
            {"rotate_y": [0.0]}
        ]
    },
    {
        "time": 4.0,
        "transform": [
            {"scale": [1.0, 1.0, 1.0]},
            {"rotate_x": [0.0]},
            {"rotate_y": [180.0]}
        ]
    },
    {
        "time": 8.0,
        "transform": [
            {"scale": [1.0, 1.0, 1.0]},
            {"rotate_x": [0.0]},
            {"rotate_y": [360.0]}
        ]
    }
]

objects = []
for name, color in parts:
    if color == BLACK:
        Ks = [0.03, 0.03, 0.03]
        exponent = 20.0
    elif color == LIGHTGRAY:
        Ks = [0.05, 0.05, 0.05]
        exponent = 28.0
    else:
        Ks = [0.08, 0.08, 0.08]
        exponent = 38.0

    objects.append({
        "filename": f"clone_trooper/{name}.json",
        "material": {
            "Kd": color,
            "Ks": Ks,
            "exponent": exponent
        },
        "transform_keyframes": keyframes
    })

scene = {
    "duration": 8.0,
    "background": [0.16, 0.18, 0.24],
    "camera": {
        "resolution": [800, 800],
        "s": 0.78,
        "transform": [
            {
                "lookat": {
                    "position": [0.0, 0.0, 5.0],
                    "target": [0.0, -0.10, 0.0],
                    "up": [0.0, 1.0, 0.0]
                }
            }
        ]
    },
    "light": {
        "light_dir": [0.55, 0.85, 0.70],
        "light_color": [0.80, 0.80, 0.80]
    },
    "objects": objects
}

scene_file = scene_dir / "clone_helmet_spin.json"
with open(scene_file, "w") as f:
    json.dump(scene, f, indent=2)

print("Generated updated Phase I-style clone helmet scene")
print(scene_file)
print(f"Parts generated: {len(parts)}")