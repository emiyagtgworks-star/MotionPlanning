import numpy as np
import matplotlib.pyplot as plt
import random

# ============================================================
# WALL SEGMENTS - dari Figure 1 Environment Map
# Format: ((x1,y1), (x2,y2))
# ============================================================
walls = [
    # BORDER LUAR
    ((-10, -10), ( 60, -10)),  # bawah
    ((-10, -10), (-10,  60)),  # kiri
    ((-10,  60), ( 60,  60)),  # atas
    (( 60, -10), ( 60,  60)),  # kanan

    # INNER WALLS (sesuai Figure 1)
    # Dinding vertikal kiri dalam: x=0, y=-10 s/d y=40
    ((0, -10), (0, 40)),

    # Dinding horizontal atas kiri: y=40, x=-10 s/d x=20
    ((-10, 40), (20, 40)),

    # Dinding vertikal tengah: x=20, y=20 s/d y=40
    ((20, 20), (20, 40)),

    # Dinding horizontal tengah: y=20, x=20 s/d x=40
    ((20, 20), (40, 20)),

    # Dinding vertikal kanan dalam: x=40, y=20 s/d y=40
    ((40, 20), (40, 40)),

    # Dinding horizontal bawah kiri: y=5, x=10 s/d x=30
    ((10, 5), (30, 5)),

    # Dinding horizontal bawah kanan: y=5, x=45 s/d x=60
    ((45, 5), (60, 5)),
]

# ============================================================
# HELPER: JARAK TITIK KE SEGMEN
# ============================================================
def dist_point_to_segment(px, py, x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return np.hypot(px - x1, py - y1)
    t = max(0.0, min(1.0, ((px-x1)*dx + (py-y1)*dy) / (dx*dx + dy*dy)))
    return np.hypot(px - (x1+t*dx), py - (y1+t*dy))

def point_hits_wall(x, y, walls, clearance=0.8):
    for (p1, p2) in walls:
        if dist_point_to_segment(x, y, p1[0], p1[1], p2[0], p2[1]) < clearance:
            return True
    return False

def segments_cross(a1, a2, b1, b2):
    """Cek apakah segmen a1-a2 berpotongan dengan b1-b2"""
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    def on_seg(p, q, r):
        eps = 1e-9
        return (min(p[0],r[0])-eps <= q[0] <= max(p[0],r[0])+eps and
                min(p[1],r[1])-eps <= q[1] <= max(p[1],r[1])+eps)

    d1 = cross(b1, b2, a1); d2 = cross(b1, b2, a2)
    d3 = cross(a1, a2, b1); d4 = cross(a1, a2, b2)

    if ((d1>0 and d2<0) or (d1<0 and d2>0)) and \
       ((d3>0 and d4<0) or (d3<0 and d4>0)):
        return True
    if d1==0 and on_seg(b1,a1,b2): return True
    if d2==0 and on_seg(b1,a2,b2): return True
    if d3==0 and on_seg(a1,b1,a2): return True
    if d4==0 and on_seg(a1,b2,a2): return True
    return False

def edge_is_free(x1, y1, x2, y2, walls, n_sample=30):
    """
    Cek apakah edge (x1,y1)-(x2,y2) bebas dari walls.
    Kombinasi: cek perpotongan geometris + sample titik.
    """
    p1, p2 = (x1,y1), (x2,y2)
    # Cek perpotongan langsung
    for (w1, w2) in walls:
        if segments_cross(p1, p2, w1, w2):
            return False
    # Cek jarak titik-titik di sepanjang edge
    for i in range(1, n_sample):
        t = i / n_sample
        sx = x1 + t*(x2-x1)
        sy = y1 + t*(y2-y1)
        if point_hits_wall(sx, sy, walls, clearance=0.6):
            return False
    return True

def in_bounds(x, y):
    return -9.5 < x < 59.5 and -9.5 < y < 59.5

# ============================================================
# NODE
# ============================================================
class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None

# ============================================================
# RRT
# ============================================================
def sample(goal, goal_bias=0.15):
    if random.random() < goal_bias:
        return goal.x, goal.y
    return random.uniform(-9, 59), random.uniform(-9, 59)

def nearest(tree, x, y):
    return min(tree, key=lambda n: (n.x-x)**2 + (n.y-y)**2)

def steer(frm, tx, ty, step=2.0):
    dx, dy = tx - frm.x, ty - frm.y
    d = np.hypot(dx, dy)
    if d <= step:
        return tx, ty
    r = step / d
    return frm.x + r*dx, frm.y + r*dy

def backtrace(node):
    path = []
    while node:
        path.append((node.x, node.y))
        node = node.parent
    return path[::-1]

def run_rrt(start, goal, walls,
            max_iter=12000, step=2.0,
            goal_thr=2.5, goal_bias=0.15):

    tree = [start]
    print(f"  Memulai RRT... (max {max_iter} iterasi)")

    for i in range(max_iter):
        rx, ry = sample(goal, goal_bias)
        nr = nearest(tree, rx, ry)
        nx, ny = steer(nr, rx, ry, step)

        if not in_bounds(nx, ny):
            continue
        # Cek node baru (skip untuk start: i==0 sudah di tree)
        if point_hits_wall(nx, ny, walls, clearance=0.8):
            continue
        if not edge_is_free(nr.x, nr.y, nx, ny, walls):
            continue

        new = Node(nx, ny)
        new.parent = nr
        tree.append(new)

        # Cek goal
        if np.hypot(nx - goal.x, ny - goal.y) <= goal_thr:
            if edge_is_free(nx, ny, goal.x, goal.y, walls):
                goal.parent = new
                tree.append(goal)
                print(f"  ✓ Goal tercapai! Iterasi: {i+1}, Nodes: {len(tree)}")
                return backtrace(goal), tree

    print("  ✗ Path tidak ditemukan.")
    return None, tree

# ============================================================
# PLOT (style mirip Figure 1)
# ============================================================
def draw_wall_dots(ax, p1, p2, spacing=1.5):
    length = np.hypot(p2[0]-p1[0], p2[1]-p1[1])
    n = max(int(length / spacing), 2)
    xs = [p1[0] + i/n*(p2[0]-p1[0]) for i in range(n+1)]
    ys = [p1[1] + i/n*(p2[1]-p1[1]) for i in range(n+1)]
    ax.plot(xs, ys, 'k.', markersize=5, zorder=3)

def visualize(start, goal, tree, path, walls):
    fig, ax = plt.subplots(figsize=(10, 10))

    # Walls
    for (p1, p2) in walls:
        draw_wall_dots(ax, p1, p2)

    # Tree
    for node in tree:
        if node.parent:
            ax.plot([node.x, node.parent.x],
                    [node.y, node.parent.y],
                    color='lightblue', linewidth=0.5, alpha=0.5, zorder=1)

    # Path
    if path:
        px, py = zip(*path)
        ax.plot(px, py, 'r-', linewidth=2.5, zorder=5, label='RRT Path')

    # Markers
    ax.plot(start.x, start.y, 'o', color='green',
            markersize=12, zorder=6, label='Start (0, 30)')
    ax.plot(goal.x, goal.y, 'x', color='blue',
            markersize=14, markeredgewidth=3,
            zorder=6, label='Goal (50, 50)')

    ax.set_xlim(-25, 70); ax.set_ylim(-15, 70)
    ax.set_xticks(range(-20, 65, 20))
    ax.set_yticks(range(-10, 65, 10))
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title(
        'RRT Motion Planning Simulation\n'
        'Start: (0, 30)  →  Goal: (50, 50)',
        fontsize=13, fontweight='bold'
    )
    ax.legend(fontsize=11, loc='lower right')
    plt.tight_layout()
    out = 'rrt_result.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"  Gambar disimpan: {out}")

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    # Start & Goal sesuai soal
    # Start (0,30) berada di tepi dinding inner (x=0), di area FREE SPACE sisi kanan
    # Goal (50,50) berada di area FREE SPACE kanan atas
    start = Node(1, 30)   # digeser 1 unit ke kanan agar benar-benar di free space
    goal  = Node(50, 50)

    print("=" * 55)
    print("   RRT Motion Planning - RE605 Motion Planning")
    print("   Politeknik Negeri Batam")
    print(f"   Start  : (0, 30)    Goal  : (50, 50)")
    print(f"   Method : Rapidly-Exploring Random Trees (RRT)")
    print("=" * 55)

    path, tree = run_rrt(start, goal, walls,
                         max_iter=12000, step=2.0,
                         goal_thr=2.5, goal_bias=0.15)

    if path:
        total = sum(
            np.hypot(path[i+1][0]-path[i][0], path[i+1][1]-path[i][1])
            for i in range(len(path)-1)
        )
        print(f"\n  Jumlah waypoints  : {len(path)}")
        print(f"  Panjang path      : {total:.2f} unit")
        print(f"\n  Waypoints (Start → Goal):")
        for i, (x,y) in enumerate(path):
            label = " ← START" if i==0 else (" ← GOAL" if i==len(path)-1 else "")
            print(f"    [{i+1:2d}] ({x:6.2f}, {y:6.2f}){label}")

    visualize(start, goal, tree, path, walls)
    print("\nSelesai!")
