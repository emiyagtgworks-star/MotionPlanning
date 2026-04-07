import math
import heapq
from sys import maxsize
import matplotlib.pyplot as plt

# =============================================
# KONFIGURASI PETA — sesuai Gambar 5-3
# =============================================
def build_obstacle_map():
    ox, oy = [], []

    # === Border kotak luar ===
    for i in range(-10, 61):
        ox.append(i);  oy.append(-10)   # bawah
    for i in range(-10, 61):
        ox.append(60); oy.append(i)     # kanan
    for i in range(-10, 61):
        ox.append(i);  oy.append(60)    # atas
    for i in range(-10, 61):
        ox.append(-10); oy.append(i)    # kiri

    # === Obstacle dalam peta ===

    # Vertikal kiri: x=20, dari y=-10 sampai y=40
    for i in range(-10, 41):
        ox.append(20); oy.append(i)

    # Vertikal kanan: x=40, dari y=0 sampai y=60
    for i in range(0, 61):
        ox.append(40); oy.append(i)

    # Horizontal bawah: y=10, dari x=-10 sampai x=10
    for i in range(-10, 11):
        ox.append(i); oy.append(10)

    # Horizontal atas: y=30, dari x=0 sampai x=20
    for i in range(0, 21):
        ox.append(i); oy.append(30)

    return ox, oy

START = (-5, -5)
GOAL  = (50, 50)

# =============================================
# UTILITY
# =============================================
def heuristic(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def get_neighbors(node, obstacle_set, x_range, y_range):
    neighbors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            nx, ny = node[0]+dx, node[1]+dy
            if nx < x_range[0] or nx > x_range[1]:
                continue
            if ny < y_range[0] or ny > y_range[1]:
                continue
            if (nx, ny) in obstacle_set:
                continue
            cost = math.sqrt(dx**2 + dy**2)
            neighbors.append(((nx, ny), cost))
    return neighbors

# =============================================
# DIJKSTRA
# =============================================
def dijkstra(start, goal, obstacle_set, x_range, y_range):
    dist   = {start: 0}
    parent = {start: None}
    pq      = [(0, start)]
    visited = set()

    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        if u == goal:
            break
        for (v, w) in get_neighbors(u, obstacle_set, x_range, y_range):
            nd = d + w
            if v not in dist or nd < dist[v]:
                dist[v]   = nd
                parent[v] = u
                heapq.heappush(pq, (nd, v))

    path, node = [], goal
    while node is not None:
        path.append(node)
        node = parent.get(node)
    path.reverse()
    return path

# =============================================
# A*
# =============================================
def astar(start, goal, obstacle_set, x_range, y_range):
    g      = {start: 0}
    parent = {start: None}
    pq      = [(heuristic(start, goal), start)]
    visited = set()

    while pq:
        _, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        if u == goal:
            break
        for (v, w) in get_neighbors(u, obstacle_set, x_range, y_range):
            ng = g[u] + w
            if v not in g or ng < g[v]:
                g[v]      = ng
                parent[v] = u
                heapq.heappush(pq, (ng + heuristic(v, goal), v))

    path, node = [], goal
    while node is not None:
        path.append(node)
        node = parent.get(node)
    path.reverse()
    return path

# =============================================
# D*
# =============================================
class State:
    def __init__(self, x, y):
        self.x = x; self.y = y
        self.parent = None
        self.state  = "."
        self.t      = "new"
        self.h      = 0
        self.k      = 0

    def cost(self, other):
        if self.state == "#" or other.state == "#":
            return maxsize
        return math.sqrt((self.x-other.x)**2 + (self.y-other.y)**2)

    def set_state(self, s):
        if s in ["s", ".", "#", "e", "*"]:
            self.state = s

class MapDstar:
    def __init__(self, row, col):
        self.row = row; self.col = col
        self.map = [[State(i, j) for j in range(col)] for i in range(row)]

    def get_neighbors(self, s):
        result = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni, nj = s.x+di, s.y+dj
                if 0 <= ni < self.row and 0 <= nj < self.col:
                    result.append(self.map[ni][nj])
        return result

    def set_obstacle(self, pts):
        for x, y in pts:
            if 0 <= x < self.row and 0 <= y < self.col:
                self.map[x][y].set_state("#")

class Dstar:
    def __init__(self, maps):
        self.map       = maps
        self.open_list = set()

    def insert(self, s, h_new):
        if   s.t == "new":   s.k = h_new
        elif s.t == "open":  s.k = min(s.k, h_new)
        elif s.t == "close": s.k = min(s.h, h_new)
        s.h = h_new; s.t = "open"
        self.open_list.add(s)

    def remove(self, s):
        if s.t == "open":
            s.t = "close"
            self.open_list.remove(s)

    def min_state(self):
        if not self.open_list: return None
        return min(self.open_list, key=lambda x: x.k)

    def get_kmin(self):
        if not self.open_list: return -1
        return min(x.k for x in self.open_list)

    def process_state(self):
        x = self.min_state()
        if x is None: return -1
        k_old = self.get_kmin()
        self.remove(x)
        if k_old < x.h:
            for y in self.map.get_neighbors(x):
                if y.h <= k_old and x.h > y.h + x.cost(y):
                    x.parent = y; x.h = y.h + x.cost(y)
        if k_old == x.h:
            for y in self.map.get_neighbors(x):
                if (y.t == "new" or
                   (y.parent == x and y.h != x.h + x.cost(y)) or
                   (y.parent != x and y.h  > x.h + x.cost(y))):
                    y.parent = x
                    self.insert(y, x.h + x.cost(y))
        else:
            for y in self.map.get_neighbors(x):
                if y.t == "new" or (y.parent == x and y.h != x.h + x.cost(y)):
                    y.parent = x; self.insert(y, x.h + x.cost(y))
                else:
                    if y.parent != x and y.h > x.h + x.cost(y):
                        self.insert(x, x.h)
                    elif (y.parent != x and x.h > y.h + x.cost(y)
                          and y.t == "close" and y.h > k_old):
                        self.insert(y, y.h)
        return self.get_kmin()

    def modify_cost(self, x):
        if x.t == "close":
            self.insert(x, x.parent.h + x.cost(x.parent))

    def modify(self, s):
        self.modify_cost(s)
        while True:
            k_min = self.process_state()
            if k_min >= s.h: break

    def run(self, start, end):
        rx, ry = [], []
        end.h = 0; end.t = "open"; end.k = 0
        self.open_list.add(end)
        while True:
            self.process_state()
            if start.t == "close":
                break
        tmp = start
        while tmp != end:
            tmp.set_state("*")
            rx.append(tmp.x); ry.append(tmp.y)
            if tmp.parent is None:
                break
            if tmp.parent.state == "#":
                self.modify(tmp); continue
            tmp = tmp.parent
        return rx, ry

# =============================================
# MAIN
# =============================================
if __name__ == "__main__":
    ox, oy       = build_obstacle_map()
    obstacle_set = set(zip(ox, oy))
    x_range      = (-10, 60)
    y_range      = (-10, 60)

    # --- Dijkstra ---
    print("Menjalankan Dijkstra...")
    path_d = dijkstra(START, GOAL, obstacle_set, x_range, y_range)
    print(f"  Panjang jalur: {len(path_d)} node")

    # --- A* ---
    print("Menjalankan A*...")
    path_a = astar(START, GOAL, obstacle_set, x_range, y_range)
    print(f"  Panjang jalur: {len(path_a)} node")

    # --- D* ---
    print("Menjalankan D*...")
    OFFSET = 10
    SIZE   = 71       # mencakup -10..60 → index 0..70
    m      = MapDstar(SIZE, SIZE)
    obs_shifted = [(x+OFFSET, y+OFFSET) for x, y in obstacle_set]
    m.set_obstacle(obs_shifted)

    sx, sy = START[0]+OFFSET, START[1]+OFFSET   # (-5+10, -5+10) = (5, 5)
    gx, gy = GOAL[0]+OFFSET,  GOAL[1]+OFFSET    # (50+10, 50+10) = (60, 60)
    s_node = m.map[sx][sy]
    e_node = m.map[gx][gy]
    dstar  = Dstar(m)
    rx, ry = dstar.run(s_node, e_node)
    rx = [v-OFFSET for v in rx]
    ry = [v-OFFSET for v in ry]
    print(f"  Panjang jalur: {len(rx)} node")

    # ---- PLOT ----
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    titles = ["Dijkstra's", "A*", "D*"]
    paths  = [path_d, path_a, list(zip(rx, ry))]
    colors = ["blue", "green", "red"]

    for ax, title, path, color in zip(axes, titles, paths, colors):
        # Obstacle & border
        ax.plot(ox, oy, ".k", markersize=2)
        # Start & Goal
        ax.plot(START[0], START[1], "og", markersize=10, label="Start")
        ax.plot(GOAL[0],  GOAL[1],  "xb", markersize=10, markeredgewidth=2, label="Goal")
        # Path
        if path:
            if title == "D*":
                if path:
                    px, py = zip(*path)
                else:
                    px, py = [], []
            else:
                px = [p[0] for p in path]
                py = [p[1] for p in path]
            ax.plot(px, py, "-", color=color, linewidth=2, label="Path")

        ax.set_title(title, fontsize=15, fontweight='bold')
        ax.set_xlim(-20, 65); ax.set_ylim(-15, 65)
        ax.set_aspect('equal')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("X"); ax.set_ylabel("Y")

    plt.suptitle("Perbandingan Simulasi: Dijkstra vs A* vs D*\nMotion Planning",
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig("simulasi_ketiga_algoritma.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("\n✅ Selesai! File tersimpan: simulasi_ketiga_algoritma.png")
