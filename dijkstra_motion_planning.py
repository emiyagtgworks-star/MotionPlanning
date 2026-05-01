"""
=============================================================
  MOTION PLANNING - DIJKSTRA'S ALGORITHM
  Studi Kasus: Lab Robotika 
  Robot: Jethexa Hexapod Robot
  Grid: 20x20
  Nama Mahasiswa: Emiya Rehulina Br Ginting
  NIM: 4222301001
=============================================================
"""

import heapq
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import time

# ============================================================
# KONFIGURASI GRID - Denah Lab Robotika Polibatam (20x20)
# ============================================================
# 0 = Free space (bisa dilalui)
# 1 = Obstacle/Dinding/Meja/Lemari
# S = Start (posisi awal Jethexa)
# G = Goal  (posisi tujuan Jethexa)

GRID_SIZE = 20

# Denah Lab Robotika Polibatam (20x20)
# Baris 0 = atas (dinding atas), Baris 19 = bawah (dinding bawah)
# Kolom 0 = kiri (dinding kiri), Kolom 19 = kanan (dinding kanan)
grid_map = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # row 0  - dinding atas
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # row 1
    [1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1],  # row 2  - meja komputer
    [1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1],  # row 3  - meja komputer
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # row 4
    [1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1],  # row 5  - lemari alat
    [1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1],  # row 6  - lemari alat
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # row 7
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # row 8
    [1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1],  # row 9  - workbench
    [1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1],  # row 10 - workbench
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # row 11
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # row 12
    [1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1],  # row 13 - kursi/bangku
    [1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1],  # row 14 - kursi/bangku
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # row 15
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1],  # row 16 - lemari dokumen
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1],  # row 17 - lemari dokumen
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # row 18
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # row 19 - dinding bawah
]

# ============================================================
# POSISI START DAN GOAL
# Koordinat: (baris, kolom) = (row, col)
# ============================================================
START = (1, 1)     # Pojok kiri atas (pintu masuk lab)
GOAL  = (18, 18)   # Pojok kanan bawah (area demo robot)


# ============================================================
# ALGORITMA DIJKSTRA
# ============================================================

def get_neighbors(pos, grid):
    """
    Mendapatkan tetangga yang valid dari posisi saat ini.
    Mendukung 8 arah: atas, bawah, kiri, kanan, dan diagonal.
    """
    row, col = pos
    rows = len(grid)
    cols = len(grid[0])

    # 8 arah gerak robot Jethexa
    directions = [
        (-1,  0, 1.0),   # atas
        ( 1,  0, 1.0),   # bawah
        ( 0, -1, 1.0),   # kiri
        ( 0,  1, 1.0),   # kanan
        (-1, -1, 1.414), # diagonal kiri-atas
        (-1,  1, 1.414), # diagonal kanan-atas
        ( 1, -1, 1.414), # diagonal kiri-bawah
        ( 1,  1, 1.414), # diagonal kanan-bawah
    ]

    neighbors = []
    for dr, dc, cost in directions:
        nr, nc = row + dr, col + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            if grid[nr][nc] == 0:  # bukan obstacle
                neighbors.append(((nr, nc), cost))
    return neighbors


def dijkstra(grid, start, goal):
    """
    Implementasi Algoritma Dijkstra untuk Motion Planning.
    
    Parameters:
        grid  : 2D list peta lingkungan (0=free, 1=obstacle)
        start : tuple (row, col) posisi awal robot
        goal  : tuple (row, col) posisi tujuan robot
    
    Returns:
        path  : list koordinat jalur optimal
        dist  : jarak total jalur optimal
        nodes_visited : jumlah node yang dikunjungi
    """
    # Priority queue: (cost, (row, col))
    pq = [(0, start)]
    
    # Jarak minimum ke setiap node
    dist_map = {start: 0}
    
    # Parent untuk rekonstruksi jalur
    parent = {start: None}
    
    # Set node yang sudah diproses
    visited = set()
    
    nodes_visited = 0

    while pq:
        current_cost, current_pos = heapq.heappop(pq)

        if current_pos in visited:
            continue
        visited.add(current_pos)
        nodes_visited += 1

        # Cek apakah sudah sampai di tujuan
        if current_pos == goal:
            break

        # Eksplorasi tetangga
        for neighbor, edge_cost in get_neighbors(current_pos, grid):
            new_cost = current_cost + edge_cost

            if neighbor not in dist_map or new_cost < dist_map[neighbor]:
                dist_map[neighbor] = new_cost
                parent[neighbor] = current_pos
                heapq.heappush(pq, (new_cost, neighbor))

    # Rekonstruksi jalur dari goal ke start
    if goal not in parent:
        return None, float('inf'), nodes_visited  # Tidak ada jalur

    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = parent[node]
    path.reverse()

    return path, dist_map.get(goal, float('inf')), nodes_visited


# ============================================================
# VISUALISASI HASIL
# ============================================================

def visualize(grid, path, start, goal, execution_time, nodes_visited, total_cost):
    """
    Membuat visualisasi grid, obstacle, dan jalur optimal Dijkstra.
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    fig.patch.set_facecolor('#1a1a2e')

    # ---- Plot 1: Peta + Jalur ----
    ax1 = axes[0]
    ax1.set_facecolor('#16213e')

    grid_array = np.array(grid, dtype=float)

    # Warna custom colormap
    cmap = plt.cm.colors.ListedColormap(['#0f3460', '#e94560'])  # free=biru gelap, obstacle=merah
    ax1.imshow(grid_array, cmap=cmap, origin='upper', aspect='equal')

    # Gambar grid lines
    for x in range(GRID_SIZE + 1):
        ax1.axhline(x - 0.5, color='#533483', linewidth=0.4, alpha=0.5)
        ax1.axvline(x - 0.5, color='#533483', linewidth=0.4, alpha=0.5)

    # Gambar jalur
    if path:
        path_rows = [p[0] for p in path]
        path_cols = [p[1] for p in path]
        ax1.plot(path_cols, path_rows, color='#00d4ff', linewidth=2.5,
                 zorder=3, label='Jalur Optimal Dijkstra')

        # Titik-titik jalur
        ax1.scatter(path_cols[1:-1], path_rows[1:-1],
                    color='#00d4ff', s=15, zorder=4, alpha=0.7)

    # Titik START
    ax1.scatter(start[1], start[0], color='#00ff88', s=250,
                marker='*', zorder=5, label=f'Start {start}')
    ax1.annotate('START\nJethexa', xy=(start[1], start[0]),
                 xytext=(start[1]+1.5, start[0]+1.5),
                 color='#00ff88', fontsize=8, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='#00ff88', lw=1.5))

    # Titik GOAL
    ax1.scatter(goal[1], goal[0], color='#ff6b35', s=250,
                marker='*', zorder=5, label=f'Goal {goal}')
    ax1.annotate('GOAL', xy=(goal[1], goal[0]),
                 xytext=(goal[1]-4, goal[0]-1.5),
                 color='#ff6b35', fontsize=8, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='#ff6b35', lw=1.5))

    # Label koordinat sumbu
    ax1.set_xticks(range(GRID_SIZE))
    ax1.set_yticks(range(GRID_SIZE))
    ax1.tick_params(colors='#a8b2d8', labelsize=7)
    ax1.set_xlabel('Kolom (X)', color='#a8b2d8', fontsize=10)
    ax1.set_ylabel('Baris (Y)', color='#a8b2d8', fontsize=10)
    ax1.set_title('Motion Planning - Dijkstra Algorithm',
                  color='white', fontsize=12, fontweight='bold', pad=10)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#0f3460', edgecolor='#533483', label='Free Space'),
        mpatches.Patch(facecolor='#e94560', edgecolor='#533483', label='Obstacle (Dinding/Meja)'),
        plt.Line2D([0], [0], color='#00d4ff', linewidth=2.5, label='Jalur Optimal Dijkstra'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='#00ff88',
                   markersize=12, label=f'Start {start}', linestyle='None'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='#ff6b35',
                   markersize=12, label=f'Goal {goal}', linestyle='None'),
    ]
    ax1.legend(handles=legend_elements, loc='upper right',
               facecolor='#16213e', edgecolor='#533483',
               labelcolor='white', fontsize=8)

    # ---- Plot 2: Statistik & Info ----
    ax2 = axes[1]
    ax2.set_facecolor('#16213e')
    ax2.axis('off')

    # === ZONA ATAS: Judul (fixed) ===
    ax2.text(0.5, 0.98, 'HASIL SIMULASI DIJKSTRA', transform=ax2.transAxes,
             ha='center', va='top', color='#00d4ff', fontsize=13, fontweight='bold')
    ax2.plot([0.03, 0.97], [0.945, 0.945], color='#533483', linewidth=1.5,
             transform=ax2.transAxes)

    # === ZONA TENGAH ATAS: Info rows (fixed y, masing-masing) ===
    info_lines = [
        ('Robot',          'Jethexa Hexapod'),
        ('Algoritma',      "Dijkstra's Algorithm"),
        ('Ukuran Grid',    f'{GRID_SIZE} x {GRID_SIZE} sel'),
        ('Posisi Start',   f'({START[0]}, {START[1]})'),
        ('Posisi Goal',    f'({GOAL[0]}, {GOAL[1]})'),
        ('Panjang Jalur',  f'{len(path)} langkah' if path else 'Tidak ditemukan'),
        ('Total Cost',     f'{total_cost:.4f} satuan' if path else '-'),
        ('Node Dikunjungi',f'{nodes_visited} node'),
        ('Waktu Eksekusi', f'{execution_time*1000:.4f} ms'),
    ]
    # 9 baris, dimulai dari y=0.915, jarak tetap 0.055
    for i, (label, value) in enumerate(info_lines):
        y = 0.915 - i * 0.055
        ax2.text(0.05, y, label, transform=ax2.transAxes,
                 ha='left', va='top', color='#a8b2d8', fontsize=9.5)
        ax2.text(0.97, y, value, transform=ax2.transAxes,
                 ha='right', va='top', color='white', fontsize=9.5, fontweight='bold')

    # Separator sebelum koordinat
    ax2.plot([0.03, 0.97], [0.415, 0.415], color='#533483', linewidth=1,
             transform=ax2.transAxes)

    # === ZONA KOORDINAT: 2 kolom kiri-kanan (fixed zone y=0.10 s/d 0.40) ===
    ax2.text(0.5, 0.395, 'KOORDINAT JALUR', transform=ax2.transAxes,
             ha='center', va='top', color='#00d4ff', fontsize=10, fontweight='bold')

    if path:
        # Tampilkan semua step dalam 2 kolom
        # Kolom kiri: step 1-11, kolom kanan: step 12-21
        col_left  = path[:11]
        col_right = path[11:]
        row_height = 0.026
        start_y = 0.365

        for i, coord in enumerate(col_left):
            y = start_y - i * row_height
            ax2.text(0.03, y, f'Step {i+1:2d}: ({coord[0]:2d},{coord[1]:2d})',
                     transform=ax2.transAxes, ha='left', va='top',
                     color='#e2e8f0', fontsize=8.5, fontfamily='monospace')

        for i, coord in enumerate(col_right):
            y = start_y - i * row_height
            step_num = 12 + i
            ax2.text(0.52, y, f'Step {step_num:2d}: ({coord[0]:2d},{coord[1]:2d})',
                     transform=ax2.transAxes, ha='left', va='top',
                     color='#e2e8f0', fontsize=8.5, fontfamily='monospace')

    # Separator sebelum status
    ax2.plot([0.03, 0.97], [0.085, 0.085], color='#533483', linewidth=1,
             transform=ax2.transAxes)

    # === ZONA BAWAH: Status (fixed y=0.06) ===
    status_color = '#00ff88' if path else '#e94560'
    status_text  = 'JALUR DITEMUKAN!' if path else 'JALUR TIDAK DITEMUKAN!'
    ax2.text(0.5, 0.065, status_text, transform=ax2.transAxes,
             ha='center', va='top', color=status_color, fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#0f3460',
                       edgecolor=status_color, linewidth=2))

    plt.tight_layout(pad=3.0)
    fig.subplots_adjust(top=0.93, bottom=0.08, left=0.06, right=0.97, wspace=0.15)
    plt.savefig('dijkstra_result.png', dpi=150, bbox_inches='tight',
                facecolor='#1a1a2e', edgecolor='none')
    print("✅ Visualisasi disimpan sebagai: dijkstra_result.png")
    plt.show()


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():
    print("=" * 60)
    print("  MOTION PLANNING - DIJKSTRA'S ALGORITHM")
    print("  Studi Kasus: Lab Robotika Polibatam")
    print("  Robot     : Jethexa Hexapod")
    print(f"  Grid      : {GRID_SIZE} x {GRID_SIZE}")
    print(f"  Start     : {START}")
    print(f"  Goal      : {GOAL}")
    print("=" * 60)

    # Validasi start dan goal
    if grid_map[START[0]][START[1]] == 1:
        print("❌ ERROR: Posisi START berada di obstacle!")
        return
    if grid_map[GOAL[0]][GOAL[1]] == 1:
        print("❌ ERROR: Posisi GOAL berada di obstacle!")
        return

    print("\n🔄 Menjalankan Dijkstra Algorithm...")
    start_time = time.time()
    path, total_cost, nodes_visited = dijkstra(grid_map, START, GOAL)
    end_time = time.time()
    execution_time = end_time - start_time

    print(f"\n📊 HASIL:")
    print(f"   ⏱️  Waktu Eksekusi : {execution_time*1000:.4f} ms")
    print(f"   🔍 Node Dikunjungi: {nodes_visited}")

    if path:
        print(f"   📏 Panjang Jalur  : {len(path)} langkah")
        print(f"   📐 Total Cost     : {total_cost:.4f} satuan")
        print(f"\n🗺️  Koordinat Jalur ({len(path)} langkah):")
        for i, coord in enumerate(path):
            print(f"   Step {i+1:3d}: ({coord[0]:2d}, {coord[1]:2d})")
    else:
        print("   ❌ Jalur tidak ditemukan!")
        total_cost = 0

    print("\n🎨 Membuat visualisasi...")
    visualize(grid_map, path, START, GOAL, execution_time, nodes_visited, total_cost)

    print("\n✅ Simulasi selesai!")
    print("=" * 60)


if __name__ == "__main__":
    main()
