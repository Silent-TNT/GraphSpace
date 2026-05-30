import type { HouseData, HouseSummary, Manifest, Room } from "./types";
import {
  ROOM_COLORS,
  formatArea,
  formatSize,
  getRoomColor,
  getRoomLabel,
} from "./types";
import { renderFloorPlan } from "./floorPlan";
import { HouseViewer3D } from "./viewer3d";

type ViewMode = "2d" | "3d" | "split";

class App {
  private manifest: Manifest | null = null;
  private currentHouse: HouseData | null = null;
  private currentFloor = 1;
  private viewMode: ViewMode = "2d";
  private viewer3d: HouseViewer3D | null = null;
  private cleanup2d: (() => void) | null = null;
  private filteredHouses: HouseSummary[] = [];

  private houseListEl = document.getElementById("house-list")!;
  private searchInput = document.getElementById("search-input") as HTMLInputElement;
  private emptyState = document.getElementById("empty-state")!;
  private viewerPanel = document.getElementById("viewer-panel")!;
  private houseTitle = document.getElementById("house-title")!;
  private houseMeta = document.getElementById("house-meta")!;
  private floorToolbar = document.getElementById("floor-toolbar")!;
  private view2d = document.getElementById("view-2d")!;
  private view3d = document.getElementById("view-3d")!;
  private canvasArea = document.querySelector(".canvas-area")!;
  private statsGrid = document.getElementById("stats-grid")!;
  private buildingInfo = document.getElementById("building-info")!;
  private legend = document.getElementById("legend")!;
  private roomDetail = document.getElementById("room-detail")!;
  private datasetCount = document.getElementById("dataset-count")!;
  private modeBanner = document.getElementById("mode-banner");

  private assetUrl(relativePath: string): string {
    const base = import.meta.env.BASE_URL;
    return `${base}${relativePath.replace(/^\//, "")}`;
  }

  async init(): Promise<void> {
    this.bindEvents();
    this.renderLegend();

    const res = await fetch(this.assetUrl("manifest.json"));
    this.manifest = (await res.json()) as Manifest;
    this.filteredHouses = this.manifest.houses;
    this.renderModeBanner();
    this.datasetCount.textContent =
      this.manifest.mode === "demo"
        ? `公开展示：${this.manifest.count} 套 Demo 样例`
        : `共 ${this.manifest.count} 套房屋（本地）`;
    this.renderHouseList();

    const deepLinkId = new URLSearchParams(window.location.search)
      .get("house")
      ?.trim()
      .toLowerCase();
    if (deepLinkId) {
      const match = this.manifest.houses.find(
        (h) => h.id.toLowerCase() === deepLinkId
      );
      if (match) {
        await this.selectHouse(match.id);
        return;
      }
    }

    if (this.filteredHouses.length > 0) {
      await this.selectHouse(this.filteredHouses[0].id);
    }
  }

  private renderModeBanner(): void {
    if (!this.modeBanner || this.manifest?.mode !== "demo") return;
    this.modeBanner.classList.remove("hidden");
    this.modeBanner.textContent =
      `公开展示 ${this.manifest.count} 套 Demo 样例。完整训练集请在本机运行 npm run dev（website/qc-viewer）。`;
  }

  private bindEvents(): void {
    this.searchInput.addEventListener("input", () => {
      const q = this.searchInput.value.trim().toLowerCase();
      this.filteredHouses = (this.manifest?.houses ?? []).filter((h) =>
        h.id.toLowerCase().includes(q)
      );
      this.renderHouseList();
    });

    document.querySelectorAll(".tab").forEach((btn) => {
      btn.addEventListener("click", () => {
        const view = (btn as HTMLElement).dataset.view as ViewMode;
        this.setViewMode(view);
      });
    });
  }

  private renderHouseList(): void {
    this.houseListEl.innerHTML = "";
    for (const house of this.filteredHouses) {
      const btn = document.createElement("button");
      btn.className = "house-item";
      if (this.currentHouse?.house_id === house.id) btn.classList.add("active");

      const bedroom = house.stats.bedroom ?? 0;
      const bathroom = house.stats.bathroom ?? 0;

      btn.innerHTML = `
        <div class="house-item-id">${house.id}</div>
        <div class="house-item-meta">${house.totalRooms} 房间 · ${bedroom} 卧 · ${bathroom} 卫 · ${house.floors.length} 层</div>
      `;
      btn.addEventListener("click", () => this.selectHouse(house.id));
      this.houseListEl.appendChild(btn);
    }
  }

  private async selectHouse(id: string): Promise<void> {
    const summary = this.manifest?.houses.find((h) => h.id === id);
    if (!summary) return;

    const res = await fetch(this.assetUrl(`dataset/${summary.file}`));
    this.currentHouse = (await res.json()) as HouseData;
    this.currentFloor = summary.floors[0] ?? 1;

    this.emptyState.classList.add("hidden");
    this.viewerPanel.classList.remove("hidden");

    this.houseTitle.textContent = this.currentHouse.house_id;
    this.houseMeta.textContent = `${summary.totalRooms} 个房间 · ${summary.floors.length} 个楼层`;

    this.renderHouseList();
    this.renderStats(summary);
    this.renderBuildingInfo(summary);
    this.renderFloorToolbar(summary.floors);
    this.renderViews();
    this.clearRoomDetail();
  }

  private renderStats(summary: HouseSummary): void {
    this.statsGrid.innerHTML = "";
    const entries = Object.entries(summary.stats).sort((a, b) => b[1] - a[1]);
    for (const [type, count] of entries) {
      const el = document.createElement("div");
      el.className = "stat-item";
      el.innerHTML = `
        <div class="stat-label">${getRoomLabel(type)}</div>
        <div class="stat-value">${count}</div>
      `;
      this.statsGrid.appendChild(el);
    }
  }

  private renderBuildingInfo(summary: HouseSummary): void {
    const { x, y, z } = summary.buildingSize;
    this.buildingInfo.innerHTML = `
      <div class="info-row"><span>宽度 (X)</span><span>${formatSize(x)}</span></div>
      <div class="info-row"><span>深度 (Y)</span><span>${formatSize(y)}</span></div>
      <div class="info-row"><span>高度 (Z)</span><span>${formatSize(z)}</span></div>
      <div class="info-row"><span>占地面积</span><span>${formatArea(x * y)}</span></div>
    `;
  }

  private renderLegend(): void {
    this.legend.innerHTML = "";
    for (const [type, color] of Object.entries(ROOM_COLORS)) {
      if (type === "default") continue;
      const el = document.createElement("div");
      el.className = "legend-item";
      el.innerHTML = `
        <span class="legend-color" style="background:${color}"></span>
        <span>${getRoomLabel(type)}</span>
      `;
      this.legend.appendChild(el);
    }
  }

  private renderFloorToolbar(floors: number[]): void {
    this.floorToolbar.innerHTML = "";
    for (const floor of floors) {
      const btn = document.createElement("button");
      btn.className = "floor-btn";
      if (floor === this.currentFloor) btn.classList.add("active");
      btn.textContent = `${floor} 层`;
      btn.addEventListener("click", () => {
        this.currentFloor = floor;
        this.renderFloorToolbar(floors);
        this.render2d();
      });
      this.floorToolbar.appendChild(btn);
    }
  }

  private setViewMode(mode: ViewMode): void {
    this.viewMode = mode;
    document.querySelectorAll(".tab").forEach((btn) => {
      btn.classList.toggle("active", (btn as HTMLElement).dataset.view === mode);
    });

    this.canvasArea.classList.toggle("split-mode", mode === "split");

    if (mode === "2d") {
      this.view2d.classList.remove("hidden");
      this.view3d.classList.add("hidden");
    } else if (mode === "3d") {
      this.view2d.classList.add("hidden");
      this.view3d.classList.remove("hidden");
    } else {
      this.view2d.classList.remove("hidden");
      this.view3d.classList.remove("hidden");
    }

    this.renderViews();
  }

  private renderViews(): void {
    if (this.viewMode === "2d" || this.viewMode === "split") {
      this.render2d();
    }
    if (this.viewMode === "3d" || this.viewMode === "split") {
      this.render3d();
    }
  }

  private render2d(): void {
    if (!this.currentHouse) return;
    this.cleanup2d?.();
    this.cleanup2d = renderFloorPlan(
      this.view2d,
      this.currentHouse.rooms,
      this.currentFloor,
      (room) => this.showRoomDetail(room)
    );
  }

  private render3d(): void {
    if (!this.currentHouse) return;

    if (!this.viewer3d) {
      this.viewer3d = new HouseViewer3D(this.view3d);
    }

    requestAnimationFrame(() => {
      this.viewer3d?.loadHouse(this.currentHouse!);
    });
  }

  private showRoomDetail(room: Room | null): void {
    if (!room) {
      this.clearRoomDetail();
      this.viewer3d?.highlightRoom(null);
      return;
    }

    const w = room.box_max[0] - room.box_min[0];
    const d = room.box_max[1] - room.box_min[1];
    const h = room.box_max[2] - room.box_min[2];
    const area = w * d;

    this.roomDetail.innerHTML = `
      <div class="detail-row"><span>ID</span><span>${room.id}</span></div>
      <div class="detail-row"><span>类型</span><span>${getRoomLabel(room.type)}</span></div>
      <div class="detail-row"><span>楼层</span><span>${room.floor}</span></div>
      <div class="detail-row"><span>面积</span><span>${formatArea(area)}</span></div>
      <div class="detail-row"><span>尺寸</span><span>${formatSize(w)} × ${formatSize(d)}</span></div>
      <div class="detail-row"><span>高度</span><span>${formatSize(h)}</span></div>
      <div class="detail-row">
        <span>颜色</span>
        <span style="display:flex;align-items:center;gap:6px">
          <span style="width:12px;height:12px;border-radius:2px;background:${getRoomColor(room.type)};display:inline-block"></span>
          ${room.type}
        </span>
      </div>
    `;

    if (this.viewMode !== "2d") {
      this.viewer3d?.highlightRoom(room.id);
    }
  }

  private clearRoomDetail(): void {
    this.roomDetail.innerHTML = `<p class="muted">点击平面图上的房间查看详情</p>`;
  }
}

const app = new App();
app.init();
