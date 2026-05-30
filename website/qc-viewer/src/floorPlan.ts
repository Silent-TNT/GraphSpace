import type { HouseData, Room } from "./types";

export function renderFloorPlan(
  container: HTMLElement,
  rooms: Room[],
  floor: number,
  onRoomSelect?: (room: Room | null) => void
): () => void {
  container.innerHTML = "";

  const floorRooms = rooms.filter((r) => r.floor === floor);
  if (floorRooms.length === 0) {
    container.innerHTML = `<p class="empty-hint">该楼层暂无房间数据</p>`;
    return () => {};
  }

  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;

  for (const room of floorRooms) {
    minX = Math.min(minX, room.box_min[0]);
    minY = Math.min(minY, room.box_min[1]);
    maxX = Math.max(maxX, room.box_max[0]);
    maxY = Math.max(maxY, room.box_max[1]);
  }

  const padding = 800;
  const viewW = maxX - minX + padding * 2;
  const viewH = maxY - minY + padding * 2;

  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", `${minX - padding} ${minY - padding} ${viewW} ${viewH}`);
  svg.setAttribute("class", "floor-plan-svg");
  svg.setAttribute("preserveAspectRatio", "xMidYMid meet");

  const bg = document.createElementNS("http://www.w3.org/2000/svg", "rect");
  bg.setAttribute("x", String(minX - padding));
  bg.setAttribute("y", String(minY - padding));
  bg.setAttribute("width", String(viewW));
  bg.setAttribute("height", String(viewH));
  bg.setAttribute("class", "floor-bg");
  svg.appendChild(bg);

  const sorted = [...floorRooms].sort((a, b) => {
    const areaA =
      (a.box_max[0] - a.box_min[0]) * (a.box_max[1] - a.box_min[1]);
    const areaB =
      (b.box_max[0] - b.box_min[0]) * (b.box_max[1] - b.box_min[1]);
    return areaB - areaA;
  });

  let selectedRect: SVGRectElement | null = null;

  for (const room of sorted) {
    const x = room.box_min[0];
    const y = room.box_min[1];
    const w = room.box_max[0] - room.box_min[0];
    const h = room.box_max[1] - room.box_min[1];

    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    rect.setAttribute("x", String(x));
    rect.setAttribute("y", String(y));
    rect.setAttribute("width", String(w));
    rect.setAttribute("height", String(h));
    rect.setAttribute("class", "room-rect");
    rect.setAttribute("data-type", room.type);
    rect.setAttribute("data-id", room.id);

    rect.addEventListener("click", (e) => {
      e.stopPropagation();
      if (selectedRect) selectedRect.classList.remove("selected");
      rect.classList.add("selected");
      selectedRect = rect;
      onRoomSelect?.(room);
    });

    svg.appendChild(rect);

    if (w > 1200 && h > 1200) {
      const cx = x + w / 2;
      const cy = y + h / 2;
      const fontSize = Math.min(w, h) * 0.08;

      const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
      text.setAttribute("x", String(cx));
      text.setAttribute("y", String(cy));
      text.setAttribute("class", "room-label");
      text.setAttribute("font-size", String(Math.max(200, Math.min(fontSize, 500))));
      text.textContent = room.type.replace(/_/g, " ");
      svg.appendChild(text);
    }
  }

  svg.addEventListener("click", () => {
    if (selectedRect) selectedRect.classList.remove("selected");
    selectedRect = null;
    onRoomSelect?.(null);
  });

  container.appendChild(svg);
  return () => {
    container.innerHTML = "";
  };
}

export function getHouseFloors(house: HouseData): number[] {
  return [...new Set(house.rooms.map((r) => r.floor))].sort((a, b) => a - b);
}
