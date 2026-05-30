export interface Room {
  id: string;
  floor: number;
  type: string;
  box_min: [number, number, number];
  box_max: [number, number, number];
}

export interface HouseMetadata {
  total_rooms?: number;
  stats?: Record<string, number>;
  building_size?: { x: number; y: number; z: number };
  constraints?: {
    pure_box_enforced?: boolean;
    origin_aligned_auto?: boolean;
    modulus?: number;
  };
}

export interface HouseData {
  house_id: string;
  rooms: Room[];
  metadata?: HouseMetadata;
}

export interface HouseSummary {
  id: string;
  file: string;
  totalRooms: number;
  stats: Record<string, number>;
  buildingSize: { x: number; y: number; z: number };
  floors: number[];
}

export interface Manifest {
  count: number;
  houses: HouseSummary[];
  mode?: "demo" | "local";
  notice?: string;
}

export const ROOM_COLORS: Record<string, string> = {
  living_room: "#4ade80",
  bedroom: "#60a5fa",
  bathroom: "#22d3ee",
  kitchen: "#fb923c",
  dining_room: "#fbbf24",
  corridor: "#94a3b8",
  entryway: "#a78bfa",
  balcony: "#86efac",
  stairs: "#64748b",
  default: "#cbd5e1",
};

export const ROOM_LABELS: Record<string, string> = {
  living_room: "客厅",
  bedroom: "卧室",
  bathroom: "卫生间",
  kitchen: "厨房",
  dining_room: "餐厅",
  corridor: "走廊",
  entryway: "玄关",
  balcony: "阳台",
  stairs: "楼梯",
};

export function getRoomColor(type: string): string {
  return ROOM_COLORS[type] ?? ROOM_COLORS.default;
}

export function getRoomLabel(type: string): string {
  return ROOM_LABELS[type] ?? type;
}

export function formatSize(mm: number): string {
  if (mm >= 1000) return `${(mm / 1000).toFixed(1)} m`;
  return `${mm} mm`;
}

export function formatArea(mm2: number): string {
  const m2 = mm2 / 1_000_000;
  return `${m2.toFixed(1)} m²`;
}
