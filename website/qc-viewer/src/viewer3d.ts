import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { HouseData, Room } from "./types";
import { getRoomColor } from "./types";

export class HouseViewer3D {
  private container: HTMLElement;
  private renderer: THREE.WebGLRenderer;
  private scene: THREE.Scene;
  private camera: THREE.PerspectiveCamera;
  private controls: OrbitControls;
  private animationId = 0;
  private resizeObserver: ResizeObserver;
  private roomMeshes: THREE.Mesh[] = [];

  constructor(container: HTMLElement) {
    this.container = container;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0f172a);

    const width = container.clientWidth || 800;
    const height = container.clientHeight || 600;

    this.camera = new THREE.PerspectiveCamera(50, width / height, 100, 200000);
    this.camera.position.set(12000, 18000, 12000);

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(width, height);
    this.renderer.shadowMap.enabled = true;
    container.appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.maxPolarAngle = Math.PI / 2.05;

    const ambient = new THREE.AmbientLight(0xffffff, 0.55);
    this.scene.add(ambient);

    const dirLight = new THREE.DirectionalLight(0xffffff, 0.85);
    dirLight.position.set(8000, 15000, 6000);
    dirLight.castShadow = true;
    this.scene.add(dirLight);

    const grid = new THREE.GridHelper(30000, 30, 0x334155, 0x1e293b);
    grid.rotation.x = Math.PI / 2;
    this.scene.add(grid);

    this.resizeObserver = new ResizeObserver(() => this.handleResize());
    this.resizeObserver.observe(container);

    this.animate();
  }

  loadHouse(house: HouseData, highlightRoomId?: string | null): void {
    for (const mesh of this.roomMeshes) {
      this.scene.remove(mesh);
      mesh.geometry.dispose();
      (mesh.material as THREE.Material).dispose();
    }
    this.roomMeshes = [];

    let maxX = 0;
    let maxY = 0;
    let maxZ = 0;

    for (const room of house.rooms) {
      maxX = Math.max(maxX, room.box_max[0]);
      maxY = Math.max(maxY, room.box_max[1]);
      maxZ = Math.max(maxZ, room.box_max[2]);
    }

    const centerX = maxX / 2;
    const centerY = maxY / 2;

    for (const room of house.rooms) {
      const mesh = this.createRoomMesh(room, highlightRoomId === room.id);
      this.scene.add(mesh);
      this.roomMeshes.push(mesh);
    }

    const span = Math.max(maxX, maxY, maxZ);
    const dist = span * 1.4;
    this.camera.position.set(centerX + dist * 0.7, dist * 0.9, centerY + dist * 0.7);
    this.controls.target.set(centerX, maxZ / 2, centerY);
    this.controls.update();
  }

  highlightRoom(roomId: string | null): void {
    for (const mesh of this.roomMeshes) {
      const isHighlight = mesh.userData.roomId === roomId;
      const mat = mesh.material as THREE.MeshStandardMaterial;
      mat.emissive.setHex(isHighlight ? 0x334155 : 0x000000);
      mat.opacity = roomId && !isHighlight ? 0.45 : 0.88;
    }
  }

  private createRoomMesh(room: Room, highlighted: boolean): THREE.Mesh {
    const w = room.box_max[0] - room.box_min[0];
    const d = room.box_max[1] - room.box_min[1];
    const h = room.box_max[2] - room.box_min[2];

    const geometry = new THREE.BoxGeometry(w, h, d);
    const color = new THREE.Color(getRoomColor(room.type));

    const material = new THREE.MeshStandardMaterial({
      color,
      transparent: true,
      opacity: highlighted ? 1 : 0.88,
      emissive: highlighted ? new THREE.Color(0x334155) : new THREE.Color(0x000000),
    });

    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.set(
      room.box_min[0] + w / 2,
      room.box_min[2] + h / 2,
      room.box_min[1] + d / 2
    );
    mesh.userData.roomId = room.id;
    mesh.userData.roomType = room.type;
    return mesh;
  }

  private handleResize(): void {
    const width = this.container.clientWidth;
    const height = this.container.clientHeight;
    if (width === 0 || height === 0) return;
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
  }

  private animate = (): void => {
    this.animationId = requestAnimationFrame(this.animate);
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  };

  dispose(): void {
    cancelAnimationFrame(this.animationId);
    this.resizeObserver.disconnect();
    for (const mesh of this.roomMeshes) {
      mesh.geometry.dispose();
      (mesh.material as THREE.Material).dispose();
    }
    this.controls.dispose();
    this.renderer.dispose();
    this.container.innerHTML = "";
  }
}
