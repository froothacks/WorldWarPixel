import type { PixelWrite } from "./ws";

function hexToRgb(hex: string): [number, number, number] {
  const n = parseInt(hex.replace("#", ""), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

/** Client-side mirror of the canvas: palette indices, last writer per pixel, and an RGBA image. */
export class PixelStore {
  readonly width: number;
  readonly height: number;
  readonly colors: Uint8Array;
  readonly owners: Uint16Array;
  readonly image: ImageData;
  private readonly rgb: [number, number, number][];
  version = 0;

  constructor(width: number, height: number, palette: string[]) {
    this.width = width;
    this.height = height;
    this.colors = new Uint8Array(width * height);
    this.owners = new Uint16Array(width * height);
    this.image = new ImageData(width, height);
    this.rgb = palette.map(hexToRgb);
    this.image.data.fill(255);
  }

  private paint(i: number, color: number) {
    this.colors[i] = color;
    const [r, g, b] = this.rgb[color] ?? [255, 0, 255];
    const o = i * 4;
    this.image.data[o] = r;
    this.image.data[o + 1] = g;
    this.image.data[o + 2] = b;
  }

  loadSnapshot(pixels: Uint8Array, owners: Uint16Array) {
    for (let i = 0; i < pixels.length; i++) this.paint(i, pixels[i]);
    this.owners.set(owners);
    this.version++;
  }

  apply(writes: PixelWrite[]) {
    for (const w of writes) {
      const i = w.y * this.width + w.x;
      this.paint(i, w.color);
      this.owners[i] = w.agentId;
    }
    this.version++;
  }

  ownerAt(x: number, y: number): number {
    return this.owners[y * this.width + x];
  }

  colorAt(x: number, y: number): number {
    return this.colors[y * this.width + x];
  }

  countOwned(agentId: number): number {
    let n = 0;
    for (let i = 0; i < this.owners.length; i++) if (this.owners[i] === agentId) n++;
    return n;
  }
}
