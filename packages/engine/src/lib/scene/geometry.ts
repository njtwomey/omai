import { group } from './build.ts'
import type { GroupNode, Scene, SceneNode } from './types.ts'

/** An axis-aligned bounding box. */
export interface Bounds {
  readonly x: number
  readonly y: number
  readonly width: number
  readonly height: number
}

function nodePoints(node: SceneNode): (readonly [number, number])[] {
  switch (node.kind) {
    case 'rect':
    case 'image':
      return [
        [node.x, node.y],
        [node.x + node.width, node.y + node.height],
      ]
    case 'circle':
      return [
        [node.cx - node.r, node.cy - node.r],
        [node.cx + node.r, node.cy + node.r],
      ]
    case 'ellipse':
      return [
        [node.cx - node.rx, node.cy - node.ry],
        [node.cx + node.rx, node.cy + node.ry],
      ]
    case 'line':
      return [
        [node.x1, node.y1],
        [node.x2, node.y2],
      ]
    case 'path':
      return [...node.points]
    case 'text':
    case 'tex':
      return [[node.x, node.y]]
    case 'group': {
      const s = node.scale ?? 1
      const [tx, ty] = node.translate ?? [0, 0]
      const theta = ((node.rotate ?? 0) * Math.PI) / 180
      const c = Math.cos(theta)
      const sn = Math.sin(theta)
      return node.children
        .flatMap(nodePoints)
        .map(([x, y]) => [tx + s * (c * x - sn * y), ty + s * (sn * x + c * y)] as const)
    }
  }
}

/**
 * The bounding box of nodes (text is counted by its anchor only).
 *
 * @param nodes - The nodes.
 * @returns The box, or a zero box for no nodes.
 * @example
 * ```ts
 * import { bounds, rect, circle } from 'whitebox-engine/scene'
 * bounds([rect(0, 0, 2, 1), circle(5, 5, 1)]) // => { x: 0, y: 0, width: 6, height: 6 }
 * ```
 */
export function bounds(nodes: readonly SceneNode[]): Bounds {
  const points = nodes.flatMap(nodePoints)
  if (points.length === 0) return { x: 0, y: 0, width: 0, height: 0 }
  let x0 = Infinity
  let y0 = Infinity
  let x1 = -Infinity
  let y1 = -Infinity
  for (const [x, y] of points) {
    x0 = Math.min(x0, x)
    y0 = Math.min(y0, y)
    x1 = Math.max(x1, x)
    y1 = Math.max(y1, y)
  }
  return { x: x0, y: y0, width: x1 - x0, height: y1 - y0 }
}

/**
 * Scale and translate nodes to fit a box, keeping their aspect ratio and centring them.
 *
 * @param nodes - The nodes.
 * @param box - The target box.
 * @param padding - Margin inside the box.
 * @returns A group with the fitted nodes.
 * @example
 * ```ts
 * import { fit, rect } from 'whitebox-engine/scene'
 * fit([rect(0, 0, 1, 1)], { x: 0, y: 0, width: 10, height: 10 }).scale // => 10
 * ```
 */
export function fit(nodes: readonly SceneNode[], box: Bounds, padding = 0): GroupNode {
  const b = bounds(nodes)
  const w = box.width - 2 * padding
  const h = box.height - 2 * padding
  const s =
    b.width === 0 && b.height === 0
      ? 1
      : Math.min(b.width === 0 ? Infinity : w / b.width, b.height === 0 ? Infinity : h / b.height)
  const tx = box.x + padding + (w - s * b.width) / 2 - s * b.x
  const ty = box.y + padding + (h - s * b.height) / 2 - s * b.y
  return group(nodes, { translate: [tx, ty], scale: s })
}

/**
 * Tile scenes into a grid (a filmstrip), each scaled to a cell.
 *
 * @param scenes - The scenes.
 * @param columns - Cells per row.
 * @param cell - Cell width and height.
 * @returns One scene containing the grid.
 * @example
 * ```ts
 * import { grid, scene } from 'whitebox-engine/scene'
 * grid([scene(10, 10, []), scene(10, 10, [])], 2, [20, 20]).width // => 40
 * ```
 */
export function grid(scenes: readonly Scene[], columns: number, cell: readonly [number, number]): Scene {
  const [cw, ch] = cell
  const rows = Math.ceil(scenes.length / columns)
  const children = scenes.map((s, i) => {
    const scale = Math.min(cw / s.width, ch / s.height)
    return group(s.children, { translate: [(i % columns) * cw, Math.floor(i / columns) * ch], scale })
  })
  return { kind: 'scene', width: columns * cw, height: rows * ch, children }
}
