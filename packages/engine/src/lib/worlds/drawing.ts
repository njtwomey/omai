/**
 * Pictures of worlds as engine data: the cell graph as a `Structure`, and a top-down `Scene` with cells filled by
 * terrain, walls, starts, goals and coloured items, optional per-cell values and paths, for omai-render to draw.
 */

import {
  circle,
  group,
  line,
  type Paint,
  path,
  polygon,
  rect,
  type Scene,
  type SceneNode,
  type Structure,
  type StructureEdge,
  type StructureNode,
  scene,
  structure,
  text,
} from '../scene/index.ts'
import { step } from './_topology.ts'
import {
  cellCentre,
  cellCoords,
  cellCorners,
  hasWall,
  type MoveOptions,
  neighbours,
  passable,
  terrainOf,
  type World,
  worldSize,
} from './world.ts'

/**
 * Describe a world's cell graph as a {@link Structure}: one node per passable cell (with its row, column, centre and
 * terrain as data, start and goal roles marked) and one edge per legal move (undirected where moves go both ways).
 *
 * @param w - The world.
 * @param options - Connectivity and corner rule, and a title.
 * @returns The structure (type `world`).
 * @concept structures
 * @example
 * ```ts
 * import { fromAscii, toStructure } from 'omai-engine/worlds'
 * const s = toStructure(fromAscii(['S.#', '..G']))
 * [s.nodes.length, s.edges.length] // => [5, 5]
 * ```
 */
export function toStructure(w: World, options: MoveOptions & { readonly title?: string } = {}): Structure {
  const nodes: StructureNode[] = []
  const edges: StructureEdge[] = []
  const starts = new Set(w.starts)
  const goals = new Set(w.goals)
  for (let cell = 0; cell < w.n; cell++) {
    if (!passable(w, cell)) continue
    const [x, y] = cellCentre(w, cell)
    const grid = w.topology.kind === 'graph' ? {} : { row: cellCoords(w, cell)[0], col: cellCoords(w, cell)[1] }
    nodes.push({
      id: `c${cell}`,
      label: w.topology.kind === 'graph' ? String(cell) : `(${cellCoords(w, cell).join(', ')})`,
      role: starts.has(cell) ? 'input' : goals.has(cell) ? 'output' : 'state',
      data: { cell, x, y, terrain: terrainOf(w, cell).name, ...grid },
    })
    for (const to of neighbours(w, cell, options)) {
      const back = neighbours(w, to, options).includes(cell)
      if (back && to < cell) continue
      edges.push({ from: `c${cell}`, to: `c${to}`, directed: !back })
    }
  }
  return structure('world', nodes, edges, {
    ...(options.title === undefined ? (w.name === undefined ? {} : { title: w.name }) : { title: options.title }),
  })
}

const COLOURS: Readonly<Record<string, Paint>> = {
  red: 'tab3',
  green: 'tab2',
  blue: 'tab0',
  purple: 'tab4',
  yellow: 'tab8',
  grey: 'tab7',
  orange: 'tab1',
  pink: 'tab6',
}

/**
 * The theme paint of an item colour name: MiniGrid's red, green, blue, purple, yellow and grey (plus orange and pink)
 * map to tab10 tokens; anything else is the accent.
 *
 * @param colour - The colour name.
 * @returns The paint.
 * @example
 * ```ts
 * import { colourPaint } from 'omai-engine/worlds'
 * colourPaint('red') // => 'tab3'
 * ```
 */
export function colourPaint(colour: string): Paint {
  return COLOURS[colour] ?? 'accent'
}

/** Options of {@link toScene}. */
export interface WorldSceneOptions {
  /** Pixels per world unit (default 32). */
  readonly cell?: number
  readonly title?: string
  /** The agent: a cell, or a continuous pose (drawn with its heading). */
  readonly agent?: number | { readonly x: number; readonly y: number; readonly heading?: number }
  /** A path through cells, drawn as a polyline through their centres. */
  readonly path?: readonly number[]
  /** A value per cell, shaded over the terrain (largest values most opaque). */
  readonly values?: ArrayLike<number>
  /** How `values` are drawn (the default is the accent colour, more opaque the larger the magnitude). */
  readonly valueStyle?: ValueStyle
}

/** How {@link toScene} draws a world's per-cell `values`. */
export interface ValueStyle {
  /**
   * The paint of a cell's disc: one paint, or a function of the value and its place in the range (0 to 1) to draw a
   * colour ramp. With a function the disc has the same opacity everywhere (`maxOpacity`), so the colour carries the
   * value; with one paint the opacity carries it.
   */
  readonly fill?: Paint | ((value: number, normalised: number) => Paint)
  /**
   * The values that map to 0 and 1 (default 0 and the largest magnitude, so only magnitude counts). Fix it to draw
   * every frame of an animation on one scale, or to let negative values map below the middle.
   */
  readonly range?: readonly [number, number]
  /** The opacity of the strongest disc (default 0.75). */
  readonly maxOpacity?: number
}

/**
 * Draw a world from above with theme tokens: cells filled by terrain, walls as thick lines, goals and starts marked,
 * items as keys, doors and pickups in their colours, optional values shaded per cell, a path and the agent. Square
 * cells are squares, hexagons hexagons, graph nodes circles joined by their edges.
 *
 * @param w - The world.
 * @param options - Scale, title, agent, path, per-cell values and how they are drawn.
 * @returns The scene.
 * @concept scene-graph
 * @example
 * ```ts
 * import { fromAscii, toScene } from 'omai-engine/worlds'
 * const s = toScene(fromAscii(['S.', '.G']), { agent: 0 })
 * [s.width, s.height] // => [64, 64]
 * s.children.some((n) => n.id === 'agent') // => true
 * ```
 */
export function toScene(w: World, options: WorldSceneOptions = {}): Scene {
  const k = options.cell ?? 32
  const t = w.topology
  const nodes: SceneNode[] = []
  const at = (p: readonly [number, number]): [number, number] => [p[0] * k, p[1] * k]
  // The largest magnitude among the finite values: one NaN must not switch off the shading of the rest.
  let hi = 0
  for (let i = 0; i < (options.values?.length ?? 0); i++) {
    const v = options.values?.[i] ?? 0
    if (Number.isFinite(v)) hi = Math.max(hi, Math.abs(v))
  }
  const style = options.valueStyle ?? {}
  const [lo, top] = style.range ?? [0, hi]
  const maxOpacity = style.maxOpacity ?? 0.75
  for (let cell = 0; cell < w.n; cell++) {
    const tr = terrainOf(w, cell)
    const look = { fill: tr.paint, stroke: 'grid' as const, strokeWidth: 1, title: `cell ${cell}: ${tr.name}` }
    if (t.kind === 'graph') {
      const [x, y] = at(cellCentre(w, cell))
      nodes.push(circle(x, y, k / 3, look))
    } else nodes.push(polygon(cellCorners(w, cell).map(at), look))
    const v = options.values?.[cell]
    if (v !== undefined && Number.isFinite(v) && (style.range !== undefined || hi > 0) && top > lo) {
      const [x, y] = at(cellCentre(w, cell))
      const share = style.range === undefined ? Math.abs(v) / hi : Math.min(1, Math.max(0, (v - lo) / (top - lo)))
      const fill = style.fill ?? 'accent'
      nodes.push(
        typeof fill === 'function'
          ? circle(x, y, k * 0.42, { fill: fill(v, share), opacity: maxOpacity })
          : circle(x, y, k * 0.42, { fill, opacity: maxOpacity * share }),
      )
    }
  }
  if (t.kind === 'graph') {
    for (let a = 0; a < w.n; a++) {
      for (const [side, b] of (t.neighbours[a] ?? []).entries()) {
        if (b < a) continue
        const [x1, y1] = at(cellCentre(w, a))
        const [x2, y2] = at(cellCentre(w, b))
        nodes.unshift(line(x1, y1, x2, y2, hasWall(w, a, side) ? { stroke: 'fg', dash: [2, 2] } : { stroke: 'muted' }))
      }
    }
  } else {
    const walls: SceneNode[] = []
    for (let cell = 0; cell < w.n; cell++) {
      const cs = cellCorners(w, cell)
      for (let side = 0; side < cs.length; side++) {
        const nb = step(t, cell, side)
        if (!hasWall(w, cell, side) || nb < cell) continue
        const [x1, y1] = at(cs[side] ?? [0, 0])
        const [x2, y2] = at(cs[(side + 1) % cs.length] ?? [0, 0])
        walls.push(line(x1, y1, x2, y2, { stroke: 'fg', strokeWidth: Math.max(2, k / 10) }))
      }
    }
    // One group, so `recolour(scene, { walls: ... })` and `SceneView`'s `tones` can restyle every wall at once.
    if (walls.length > 0) nodes.push(group(walls, { id: 'walls' }))
  }
  const label = (cell: number, s: string, fill: Paint): void => {
    const [x, y] = at(cellCentre(w, cell))
    nodes.push(text(x, y, s, { anchor: 'middle', baseline: 'middle', size: k / 2.5, fill, weight: 'bold' }))
  }
  for (const g of w.goals) {
    const [x, y] = at(cellCentre(w, g))
    nodes.push(rect(x - k * 0.35, y - k * 0.35, k * 0.7, k * 0.7, { fill: 'tab2', opacity: 0.6, radius: k / 8 }))
    label(g, 'G', 'fg')
  }
  for (const s of w.starts) label(s, 'S', 'muted')
  for (const it of w.items) {
    const [x, y] = at(cellCentre(w, it.cell))
    const fill = colourPaint(it.colour)
    if (it.kind === 'door')
      nodes.push(rect(x - k * 0.4, y - k * 0.4, k * 0.8, k * 0.8, { stroke: fill, strokeWidth: 3 }))
    else if (it.kind === 'key') {
      nodes.push(circle(x - k * 0.12, y, k * 0.14, { stroke: fill, strokeWidth: 3 }))
      nodes.push(line(x, y, x + k * 0.3, y, { stroke: fill, strokeWidth: 3 }))
    } else nodes.push(circle(x, y, k * 0.15, { fill }))
  }
  if (options.path !== undefined && options.path.length > 1) {
    nodes.push(
      path(
        options.path.map((c) => at(cellCentre(w, c))),
        { stroke: 'accent', strokeWidth: 3, id: 'path' },
      ),
    )
  }
  const agent = options.agent
  if (agent !== undefined) {
    const [x, y] = typeof agent === 'number' ? at(cellCentre(w, agent)) : at([agent.x, agent.y])
    nodes.push(circle(x, y, k / 3, { fill: 'tab1', id: 'agent', title: 'agent' }))
    if (typeof agent !== 'number' && agent.heading !== undefined) {
      const h = agent.heading
      nodes.push(line(x, y, x + (k / 2) * Math.cos(h), y + (k / 2) * Math.sin(h), { stroke: 'fg', strokeWidth: 2 }))
    }
  }
  const [width, height] = worldSize(w)
  const title = options.title ?? w.name
  return scene(Math.ceil(width * k), Math.ceil(height * k), nodes, title === undefined ? {} : { title })
}
