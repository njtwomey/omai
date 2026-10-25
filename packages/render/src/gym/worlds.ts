/**
 * Drawing `worlds` worlds: `worldView` shows any world from above (terrain, walls, items, goals, an agent and its path)
 * with the overlays of search and reinforcement learning (frontiers, explored cells, policy arrows, values on cells),
 * `raycastViews` gives the first-person ray-cast view beside its top-down map and view cone, and `pathFrames` animates
 * a path.
 */

import { imageToPixels, type RaycastFrame, type RaycastRenderOptions, renderRaycast } from 'mdml-engine/graphics'
import {
  arrow,
  circle,
  group,
  image,
  line,
  polygon,
  rect,
  type Scene,
  type SceneNode,
  scene,
  text,
} from 'mdml-engine/scene'
import { cellCentre, cellCorners, type Pose, toScene, type World, type WorldSceneOptions } from 'mdml-engine/worlds'

/** The unit vector (x right, y down) of each direction name, for policy arrows. */
const VECTORS: Readonly<Record<string, readonly [number, number]>> = {
  N: [0, -1],
  E: [1, 0],
  S: [0, 1],
  W: [-1, 0],
  NE: [Math.SQRT1_2, -Math.SQRT1_2],
  SE: [Math.SQRT1_2, Math.SQRT1_2],
  SW: [-Math.SQRT1_2, Math.SQRT1_2],
  NW: [-Math.SQRT1_2, -Math.SQRT1_2],
  UP: [0, -1],
  RIGHT: [1, 0],
  DOWN: [0, 1],
  LEFT: [-1, 0],
}

const HEX: Readonly<Record<string, readonly [number, number]>> = Object.fromEntries(
  ['E', 'SE', 'SW', 'W', 'NW', 'NE'].map((d, i) => [d, [Math.cos((i * Math.PI) / 3), Math.sin((i * Math.PI) / 3)]]),
)

/** Options of {@link worldView}: the engine's world scene options, and overlays for search and reinforcement learning. */
export interface WorldViewOptions extends WorldSceneOptions {
  /**
   * A policy: the action of each cell, or a probability per action of each cell (arrows scaled by probability).
   * Actions are named by `actions`.
   */
  readonly policy?: ArrayLike<number> | readonly ArrayLike<number>[]
  /** The direction each action moves (`N`, `E`, `S`, `W`, diagonals, or `UP`, `RIGHT`, `DOWN`, `LEFT`); default the world's directions in order. */
  readonly actions?: readonly string[]
  /** Numbers to print in each cell (values, visit counts); `NaN` prints nothing. */
  readonly labels?: ArrayLike<number>
  /** Cells on a search's frontier (outlined in the accent) and already explored (tinted). */
  readonly frontier?: readonly number[]
  readonly explored?: readonly number[]
}

function centre(w: World, cell: number, k: number): [number, number] {
  const [x, y] = cellCentre(w, cell)
  return [x * k, y * k]
}

/**
 * One way to draw any `worlds` world from above: the engine's `toScene` (terrain, walls, items, starts and goals, an
 * agent, a path, values shaded per cell), with the overlays a search or a learning agent needs: explored cells tinted
 * and the frontier outlined, a policy as arrows (sized by probability when it is a distribution), and numbers per cell.
 *
 * @param w - The world.
 * @param options - The scene options, policy and its actions, labels, frontier and explored cells.
 * @returns The scene.
 * @example
 * ```ts
 * import { fromAscii } from 'mdml-engine/worlds'
 * import { worldView } from 'mdml-render/gym'
 * worldView(fromAscii(['S.', '.G']), { policy: [1, 2, 1, 0] }).children.some((n) => n.id === 'policy') // => true
 * ```
 */
export function worldView(w: World, options: WorldViewOptions = {}): Scene {
  const k = options.cell ?? 32
  const base = toScene(w, options)
  const under: SceneNode[] = []
  const over: SceneNode[] = []
  const cellShape = (cell: number, style: Parameters<typeof polygon>[1]): SceneNode =>
    polygon(
      cellCorners(w, cell).map(([x, y]): [number, number] => [x * k, y * k]),
      style,
    )
  if (w.topology.kind !== 'graph') {
    for (const c of options.explored ?? []) under.push(cellShape(c, { fill: 'accent', opacity: 0.18 }))
    for (const c of options.frontier ?? [])
      over.push(cellShape(c, { fill: 'none', stroke: 'accent', strokeWidth: 2.5 }))
  } else {
    for (const c of options.explored ?? []) {
      const [x, y] = centre(w, c, k)
      under.push(circle(x, y, k * 0.42, { fill: 'accent', opacity: 0.25 }))
    }
    for (const c of options.frontier ?? []) {
      const [x, y] = centre(w, c, k)
      over.push(circle(x, y, k * 0.42, { fill: 'none', stroke: 'accent', strokeWidth: 2.5 }))
    }
  }
  const arrows: SceneNode[] = []
  if (options.policy !== undefined && w.topology.kind !== 'graph') {
    const names =
      options.actions ?? (w.topology.kind === 'hex' ? ['E', 'SE', 'SW', 'W', 'NW', 'NE'] : ['N', 'E', 'S', 'W'])
    const table = w.topology.kind === 'hex' ? { ...VECTORS, ...HEX } : VECTORS
    const policy = options.policy
    for (let cell = 0; cell < w.n; cell++) {
      const row = (policy as readonly ArrayLike<number>[])[cell]
      const probs: number[] =
        typeof row === 'object' && row !== null
          ? Array.from(row)
          : names.map((_, a) => (a === (policy as ArrayLike<number>)[cell] ? 1 : 0))
      const [cx, cy] = centre(w, cell, k)
      probs.forEach((p, a) => {
        const v = table[names[a] ?? '']
        if (v === undefined || !(p > 0.01)) return
        const len = k * 0.38 * Math.sqrt(p)
        arrows.push(
          arrow(cx, cy, cx + v[0] * len, cy + v[1] * len, {
            stroke: 'fg',
            strokeWidth: 1.5,
            opacity: 0.4 + 0.6 * p,
          }),
        )
      })
    }
  }
  if (arrows.length > 0) over.push(group(arrows, { id: 'policy' }))
  if (options.labels !== undefined) {
    const labels: SceneNode[] = []
    for (let cell = 0; cell < Math.min(w.n, options.labels.length); cell++) {
      const v = options.labels[cell] ?? Number.NaN
      if (!Number.isFinite(v)) continue
      const [cx, cy] = centre(w, cell, k)
      const shown = Number.isInteger(v) ? String(v) : v.toFixed(Math.abs(v) >= 10 ? 1 : 2)
      labels.push(text(cx, cy + k * 0.3, shown, { size: Math.max(8, k * 0.26), anchor: 'middle', fill: 'muted' }))
    }
    over.push(group(labels, { id: 'labels' }))
  }
  // The engine's scene first (terrain at the bottom), the tint under its marks would hide the terrain, so tints go on
  // top at low opacity.
  return { ...base, children: [...base.children, ...under, ...over] }
}

/** What {@link raycastViews} returns: the first-person view, the map with the view cone, and the frame they come from. */
export interface RaycastViews {
  readonly firstPerson: Scene
  readonly topDown: Scene
  readonly frame: RaycastFrame
}

/**
 * The first-person view of a square-grid world by column ray casting (the engine's `renderRaycast`), with its top-down
 * companion: the map, the eye, and every `rayEvery`-th column's ray out to its hit, so the cone shows what each column
 * of the picture sees.
 *
 * @param w - A square-grid world.
 * @param pose - The eye: position in cells and heading in radians.
 * @param options - The renderer's options (image size, field of view, surfaces, fog, sprites), the map's cell size
 * (default 32), how often to draw a ray (default every 4th column) and the first-person view's drawn width (default 320).
 * @returns The two scenes and the frame.
 * @example
 * ```ts
 * import { fromAscii } from 'mdml-engine/worlds'
 * import { raycastViews } from 'mdml-render/gym'
 * const v = raycastViews(fromAscii(['#####', '#...#', '#####']), { x: 1.5, y: 1.5, heading: 0 }, { width: 16, height: 12 })
 * v.firstPerson.children[0].kind // => 'image'
 * ```
 */
export function raycastViews(
  w: World,
  pose: Pose,
  options: RaycastRenderOptions & {
    readonly cell?: number
    readonly rayEvery?: number
    readonly viewWidth?: number
  } = {},
): RaycastViews {
  const frame = renderRaycast(w, pose, options)
  const { image: img } = frame
  const viewWidth = options.viewWidth ?? 320
  const viewHeight = (viewWidth * img.height) / img.width
  const firstPerson = scene(viewWidth, viewHeight, [
    image(0, 0, viewWidth, viewHeight, imageToPixels(img), { id: 'view', smoothing: false }),
  ])
  const k = options.cell ?? 32
  const every = Math.max(1, options.rayEvery ?? 4)
  const [ex, ey] = [pose.x * k, pose.y * k]
  const rays: SceneNode[] = frame.columns
    .filter((c, i) => i % every === 0 || i === frame.columns.length - 1 || c.hit === null)
    .flatMap((c) => {
      const d = Number.isFinite(c.perpendicular)
        ? c.perpendicular / Math.max(Math.cos(c.angle - pose.heading), 1e-6)
        : 2 * Math.max(w.n, 1)
      return [
        line(ex, ey, ex + Math.cos(c.angle) * d * k, ey + Math.sin(c.angle) * d * k, {
          stroke: 'accent',
          opacity: 0.5,
        }),
      ]
    })
  const map = toScene(w, { cell: k })
  const eye: SceneNode[] = [
    circle(ex, ey, k * 0.18, { fill: 'accent', id: 'eye' }),
    arrow(ex, ey, ex + Math.cos(pose.heading) * k * 0.45, ey + Math.sin(pose.heading) * k * 0.45, {
      stroke: 'accent',
      strokeWidth: 2,
    }),
  ]
  const clip = rect(0, 0, map.width, map.height, { fill: 'none', stroke: 'grid' })
  const topDown: Scene = { ...map, children: [...map.children, group(rays, { id: 'rays' }), ...eye, clip] }
  return { firstPerson, topDown, frame }
}

/** A frame of an agent path through a world, for filmstrips and animations: the world with the path so far. */
export function pathFrames(w: World, cells: readonly number[], options: WorldViewOptions = {}): Scene[] {
  return cells.map((cell, i) => worldView(w, { ...options, agent: cell, path: cells.slice(0, i + 1) }))
}
