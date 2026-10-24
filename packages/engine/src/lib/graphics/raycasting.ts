import { type Json, WhiteboxError } from '../base/index.ts'
import {
  circle,
  line,
  palettes,
  parsePaint,
  polygon,
  rect,
  type Scene,
  type SceneNode,
  scene,
  type Vec3,
} from '../scene/index.ts'
import { concreteOrThrow, fromStorage } from '../tensor/index.ts'
import { type Algorithm, channels, type RunOptions, runAlgorithm } from '../trace/index.ts'
import { cellIndex, hasWall, type Pose, terrainOf, toScene, type World } from '../worlds/index.ts'
import { type FloatImage, floatImage, sampleNearest } from './textures.ts'
import { srgbToLinear } from './tonemapping.ts'

// Grid ray casting in the style of Wolfenstein 3D: rays from an eye over a square world (x the column, y the row,
// growing down, headings clockwise from east, as `worlds/continuous.ts`), stopped by opaque cells, thin walls and the
// map edge.

/** One cell boundary a ray crossed: the cell entered, the distance, the crossing point and which grid line it was. */
export interface DdaStep {
  readonly cell: number
  readonly t: number
  readonly point: [number, number]
  /** `'x'` for a vertical grid line (x = constant), `'y'` for a horizontal one. */
  readonly side: 'x' | 'y'
}

/** Where a grid ray stops. */
export interface GridHit {
  /** Euclidean distance from the origin. */
  readonly distance: number
  readonly point: [number, number]
  /** The cell hit (the opaque cell, or the cell beyond a thin wall; -1 at the map edge). */
  readonly cell: number
  readonly side: 'x' | 'y'
  /** The face of the hit cell, as a side index (0 N, 1 E, 2 S, 3 W). */
  readonly face: number
  /** The wall's normal, facing the ray. */
  readonly normal: [number, number]
  /** Where along the face the ray hit, in [0, 1), left to right as the viewer sees it (the texture column). */
  readonly u: number
  /** Whether a thin wall between cells (rather than an opaque cell or the edge) stopped the ray. */
  readonly thin: boolean
}

/** A grid ray: the hit, the cells visited in order (from the start cell), and every boundary crossing. */
export interface DdaResult {
  readonly hit: GridHit | null
  readonly cells: readonly number[]
  readonly steps: readonly DdaStep[]
}

function squareSize(w: World): [number, number] {
  if (w.topology.kind !== 'square') throw new WhiteboxError('value', 'graphics: ray casting needs a square-grid world')
  return [w.topology.rows, w.topology.cols]
}

const opaque = (w: World, cell: number): boolean => terrainOf(w, cell).opaque

/**
 * Cast a ray through a square world by the DDA of Amanatides and Woo: from the start cell, step to whichever cell
 * boundary (vertical or horizontal grid line) the ray reaches first, until it crosses a thin wall, enters an opaque
 * cell or leaves the map, or passes `maxDistance`. Each step costs a comparison and an addition, and the cells visited
 * are exactly the cells the ray passes through. A ray starting in an opaque cell or off the map stops at once.
 *
 * @param w - A square-grid world.
 * @param origin - The start `[x, y]`.
 * @param angle - The direction in radians (0 east, clockwise on screen).
 * @param options - Range (default the map's diagonal plus one) and a step cap (default `rows + cols + 2`).
 * @returns The hit, the cells visited and the crossings.
 * @throws WhiteboxError with code `value` for other topologies.
 * @concept ray-casting
 * @example
 * ```ts
 * import { ddaCast } from 'whitebox-engine/graphics'
 * import { fromAscii } from 'whitebox-engine/worlds'
 * const r = ddaCast(fromAscii(['#####', '#...#', '#####']), [1.5, 1.5], 0)
 * [r.hit?.distance, r.hit?.cell, r.hit?.face, r.cells] // => [2.5, 9, 3, [6, 7, 8, 9]]
 * ```
 */
export function ddaCast(
  w: World,
  origin: readonly [number, number],
  angle: number,
  options: { readonly maxDistance?: number; readonly maxSteps?: number } = {},
): DdaResult {
  const [rows, cols] = squareSize(w)
  const maxDistance = options.maxDistance ?? Math.hypot(rows, cols) + 1
  const maxSteps = options.maxSteps ?? rows + cols + 2
  const [x, y] = origin
  const dx = Math.cos(angle)
  const dy = Math.sin(angle)
  let cx = Math.floor(x)
  let cy = Math.floor(y)
  const start = cellIndex(w, cy, cx)
  const cells: number[] = []
  const steps: DdaStep[] = []
  const stop = (t: number, cell: number, side: 'x' | 'y', face: number, thin: boolean): DdaResult => {
    const px = x + t * dx
    const py = y + t * dy
    const normal: [number, number] = side === 'x' ? [dx > 0 ? -1 : 1, 0] : [0, dy > 0 ? -1 : 1]
    const fx = px - Math.floor(px)
    const fy = py - Math.floor(py)
    const u = side === 'x' ? (dx > 0 ? fy : 1 - fy) : dy > 0 ? 1 - fx : fx
    return { hit: { distance: t, point: [px, py], cell, side, face, normal, u: u === 1 ? 0 : u, thin }, cells, steps }
  }
  if (start < 0 || opaque(w, start)) {
    if (start >= 0) cells.push(start)
    return stop(0, start, 'x', dx > 0 ? 3 : 1, false)
  }
  cells.push(start)
  const stepX = dx > 0 ? 1 : -1
  const stepY = dy > 0 ? 1 : -1
  const deltaX = dx === 0 ? Number.POSITIVE_INFINITY : Math.abs(1 / dx)
  const deltaY = dy === 0 ? Number.POSITIVE_INFINITY : Math.abs(1 / dy)
  let tx = dx > 0 ? (cx + 1 - x) * deltaX : dx < 0 ? (x - cx) * deltaX : Number.POSITIVE_INFINITY
  let ty = dy > 0 ? (cy + 1 - y) * deltaY : dy < 0 ? (y - cy) * deltaY : Number.POSITIVE_INFINITY
  let current = start
  for (let k = 0; k < maxSteps; k++) {
    const alongX = tx <= ty
    const t = alongX ? tx : ty
    if (!(t <= maxDistance)) return { hit: null, cells, steps }
    const side: 'x' | 'y' = alongX ? 'x' : 'y'
    // The side of the current cell the ray leaves through, and the face of the next cell it enters through.
    const leaving = alongX ? (stepX > 0 ? 1 : 3) : stepY > 0 ? 2 : 0
    const entering = (leaving + 2) % 4
    if (alongX) {
      cx += stepX
      tx += deltaX
    } else {
      cy += stepY
      ty += deltaY
    }
    const next = cellIndex(w, cy, cx)
    if (hasWall(w, current, leaving)) return stop(t, next, side, entering, true)
    if (next < 0) return stop(t, -1, side, entering, false)
    steps.push({ cell: next, t, point: [x + t * dx, y + t * dy], side })
    cells.push(next)
    if (opaque(w, next)) return stop(t, next, side, entering, false)
    current = next
  }
  return { hit: null, cells, steps }
}

/**
 * Fish-eye correction: the distance from the camera plane rather than from the eye, `d cos(angle - heading)`. Walls
 * drawn by Euclidean distance bulge towards the edges of the view; by perpendicular distance they stay straight.
 *
 * @param distance - Euclidean distance along the ray.
 * @param angle - The ray's direction.
 * @param heading - The view direction.
 * @returns The perpendicular distance.
 * @concept ray-casting
 * @example
 * ```ts
 * import { fisheyeCorrect } from 'whitebox-engine/graphics'
 * fisheyeCorrect(2, Math.PI / 3, 0) // => 1.0000000000000002
 * ```
 */
export function fisheyeCorrect(distance: number, angle: number, heading: number): number {
  return distance * Math.cos(angle - heading)
}

/** Options shared by the column caster and renderer. */
export interface RaycastOptions {
  /** Image width in pixels, one ray per column (default 64). */
  readonly width?: number
  /** Image height in pixels (default 48). */
  readonly height?: number
  /** Horizontal field of view in radians (default pi / 3). */
  readonly fov?: number
  /** Ray range (default the map's diagonal plus one). */
  readonly maxDistance?: number
  /** DDA step cap per ray (default `rows + cols + 2`). */
  readonly maxSteps?: number
  /** Eye height above the floor, in wall heights (default 0.5). */
  readonly eyeHeight?: number
}

/**
 * The direction of the ray of each column: columns sample the camera plane evenly (centre of column `x` at
 * `cameraX = 2 (x + 0.5) / width - 1`), so `angle = heading + atan(cameraX tan(fov / 2))`; rays are denser at the
 * centre than an even fan of angles, which is what keeps walls straight.
 *
 * @param heading - The view direction.
 * @param width - Columns.
 * @param fov - Horizontal field of view.
 * @returns The angles, left to right.
 * @concept ray-casting
 * @example
 * ```ts
 * import { columnAngles } from 'whitebox-engine/graphics'
 * columnAngles(0, 2, Math.PI / 2).map((a) => Math.round(a * 1e9) / 1e9) // => [-0.463647609, 0.463647609]
 * ```
 */
export function columnAngles(heading: number, width: number, fov: number): number[] {
  const t = Math.tan(fov / 2)
  return Array.from({ length: width }, (_, x) => heading + Math.atan(((2 * (x + 0.5)) / width - 1) * t))
}

/** The ray of one screen column and the wall slice it draws. */
export interface Column {
  readonly column: number
  readonly angle: number
  readonly hit: GridHit | null
  /** Fish-eye corrected distance (Infinity when nothing is hit). */
  readonly perpendicular: number
  /** The wall slice's top and bottom on screen, in pixel rows (unclamped). */
  readonly top: number
  readonly bottom: number
  /** Cells the ray visited. */
  readonly cells: readonly number[]
}

/** The state of {@link raycastAlgorithm}: the columns cast so far. */
export interface RaycastState {
  readonly columns: readonly Column[]
}

function settings(w: World, options: RaycastOptions) {
  const [rows, cols] = squareSize(w)
  const width = options.width ?? 64
  const height = options.height ?? 48
  const fov = options.fov ?? Math.PI / 3
  if (!(Number.isInteger(width) && Number.isInteger(height) && width > 0 && height > 0)) {
    throw new WhiteboxError('value', 'raycast: the image size must be positive integers')
  }
  if (!(fov > 0 && fov < Math.PI)) throw new WhiteboxError('value', 'raycast: fov must be in (0, pi)')
  const eyeHeight = options.eyeHeight ?? 0.5
  // Focal length in pixels: half the width over tan(fov / 2); square pixels, so the same vertically.
  const focal = width / 2 / Math.tan(fov / 2)
  return {
    width,
    height,
    fov,
    eyeHeight,
    focal,
    maxDistance: options.maxDistance ?? Math.hypot(rows, cols) + 1,
    maxSteps: options.maxSteps ?? rows + cols + 2,
  }
}

/**
 * Column ray casting as a traced {@link Algorithm}, one step per screen column: cast the column's ray with
 * {@link ddaCast}, correct the fish-eye, and project the wall slice (`top = h/2 - f (1 - eye) / d`,
 * `bottom = h/2 + f eye / d`, with `f` the focal length in pixels and `d` the perpendicular distance). Each step emits
 * the distances and a `hit` message, so a figure can replay the screen filling column by column.
 *
 * @param w - A square-grid world.
 * @param pose - The eye.
 * @param options - Image size, field of view, range, step cap and eye height.
 * @returns The algorithm.
 * @concept ray-casting
 * @example
 * ```ts
 * import { raycastAlgorithm } from 'whitebox-engine/graphics'
 * import { runAlgorithm } from 'whitebox-engine/trace'
 * import { fromAscii } from 'whitebox-engine/worlds'
 * const a = raycastAlgorithm(fromAscii(['#####', '#...#', '#####']), { x: 1.5, y: 1.5, heading: 0 }, { width: 8 })
 * runAlgorithm(a, { maxSteps: 100 }).steps // => 8
 * ```
 */
export function raycastAlgorithm(w: World, pose: Pose, options: RaycastOptions = {}): Algorithm<RaycastState> {
  const st = settings(w, options)
  const angles = columnAngles(pose.heading, st.width, st.fov)
  return {
    name: 'graphics/raycast',
    config: { width: st.width, height: st.height, fov: st.fov, x: pose.x, y: pose.y, heading: pose.heading } as Json,
    schema: channels({
      metrics: {
        distance: { description: 'Euclidean distance to the wall hit by this column' },
        perpendicular: { description: 'fish-eye corrected distance' },
        cells: { description: 'cells visited by the DDA' },
      },
      messages: {
        hit: {
          description: 'the wall a column ray hit',
          payload: { column: 'number', cell: 'number', side: 'string' },
        },
      },
    }),
    init: () => ({ columns: [] }),
    step: (s) => {
      const x = s.columns.length
      const angle = angles[x] ?? pose.heading
      const r = ddaCast(w, [pose.x, pose.y], angle, { maxDistance: st.maxDistance, maxSteps: st.maxSteps })
      const perpendicular =
        r.hit === null ? Number.POSITIVE_INFINITY : Math.max(1e-9, fisheyeCorrect(r.hit.distance, angle, pose.heading))
      const top = st.height / 2 - (st.focal * (1 - st.eyeHeight)) / perpendicular
      const bottom = st.height / 2 + (st.focal * st.eyeHeight) / perpendicular
      return { columns: [...s.columns, { column: x, angle, hit: r.hit, perpendicular, top, bottom, cells: r.cells }] }
    },
    done: (s) => (s.columns.length >= st.width ? 'converged' : undefined),
    observe: (s) => {
      const c = s.columns[s.columns.length - 1]
      if (c === undefined) return {}
      return {
        metrics: {
          distance: c.hit?.distance ?? -1,
          perpendicular: Number.isFinite(c.perpendicular) ? c.perpendicular : -1,
          cells: c.cells.length,
        },
        messages: [
          { kind: 'hit', payload: { column: c.column, cell: c.hit?.cell ?? -1, side: c.hit?.side ?? 'none' } },
        ],
      }
    },
  }
}

/**
 * Cast every column (see {@link raycastAlgorithm}).
 *
 * @param w - A square-grid world.
 * @param pose - The eye.
 * @param options - Image size, field of view, range, step cap, eye height, and a tracer and policy for the run.
 * @returns The columns, left to right.
 * @concept ray-casting
 * @example
 * ```ts
 * import { castColumns } from 'whitebox-engine/graphics'
 * import { fromAscii } from 'whitebox-engine/worlds'
 * const cols = castColumns(fromAscii(['#####', '#...#', '#####']), { x: 1.5, y: 1.5, heading: 0 }, { width: 4, height: 10 })
 * [cols.length, cols[1].hit?.cell, Math.round(cols[1].perpendicular * 1e9) / 1e9] // => [4, 9, 2.5]
 * ```
 */
export function castColumns(
  w: World,
  pose: Pose,
  options: RaycastOptions & Partial<Pick<RunOptions, 'tracer' | 'policy' | 'parent'>> = {},
): Column[] {
  const { tracer, policy, parent, ...rest } = options
  const run = runAlgorithm(raycastAlgorithm(w, pose, rest), {
    maxSteps: (rest.width ?? 64) + 1,
    ...(tracer === undefined ? {} : { tracer }),
    ...(policy === undefined ? {} : { policy }),
    ...(parent === undefined ? {} : { parent }),
  })
  return [...run.state.columns]
}

// ------------------------------------------------------------------------------------------- first-person frames

/** A billboard sprite standing on the floor at `(x, y)`: textured (alpha below 0.5 is transparent) or flat. */
export interface Sprite {
  readonly x: number
  readonly y: number
  readonly texture?: FloatImage
  readonly colour?: Vec3
  /** Height (and width) in wall heights (default 0.6). */
  readonly size?: number
}

/** A surface's look: a texture tiled once per cell, or a flat colour. */
export type Surface = FloatImage | Vec3

/** Options of {@link renderRaycast}. */
export interface RaycastRenderOptions extends RaycastOptions {
  /** Wall surfaces by terrain name (default: the terrain's paint as a flat colour). */
  readonly walls?: Readonly<Record<string, Surface>>
  /** The map edge's surface (default grey). */
  readonly edge?: Surface
  /** Floor surfaces by terrain name, and the default floor. */
  readonly floors?: Readonly<Record<string, Surface>>
  readonly floor?: Surface
  readonly ceiling?: Surface
  /** Brightness of walls facing north or south (y sides), for depth cues (default 0.7). */
  readonly sideShade?: number
  /** Exponential fog: colour and density per unit distance (default none). */
  readonly fog?: { readonly colour: Vec3; readonly density: number }
  readonly sprites?: readonly Sprite[]
}

/** A sprite as drawn: its index, distance in front of the eye, screen centre and size, and the columns it covers. */
export interface DrawnSprite {
  readonly index: number
  readonly depth: number
  readonly screenX: number
  readonly size: number
  readonly columns: readonly [number, number]
  /** How many of those columns it was nearer than the wall in (0 when the walls hide it). */
  readonly visible: number
}

/** A rendered first-person frame and how it was made. */
export interface RaycastFrame {
  readonly image: FloatImage
  readonly columns: readonly Column[]
  /** The depth buffer of the walls: perpendicular distance per column. */
  readonly depth: Float64Array
  /** Sprites in the order they were drawn (far to near). */
  readonly sprites: readonly DrawnSprite[]
}

function paintColour(paint: string): Vec3 {
  const c = parsePaint(paint as Parameters<typeof parsePaint>[0], palettes.light) ?? [128, 128, 128, 1]
  const lin = concreteOrThrow(
    srgbToLinear(fromStorage(Float64Array.from([c[0] / 255, c[1] / 255, c[2] / 255]), [3], 'float64')),
    'raycast',
  ).data
  return [lin[0] ?? 0, lin[1] ?? 0, lin[2] ?? 0]
}

function surfaceAt(s: Surface, u: number, v: number): Vec3 {
  if (Array.isArray(s)) return s as Vec3
  const c = sampleNearest(s as FloatImage, u, v)
  return [c[0], c[1], c[2]]
}

/**
 * Render a first-person frame of a world by column ray casting: per column, the wall slice textured by the hit's `u`
 * and the row's height on the slice (y sides shaded); below and above it the floor and ceiling by floor casting (each
 * pixel row lies at distance `f eye / (y - h/2)` along every column's ray, giving the floor point and so its texel);
 * then sprites sorted far to near, each column of a sprite drawn only where it is nearer than the wall's depth buffer.
 * Everything is deterministic and returned with the columns and the depth buffer, so figures can show each stage.
 *
 * @param w - A square-grid world.
 * @param pose - The eye.
 * @param options - Image size, field of view, surfaces, side shading, fog and sprites.
 * @returns The frame.
 * @concept ray-casting
 * @example
 * ```ts
 * import { renderRaycast } from 'whitebox-engine/graphics'
 * import { fromAscii } from 'whitebox-engine/worlds'
 * const room = fromAscii(['#######', '#.....#', '#.....#', '#.....#', '#######'])
 * const f = renderRaycast(room, { x: 1.5, y: 2.5, heading: 0 }, { width: 8, height: 6, floor: [0, 1, 0], ceiling: [0, 0, 1] })
 * [f.image.rgb.slice(0, 3), f.image.rgb.slice(3 * 40, 3 * 40 + 3)].map((c) => Array.from(c)) // => [[0, 0, 1], [0, 1, 0]]
 * ```
 */
export function renderRaycast(w: World, pose: Pose, options: RaycastRenderOptions = {}): RaycastFrame {
  const st = settings(w, options)
  const { width, height, focal, eyeHeight } = st
  const columns = castColumns(w, pose, options)
  const image = floatImage(width, height)
  const depth = Float64Array.from(columns, (c) => c.perpendicular)
  const sideShade = options.sideShade ?? 0.7
  const wallCache = new Map<string, Surface>()
  const wallSurface = (cell: number): Surface => {
    if (cell < 0) return options.edge ?? [0.3, 0.3, 0.3]
    const t = terrainOf(w, cell)
    const given = options.walls?.[t.name]
    if (given !== undefined) return given
    const cached = wallCache.get(t.name) ?? paintColour(t.paint)
    wallCache.set(t.name, cached)
    return cached
  }
  const floorDefault: Surface = options.floor ?? [0.35, 0.35, 0.35]
  const ceilingDefault: Surface = options.ceiling ?? [0.6, 0.6, 0.6]
  const fog = (c: Vec3, d: number): Vec3 => {
    if (options.fog === undefined) return c
    const k = Math.exp(-options.fog.density * d)
    const f = options.fog.colour
    return [c[0] * k + f[0] * (1 - k), c[1] * k + f[1] * (1 - k), c[2] * k + f[2] * (1 - k)]
  }
  const put = (x: number, y: number, c: Vec3): void => image.rgb.set(c, 3 * (y * width + x))
  const fx = Math.cos(pose.heading)
  const fy = Math.sin(pose.heading)
  const plane = Math.tan(st.fov / 2)
  for (const col of columns) {
    const x = col.column
    // The column's ray direction with unit component along the view: a step of 1 along it is a step of 1 in depth.
    const cameraX = (2 * (x + 0.5)) / width - 1
    const rx = fx - fy * cameraX * plane
    const ry = fy + fx * cameraX * plane
    for (let y = 0; y < height; y++) {
      const yc = y + 0.5
      if (col.hit !== null && yc >= col.top && yc < col.bottom) {
        const v = (yc - col.top) / (col.bottom - col.top)
        let c = surfaceAt(wallSurface(col.hit.cell), col.hit.u, v)
        if (col.hit.side === 'y') c = [c[0] * sideShade, c[1] * sideShade, c[2] * sideShade]
        put(x, y, fog(c, col.perpendicular))
        continue
      }
      const below = yc > height / 2
      const rowDistance = below
        ? (focal * eyeHeight) / (yc - height / 2)
        : (focal * (1 - eyeHeight)) / (height / 2 - yc)
      const px = pose.x + rowDistance * rx
      const py = pose.y + rowDistance * ry
      const u = px - Math.floor(px)
      const v = py - Math.floor(py)
      let surface: Surface = below ? floorDefault : ceilingDefault
      if (below && options.floors !== undefined) {
        const cell = cellIndex(w, Math.floor(py), Math.floor(px))
        if (cell >= 0) surface = options.floors[terrainOf(w, cell).name] ?? floorDefault
      }
      put(x, y, fog(surfaceAt(surface, u, v), rowDistance))
    }
  }
  const drawn: DrawnSprite[] = []
  const sprites = (options.sprites ?? []).map((s, index) => {
    const ox = s.x - pose.x
    const oy = s.y - pose.y
    // Right of the view is the heading turned a quarter clockwise on screen: (-fy, fx).
    return { s, index, depth: ox * fx + oy * fy, lateral: -ox * fy + oy * fx }
  })
  sprites.sort((a, b) => b.depth - a.depth || a.index - b.index)
  for (const { s, index, depth: d, lateral } of sprites) {
    if (d < 1e-3) continue
    const size = (focal * (s.size ?? 0.6)) / d
    const screenX = width / 2 + (focal * lateral) / d
    const bottom = height / 2 + (focal * eyeHeight) / d
    const top = bottom - size
    const left = screenX - size / 2
    const x0 = Math.max(0, Math.ceil(left - 0.5))
    const x1 = Math.min(width, Math.ceil(left + size - 0.5))
    if (x1 <= x0) continue
    let visible = 0
    for (let x = x0; x < x1; x++) {
      if (!(d < (depth[x] ?? Number.POSITIVE_INFINITY))) continue
      visible++
      const u = (x + 0.5 - left) / size
      for (let y = Math.max(0, Math.ceil(top - 0.5)); y < Math.min(height, Math.ceil(bottom - 0.5)); y++) {
        const v = (y + 0.5 - top) / size
        let c: Vec3 = s.colour ?? [1, 0.8, 0.2]
        if (s.texture !== undefined) {
          const t = sampleNearest(s.texture, u, v, 'clamp')
          if (t[3] < 0.5) continue
          c = [t[0], t[1], t[2]]
        }
        put(x, y, fog(c, d))
      }
    }
    drawn.push({ index, depth: d, screenX, size, columns: [x0, x1], visible })
  }
  return { image, columns, depth, sprites: drawn }
}

// ---------------------------------------------------------------------------------------- top-down view cones

/** One ray of a view-cone figure. */
export interface ViewConeRay {
  readonly angle: number
  readonly from: [number, number]
  /** Where it stopped (the hit, or the end of its range). */
  readonly to: [number, number]
  readonly hit: GridHit | null
  readonly cells: readonly number[]
  readonly steps: readonly DdaStep[]
}

/**
 * The rays of a view cone as data: `rays` column rays across the field of view (as {@link columnAngles} spaces them),
 * each with its DDA cells and crossings.
 *
 * @param w - A square-grid world.
 * @param pose - The eye.
 * @param options - Ray count (default 9), field of view, range and step cap.
 * @returns The rays, left to right.
 * @concept ray-casting
 * @example
 * ```ts
 * import { viewConeRays } from 'whitebox-engine/graphics'
 * import { fromAscii } from 'whitebox-engine/worlds'
 * const rays = viewConeRays(fromAscii(['#####', '#...#', '#####']), { x: 1.5, y: 1.5, heading: 0 }, { rays: 3 })
 * rays.map((r) => r.hit?.cell) // => [2, 9, 12]
 * ```
 */
export function viewConeRays(
  w: World,
  pose: Pose,
  options: {
    readonly rays?: number
    readonly fov?: number
    readonly maxDistance?: number
    readonly maxSteps?: number
  } = {},
): ViewConeRay[] {
  const [rows, cols] = squareSize(w)
  const n = Math.max(1, Math.floor(options.rays ?? 9))
  const range = options.maxDistance ?? Math.hypot(rows, cols) + 1
  return columnAngles(pose.heading, n, options.fov ?? Math.PI / 3).map((angle) => {
    const r = ddaCast(w, [pose.x, pose.y], angle, {
      maxDistance: range,
      ...(options.maxSteps === undefined ? {} : { maxSteps: options.maxSteps }),
    })
    const d = r.hit?.distance ?? range
    return {
      angle,
      from: [pose.x, pose.y],
      to: r.hit?.point ?? [pose.x + d * Math.cos(angle), pose.y + d * Math.sin(angle)],
      hit: r.hit,
      cells: r.cells,
      steps: r.steps,
    }
  })
}

/**
 * A top-down figure of ray casting: the world (drawn by `worlds`' `toScene`), the agent, its view cone, each ray to its
 * hit, and for one highlighted ray the cells its DDA visited and the grid crossings it tested.
 *
 * @param w - A square-grid world.
 * @param pose - The eye.
 * @param options - Ray count, field of view, range, pixels per cell (default 32) and the highlighted ray (default the
 *   middle one; -1 for none).
 * @returns The scene.
 * @concept ray-casting
 * @example
 * ```ts
 * import { viewConeScene } from 'whitebox-engine/graphics'
 * import { fromAscii } from 'whitebox-engine/worlds'
 * const s = viewConeScene(fromAscii(['#####', '#...#', '#####']), { x: 1.5, y: 1.5, heading: 0 }, { rays: 5 })
 * [s.width, s.children.filter((n) => n.id?.startsWith('ray-')).length] // => [160, 5]
 * ```
 */
export function viewConeScene(
  w: World,
  pose: Pose,
  options: {
    readonly rays?: number
    readonly fov?: number
    readonly maxDistance?: number
    readonly cell?: number
    readonly highlight?: number
  } = {},
): Scene {
  const k = options.cell ?? 32
  const base = toScene(w, { cell: k, agent: pose })
  const rays = viewConeRays(w, pose, options)
  const highlight = options.highlight ?? Math.floor(rays.length / 2)
  const nodes: SceneNode[] = []
  const chosen = rays[highlight]
  if (chosen !== undefined) {
    for (const c of chosen.cells) {
      const cols = w.topology.kind === 'square' ? w.topology.cols : 1
      nodes.push(
        rect((c % cols) * k, Math.floor(c / cols) * k, k, k, { fill: 'accent', opacity: 0.25, id: `dda-cell-${c}` }),
      )
    }
  }
  const cone: [number, number][] = [
    [pose.x * k, pose.y * k],
    ...rays.map((r): [number, number] => [r.to[0] * k, r.to[1] * k]),
  ]
  nodes.push(polygon(cone, { fill: 'accent', opacity: 0.12, id: 'cone' }))
  for (const [i, r] of rays.entries()) {
    nodes.push(
      line(r.from[0] * k, r.from[1] * k, r.to[0] * k, r.to[1] * k, {
        stroke: 'accent',
        strokeWidth: i === highlight ? 2 : 1,
        id: `ray-${i}`,
      }),
    )
    if (r.hit !== null)
      nodes.push(
        circle(r.hit.point[0] * k, r.hit.point[1] * k, 2.5, {
          fill: r.hit.side === 'x' ? 'tab3' : 'tab0',
          id: `hit-${i}`,
        }),
      )
  }
  if (chosen !== undefined) {
    for (const [j, s] of chosen.steps.entries()) {
      nodes.push(
        circle(s.point[0] * k, s.point[1] * k, 1.5, { fill: s.side === 'x' ? 'tab3' : 'tab0', id: `crossing-${j}` }),
      )
    }
  }
  return scene(base.width, base.height, [...base.children, ...nodes], {
    title: 'ray casting: view cone',
    ...(base.background === undefined ? {} : { background: base.background }),
  })
}
