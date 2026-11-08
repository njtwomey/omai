/**
 * Iterated function systems: a handful of affine contractions of the plane whose union of images of a set is the set
 * (the attractor). Sampled by the chaos game (apply a randomly chosen map to a point, over and over; the points trace
 * the attractor) or built whole by the deterministic iteration (map a picture of the plane by all the maps at once, the
 * collage theorem: it converges to the attractor whatever it started from). Sierpinski's triangle, its carpet and
 * Barnsley's fern, the similarity dimension that predicts the dimension of the first two exactly, and the two
 * iterations as traced algorithms.
 */

import { OmaiError } from '../../base/index.ts'
import { foldIn } from '../../random/index.ts'
import {
  type Algorithm,
  type Contract,
  contractSchema,
  defineContract,
  scalar,
  stateFromJson,
  stateToJson,
  tensor,
} from '../../trace/index.ts'
import { draws, positiveInt } from '../automata/_core.ts'

/** An affine map of the plane, `(x, y) -> (a x + b y + e, c x + d y + f)`. */
export interface PlanarAffineMap {
  readonly a: number
  readonly b: number
  readonly c: number
  readonly d: number
  readonly e: number
  readonly f: number
}

/** A system of affine contractions, with the chance of picking each map in the chaos game. */
export interface Ifs {
  readonly name: string
  readonly maps: readonly PlanarAffineMap[]
  /** The probabilities of the maps (default proportional to each map's area scaling, which fills the attractor evenly). */
  readonly probabilities?: readonly number[]
}

const similarity = (s: number, tx: number, ty: number): PlanarAffineMap => ({ a: s, b: 0, c: 0, d: s, e: tx, f: ty })

/** Sierpinski's triangle: three maps halving the plane toward the corners of an equilateral triangle. Dimension log 3 / log 2. */
export const SIERPINSKI: Ifs = {
  name: 'sierpinski',
  maps: [similarity(0.5, 0, 0), similarity(0.5, 0.5, 0), similarity(0.5, 0.25, Math.sqrt(3) / 4)],
}

/** Sierpinski's carpet: eight maps by a third, around the middle square. Dimension log 8 / log 3. */
export const SIERPINSKI_CARPET: Ifs = {
  name: 'sierpinski-carpet',
  maps: [0, 1, 2]
    .flatMap((i) => [0, 1, 2].map((j) => [i, j] as const))
    .filter(([i, j]) => !(i === 1 && j === 1))
    .map(([i, j]) => similarity(1 / 3, i / 3, j / 3)),
}

/** Barnsley's fern: four affine maps (stem, the main frond, and the two lowest leaflets) with probabilities 1, 85, 7, 7 per cent. */
export const BARNSLEY_FERN: Ifs = {
  name: 'barnsley-fern',
  maps: [
    { a: 0, b: 0, c: 0, d: 0.16, e: 0, f: 0 },
    { a: 0.85, b: 0.04, c: -0.04, d: 0.85, e: 0, f: 1.6 },
    { a: 0.2, b: -0.26, c: 0.23, d: 0.22, e: 0, f: 1.6 },
    { a: -0.15, b: 0.28, c: 0.26, d: 0.24, e: 0, f: 0.44 },
  ],
  probabilities: [0.01, 0.85, 0.07, 0.07],
}

/**
 * Apply a map to a point.
 *
 * @param m - The map.
 * @param x - The abscissa.
 * @param y - The ordinate.
 * @returns The image `[x, y]`.
 * @concept iterated-function-systems
 * @example
 * ```ts
 * import { fractals } from 'omai-engine/dynamics'
 * fractals.applyAffine({ a: 0.5, b: 0, c: 0, d: 0.5, e: 1, f: 2 }, 2, 4) // => [2, 4]
 * ```
 */
export function applyAffine(m: PlanarAffineMap, x: number, y: number): [number, number] {
  return [m.a * x + m.b * y + m.e, m.c * x + m.d * y + m.f]
}

/**
 * The probabilities of the chaos game: those given, or each map's share of the total area scaling `|ad - bc|` (maps
 * that shrink the plane less are visited more often, so the points fill the attractor evenly).
 *
 * @param ifs - The system.
 * @returns Probabilities summing to 1.
 * @throws OmaiError with code `value` for given probabilities that are negative, the wrong number, or do not sum to 1.
 * @concept iterated-function-systems
 * @example
 * ```ts
 * import { fractals } from 'omai-engine/dynamics'
 * fractals.ifsProbabilities(fractals.SIERPINSKI).map((p) => p.toFixed(3)) // => ['0.333', '0.333', '0.333']
 * ```
 */
export function ifsProbabilities(ifs: Ifs): number[] {
  if (ifs.probabilities !== undefined) {
    const sum = ifs.probabilities.reduce((a, b) => a + b, 0)
    if (
      ifs.probabilities.length !== ifs.maps.length ||
      ifs.probabilities.some((p) => !(p >= 0)) ||
      Math.abs(sum - 1) > 1e-9
    )
      throw new OmaiError(
        'value',
        `ifsProbabilities: ${ifs.name} needs one non-negative probability per map, summing to 1`,
      )
    return [...ifs.probabilities]
  }
  const areas = ifs.maps.map((m) => Math.abs(m.a * m.d - m.b * m.c))
  const total = areas.reduce((a, b) => a + b, 0)
  if (!(total > 0)) throw new OmaiError('value', `ifsProbabilities: ${ifs.name} has only degenerate maps`)
  return areas.map((a) => a / total)
}

/**
 * The similarity dimension: the `D` with `sum_i r_i^D = 1`, where `r_i = sqrt |det|` is each map's scaling. It is the
 * dimension of the attractor when the maps are similarities and the pieces barely overlap (open set condition), as for
 * Sierpinski's triangle and carpet; for maps that also shear or squash (the fern) it is only an estimate.
 *
 * @param ifs - The system.
 * @returns D, found by bisection.
 * @throws OmaiError with code `value` if a map is not a contraction (scaling at least 1) or is degenerate in every map.
 * @concept iterated-function-systems
 * @example
 * ```ts
 * import { fractals } from 'omai-engine/dynamics'
 * Math.round(fractals.similarityDimension(fractals.SIERPINSKI) * 1e9) / 1e9 // => 1.584962501
 * ```
 */
export function similarityDimension(ifs: Ifs): number {
  const r = ifs.maps.map((m) => Math.sqrt(Math.abs(m.a * m.d - m.b * m.c))).filter((x) => x > 0)
  if (r.length === 0 || r.some((x) => x >= 1))
    throw new OmaiError('value', `similarityDimension: ${ifs.name} must be made of contractions`)
  const f = (D: number): number => r.reduce((s, x) => s + x ** D, 0) - 1
  let lo = 0
  let hi = 10
  for (let i = 0; i < 200; i++) {
    const mid = (lo + hi) / 2
    if (f(mid) > 0) lo = mid
    else hi = mid
  }
  return (lo + hi) / 2
}

/** Options of {@link chaosGame}. */
export interface ChaosGameOptions {
  /** The system (default Sierpinski's triangle). */
  readonly ifs?: Ifs
  /** Points added per step (default 100). */
  readonly batch?: number
  /** Iterations thrown away before the first point is kept (default 20, enough to land on the attractor). */
  readonly burn?: number
  /** The most points kept (default 200000); further points are generated but not stored. */
  readonly keep?: number
}

/** The state of {@link chaosGame}. */
export interface ChaosGameState {
  /** The current point. */
  readonly point: Float64Array
  /** The kept points so far, flat `[x, y, ...]`. */
  readonly cloud: Float64Array
  /** Points generated (kept or not). */
  readonly generated: number
}

/** The paths of {@link chaosGame} and {@link ifsDeterministic}. */
export type IfsPath = 'points/cloud' | 'points/count' | 'ifs/maps' | 'ifs/dimension'

/** The log contract of {@link chaosGame}. */
export const chaosGameContract: Contract<ChaosGameState, IfsPath> = defineContract({
  'points/cloud': tensor({ role: 'state', shape: ['n', 2], description: 'the points so far' }),
  'points/count': scalar({ description: 'points generated' }),
  'ifs/maps': tensor({ role: 'param', shape: ['maps', 6], description: 'a, b, c, d, e, f of every map' }),
  'ifs/dimension': scalar({ description: 'the similarity dimension of the system' }),
})

const mapRows = (ifs: Ifs): number[] => ifs.maps.flatMap((m) => [m.a, m.b, m.c, m.d, m.e, m.f])

/**
 * The chaos game as a traced algorithm: each step applies `batch` randomly chosen maps in turn (chosen with the
 * system's probabilities, from the step's key), keeping every image. The cloud accumulates and fills out the attractor.
 * The state serialises.
 *
 * @param options - The system, batch, burn-in and the most points kept.
 * @returns The algorithm.
 * @throws OmaiError with code `value` for bad probabilities, a non-contraction (for the dimension) or a non-positive batch.
 * @concept iterated-function-systems
 * @example
 * ```ts
 * import { fractals } from 'omai-engine/dynamics'
 * import { runAlgorithm } from 'omai-engine/trace'
 * runAlgorithm(fractals.chaosGame({ batch: 50 }), { seed: 0, maxSteps: 4 }).state.cloud.length // => 400
 * ```
 */
export function chaosGame(options: ChaosGameOptions = {}): Algorithm<ChaosGameState, IfsPath> {
  const ifs = options.ifs ?? SIERPINSKI
  const batch = options.batch ?? 100
  const burn = options.burn ?? 20
  const keep = options.keep ?? 200_000
  positiveInt('chaosGame', 'batch', batch)
  positiveInt('chaosGame', 'keep', keep)
  const p = ifsProbabilities(ifs)
  const cumulative = p.map((_, i) => p.slice(0, i + 1).reduce((a, b) => a + b, 0))
  const pick = (u: number): number => {
    const i = cumulative.findIndex((c) => u < c)
    return i < 0 ? cumulative.length - 1 : i
  }
  return {
    name: 'dynamics/fractals/chaos-game',
    config: { ifs: ifs.name, maps: mapRows(ifs), probabilities: p, batch, burn, keep },
    schema: contractSchema(chaosGameContract),
    init: (key) => {
      const u = draws(key, burn + 2)
      let x = u[burn] ?? 0
      let y = u[burn + 1] ?? 0
      for (let i = 0; i < burn; i++) [x, y] = applyAffine(ifs.maps[pick(u[i] ?? 0)] as PlanarAffineMap, x, y)
      return { point: Float64Array.of(x, y), cloud: new Float64Array(0), generated: 0 }
    },
    step: (s, ctx) => {
      const u = draws(foldIn(ctx.key, 0), batch)
      const room = Math.max(0, Math.min(batch, keep - s.cloud.length / 2))
      const cloud = new Float64Array(s.cloud.length + 2 * room)
      cloud.set(s.cloud)
      let x = s.point[0] ?? 0
      let y = s.point[1] ?? 0
      for (let i = 0; i < batch; i++) {
        ;[x, y] = applyAffine(ifs.maps[pick(u[i] ?? 0)] as PlanarAffineMap, x, y)
        if (i < room) {
          cloud[s.cloud.length + 2 * i] = x
          cloud[s.cloud.length + 2 * i + 1] = y
        }
      }
      return { point: Float64Array.of(x, y), cloud, generated: s.generated + batch }
    },
    observe: (s) => ({
      metrics: {
        'points/count': s.generated,
        'ifs/dimension': safeDimension(ifs),
      },
      tensors: {
        'points/cloud': { role: 'state', value: s.cloud, shape: [s.cloud.length / 2, 2] },
        'ifs/maps': { role: 'param', value: mapRows(ifs), shape: [ifs.maps.length, 6] },
      },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as ChaosGameState,
  }
}

function safeDimension(ifs: Ifs): number {
  try {
    return similarityDimension(ifs)
  } catch {
    return Number.NaN
  }
}

/** Options of {@link ifsDeterministic}. */
export interface IfsDeterministicOptions {
  /** The system (default Sierpinski's triangle). */
  readonly ifs?: Ifs
  /** The window the pixels cover, `[xmin, xmax, ymin, ymax]` (default the unit square, which holds the defaults). */
  readonly box?: readonly [number, number, number, number]
  /** Pixels along each side (default 256). */
  readonly resolution?: number
}

/** The state of {@link ifsDeterministic}. */
export interface IfsDeterministicState {
  /** Lit pixels, row-major with row 0 at the top (`ymax`). */
  readonly mask: Uint8Array
  /** The number of lit pixels. */
  readonly lit: number
  /** Pixels that changed in the last step. */
  readonly changed: number
  readonly iteration: number
}

/** The paths of {@link ifsDeterministic}. */
export type IfsDeterministicPath = 'set/mask' | 'set/pixels' | 'set/changed' | 'ifs/dimension'

/** The log contract of {@link ifsDeterministic}. */
export const ifsDeterministicContract: Contract<IfsDeterministicState, IfsDeterministicPath> = defineContract({
  'set/mask': tensor({ role: 'state', shape: ['resolution', 'resolution'], description: '1 for a lit pixel' }),
  'set/pixels': scalar({ description: 'lit pixels' }),
  'set/changed': scalar({ description: 'pixels that changed in the step' }),
  'ifs/dimension': scalar({ description: 'the similarity dimension of the system' }),
})

/**
 * The deterministic iteration of a system of maps on a pixel grid, as a traced algorithm: start with every pixel lit (a
 * square of the plane) and at each step map the centre of every lit pixel by every map, lighting the pixel each image
 * falls in. The lit set converges to the attractor at the grid's resolution whatever the start (the collage theorem), in
 * a number of steps that grows with the log of the resolution; it stops when a step changes nothing. The state serialises.
 *
 * @param options - The system, the window and the resolution.
 * @returns The algorithm.
 * @throws OmaiError with code `value` for a resolution below 2 or an empty window.
 * @concept iterated-function-systems
 * @example
 * ```ts
 * import { fractals } from 'omai-engine/dynamics'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const s = runAlgorithm(fractals.ifsDeterministic({ resolution: 8 }), { seed: 0, maxSteps: 30 })
 * [s.status, s.state.changed] // => ['converged', 0]
 * ```
 */
export function ifsDeterministic(
  options: IfsDeterministicOptions = {},
): Algorithm<IfsDeterministicState, IfsDeterministicPath> {
  const ifs = options.ifs ?? SIERPINSKI
  const [x0, x1, y0, y1] = options.box ?? [0, 1, 0, 1]
  const N = options.resolution ?? 256
  if (!(Number.isInteger(N) && N >= 2))
    throw new OmaiError('value', `ifsDeterministic: resolution must be an integer of at least 2, got ${N}`)
  if (!(x1 > x0 && y1 > y0)) throw new OmaiError('value', 'ifsDeterministic: the window is empty')
  const cx = (c: number): number => x0 + ((c + 0.5) / N) * (x1 - x0)
  const cy = (r: number): number => y1 - ((r + 0.5) / N) * (y1 - y0)
  return {
    name: 'dynamics/fractals/ifs-deterministic',
    config: { ifs: ifs.name, maps: mapRows(ifs), box: [x0, x1, y0, y1], resolution: N },
    schema: contractSchema(ifsDeterministicContract),
    init: () => ({ mask: new Uint8Array(N * N).fill(1), lit: N * N, changed: 0, iteration: 0 }),
    step: (s) => {
      const mask = new Uint8Array(N * N)
      for (let r = 0; r < N; r++)
        for (let c = 0; c < N; c++) {
          if (s.mask[r * N + c] === 0) continue
          for (const m of ifs.maps) {
            const [x, y] = applyAffine(m, cx(c), cy(r))
            const col = Math.floor(((x - x0) / (x1 - x0)) * N)
            const row = Math.floor(((y1 - y) / (y1 - y0)) * N)
            if (col >= 0 && col < N && row >= 0 && row < N) mask[row * N + col] = 1
          }
        }
      let lit = 0
      let changed = 0
      for (let i = 0; i < mask.length; i++) {
        lit += mask[i] ?? 0
        if (mask[i] !== s.mask[i]) changed++
      }
      return { mask, lit, changed, iteration: s.iteration + 1 }
    },
    done: (s) => (s.iteration > 0 && s.changed === 0 ? 'converged' : undefined),
    observe: (s) => ({
      metrics: { 'set/pixels': s.lit, 'set/changed': s.changed, 'ifs/dimension': safeDimension(ifs) },
      tensors: { 'set/mask': { role: 'state', value: s.mask, shape: [N, N] } },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as IfsDeterministicState,
  }
}
