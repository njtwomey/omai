/**
 * Lattice gases: particles hopping on a lattice, at most one per direction per site, colliding by a handful of rules
 * that conserve their number and their momentum, and from those microscopic rules a fluid emerging when the occupancy
 * is coarse-grained into a density and a velocity. HPP on the square lattice (four directions; a head-on pair turns
 * through a right angle) and FHP-I on the triangular lattice (six directions; head-on pairs turn either way, and the
 * symmetric triples flip), which has the isotropy that HPP lacks. Solid sites bounce particles back. The collisions,
 * the coarse-graining and the gas as a traced algorithm that logs the occupancy and the conserved mass and momentum.
 */

import { MdmlError } from '../../base/index.ts'
import { foldIn } from '../../random/index.ts'
import {
  type Algorithm,
  type Contract,
  contractSchema,
  defineContract,
  derived,
  scalar,
  tensor,
} from '../../trace/index.ts'
import { codec, draws, mod, positiveInt, unit } from './_core.ts'

/** The lattice: `hpp` has 4 directions (east, north, west, south), `fhp` 6 (every 60 degrees from east). */
export type LatticeGasModel = 'hpp' | 'fhp'

const DIRECTIONS: Readonly<Record<LatticeGasModel, number>> = { hpp: 4, fhp: 6 }

/** The unit momentum of each direction of a model, `[x, y]` with y up. */
export function gasVelocities(model: LatticeGasModel): [number, number][] {
  const n = DIRECTIONS[model]
  return Array.from({ length: n }, (_, d) => [Math.cos((2 * Math.PI * d) / n), Math.sin((2 * Math.PI * d) / n)])
}

/** Options of {@link latticeGas}. */
export interface LatticeGasOptions {
  /** The lattice (default `'hpp'`). */
  readonly model?: LatticeGasModel
  /** Columns and rows, periodic (default 64 by 64; `fhp` needs an even number of rows). */
  readonly width?: number
  readonly height?: number
  /** The chance a direction at a site is occupied at the start (default 0.2). */
  readonly density?: number
  /** A push on the start: occupancy of direction `d` becomes `density (1 + 2 (bx ex + by ey))`, clipped to [0, 1] (default none). */
  readonly bias?: readonly [number, number]
  /** Solid sites (nonzero) that reflect particles straight back (row-major; default none). */
  readonly solid?: Uint8Array
  /** The block side for the coarse-grained fields (default 8). */
  readonly block?: number
}

/** The state of {@link latticeGas}: the bits of each site are its directions, bit `d` set when a particle moves in `d`. */
export interface LatticeGasState {
  readonly cells: Uint8Array
  readonly time: number
}

/** The paths of {@link latticeGas}. */
export type LatticeGasPath =
  | 'lattice/occupancy'
  | 'lattice/mass'
  | 'lattice/momentum'
  | 'field/density'
  | 'field/velocity'
  | 'field/vorticity'

const popcount = (x: number): number => {
  let n = 0
  for (let v = x; v > 0; v >>= 1) n += v & 1
  return n
}

/** The post-collision state of every pre-collision state (for the second chirality too in the FHP case). */
function collisionTable(model: LatticeGasModel, chirality: 0 | 1): Uint8Array {
  const n = DIRECTIONS[model]
  const table = new Uint8Array(1 << n)
  for (let s = 0; s < table.length; s++) table[s] = s
  if (model === 'hpp') {
    table[0b0101] = 0b1010
    table[0b1010] = 0b0101
    return table
  }
  // FHP-I: a lone head-on pair {d, d+3} turns to {d+1, d+4} or {d-1, d+2}; the triples {0,2,4} and {1,3,5} swap.
  for (let d = 0; d < 3; d++) {
    const pair = (1 << d) | (1 << (d + 3))
    const turn = chirality === 0 ? 1 : 5
    table[pair] = (1 << ((d + turn) % 6)) | (1 << ((d + turn + 3) % 6))
  }
  table[0b010101] = 0b101010
  table[0b101010] = 0b010101
  return table
}

const HPP_TABLE = collisionTable('hpp', 0)
const FHP_TABLES = [collisionTable('fhp', 0), collisionTable('fhp', 1)] as const

/**
 * Where a particle moving in direction `d` from site `(r, c)` lands, on a periodic lattice. For `fhp` the rows are
 * staggered (odd rows shifted half a cell right), so the diagonal neighbours depend on the row.
 *
 * @param model - The lattice.
 * @param r - Row.
 * @param c - Column.
 * @param d - Direction.
 * @param width - Columns.
 * @param height - Rows.
 * @returns The row and column of the neighbour.
 * @concept lattice-gas
 * @example
 * ```ts
 * import { automata } from 'mdml-engine/dynamics'
 * [automata.gasNeighbour('hpp', 2, 3, 1, 8, 8), automata.gasNeighbour('fhp', 1, 3, 1, 8, 8)] // => [[1, 3], [0, 4]]
 * ```
 */
export function gasNeighbour(
  model: LatticeGasModel,
  r: number,
  c: number,
  d: number,
  width: number,
  height: number,
): [number, number] {
  if (model === 'hpp') {
    const dr = [0, -1, 0, 1][d] ?? 0
    const dc = [1, 0, -1, 0][d] ?? 0
    return [mod(r + dr, height), mod(c + dc, width)]
  }
  const odd = r % 2 === 1 ? 1 : 0
  // East, north-east, north-west, west, south-west, south-east.
  const dr = [0, -1, -1, 0, 1, 1][d] ?? 0
  const dc = [1, odd, odd - 1, -1, odd - 1, odd][d] ?? 0
  return [mod(r + dr, height), mod(c + dc, width)]
}

/**
 * The mass (particles) and momentum of a gas.
 *
 * @param cells - The occupancy bits.
 * @param model - The lattice.
 * @returns The number of particles and the total momentum `[px, py]`.
 * @concept lattice-gas
 * @example
 * ```ts
 * import { automata } from 'mdml-engine/dynamics'
 * const g = automata.gasMassMomentum(Uint8Array.of(0b0011, 0b0100), 'hpp') // east+north, west
 * [g.mass, g.momentum.map(Math.round)] // => [3, [0, 1]]
 * ```
 */
export function gasMassMomentum(
  cells: Uint8Array,
  model: LatticeGasModel,
): { mass: number; momentum: [number, number] } {
  const e = gasVelocities(model)
  let mass = 0
  let px = 0
  let py = 0
  for (const s of cells) {
    if (s === 0) continue
    mass += popcount(s)
    for (let d = 0; d < e.length; d++)
      if (s & (1 << d)) {
        px += e[d]?.[0] ?? 0
        py += e[d]?.[1] ?? 0
      }
  }
  return { mass, momentum: [px, py] }
}

/**
 * Coarse-grain a gas: in each block of `block` by `block` sites, the mean number of particles per site (density) and the
 * mean momentum per particle (velocity, 0 in an empty block). The fluid fields the microscopic rules give rise to.
 *
 * @param cells - The occupancy bits.
 * @param width - Columns.
 * @param height - Rows.
 * @param block - The block side (the sizes must divide).
 * @param model - The lattice.
 * @returns Density and velocity per block, row-major `[height / block, width / block]`.
 * @throws MdmlError with code `value` if the block does not divide the lattice.
 * @concept lattice-gas
 * @example
 * ```ts
 * import { automata } from 'mdml-engine/dynamics'
 * const f = automata.coarseGrainGas(Uint8Array.of(0b0001, 0b0001, 0, 0), 2, 2, 2, 'hpp')
 * [Array.from(f.density), Array.from(f.vx), Array.from(f.vy)] // => [[0.5], [1], [0]]
 * ```
 */
export function coarseGrainGas(
  cells: Uint8Array,
  width: number,
  height: number,
  block: number,
  model: LatticeGasModel,
): { density: Float64Array; vx: Float64Array; vy: Float64Array; columns: number; rows: number } {
  if (!(Number.isInteger(block) && block >= 1 && width % block === 0 && height % block === 0))
    throw new MdmlError('value', `coarseGrainGas: block ${block} must divide ${width} by ${height}`)
  const columns = width / block
  const rows = height / block
  const e = gasVelocities(model)
  const density = new Float64Array(columns * rows)
  const vx = new Float64Array(columns * rows)
  const vy = new Float64Array(columns * rows)
  for (let r = 0; r < height; r++)
    for (let c = 0; c < width; c++) {
      const s = cells[r * width + c] ?? 0
      if (s === 0) continue
      const b = Math.floor(r / block) * columns + Math.floor(c / block)
      density[b] = (density[b] ?? 0) + popcount(s)
      for (let d = 0; d < e.length; d++)
        if (s & (1 << d)) {
          vx[b] = (vx[b] ?? 0) + (e[d]?.[0] ?? 0)
          vy[b] = (vy[b] ?? 0) + (e[d]?.[1] ?? 0)
        }
    }
  for (let b = 0; b < density.length; b++) {
    const n = density[b] ?? 0
    vx[b] = n === 0 ? 0 : (vx[b] ?? 0) / n
    vy[b] = n === 0 ? 0 : (vy[b] ?? 0) / n
    density[b] = n / (block * block)
  }
  return { density, vx, vy, columns, rows }
}

/**
 * The log contract of {@link latticeGas}. `field/vorticity` is derived on demand from the coarse-grained velocity.
 *
 * @param model - The lattice.
 * @param width - Columns.
 * @param height - Rows.
 * @param block - The coarse-graining block.
 * @returns The contract.
 * @example
 * ```ts
 * import { automata } from 'mdml-engine/dynamics'
 * Object.keys(automata.latticeGasContract('hpp', 8, 8, 4)).length // => 6
 * ```
 */
export function latticeGasContract(
  model: LatticeGasModel,
  width: number,
  height: number,
  block: number,
): Contract<LatticeGasState, LatticeGasPath> {
  return defineContract({
    'lattice/occupancy': tensor({
      role: 'state',
      shape: ['height', 'width'],
      description: 'the direction bits of every site',
    }),
    'lattice/mass': scalar({ description: 'particles (conserved)' }),
    'lattice/momentum': tensor({ role: 'state', shape: [2], description: 'total momentum (conserved without solids)' }),
    'field/density': tensor({
      role: 'state',
      shape: ['rows', 'columns'],
      description: 'particles per site in each block',
    }),
    'field/velocity': tensor({
      role: 'state',
      shape: ['rows', 'columns', 2],
      description: 'mean velocity of the particles in each block',
    }),
    'field/vorticity': derived<LatticeGasState>({
      description: 'the curl of the coarse-grained velocity (central differences, blocks on a torus)',
      cost: 'cheap',
      compute: (s) => {
        const f = coarseGrainGas(s.cells, width, height, block, model)
        return Array.from(curl2d(f.vx, f.vy, f.columns, f.rows))
      },
    }),
  })
}

/**
 * The curl `d uy / dx - d ux / dy` of a velocity field on a periodic grid, by central differences with unit spacing.
 *
 * @param ux - The x component, row-major (row 0 at the top; y points up, so rows increase downward).
 * @param uy - The y component.
 * @param width - Columns.
 * @param height - Rows.
 * @returns The vorticity per site.
 * @concept lattice-gas
 * @example
 * ```ts
 * import { automata } from 'mdml-engine/dynamics'
 * // Solid-body rotation u = (-y, x) about the middle of a 5 x 5 patch has vorticity 2 (rows run down, so y = -row).
 * const ux = Float64Array.from({ length: 25 }, (_, i) => Math.floor(i / 5) - 2)
 * const uy = Float64Array.from({ length: 25 }, (_, i) => (i % 5) - 2)
 * automata.curl2d(ux, uy, 5, 5)[12] // => 2
 * ```
 */
export function curl2d(ux: ArrayLike<number>, uy: ArrayLike<number>, width: number, height: number): Float64Array {
  const out = new Float64Array(width * height)
  for (let r = 0; r < height; r++)
    for (let c = 0; c < width; c++) {
      const at = (rr: number, cc: number): number => mod(rr, height) * width + mod(cc, width)
      const duyDx = ((uy[at(r, c + 1)] ?? 0) - (uy[at(r, c - 1)] ?? 0)) / 2
      // y points up while rows run down, so d/dy = -d/drow.
      const duxDy = -((ux[at(r + 1, c)] ?? 0) - (ux[at(r - 1, c)] ?? 0)) / 2
      out[r * width + c] = duyDx - duxDy
    }
  return out
}

/**
 * A lattice gas as a traced algorithm: each step collides every site by the model's rule and then streams every
 * particle one site in its direction (periodic), with solid sites reflecting particles straight back. The state is the
 * occupancy bits and serialises. Mass is conserved always, momentum as well when there are no solids; both are logged.
 * The FHP collisions that can turn either way choose with the step's key.
 *
 * @param options - The lattice, size, density, bias, solids and coarse-graining block.
 * @returns The algorithm.
 * @throws MdmlError with code `value` for a bad size, density, solid mask, a block that does not divide the lattice, or odd rows for `fhp`.
 * @concept lattice-gas
 * @example
 * ```ts
 * import { automata } from 'mdml-engine/dynamics'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const alg = automata.latticeGas({ width: 16, height: 16, block: 4 })
 * const [m0, m1] = [0, 50].map((t) => automata.gasMassMomentum(runAlgorithm(alg, { seed: 1, maxSteps: t }).state.cells, 'hpp').mass)
 * m1 === m0 // => true
 * ```
 */
export function latticeGas(options: LatticeGasOptions = {}): Algorithm<LatticeGasState, LatticeGasPath> {
  const model = options.model ?? 'hpp'
  const width = options.width ?? 64
  const height = options.height ?? 64
  const density = options.density ?? 0.2
  const block = options.block ?? 8
  const bias = options.bias ?? [0, 0]
  const solid = options.solid
  positiveInt('latticeGas', 'width', width)
  positiveInt('latticeGas', 'height', height)
  unit('latticeGas', 'density', density)
  if (model === 'fhp' && height % 2 !== 0)
    throw new MdmlError('value', 'latticeGas: the fhp lattice needs an even number of rows')
  if (solid !== undefined && solid.length !== width * height)
    throw new MdmlError('value', 'latticeGas: solid must be width x height')
  if (width % block !== 0 || height % block !== 0)
    throw new MdmlError('value', `latticeGas: block ${block} must divide ${width} by ${height}`)
  const n = DIRECTIONS[model]
  const e = gasVelocities(model)
  const sites = width * height
  // Where each direction of each site streams to.
  const target = Int32Array.from({ length: n * sites }, (_, i) => {
    const d = Math.floor(i / sites)
    const s = i % sites
    const [r, c] = gasNeighbour(model, Math.floor(s / width), s % width, d, width, height)
    return r * width + c
  })
  const reverse = (s: number): number => {
    let out = 0
    for (let d = 0; d < n; d++) if (s & (1 << d)) out |= 1 << ((d + n / 2) % n)
    return out
  }
  const bounce = Uint8Array.from({ length: 1 << n }, (_, s) => reverse(s))
  const contract = latticeGasContract(model, width, height, block)
  return {
    name: `dynamics/automata/lattice-gas-${model}`,
    config: {
      model,
      width,
      height,
      density,
      bias: [...bias],
      block,
      solids: solid === undefined ? 0 : solid.reduce((a, b) => a + (b ? 1 : 0), 0),
    },
    schema: contractSchema(contract),
    init: (key) => {
      const u = draws(key, n * sites)
      const cells = new Uint8Array(sites)
      for (let s = 0; s < sites; s++) {
        if (solid?.[s]) continue
        for (let d = 0; d < n; d++) {
          const push = (bias[0] * (e[d]?.[0] ?? 0) + bias[1] * (e[d]?.[1] ?? 0)) * 2
          const p = Math.min(1, Math.max(0, density * (1 + push)))
          if ((u[d * sites + s] ?? 1) < p) cells[s] = (cells[s] ?? 0) | (1 << d)
        }
      }
      return { cells, time: 0 }
    },
    step: (s, ctx) => {
      const flips = model === 'fhp' ? draws(foldIn(ctx.key, 0), sites) : undefined
      const next = new Uint8Array(sites)
      for (let i = 0; i < sites; i++) {
        const before = s.cells[i] ?? 0
        let after: number
        if (solid?.[i]) after = bounce[before] ?? 0
        else if (model === 'hpp') after = HPP_TABLE[before] ?? 0
        else after = (FHP_TABLES[(flips?.[i] ?? 0) < 0.5 ? 0 : 1] as Uint8Array)[before] ?? 0
        if (after === 0) continue
        for (let d = 0; d < n; d++)
          if (after & (1 << d))
            next[target[d * sites + i] as number] = (next[target[d * sites + i] as number] ?? 0) | (1 << d)
      }
      return { cells: next, time: s.time + 1 }
    },
    observe: (s) => {
      const { mass, momentum } = gasMassMomentum(s.cells, model)
      const f = coarseGrainGas(s.cells, width, height, block, model)
      const velocity = new Float64Array(2 * f.vx.length)
      f.vx.forEach((v, i) => {
        velocity[2 * i] = v
        velocity[2 * i + 1] = f.vy[i] ?? 0
      })
      return {
        metrics: { 'lattice/mass': mass },
        tensors: {
          'lattice/occupancy': { role: 'state', value: s.cells, shape: [height, width] },
          'lattice/momentum': { role: 'state', value: momentum, shape: [2] },
          'field/density': { role: 'state', value: f.density, shape: [f.rows, f.columns] },
          'field/velocity': { role: 'state', value: velocity, shape: [f.rows, f.columns, 2] },
        },
      }
    },
    ...codec<LatticeGasState>(),
  }
}
