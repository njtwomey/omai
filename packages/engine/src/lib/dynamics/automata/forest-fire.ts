/**
 * The Drossel-Schwabl forest-fire model: trees grow on empty sites at rate p, lightning strikes a tree at rate f, and a
 * struck tree burns its whole connected cluster of trees at once. When lightning is rare against growth (f much less
 * than p) the fire sizes follow a power law up to the scale 1/(f/p), a second model of self-organised criticality.
 * The model as a traced algorithm, with the one exact result there is (the density of a forest that never burns).
 */

import { MdmlError } from '../../base/index.ts'
import { type Algorithm, type Contract, contractSchema, defineContract, scalar, tensor } from '../../trace/index.ts'
import { codec, draws, mod, positiveInt, unit } from './_core.ts'

/** Options of {@link forestFire}. */
export interface ForestFireOptions {
  /** Columns and rows (default 64 by 64). */
  readonly width?: number
  readonly height?: number
  /** Chance an empty site grows a tree each step (default 0.05). */
  readonly growth?: number
  /** Chance a tree is struck by lightning each step (default 0.0005). */
  readonly lightning?: number
  /** Whether the forest is a torus (default true). */
  readonly wrap?: boolean
  /** The share of sites that are trees at the start (default 0). */
  readonly density?: number
  /** The starting forest itself, row-major with 1 for a tree (replaces `density`). */
  readonly cells?: Uint8Array
}

/** The state of {@link forestFire}. */
export interface ForestFireState {
  /** 1 where a tree stands, 0 where the ground is empty. */
  readonly cells: Uint8Array
  /** Trees burnt in the last step (all fires together). */
  readonly fireSize: number
  /** Fires started in the last step. */
  readonly fires: number
}

/** The paths of {@link forestFire}. */
export type ForestFirePath = 'forest/board' | 'forest/density' | 'forest/fireSize' | 'forest/fires'

/** The log contract of {@link forestFire}. */
export const forestFireContract: Contract<ForestFireState, ForestFirePath> = defineContract({
  'forest/board': tensor({ role: 'state', shape: ['height', 'width'], description: '1 where a tree stands' }),
  'forest/density': scalar({ description: 'the share of sites that are trees' }),
  'forest/fireSize': scalar({ description: 'trees burnt in the last step' }),
  'forest/fires': scalar({ description: 'fires started in the last step' }),
})

/**
 * The expected share of sites that are trees after `t` steps of a forest that starts empty and never burns (no
 * lightning): each site is a tree unless it has failed to grow `t` times, so `1 - (1 - p)^t`.
 *
 * @param growth - The growth chance p per step.
 * @param t - Steps.
 * @returns The expected tree density.
 * @concept forest-fire
 * @example
 * ```ts
 * import { automata } from 'mdml-engine/dynamics'
 * automata.treeDensityWithoutFire(0.5, 3) // => 0.875
 * ```
 */
export function treeDensityWithoutFire(growth: number, t: number): number {
  return 1 - (1 - growth) ** t
}

/**
 * The Drossel-Schwabl forest fire as a traced algorithm. Each step the ground grows trees (each empty site with chance
 * `growth`), then lightning picks trees (each with chance `lightning`) and every picked tree still standing burns its
 * whole cluster of trees (neighbours on the four sides) in that step. `forest/fireSize` is the trees burnt in the step,
 * the quantity whose distribution is the power law. The state serialises.
 *
 * @param options - Size, growth, lightning, wrap and starting density.
 * @returns The algorithm.
 * @throws MdmlError with code `value` for a bad size or a chance outside [0, 1].
 * @concept forest-fire
 * @example
 * ```ts
 * import { automata } from 'mdml-engine/dynamics'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * // With certain growth and no lightning the forest fills at once.
 * const s = runAlgorithm(automata.forestFire({ width: 4, height: 4, growth: 1, lightning: 0 }), { seed: 0, maxSteps: 1 }).state
 * s.cells.reduce((a, b) => a + b, 0) // => 16
 * ```
 */
export function forestFire(options: ForestFireOptions = {}): Algorithm<ForestFireState, ForestFirePath> {
  const width = options.width ?? 64
  const height = options.height ?? 64
  const p = options.growth ?? 0.05
  const f = options.lightning ?? 0.0005
  const wrap = options.wrap ?? true
  const density = options.density ?? 0
  const given = options.cells
  positiveInt('forestFire', 'width', width)
  positiveInt('forestFire', 'height', height)
  unit('forestFire', 'growth', p)
  unit('forestFire', 'lightning', f)
  unit('forestFire', 'density', density)
  if (given !== undefined && given.length !== width * height)
    throw new MdmlError('shape', 'forestFire: the starting cells must be width x height')
  const n = width * height
  const neighbours = (i: number): number[] => {
    const r = Math.floor(i / width)
    const c = i % width
    const out: number[] = []
    if (wrap) {
      out.push(mod(r - 1, height) * width + c, mod(r + 1, height) * width + c)
      out.push(r * width + mod(c - 1, width), r * width + mod(c + 1, width))
    } else {
      if (r > 0) out.push(i - width)
      if (r < height - 1) out.push(i + width)
      if (c > 0) out.push(i - 1)
      if (c < width - 1) out.push(i + 1)
    }
    return out
  }
  return {
    name: 'dynamics/automata/forest-fire',
    config: {
      width,
      height,
      growth: p,
      lightning: f,
      wrap,
      density,
      cells: given === undefined ? null : Array.from(given),
    },
    schema: contractSchema(forestFireContract),
    init: (key) => {
      if (given !== undefined) return { cells: Uint8Array.from(given), fireSize: 0, fires: 0 }
      const u = draws(key, n)
      return { cells: Uint8Array.from({ length: n }, (_, i) => ((u[i] ?? 1) < density ? 1 : 0)), fireSize: 0, fires: 0 }
    },
    step: (s, ctx) => {
      const u = draws(ctx.key, 2 * n)
      const cells = Uint8Array.from(s.cells)
      for (let i = 0; i < n; i++) if (cells[i] === 0 && (u[i] ?? 1) < p) cells[i] = 1
      let fireSize = 0
      let fires = 0
      for (let i = 0; i < n; i++) {
        if (cells[i] !== 1 || (u[n + i] ?? 1) >= f) continue
        fires++
        const stack = [i]
        cells[i] = 0
        while (stack.length > 0) {
          const j = stack.pop() as number
          fireSize++
          for (const k of neighbours(j))
            if (cells[k] === 1) {
              cells[k] = 0
              stack.push(k)
            }
        }
      }
      return { cells, fireSize, fires }
    },
    observe: (s) => {
      let trees = 0
      for (const c of s.cells) trees += c
      return {
        metrics: { 'forest/density': trees / n, 'forest/fireSize': s.fireSize, 'forest/fires': s.fires },
        tensors: { 'forest/board': { role: 'state', value: s.cells, shape: [height, width] } },
      }
    },
    ...codec<ForestFireState>(),
  }
}
