/**
 * The reference of the covariance atoms: the posterior of three coordinates (the three entries of a 2 by 2 precision matrix, or two precisions and a
 * correlation) on a grid that follows the mass, from a log density in which everything else (the means, the random effects) has been
 * integrated out in closed form. Three coordinates are all a 2 by 2 covariance has, so the posterior is a grid, exact to the grid's
 * resolution and independent of the rule under test: the marginals of each coordinate (mean, sd, density), the expectation of any function of
 * the three, and a Gaussian mixture's density over the grid (the marginal of a coordinate that is Gaussian given the three).
 */

import { lgamma } from '../../../numerics/special/index.ts'
import { item } from '../../../tensor/index.ts'
import { type ExactMarginal, gaussianMixtureMarginal } from './marginal.ts'

/** The box of one coordinate: where the grid starts and the limits it must stay inside. */
export interface CoordinateBox {
  readonly lo: number
  readonly hi: number
  /** The coordinate's own limits (positive, or a correlation in (-1, 1)). */
  readonly min: number
  readonly max: number
}

/** The grid posterior of three coordinates. */
export interface Posterior3 {
  /** The marginal of coordinate `i`: moments and density. */
  marginal(i: 0 | 1 | 2): ExactMarginal
  /** The posterior expectation of a function of the three. */
  expect(f: (a: number, b: number, c: number) => number): number
  /** The density at `x` of a variable that is Gaussian (mean, variance) given the three: the mixture over the grid. */
  mixture(
    given: (a: number, b: number, c: number) => { readonly mean: number; readonly variance: number },
  ): ExactMarginal
  /** log of the integral of the (unnormalised) density. */
  readonly logEvidence: number
}

interface Node {
  readonly x: readonly [number, number, number]
  readonly w: number
}

const GRIDS = [34, 52, 70]

/**
 * The grid posterior of `exp(logDensity)` over three coordinates, found by three passes: each pass puts a midpoint grid on the box and
 * the next box is the mean plus and minus seven standard deviations of the pass before it, kept inside the coordinates' limits.
 *
 * @param boxes - The starting box of each coordinate, and its limits.
 * @param logDensity - log of the posterior density up to a constant (`-Infinity` outside the support, for example where a matrix is not positive definite).
 * @returns The posterior (computed when first read from).
 */
export function posterior3(
  boxes: readonly [CoordinateBox, CoordinateBox, CoordinateBox],
  logDensity: (a: number, b: number, c: number) => number,
): Posterior3 {
  // The grid is computed when something is first read from it, so a problem made for a calibration seed (which never reads its
  // reference) does not pay for it.
  let state: Solved | undefined
  const solved = (): Solved => {
    state ??= solve(boxes, logDensity)
    return state
  }
  const marginal = (i: 0 | 1 | 2): ExactMarginal => ({
    get mean() {
      return solved().marginals[i]?.mean ?? Number.NaN
    },
    get sd() {
      return solved().marginals[i]?.sd ?? Number.NaN
    },
    pdf: (x) => solved().density[i]?.(x) ?? 0,
    get range() {
      const m = solved().marginals[i] as Solved['marginals'][number]
      const b = boxes[i] as CoordinateBox
      return [Math.max(b.min, m.mean - 8 * m.sd), Math.min(b.max, m.mean + 8 * m.sd)] as const
    },
  })
  return {
    get logEvidence() {
      return solved().logZ
    },
    marginal,
    expect: (f) => solved().nodes.reduce((acc, n) => acc + n.w * f(n.x[0], n.x[1], n.x[2]), 0),
    mixture: (given) => {
      let mix: ExactMarginal | undefined
      const built = (): ExactMarginal => {
        if (mix !== undefined) return mix
        // The nodes are pruned to the mass that matters (the heaviest, to all but 1e-6 of it) before the conditionals are asked for.
        const sorted = [...solved().nodes].sort((p, q) => q.w - p.w)
        const kept: Node[] = []
        let mass = 0
        for (const node of sorted) {
          kept.push(node)
          mass += node.w
          if (mass > 1 - 1e-6 || kept.length >= 6000) break
        }
        mix = gaussianMixtureMarginal(
          kept.map((node) => ({ weight: node.w / mass, ...given(node.x[0], node.x[1], node.x[2]) })),
          (mean, sd) => [mean - 8 * sd, mean + 8 * sd],
        )
        return mix
      }
      return {
        get mean() {
          return built().mean
        },
        get sd() {
          return built().sd
        },
        pdf: (x) => built().pdf(x),
        get range() {
          return built().range as readonly [number, number]
        },
      }
    },
  }
}

interface Solved {
  readonly logZ: number
  readonly nodes: readonly Node[]
  readonly marginals: readonly { x: number[]; p: number[]; mean: number; sd: number }[]
  readonly density: readonly ((x: number) => number)[]
}

/** Three passes of a midpoint grid, each over the box the pass before it found. */
function solve(
  boxes: readonly [CoordinateBox, CoordinateBox, CoordinateBox],
  logDensity: (a: number, b: number, c: number) => number,
): Solved {
  let box = boxes.map((b) => [b.lo, b.hi] as [number, number])
  let nodes: Node[] = []
  let logZ = 0
  let marginals: { x: number[]; p: number[]; mean: number; sd: number }[] = []
  for (const n of GRIDS) {
    const steps = box.map(([lo, hi]) => (hi - lo) / n)
    const axes = box.map(([lo], k) => Array.from({ length: n }, (_, i) => lo + (i + 0.5) * (steps[k] as number)))
    const raw = new Float64Array(n * n * n)
    let top = Number.NEGATIVE_INFINITY
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++)
        for (let k = 0; k < n; k++) {
          const v = logDensity(
            (axes[0] as number[])[i] as number,
            (axes[1] as number[])[j] as number,
            (axes[2] as number[])[k] as number,
          )
          raw[(i * n + j) * n + k] = v
          if (v > top) top = v
        }
    let total = 0
    for (let q = 0; q < raw.length; q++) {
      const e = Math.exp((raw[q] as number) - top)
      raw[q] = e
      total += e
    }
    const cell = (steps[0] as number) * (steps[1] as number) * (steps[2] as number)
    logZ = top + Math.log(total * cell)
    const next: Node[] = []
    const mass = [new Float64Array(n), new Float64Array(n), new Float64Array(n)] as const
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++)
        for (let k = 0; k < n; k++) {
          const w = (raw[(i * n + j) * n + k] as number) / total
          if (w === 0) continue
          mass[0][i] = (mass[0][i] as number) + w
          mass[1][j] = (mass[1][j] as number) + w
          mass[2][k] = (mass[2][k] as number) + w
          if (w > 1e-12)
            next.push({
              x: [
                (axes[0] as number[])[i] as number,
                (axes[1] as number[])[j] as number,
                (axes[2] as number[])[k] as number,
              ],
              w,
            })
        }
    nodes = next
    marginals = mass.map((m, c) => {
      const x = axes[c] as number[]
      let mean = 0
      let second = 0
      for (let i = 0; i < n; i++) {
        mean += (m[i] as number) * (x[i] as number)
        second += (m[i] as number) * (x[i] as number) ** 2
      }
      return {
        x,
        p: Array.from(m, (v) => v / (steps[c] as number)),
        mean,
        sd: Math.sqrt(Math.max(second - mean * mean, 1e-300)),
      }
    })
    box = marginals.map((m, c) => {
      const b = boxes[c] as CoordinateBox
      return [Math.max(b.min, m.mean - 7 * m.sd), Math.min(b.max, m.mean + 7 * m.sd)] as [number, number]
    })
  }
  const density = marginals.map((m) => {
    const h = (m.x[1] as number) - (m.x[0] as number)
    return (x: number): number => {
      const t = (x - (m.x[0] as number)) / h
      const i = Math.floor(t)
      if (i < 0 || i >= m.x.length - 1) return 0
      const f = t - i
      return (m.p[i] as number) * (1 - f) + (m.p[i + 1] as number) * f
    }
  })
  return { logZ, nodes, marginals, density }
}

/** `log` of the Wishart density `W(L; nu, S)` of a 2 by 2 matrix given as its three entries (`-Infinity` outside the positive definite matrices). */
export function logWishart2(
  nu: number,
  scale: readonly [number, number, number, number],
): (l11: number, l12: number, l22: number) => number {
  const det = scale[0] * scale[3] - scale[1] * scale[2]
  const inverse = [scale[3] / det, -scale[1] / det, -scale[2] / det, scale[0] / det] as const
  // log of 2^(nu d / 2) |S|^(nu / 2) Gamma_2(nu / 2) for d = 2
  const logNorm =
    nu * Math.LN2 +
    0.5 * nu * Math.log(det) +
    0.5 * Math.log(Math.PI) +
    item(lgamma(nu / 2)) +
    item(lgamma((nu - 1) / 2))
  return (a, b, c) => {
    const d = a * c - b * b
    if (!(a > 0 && d > 0)) return Number.NEGATIVE_INFINITY
    const trace = inverse[0] * a + 2 * inverse[1] * b + inverse[3] * c
    return 0.5 * (nu - 3) * Math.log(d) - 0.5 * trace - logNorm
  }
}
