// The latent-variable row against brute-force integrals that use none of its factorisation. (1) One output of two weights: the exact
// tilted distribution of (coordinate, weight 1, weight 2) on a dense grid of all three, the message to the coordinate and to the
// weights from its moments. (2) Several outputs, one or two coordinates: the coordinates on a dense grid and, at each node, the
// weights of each output updated by an explicit matrix solve (not the rank-one formulas of the rule); the mixture moments and the
// evidence by sums over the nodes. The operator's own limits (a parameter out of range, a vector of the wrong size) are checked too.
import { describe, expect, it } from 'vitest'
import { massOf, messageArg } from '../../core/args.ts'
import { ParameterError } from '../../core/errors.ts'
import { gaussian, gaussianEta } from '../../core/families.ts'
import { gaussianMv, gaussianMvEta } from '../../core/families-vector.ts'
import type { Arg, Params } from '../../core/types.ts'
import { spd } from '../../numerics/_dense.ts'
import { defaultOperators } from '../registry.ts'
import { gplvmRow } from './gplvm-row.ts'

const LOG_2PI = Math.log(2 * Math.PI)

interface Case {
  readonly coords: number
  readonly features: number
  readonly outputs: number
  readonly omega: readonly number[]
  readonly phase: readonly number[]
  readonly noise: number
  readonly y: readonly number[]
  /** The cavity of each coordinate (mean, variance), and of the weights: a mean and an `M x M` covariance block per output. */
  readonly x: readonly (readonly [number, number])[]
  readonly means: readonly (readonly number[])[]
  readonly blocks: readonly (readonly number[])[]
}

/** The cavities as arguments: the coordinates, then the weights of each output. */
function argsOf(c: Case): Arg[] {
  return [
    ...c.x.map(([m, v]) => messageArg(gaussian, gaussianEta(m, v))),
    ...c.means.map((mean, d) => messageArg(gaussianMv(c.features), gaussianMvEta(mean, c.blocks[d] as number[]))),
  ]
}

const paramsOf = (c: Case): Params => ({
  coords: c.coords,
  features: c.features,
  outputs: c.outputs,
  omega: [...c.omega],
  phase: [...c.phase],
  noise: c.noise,
  y: [...c.y],
})

const phiAt = (c: Case, x: readonly number[]): number[] =>
  Array.from({ length: c.features }, (_, m) => {
    let angle = c.phase[m] as number
    for (let i = 0; i < c.coords; i++) angle += (c.omega[i * c.features + m] as number) * (x[i] as number)
    return Math.cos(angle)
  })

const logNormal = (v: number, mean: number, variance: number): number =>
  -0.5 * (LOG_2PI + Math.log(variance) + (v - mean) ** 2 / variance)

/** The message a Gaussian moment match gives: its natural parameters less the cavity's. */
function scalarMessage(mean: number, variance: number, cavity: readonly [number, number]): number[] {
  const [cm, cv] = cavity
  return [mean / variance - cm / cv, 1 / variance - 1 / cv]
}

/** Compare two messages entry by entry, relative to the larger of one and the entry. */
function same(got: readonly number[], want: readonly number[], tolerance: number, what: string): void {
  expect(got.length, what).toBe(want.length)
  for (let i = 0; i < want.length; i++)
    expect(
      Math.abs((got[i] as number) - (want[i] as number)),
      `${what} entry ${i}: ${got[i]} against ${want[i]}`,
    ).toBeLessThan(tolerance * Math.max(1, Math.abs(want[i] as number)))
}

/** The reference by a dense grid over the coordinates, the weights updated at each node by an explicit solve. */
function referenceOverNodes(
  c: Case,
  points: number,
): {
  readonly logZ: number
  readonly mean: number[]
  readonly variance: number[]
  readonly weightMean: number[][]
  readonly weightCovariance: number[][]
} {
  const axes = c.x.map(([m, v]) =>
    Array.from({ length: points }, (_, k) => m + Math.sqrt(v) * (-8 + (16 * k) / (points - 1))),
  )
  const step = c.x.map(([, v]) => (16 * Math.sqrt(v)) / (points - 1))
  const nodes: number[][] = []
  const logs: number[] = []
  const total = points ** c.coords
  const updated: { mean: number[][]; covariance: number[][] }[] = []
  for (let k = 0; k < total; k++) {
    let rest = k
    const x: number[] = []
    for (let i = 0; i < c.coords; i++) {
      x.push((axes[i] as number[])[rest % points] as number)
      rest = Math.floor(rest / points)
    }
    const phi = phiAt(c, x)
    let lp = 0
    for (let i = 0; i < c.coords; i++)
      lp += logNormal(
        x[i] as number,
        (c.x[i] as readonly number[])[0] as number,
        (c.x[i] as readonly number[])[1] as number,
      )
    const post: { mean: number[]; covariance: number[] }[] = []
    for (let d = 0; d < c.outputs; d++) {
      const v = c.blocks[d] as readonly number[]
      const m = c.means[d] as readonly number[]
      // The weights' prior precision, the observation's precision phi phi' / noise^2, and the posterior by one solve.
      const prior = spd(v, c.features)
      if (prior === undefined) throw new Error('block not positive definite')
      const precision = prior.inverse()
      const jm = Array.from({ length: c.features }, (_, a) => {
        let s = 0
        for (let b = 0; b < c.features; b++) s += (precision[a * c.features + b] as number) * (m[b] as number)
        return s
      })
      const full = precision.map(
        (p, i) => p + ((phi[Math.floor(i / c.features)] as number) * (phi[i % c.features] as number)) / c.noise ** 2,
      )
      const rhs = jm.map((h, a) => h + ((phi[a] as number) * (c.y[d] as number)) / c.noise ** 2)
      const factor = spd(full, c.features)
      if (factor === undefined) throw new Error('posterior not positive definite')
      post.push({ mean: Array.from(factor.solve(rhs)), covariance: Array.from(factor.inverse()) })
      // The marginal likelihood of the output: a Gaussian of y with the predictive mean and variance.
      let mu = 0
      let quad = 0
      for (let a = 0; a < c.features; a++) {
        mu += (phi[a] as number) * (m[a] as number)
        for (let b = 0; b < c.features; b++)
          quad += (phi[a] as number) * (v[a * c.features + b] as number) * (phi[b] as number)
      }
      lp += logNormal(c.y[d] as number, mu, quad + c.noise ** 2)
    }
    nodes.push(x)
    logs.push(lp)
    updated.push({ mean: post.map((p) => p.mean), covariance: post.map((p) => p.covariance) })
  }
  const top = logs.reduce((a, b) => Math.max(a, b), Number.NEGATIVE_INFINITY)
  const w = logs.map((l) => Math.exp(l - top))
  const sum = w.reduce((a, b) => a + b, 0)
  const volume = step.reduce((a, b) => a * b, 1)
  const mean = Array.from(
    { length: c.coords },
    (_, i) => w.reduce((s, wk, k) => s + wk * ((nodes[k] as number[])[i] as number), 0) / sum,
  )
  const variance = Array.from(
    { length: c.coords },
    (_, i) =>
      w.reduce((s, wk, k) => s + wk * (((nodes[k] as number[])[i] as number) - (mean[i] as number)) ** 2, 0) / sum,
  )
  const weightMean: number[][] = []
  const weightCovariance: number[][] = []
  for (let d = 0; d < c.outputs; d++) {
    const m = Array.from(
      { length: c.features },
      (_, a) =>
        w.reduce((s, wk, k) => s + wk * (((updated[k] as { mean: number[][] }).mean[d] as number[])[a] as number), 0) /
        sum,
    )
    const cov = Array.from({ length: c.features * c.features }, (_, e) => {
      const a = Math.floor(e / c.features)
      const b = e % c.features
      let s = 0
      for (let k = 0; k < total; k++) {
        const u = updated[k] as { mean: number[][]; covariance: number[][] }
        s +=
          (w[k] as number) *
          (((u.covariance[d] as number[])[e] as number) +
            ((u.mean[d] as number[])[a] as number) * ((u.mean[d] as number[])[b] as number))
      }
      return s / sum - (m[a] as number) * (m[b] as number)
    })
    weightMean.push(m)
    weightCovariance.push(cov)
  }
  return { logZ: top + Math.log(sum * volume), mean, variance, weightMean, weightCovariance }
}

/** The messages and the evidence the operator gives, against the reference built from `referenceOverNodes`. */
function checkAgainstNodes(c: Case, points: number, tolerance: number): void {
  const args = argsOf(c)
  const params = paramsOf(c)
  const ref = referenceOverNodes(c, points)
  for (let i = 0; i < c.coords; i++) {
    const got = gplvmRow.ep?.({ mode: 'ep', to: i, args, params, power: 1 }).message as number[]
    same(
      got,
      scalarMessage(ref.mean[i] as number, ref.variance[i] as number, c.x[i] as [number, number]),
      tolerance,
      `message to x${i}`,
    )
  }
  for (let d = 0; d < c.outputs; d++) {
    const got = gplvmRow.ep?.({ mode: 'ep', to: c.coords + d, args, params, power: 1 }).message as number[]
    const cavity = (args[c.coords + d] as Arg).eta as number[]
    const precision = (
      spd(ref.weightCovariance[d] as number[], c.features) as NonNullable<ReturnType<typeof spd>>
    ).inverse()
    const h = Array.from({ length: c.features }, (_, a) => {
      let total = 0
      for (let b = 0; b < c.features; b++)
        total += (precision[a * c.features + b] as number) * ((ref.weightMean[d] as number[])[b] as number)
      return total
    })
    same(
      got,
      [...h, ...precision].map((v, i) => v - (cavity[i] as number)),
      tolerance,
      `message to the weights of output ${d}`,
    )
  }
  const evidence = gplvmRow.logEvidence?.({ mode: 'ep', to: 0, args, params, power: 1 }) as number
  let masses = 0
  for (const a of args) masses += massOf(a)
  expect(Math.abs(evidence - (ref.logZ + masses)), `evidence ${evidence} against ${ref.logZ + masses}`).toBeLessThan(
    tolerance,
  )
}

const IDENTITY2 = [0.8, 0.1, 0.1, 0.6]

describe('gplvm-row against a dense grid of the coordinates with an explicit solve for the weights', () => {
  it('one coordinate, three outputs of three weights, a multimodal tilt', () => {
    checkAgainstNodes(
      {
        coords: 1,
        features: 3,
        outputs: 3,
        omega: [2.5, -1.7, 3.1],
        phase: [0.3, 1.1, -0.4],
        noise: 0.2,
        y: [0.4, -0.7, 0.2],
        x: [[0.2, 1]],
        means: [
          [0.5, -0.2, 0.3],
          [-0.4, 0.6, 0.1],
          [0.2, 0.2, -0.5],
        ],
        blocks: [
          [0.9, 0.1, 0, 0.1, 0.7, 0.05, 0, 0.05, 0.8],
          [0.6, -0.1, 0.1, -0.1, 0.9, 0, 0.1, 0, 0.5],
          [1.1, 0.2, 0, 0.2, 0.8, 0.1, 0, 0.1, 0.7],
        ],
      },
      4001,
      2e-3,
    )
  })

  it('one coordinate with a small noise and many outputs: the tilt is narrow against its cavity', () => {
    checkAgainstNodes(
      {
        coords: 1,
        features: 2,
        outputs: 5,
        omega: [1.2, -0.8],
        phase: [0.1, 0.9],
        noise: 0.05,
        y: [0.3, -0.2, 0.6, 0.1, -0.5],
        x: [[0, 1]],
        means: [
          [0.8, 0.2],
          [-0.3, 0.7],
          [0.5, 0.5],
          [0.1, -0.4],
          [-0.6, 0.3],
        ],
        blocks: [IDENTITY2, IDENTITY2, IDENTITY2, IDENTITY2, IDENTITY2],
      },
      20001,
      1e-4,
    )
  })

  it('two coordinates, two outputs of three weights', () => {
    checkAgainstNodes(
      {
        coords: 2,
        features: 3,
        outputs: 2,
        omega: [1.3, -0.9, 2.1, 0.7, 1.5, -1.1],
        phase: [0.2, -0.6, 1.0],
        noise: 0.3,
        y: [0.5, -0.3],
        x: [
          [0.1, 1],
          [-0.2, 1.2],
        ],
        means: [
          [0.4, -0.5, 0.3],
          [-0.2, 0.6, 0.2],
        ],
        blocks: [
          [0.9, 0.1, 0, 0.1, 0.7, 0.05, 0, 0.05, 0.8],
          [0.6, -0.1, 0.1, -0.1, 0.9, 0, 0.1, 0, 0.5],
        ],
      },
      241,
      1e-3,
    )
  })
})

/** The exact tilted distribution of (x, w1, w2) for one output, by a grid of all three. */
function threeDimensional(c: Case): {
  readonly mean: number
  readonly variance: number
  readonly weightMean: number[]
  readonly weightCovariance: number[]
  readonly logZ: number
} {
  const [mx, vx] = c.x[0] as [number, number]
  const sx = Math.sqrt(vx)
  const v = c.blocks[0] as number[]
  const m = c.means[0] as number[]
  const factor = spd(v, 2) as NonNullable<ReturnType<typeof spd>>
  const precision = factor.inverse()
  const sd = [Math.sqrt(v[0] as number), Math.sqrt(v[3] as number)]
  const nx = 801
  const nw = 61
  const dx = (16 * sx) / (nx - 1)
  const dw = [(16 * (sd[0] as number)) / (nw - 1), (16 * (sd[1] as number)) / (nw - 1)]
  const nodeAt = (i: number, a: number, b: number): readonly [number, number, number] => [
    mx + sx * (-8 + (16 * i) / (nx - 1)),
    (m[0] as number) + (sd[0] as number) * (-8 + (16 * a) / (nw - 1)),
    (m[1] as number) + (sd[1] as number) * (-8 + (16 * b) / (nw - 1)),
  ]
  const logAt = (i: number, a: number, b: number): number => {
    const [x, w0, w1] = nodeAt(i, a, b)
    const phi = phiAt(c, [x])
    const e0 = w0 - (m[0] as number)
    const e1 = w1 - (m[1] as number)
    const quad =
      e0 * ((precision[0] as number) * e0 + (precision[1] as number) * e1) +
      e1 * ((precision[2] as number) * e0 + (precision[3] as number) * e1)
    return (
      logNormal(x, mx, vx) -
      0.5 * (2 * LOG_2PI + factor.logDet + quad) +
      logNormal(c.y[0] as number, (phi[0] as number) * w0 + (phi[1] as number) * w1, c.noise ** 2)
    )
  }
  let top = Number.NEGATIVE_INFINITY
  for (let i = 0; i < nx; i++)
    for (let a = 0; a < nw; a++) for (let b = 0; b < nw; b++) top = Math.max(top, logAt(i, a, b))
  let sum = 0
  let sx1 = 0
  let sx2 = 0
  const sw = [0, 0]
  const sww = [0, 0, 0, 0]
  for (let i = 0; i < nx; i++)
    for (let a = 0; a < nw; a++)
      for (let b = 0; b < nw; b++) {
        const wk = Math.exp(logAt(i, a, b) - top)
        const [x, w0, w1] = nodeAt(i, a, b)
        sum += wk
        sx1 += wk * x
        sx2 += wk * x * x
        sw[0] = (sw[0] as number) + wk * w0
        sw[1] = (sw[1] as number) + wk * w1
        sww[0] = (sww[0] as number) + wk * w0 * w0
        sww[1] = (sww[1] as number) + wk * w0 * w1
        sww[3] = (sww[3] as number) + wk * w1 * w1
      }
  const mean = sx1 / sum
  const wm = [(sw[0] as number) / sum, (sw[1] as number) / sum]
  const cov00 = (sww[0] as number) / sum - (wm[0] as number) ** 2
  const cov01 = (sww[1] as number) / sum - (wm[0] as number) * (wm[1] as number)
  const cov11 = (sww[3] as number) / sum - (wm[1] as number) ** 2
  return {
    mean,
    variance: sx2 / sum - mean * mean,
    weightMean: wm,
    weightCovariance: [cov00, cov01, cov01, cov11],
    logZ: top + Math.log(sum * dx * (dw[0] as number) * (dw[1] as number)),
  }
}

describe('gplvm-row against the exact tilted distribution of the coordinate and the weights together', () => {
  const c: Case = {
    coords: 1,
    features: 2,
    outputs: 1,
    omega: [1.6, -2.2],
    phase: [0.4, 0.9],
    noise: 0.3,
    y: [0.45],
    x: [[0.1, 1]],
    means: [[0.3, -0.2]],
    blocks: [[0.8, 0.2, 0.2, 0.6]],
  }

  it('is the moment match of the coordinate and of the weights (one output: one block, no correlation dropped)', () => {
    const exact = threeDimensional(c)
    const args = argsOf(c)
    const params = paramsOf(c)
    const x = gplvmRow.ep?.({ mode: 'ep', to: 0, args, params, power: 1 }).message as number[]
    same(x, scalarMessage(exact.mean, exact.variance, [0.1, 1]), 1e-5, 'message to the coordinate')
    const w = gplvmRow.ep?.({ mode: 'ep', to: 1, args, params, power: 1 }).message as number[]
    const precision = (spd(exact.weightCovariance, 2) as NonNullable<ReturnType<typeof spd>>).inverse()
    const h = [0, 1].map(
      (a) =>
        (precision[a * 2] as number) * (exact.weightMean[0] as number) +
        (precision[a * 2 + 1] as number) * (exact.weightMean[1] as number),
    )
    const cavity = (args[1] as Arg).eta as number[]
    same(
      w,
      [...h, ...precision].map((v, i) => v - (cavity[i] as number)),
      1e-5,
      'message to the weights',
    )
    const evidence = gplvmRow.logEvidence?.({ mode: 'ep', to: 0, args, params, power: 1 }) as number
    expect(Math.abs(evidence - (exact.logZ + massOf(args[0] as Arg) + massOf(args[1] as Arg)))).toBeLessThan(1e-5)
  })
})

describe('gplvm-row with a second output', () => {
  it('sends output 0 a different message once there is a second measurement to explain (its weights are not ignored)', () => {
    const base: Case = {
      coords: 1,
      features: 2,
      outputs: 1,
      omega: [1.6, -2.2],
      phase: [0.4, 0.9],
      noise: 0.3,
      y: [0.45],
      x: [[0.1, 1]],
      means: [[0.3, -0.2]],
      blocks: [[0.8, 0.2, 0.2, 0.6]],
    }
    const two: Case = {
      ...base,
      outputs: 2,
      y: [0.45, -0.3],
      means: [
        [0.3, -0.2],
        [0.1, 0.4],
      ],
      blocks: [
        [0.8, 0.2, 0.2, 0.6],
        [0.7, 0, 0, 0.9],
      ],
    }
    const one = gplvmRow.ep?.({ mode: 'ep', to: 1, args: argsOf(base), params: paramsOf(base), power: 1 })
      .message as number[]
    const both = gplvmRow.ep?.({ mode: 'ep', to: 1, args: argsOf(two), params: paramsOf(two), power: 1 })
      .message as number[]
    expect(Math.max(...one.map((v, i) => Math.abs(v - (both[i] as number))))).toBeGreaterThan(1e-3)
  })
})

describe('gplvm-row declarations', () => {
  const ops = defaultOperators()
  const weights = gaussianMv(2)
  const small: Case = {
    coords: 1,
    features: 2,
    outputs: 1,
    omega: [1, 2],
    phase: [0, 0],
    noise: 0.1,
    y: [0.5],
    x: [[0, 1]],
    means: [[0, 0]],
    blocks: [[1, 0, 0, 1]],
  }
  const base = paramsOf(small)

  it('is in the registry with a port per coordinate and a vector port per output', () => {
    expect(ops['gplvm-row']).toBe(gplvmRow)
    const ports = gplvmRow.ports as (p: Params) => readonly { name: string }[]
    expect(ports({ coords: 2, outputs: 3 }).map((p) => p.name)).toEqual(['x0', 'x1', 'w0', 'w1', 'w2'])
  })

  it('has no variational rule and says why', () => {
    expect(gplvmRow.support.vmp).toBeUndefined()
    expect(gplvmRow.noRule?.vmp).toMatch(/mean field/)
  })

  it('refuses a parameter or a family it cannot take, with a reason', () => {
    expect(gplvmRow.check?.(base, [gaussian, weights])).toBeUndefined()
    const three = { ...base, coords: 3, omega: [1, 2, 3, 4, 5, 6] }
    expect(gplvmRow.check?.(three, [gaussian, gaussian, gaussian, weights])).toMatch(/1 to 2/)
    expect(gplvmRow.check?.({ ...base, omega: [1] }, [gaussian, weights])).toMatch(/omega must have 1 x 2/)
    expect(gplvmRow.check?.({ ...base, y: [0.1, 0.2] }, [gaussian, weights])).toMatch(/y must have 1/)
    expect(gplvmRow.check?.(base, [gaussian, gaussianMv(3)])).toMatch(/output 0 have 3 entries, not 2/)
    expect(gplvmRow.check?.(base, [gaussian])).toMatch(/one port each/)
    expect(gplvmRow.check?.({ ...base, noise: -1 }, [gaussian, weights])).toMatch(/noise/)
  })

  it('throws a parameter error for a noise that is not positive when it is asked for a message', () => {
    const args = argsOf(small)
    expect(() => gplvmRow.ep?.({ mode: 'ep', to: 0, args, params: { ...base, noise: -1 }, power: 1 })).toThrow(
      ParameterError,
    )
  })
})
