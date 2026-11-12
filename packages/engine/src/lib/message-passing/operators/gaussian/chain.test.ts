// gaussian-chain: the pass along the chain against dense Gaussian algebra (the joint of the states, the drivers and the cavities at one
// precision), the tilted moments and the evidence against a fine grid over the precision, the messages against the tilted moments, a
// chain of two states against the gaussian-gamma-precision factor, and the cavities a chain can be shown (flat ones, a ridge).
import { describe, expect, it } from 'vitest'
import { add, arange, cos, mul, sin, toFlat } from '../../../tensor/index.ts'
import { messageArg, valueArg } from '../../core/args.ts'
import { ImproperMessageError, ParameterError } from '../../core/errors.ts'
import { gamma, gammaEta, gaussian, gaussianEta } from '../../core/families.ts'
import { compileGraph } from '../../core/graph.ts'
import type { Arg, OperatorContext } from '../../core/types.ts'
import { float64Of } from '../../numerics/_arrays.ts'
import { spd } from '../../numerics/_dense.ts'
import { defaultOperators } from '../registry.ts'
import { chainAt, chainSpec, chainTilt, gaussianChain } from './chain.ts'
import { gaussianGammaPrecision } from './gaussian.ts'

const SPEC = { n: 6, a: 0.7, drive: [0.3, -0.5], intercept: 0.2, scale: 1.5, offset: 0.1, ridge: 0 }
const N = SPEC.n + SPEC.drive.length
const noise = (i: number): number => (((Math.sin(i * 12.9898) * 43758.5453) % 1) + 1) % 1
const H = Float64Array.from({ length: N }, (_, i) => 2 * noise(i) - 1)
const J = Float64Array.from({ length: N }, (_, i) => 0.3 + noise(i + 50))
// one cavity is flat: the chain alone has to make the joint proper
J[2] = 0
H[2] = 0

/** The same integral and marginals by building the joint precision of all the variables and inverting it. */
function dense(spec: typeof SPEC, h: ArrayLike<number>, j: ArrayLike<number>, lambda: number) {
  const { n } = spec
  const size = n + spec.drive.length
  const P = new Float64Array(size * size)
  const r = new Float64Array(size)
  for (let i = 0; i < size; i++) {
    P[i * size + i] = (P[i * size + i] as number) + (j[i] as number) + (i < n ? spec.ridge : 0)
    r[i] = (r[i] as number) + (h[i] as number)
  }
  for (let t = 1; t < n; t++) {
    const e = new Float64Array(size)
    e[t] = 1
    e[t - 1] = (e[t - 1] as number) - spec.a
    for (const [k, c] of spec.drive.entries()) e[n + k] = (e[n + k] as number) - c
    for (let p = 0; p < size; p++) {
      for (let q = 0; q < size; q++)
        P[p * size + q] = (P[p * size + q] as number) + lambda * (e[p] as number) * (e[q] as number)
      r[p] = (r[p] as number) + lambda * spec.intercept * (e[p] as number)
    }
  }
  const f = spd(P, size) as NonNullable<ReturnType<typeof spd>>
  const mean = f.solve(r)
  const inverse = f.inverse()
  let quad = 0
  for (let i = 0; i < size; i++) quad += (r[i] as number) * (mean[i] as number)
  const logZ =
    0.5 * (n - 1) * (Math.log(lambda) - Math.log(2 * Math.PI)) -
    0.5 * lambda * (n - 1) * spec.intercept ** 2 +
    0.5 * (size * Math.log(2 * Math.PI) - f.logDet + quad)
  return { logZ, mean, variance: Float64Array.from({ length: size }, (_, i) => inverse[i * size + i] as number) }
}

describe('the pass along the chain at one precision', () => {
  it('has the integral, the means and the variances of the dense joint, with drivers, an intercept and a flat cavity', () => {
    for (const lambda of [0.3, 2, 40]) {
      const fast = chainAt(SPEC, H, J, lambda, true)
      const slow = dense(SPEC, H, J, lambda)
      expect(fast?.logZ).toBeCloseTo(slow.logZ, 10)
      for (let p = 0; p < N; p++) {
        expect(fast?.mean[p] as number).toBeCloseTo(slow.mean[p] as number, 10)
        expect(fast?.variance[p] as number).toBeCloseTo(slow.variance[p] as number, 10)
      }
    }
  })

  it('works with no drivers (a random walk, an autoregression) and with the integral alone', () => {
    const spec = { ...SPEC, drive: [] as number[], a: 1, ridge: 0 }
    const h = H.slice(0, spec.n)
    const j = J.slice(0, spec.n).map((v) => v + 0.2)
    const fast = chainAt(spec, h, j, 3, true)
    const slow = dense(spec, h, j, 3)
    expect(fast?.logZ).toBeCloseTo(slow.logZ, 10)
    expect(chainAt(spec, h, j, 3, false)?.logZ).toBeCloseTo(slow.logZ, 10)
    for (let p = 0; p < spec.n; p++) expect(fast?.variance[p] as number).toBeCloseTo(slow.variance[p] as number, 10)
  })

  it('is undefined where the joint precision is not positive definite (nothing pins the chain)', () => {
    const spec = { ...SPEC, ridge: 0, drive: [] as number[], a: 1 }
    expect(chainAt(spec, new Float64Array(spec.n), new Float64Array(spec.n), 2, true)).toBeUndefined()
  })
})

describe('the tilted distribution over the precision', () => {
  const shape = 3
  const rate = 0.4
  const tilt = chainTilt(SPEC, H, J, gammaEta(shape, rate))

  it('has the evidence and the moments of a fine grid over log tau against the gamma cavity', () => {
    const G = 3000
    const lo = -6
    const step = 14 / (G - 1)
    const ts = toFlat(add(lo, mul(arange(G), step)))
    const joint = ts.map((t) => dense(SPEC, H, J, SPEC.scale * Math.exp(t) + SPEC.offset))
    const logw = ts.map((t, g) => shape * t - rate * Math.exp(t) + (joint[g]?.logZ as number))
    const top = Math.max(...logw)
    const w = logw.map((v) => Math.exp(v - top))
    const z = w.reduce((s, v) => s + v, 0)
    expect(tilt.logZ).toBeCloseTo(top + Math.log(z * step), 6)
    let eTau = 0
    let eLog = 0
    for (let g = 0; g < G; g++) {
      eTau += ((w[g] as number) * Math.exp(ts[g] as number)) / z
      eLog += ((w[g] as number) * (ts[g] as number)) / z
    }
    expect(tilt.tau).toBeCloseTo(eTau, 6)
    expect(tilt.logTau).toBeCloseTo(eLog, 6)
    for (let p = 0; p < N; p++) {
      let m = 0
      for (let g = 0; g < G; g++)
        m += ((w[g] as number) * (((joint[g] as ReturnType<typeof dense>).mean as Float64Array)[p] as number)) / z
      let v = 0
      for (let g = 0; g < G; g++) {
        const jg = joint[g] as ReturnType<typeof dense>
        v += ((w[g] as number) * ((jg.variance[p] as number) + ((jg.mean[p] as number) - m) ** 2)) / z
      }
      expect(tilt.mean[p] as number).toBeCloseTo(m, 6)
      expect(tilt.variance[p] as number).toBeCloseTo(v, 6)
    }
  })
})

describe('a cavity with an exponential tail', () => {
  // A state whose reading is a return near zero (the log-variance likelihood has a flat tail): its cavity has a mean far away and
  // almost no precision. The chain integrated over the precision would let the precision run to zero to reach it; the ridge, a weak
  // prior on a state whose cavity is uninformative, centred at the chain's own estimate of the state, keeps the precision where the
  // other states put it, and moves a chain whose cavities all have precision of their own by nothing.
  const n = 20
  const h = float64Of(mul(sin(arange(n + 1)), 0.2))
  const j = float64Of(add(0.5, mul(cos(arange(n + 1)), 0.1)))
  h[7] = -0.45
  j[7] = 1e-4
  h[n] = 0
  j[n] = 1
  const spec = { n, a: 0.7, drive: [0.3], intercept: 0, scale: 1, offset: 0, ridge: 0 }

  it('sends the precision to zero without a ridge and leaves it with the other states with one', () => {
    expect(chainTilt(spec, h, j, gammaEta(3, 0.3)).tau).toBeLessThan(0.01)
    const ridged = chainTilt({ ...spec, ridge: 0.05 }, h, j, gammaEta(3, 0.3))
    expect(ridged.tau).toBeGreaterThan(1)
    expect(ridged.tau).toBeLessThan(40)
  })

  it('moves a chain whose cavities are all informative by almost nothing', () => {
    const j2 = j.map((v, i) => (i === 7 ? 0.6 : v))
    const h2 = h.map((v, i) => (i === 7 ? 0.1 : v))
    const plain = chainTilt(spec, h2, j2, gammaEta(3, 0.3))
    const ridged = chainTilt({ ...spec, ridge: 0.05 }, h2, j2, gammaEta(3, 0.3))
    expect(ridged.tau).toBeCloseTo(plain.tau, 3)
    for (let p = 0; p <= n; p++) expect(ridged.mean[p] as number).toBeCloseTo(plain.mean[p] as number, 3)
    expect(ridged.logZ).toBeCloseTo(plain.logZ, 3)
  })
})

describe('the operator', () => {
  const args = (): Arg[] => [
    messageArg(gamma, gammaEta(3, 0.4)),
    ...Array.from({ length: N }, (_, p) => messageArg(gaussian, [H[p] as number, J[p] as number])),
  ]
  const params = {
    length: SPEC.n,
    coefficient: SPEC.a,
    drive: SPEC.drive,
    intercept: SPEC.intercept,
    scale: SPEC.scale,
    offset: SPEC.offset,
    ridge: SPEC.ridge,
  }
  const ctx = (to: number, memo?: Map<string, unknown>): OperatorContext => ({
    mode: 'ep',
    to,
    args: args(),
    params,
    power: 1,
    ...(memo === undefined ? {} : { memo }),
  })
  const tilt = chainTilt(SPEC, H, J, gammaEta(3, 0.4))

  it('has the precision, the states and the drivers as ports, and a message that is the tilted moments over the cavity', () => {
    expect(gaussianChain.ports).toBeTypeOf('function')
    const ports = typeof gaussianChain.ports === 'function' ? gaussianChain.ports(params) : gaussianChain.ports
    expect(ports.map((p) => p.name)).toEqual(['tau', 'x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'u0', 'u1'])
    // to a state: the Gaussian of the tilted mean and variance, divided by the cavity
    const to = 4
    const m = gaussianChain.ep?.(ctx(to)).message as number[]
    const v = tilt.variance[to - 1] as number
    expect(m[1]).toBeCloseTo(1 / v - (J[to - 1] as number), 8)
    expect(m[0]).toBeCloseTo((tilt.mean[to - 1] as number) / v - (H[to - 1] as number), 8)
    // to the precision: a gamma that, with the cavity, has the tilted E tau as its mean (shape over rate)
    const g = gaussianChain.ep?.(ctx(0)).message as number[]
    const shape = (g[0] as number) + 2 + 1
    const rate = -((g[1] as number) - 0.4)
    expect(shape / rate).toBeCloseTo(tilt.tau, 6)
  })

  it('shares one pass between the ports of the same cavities (the engine cache) and gives the evidence of the tilt', () => {
    const memo = new Map<string, unknown>()
    const first = [0, 1, 4, 8].map((to) => gaussianChain.ep?.(ctx(to, memo)).message)
    expect(memo.size).toBe(1)
    expect(first).toEqual([0, 1, 4, 8].map((to) => gaussianChain.ep?.(ctx(to)).message))
    expect(gaussianChain.logEvidence?.(ctx(0, memo))).toBeCloseTo(tilt.logZ, 12)
    expect(memo.size).toBe(1)
  })

  it('an observed state is a model mistake: the compiler refuses it, and a hand-built call throws a ParameterError (not an improper message)', () => {
    expect(() =>
      compileGraph(
        {
          variables: [
            { name: 'tau', family: { kind: 'gamma' } },
            { name: 'x0', family: { kind: 'gaussian' } },
            { name: 'x1', family: { kind: 'gaussian' }, observed: 1 },
          ],
          factors: [{ operator: 'gaussian-chain', args: ['tau', 'x0', 'x1'], params: { length: 2 } }],
        },
        defaultOperators(),
      ),
    ).toThrow(/cannot take the observed variable x1/)
    const observed: Arg[] = args()
    observed[2] = valueArg(gaussian, 0.5)
    let caught: unknown
    try {
      gaussianChain.ep?.({ ...ctx(1), args: observed })
    } catch (e) {
      caught = e
    }
    expect(caught).toBeInstanceOf(ParameterError)
    expect(caught).not.toBeInstanceOf(ImproperMessageError)
    expect((caught as Error).message).toMatch(/port 1 cannot be observed/)
  })

  it('refuses a length under two and negative precision parameters', () => {
    expect(() => chainSpec({ length: 1 })).toThrow()
    expect(() => chainSpec({ length: 3, scale: 0, offset: 0 })).toThrow()
    expect(() => chainSpec({ length: 3, ridge: -1 })).toThrow()
  })
})

describe('a chain of two states against the scalar factor', () => {
  it('is the gaussian-gamma-precision factor of the step when the mean is the previous state (a = 1)', () => {
    const cavity = [messageArg(gaussian, gaussianEta(0.4, 1.3)), messageArg(gaussian, gaussianEta(-0.6, 0.8))]
    const tau = messageArg(gamma, gammaEta(2.5, 1.5))
    const chain = (to: number): number[] =>
      gaussianChain.ep?.({ mode: 'ep', to, args: [tau, ...cavity], params: { length: 2, coefficient: 1 }, power: 1 })
        .message as number[]
    // x ~ N(w, 1 / tau) with the pair (x, w) = (x1, x0): the scalar factor's arguments are (x, mean, precision)
    const scalar = (to: number): number[] =>
      gaussianGammaPrecision.ep?.({
        mode: 'ep',
        to,
        args: [cavity[1] as Arg, cavity[0] as Arg, tau],
        params: {},
        power: 1,
      }).message as number[]
    for (const [c, s] of [
      [chain(2), scalar(0)],
      [chain(1), scalar(1)],
      [chain(0), scalar(2)],
    ] as const)
      for (let i = 0; i < 2; i++) expect(c[i] as number).toBeCloseTo(s[i] as number, 2)
  })
})
