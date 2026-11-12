import { describe, expect, it } from 'vitest'
import { add, arange, div, mul, toFlat } from '../../../tensor/index.ts'
import { messageArg, valueArg } from '../../core/args.ts'
import { ParameterError } from '../../core/errors.ts'
import { gamma, gammaEta, gaussian, gaussianEta } from '../../core/families.ts'
import { discrete, discreteEta, discreteProbs } from '../../core/families-vector.ts'
import { lgammaOf } from '../../numerics/_scalar.ts'
import { defaultOperators } from '../registry.ts'

const ops = defaultOperators()
const gate = ops['gate']

const norm = (x: number, m: number, v: number): number =>
  Math.exp((-0.5 * (x - m) ** 2) / v) / Math.sqrt(2 * Math.PI * v)

/** The probabilities of z and the mean and variance of a and b under the gate with two branches, by a two-dimensional grid. */
function brute(
  pz: readonly number[],
  qa: readonly [number, number],
  qb: readonly [number, number],
  f: readonly [(a: number, b: number) => number, (a: number, b: number) => number],
) {
  const n = 500
  const grid = (m: number, v: number): number[] =>
    toFlat(add(m, mul(add(-9, div(mul(arange(n), 18), n - 1)), Math.sqrt(v))))
  const as = grid(...qa)
  const bs = grid(...qb)
  const da = (as[1] as number) - (as[0] as number)
  const db = (bs[1] as number) - (bs[0] as number)
  const mass = [0, 0]
  const m = { a: [0, 0], b: [0, 0] }
  const s = { a: [0, 0], b: [0, 0] }
  for (let k = 0; k < 2; k++)
    for (const a of as)
      for (const b of bs) {
        const w =
          (pz[k] as number) *
          norm(a, ...qa) *
          norm(b, ...qb) *
          (f[k] as (a: number, b: number) => number)(a, b) *
          da *
          db
        mass[k] = (mass[k] as number) + w
        m.a[k] = (m.a[k] as number) + w * a
        m.b[k] = (m.b[k] as number) + w * b
        s.a[k] = (s.a[k] as number) + w * a * a
        s.b[k] = (s.b[k] as number) + w * b * b
      }
  const [m0, m1] = mass as [number, number]
  const z = m0 + m1
  const stat = (mm: number[], ss: number[]) => {
    const mean = ((mm[0] as number) + (mm[1] as number)) / z
    return { mean, variance: ((ss[0] as number) + (ss[1] as number)) / z - mean * mean }
  }
  return { pz: [m0 / z, m1 / z], a: stat(m.a, s.a), b: stat(m.b, s.b) }
}

describe('the gate', () => {
  it('a mixture of two regressions with their own random means: the messages to z, a and b are the tilted moments', () => {
    const y = 0.7
    const qa = [0.2, 0.9] as const
    const qb = [1.5, 1.4] as const
    const params = {
      arity: 2,
      branches: [
        { operator: 'gaussian', args: [{ const: y }, 0], params: { precision: 4 } },
        { operator: 'gaussian', args: [{ const: y }, 1], params: { precision: 1 } },
      ],
    }
    const pz = [0.3, 0.7]
    const args = [
      messageArg(discrete(2), discreteEta(pz)),
      messageArg(gaussian, gaussianEta(...qa)),
      messageArg(gaussian, gaussianEta(...qb)),
    ]
    const ref = brute(pz, qa, qb, [(a) => norm(y, a, 0.25), (_a, b) => norm(y, b, 1)])
    const toZ = gate?.ep?.({ mode: 'ep', to: 0, args, params, power: 1 }).message ?? []
    // The message to z times its cavity is the tilted probability.
    const tilted = discreteProbs(toZ.map((v, i) => v + Math.log(pz[i] as number)))
    expect(tilted[0]).toBeCloseTo(ref.pz[0] as number, 4)
    for (const [port, q, want] of [
      [1, qa, ref.a],
      [2, qb, ref.b],
    ] as const) {
      const out = gate?.ep?.({ mode: 'ep', to: port, args, params, power: 1 }).message ?? []
      const j = 1 / q[1] + (out[1] ?? 0)
      expect(1 / j).toBeCloseTo(want.variance, 4)
      expect((q[0] / q[1] + (out[0] ?? 0)) / j).toBeCloseTo(want.mean, 4)
    }
  })

  it('a branch that does not involve a variable leaves its cavity in the mixture; a branch without an operator is the constant one', () => {
    // Zero inflation: z = 0 says y is exactly the baseline (no factor on a), z = 1 says y is a plus noise.
    const params = {
      arity: 1,
      branches: [{}, { operator: 'gaussian', args: [{ const: 3 }, 0], params: { precision: 2 } }],
    }
    const pz = [0.4, 0.6]
    const qa = [0, 1] as const
    const args = [messageArg(discrete(2), discreteEta(pz)), messageArg(gaussian, gaussianEta(...qa))]
    const out = gate?.ep?.({ mode: 'ep', to: 1, args, params, power: 1 }).message ?? []
    // Exact: 0.4 N(a; 0, 1) + 0.6 N(3; a, 0.5) N(a; 0, 1) (normalised): the means by closed forms.
    const z1 = norm(3, 0, 1.5)
    const w1 = (0.6 * z1) / (0.4 + 0.6 * z1)
    const post1 = { mean: (3 * 1) / 1.5, variance: 1 / (1 + 2) }
    const mean = w1 * post1.mean
    const second = w1 * (post1.variance + post1.mean ** 2) + (1 - w1) * 1
    const j = 1 + (out[1] ?? 0)
    expect(1 / j).toBeCloseTo(second - mean * mean, 8)
    expect((out[0] ?? 0) / j).toBeCloseTo(mean, 8)
  })

  it('a latent value with a known mean per state (the bimodal target): the tilted variance is the mixture variance', () => {
    const params = {
      arity: 1,
      branches: [-3, 3].map((m) => ({ operator: 'gaussian', args: [0, { const: m }], params: { precision: 1 } })),
    }
    const args = [messageArg(discrete(2), discreteEta([0.5, 0.5])), messageArg(gaussian, gaussianEta(0, 100))]
    const out = gate?.ep?.({ mode: 'ep', to: 1, args, params, power: 1 }).message ?? []
    // The mixture of N(-3, 1) and N(3, 1) has variance 1 + 9 = 10; the cavity N(0, 100) multiplies each by a nearly flat factor.
    const j = 1 / 100 + (out[1] ?? 0)
    expect(1 / j).toBeGreaterThan(8)
    expect(1 / j).toBeLessThan(10.5)
  })

  it('a power of the gate: the messages are those of the factor to the power p against its cavities, by a grid, and power one is the plain rule', () => {
    // Two Poisson branches on one gamma rate: counts 2 and 9 chosen by z; the cavity of the rate is Gamma(3, 0.5), that of z (0.3, 0.7).
    const params = {
      arity: 1,
      branches: [
        { operator: 'poisson', args: [{ const: 2 }, 0], params: { exposure: 1 } },
        { operator: 'poisson', args: [{ const: 9 }, 0], params: { exposure: 1 } },
      ],
    }
    const pz = [0.3, 0.7]
    const cavity = gammaEta(3, 0.5)
    const args = [messageArg(discrete(2), discreteEta(pz)), messageArg(gamma, cavity)]
    const counts = [2, 9]
    for (const power of [1, 0.5, 0.25]) {
      // The grid: log r over a wide range, the cavity density times the branch likelihood to the power p.
      const n = 40000
      const lo = Math.log(1e-3)
      const hi = Math.log(120)
      const step = (hi - lo) / (n - 1)
      const mass = [0, 0]
      let eLog = 0
      let eR = 0
      for (let i = 0; i < n; i++) {
        const t = lo + i * step
        const r = Math.exp(t)
        const density = Math.exp(3 * t - 0.5 * r) // the cavity in t = log r, up to its constant
        for (const [k, count] of counts.entries()) {
          const w = (pz[k] as number) * density * Math.exp(power * (count * t - r - lgammaOf(count + 1)))
          mass[k] = (mass[k] as number) + w
          eLog += w * t
          eR += w * r
        }
      }
      const total = (mass[0] as number) + (mass[1] as number)
      const projected = gamma.fromMoments([eLog / total, eR / total])
      const want = projected.map((v, i) => (v - (cavity[i] ?? 0)) / power)
      const got = gate?.ep?.({ mode: 'ep', to: 1, args, params, power }).message as number[]
      for (const [i, v] of want.entries()) expect(got[i] as number).toBeCloseTo(v, 3)
      // The message to z is the evidence of each branch to the power 1 / p: its probabilities are proportional to Z_k^(1 / p).
      const evidence = counts.map((count) => {
        let z = 0
        for (let i = 0; i < n; i++) {
          const t = lo + i * step
          z += Math.exp(3 * t - 0.5 * Math.exp(t) + power * (count * t - Math.exp(t) - lgammaOf(count + 1)))
        }
        return z * step
      })
      const raw = evidence.map((z) => z ** (1 / power))
      const sum = raw.reduce((a, b) => a + b, 0)
      const toZ = discreteProbs(gate?.ep?.({ mode: 'ep', to: 0, args, params, power }).message as number[])
      for (const [k, v] of raw.entries()) expect(toZ[k] as number).toBeCloseTo(v / sum, 4)
    }
  })

  it('refuses a power on a branch that has no closed-form message over one variable', () => {
    const params = {
      arity: 2,
      branches: [{ operator: 'gaussian', args: [0, 1], params: { precision: 2 } }, {}],
    }
    const args = [
      messageArg(discrete(2), discreteEta([0.5, 0.5])),
      messageArg(gaussian, gaussianEta(0, 1)),
      messageArg(gaussian, gaussianEta(0, 1)),
    ]
    expect(() => gate?.ep?.({ mode: 'ep', to: 1, args, params, power: 0.5 })).toThrow(/closed-form message/)
  })

  it('checks its branches against the registry and the states of z', () => {
    expect(gate?.check?.({ arity: 1, branches: [{}, {}, {}] }, [discrete(2), gaussian])).toMatch(/3 branches/)
    expect(
      gate?.check?.({ arity: 1, branches: [{ operator: 'nope', args: [0] }, {}] }, [discrete(2), gaussian]),
    ).toMatch(/no operator/)
    expect(gate?.check?.({ arity: 1, branches: [{}, {}] }, [discrete(2), gaussian])).toBeUndefined()
  })

  it('the variational rule: the message to z is the expected log density of each branch, to a variable the responsibility-weighted message', () => {
    const params = {
      arity: 1,
      branches: [
        { operator: 'gaussian', args: [{ const: 1 }, 0], params: { precision: 2 } },
        { operator: 'gaussian', args: [{ const: 1 }, 0], params: { precision: 8 } },
      ],
    }
    const args = [messageArg(discrete(2), discreteEta([0.25, 0.75])), messageArg(gaussian, gaussianEta(0.5, 4))]
    const toZ = gate?.vmp?.({ mode: 'vmp', to: 0, args, params, power: 1 }).message ?? []
    // E[log N(1; a, 1/tau)] = 0.5 log(tau / 2 pi) - tau / 2 ((1 - 0.5)^2 + 0.25)
    const e = (tau: number): number => 0.5 * Math.log(tau / (2 * Math.PI)) - 0.5 * tau * (0.25 + 4)
    expect(toZ[0]).toBeCloseTo(e(2), 10)
    expect(toZ[1]).toBeCloseTo(e(8), 10)
    const toA = gate?.vmp?.({ mode: 'vmp', to: 1, args, params, power: 1 }).message ?? []
    // The message of a Gaussian factor to its mean is [tau x, tau]; responsibilities 0.25 and 0.75.
    expect(toA[1]).toBeCloseTo(0.25 * 2 + 0.75 * 8, 10)
    expect(toA[0]).toBeCloseTo(0.25 * 2 * 1 + 0.75 * 8 * 1, 10)
    expect(valueArg(gaussian, 1).value).toBe(1)
  })

  it('refuses a variational gate over a branch whose operator has no expected log or no variational message, naming it', () => {
    const args = [messageArg(discrete(2), discreteEta([0.5, 0.5])), messageArg(gaussian, gaussianEta(0.5, 4))]
    const ok = { operator: 'gaussian', args: [{ const: 1 }, 0], params: { precision: 2 } }
    // `equal` has a variational message but no expected log, `greater-than` has neither: each was a log density of 0 or skipped.
    for (const operator of ['equal', 'greater-than']) {
      const params = { arity: 1, branches: [ok, { operator, args: [0, { const: 0 }], params: {} }] }
      const why = gate?.unsupported?.('vmp', params)
      expect(why, operator).toMatch(new RegExp(`branch 1 \\(${operator}\\)`))
      expect(gate?.unsupported?.('ep', params)).toBeUndefined()
      expect(() => gate?.vmp?.({ mode: 'vmp', to: 0, args, params, power: 1 })).toThrow(ParameterError)
      expect(() => gate?.expectedLog?.({ mode: 'vmp', to: 0, args, params, power: 1 })).toThrow(ParameterError)
    }
    // Branches with their rules, and a constant branch (a log density of 0), are as they were.
    const fine = { arity: 1, branches: [ok, {}] }
    expect(gate?.unsupported?.('vmp', fine)).toBeUndefined()
    expect(gate?.vmp?.({ mode: 'vmp', to: 1, args, params: fine, power: 1 }).message).toHaveLength(2)
  })
})
