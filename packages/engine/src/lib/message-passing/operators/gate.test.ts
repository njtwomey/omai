import { describe, expect, it } from 'vitest'
import { gaussian, gaussianEta } from '../families.ts'
import { discrete, discreteEta, discreteProbs } from '../families-vector.ts'
import { messageArg, valueArg } from './args.ts'
import { defaultOperators } from './registry.ts'

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
    Array.from({ length: n }, (_, i) => m + Math.sqrt(v) * (-9 + (18 * i) / (n - 1)))
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
})
