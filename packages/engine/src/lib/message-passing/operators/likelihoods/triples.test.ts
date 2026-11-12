// The three-variable numeric sites against a brute-force integral of the exact tilted density (the shared reference of
// `testing/message-passing.ts`: a composite Gauss-Legendre product rule that zooms onto the mass, no code of the operator in it).
import { describe, expect, it } from 'vitest'
import { type BrutePort, bruteTilted } from '../../../../testing/message-passing.ts'
import { massOf, messageArg, valueArg } from '../../core/args.ts'
import { gamma, gammaEta, gaussian, gaussianEta } from '../../core/families.ts'
import type { Arg, FactorOperator } from '../../core/types.ts'
import { logDiffPhi } from '../../numerics/_scalar.ts'
import { defaultOperators } from '../registry.ts'
import { type TripleLikelihood, tripleLikelihoodOperator, tripleLikelihoods } from './triples.ts'

const op: FactorOperator = tripleLikelihoodOperator()

const cavity = (m: number, v: number): Arg => messageArg(gaussian, gaussianEta(m, v))
const brutePort = (m: number, v: number): BrutePort => ({ kind: 'gaussian', eta: gaussianEta(m, v) })

/** The tilted moments of every free port and the evidence of one case against the brute force (a Gaussian cavity per port or a value). */
function check(
  likelihood: string,
  y: number,
  extra: Record<string, number>,
  cavities: readonly ({ m: number; v: number } | number)[],
  options: { moments?: number; evidence?: number; power?: number } = {},
): void {
  const params = { likelihood, y, ...extra }
  const power = options.power ?? 1
  const density = tripleLikelihoods()[likelihood]?.density(y, params) as (a: number, b: number, c: number) => number
  const args = cavities.map((c) => (typeof c === 'number' ? valueArg(gaussian, c) : cavity(c.m, c.v)))
  const ports: BrutePort[] = cavities.map((c) =>
    typeof c === 'number' ? { kind: 'gaussian', value: c } : brutePort(c.m, c.v),
  )
  const logf = (x: readonly number[]): number => power * density(x[0] as number, x[1] as number, x[2] as number)
  let firstZ = Number.NaN
  for (const to of [0, 1, 2]) {
    if (typeof cavities[to] === 'number') continue
    const t = op.tilted?.({ mode: 'ep', to, args, params, power })
    const want = bruteTilted(ports, logf, to, { panels: 8, order: 12 })
    if (Number.isNaN(firstZ)) firstZ = want.logZ
    for (const [i, w] of want.moments.entries()) {
      const got = t?.moments[i] as number
      expect(Math.abs(got - w), `${likelihood} port ${to} moment ${i}: ${got} against ${w}`).toBeLessThan(
        (options.moments ?? 3e-3) * (1 + Math.abs(w)),
      )
    }
    expect(
      Math.abs((t?.logZ as number) - want.logZ),
      `${likelihood} evidence ${t?.logZ} against ${want.logZ}`,
    ).toBeLessThan(options.evidence ?? 3e-3)
  }
  if (power === 1) {
    const z = op.logEvidence?.({ mode: 'ep', to: 0, args, params, power: 1 }) as number
    expect(Math.abs(z - firstZ)).toBeLessThan(options.evidence ?? 3e-3)
  }
}

describe('likelihood-triple: tilted moments and log evidence against a brute-force integral', () => {
  it('ordinal-probit-cuts, a middle category', () => {
    check('ordinal-probit-cuts', 1, { categories: 4 }, [
      { m: 0.4, v: 1 },
      { m: -0.8, v: 0.5 },
      { m: 0.7, v: 0.6 },
    ])
  })
  it('ordinal-logit-cuts, a middle category with cavities that are not close to the cuts', () => {
    check('ordinal-logit-cuts', 2, { categories: 4 }, [
      { m: -0.5, v: 1.5 },
      { m: 0.2, v: 0.8 },
      { m: 1.1, v: 0.4 },
    ])
  })
  it('the first and the last category have one cut: the other cut is ignored, whatever its value', () => {
    check('ordinal-probit-cuts', 0, { categories: 3 }, [{ m: 0.3, v: 1 }, 123, { m: 0.2, v: 0.7 }])
    check('ordinal-logit-cuts', 2, { categories: 3 }, [{ m: 0.3, v: 1 }, { m: -0.2, v: 0.7 }, -77])
    // the same with the ignored cut a variable: it does not move (its tilted distribution is its cavity)
    const params = { likelihood: 'ordinal-probit-cuts', y: 0, categories: 3 }
    const t = op.tilted?.({
      mode: 'ep',
      to: 1,
      args: [cavity(0.3, 1), cavity(5, 2), cavity(0.2, 0.7)],
      params,
      power: 1,
    })
    expect(t?.moments[0]).toBeCloseTo(5, 3)
    expect(t?.moments[1]).toBeCloseTo(27, 2)
  })
  it('a known cut beside a random one, and a known predictor', () => {
    check('ordinal-probit-cuts', 1, { categories: 3 }, [{ m: 0, v: 1 }, -0.6, { m: 0.9, v: 0.5 }])
    check('ordinal-logit-cuts', 1, { categories: 3 }, [0.2, { m: -0.4, v: 0.5 }, { m: 0.8, v: 0.5 }])
  })
  it('ordinal-probit-cuts with cavities wider than the likelihood', () => {
    check(
      'ordinal-probit-cuts',
      1,
      { categories: 3 },
      [
        { m: 0, v: 9 },
        { m: -1, v: 4 },
        { m: 1, v: 4 },
      ],
      { moments: 2e-3, evidence: 2e-3 },
    )
  })
  it('student-t-full, a log scale and a log df that are Gaussian', () => {
    check('student-t-full', 2.4, {}, [
      { m: 0.3, v: 1 },
      { m: -0.2, v: 0.4 },
      { m: 1.2, v: 0.5 },
    ])
  })
  it('student-t-full, an outlier against a wide location cavity, against a grid refined until it stopped moving', () => {
    // The limit of the grid: where the scale is small the density is a spike of the width of the scale in the location. The numbers are
    // those of a grid of 320 by 240 by 240 nodes (log mass -2.6079, tilted means -2.6112, -0.5037 and 1.5197).
    const params = { likelihood: 'student-t-full', y: -3.1 }
    const args = [cavity(0, 6), cavity(-0.5, 0.6), cavity(1.5, 0.8)]
    const want = [-2.6112, -0.5037, 1.5197]
    for (const to of [0, 1, 2]) {
      const t = op.tilted?.({ mode: 'ep', to, args, params, power: 1 })
      expect(Math.abs((t?.moments[0] as number) - (want[to] as number)), `port ${to}`).toBeLessThan(0.05)
      expect(Math.abs((t?.logZ as number) - (-2.6079 + args.reduce((z, a) => z + massOf(a), 0)))).toBeLessThan(0.05)
    }
  })
  it('student-t-full with a known scale is the pair of the location and the degrees of freedom', () => {
    check('student-t-full', 0.9, {}, [{ m: 0.2, v: 1 }, Math.log(0.8), { m: 1, v: 0.6 }])
  })
  it('a power raises the density', () => {
    check(
      'ordinal-probit-cuts',
      1,
      { categories: 3 },
      [
        { m: 0.4, v: 1 },
        { m: -0.8, v: 0.5 },
        { m: 0.7, v: 0.6 },
      ],
      {
        power: 0.5,
      },
    )
  })
  it('a gamma cavity on the degrees of freedom is read through its log', () => {
    const params = { likelihood: 'student-t-full', y: 1.7 }
    const [shape, rate] = [3, 0.6]
    const args = [cavity(0.2, 1), cavity(-0.1, 0.5), messageArg(gamma, gammaEta(shape, rate))]
    const density = tripleLikelihoods()['student-t-full']?.density(1.7, params) as (
      a: number,
      b: number,
      c: number,
    ) => number
    // The gamma density in t = log df is df^shape exp(-rate df) up to its normaliser: a flat Gaussian port with it in the integrand.
    const flat: BrutePort = { kind: 'gaussian', eta: [0, 0], range: [-9, 5] }
    const ports = [brutePort(0.2, 1), brutePort(-0.1, 0.5), flat]
    const logf = (x: readonly number[]): number =>
      density(x[0] as number, x[1] as number, x[2] as number) +
      shape * (x[2] as number) -
      rate * Math.exp(x[2] as number)
    const brute = (f: (x: readonly number[]) => number, to: number) =>
      bruteTilted(ports, f, to, { panels: 6, order: 12 })
    const location = op.tilted?.({ mode: 'ep', to: 0, args, params, power: 1 })
    const wantLocation = brute(logf, 0)
    expect(Math.abs((location?.moments[0] as number) - (wantLocation.moments[0] as number))).toBeLessThan(5e-3)
    expect(Math.abs((location?.moments[1] as number) - (wantLocation.moments[1] as number))).toBeLessThan(1e-2)
    // the moments of a gamma port are E df and E log df (the exposition of the family: mean of the log, mean of the value)
    const df = op.tilted?.({ mode: 'ep', to: 2, args, params, power: 1 })
    const base = brute(logf, 2)
    const weighted = brute((x) => logf(x) + (x[2] as number), 2)
    expect(Math.abs((df?.moments[0] as number) - (base.moments[0] as number))).toBeLessThan(5e-3)
    expect(Math.abs((df?.moments[1] as number) - Math.exp(weighted.logZ - base.logZ))).toBeLessThan(5e-3)
  })
  it('the ordinal probit by the latent score is the grid of the same density, in every category, with an ordering that is not certain', () => {
    const row = tripleLikelihoods()['ordinal-probit-cuts'] as TripleLikelihood
    const { integrate: _exact, ...gridRow } = row
    const byGrid = tripleLikelihoodOperator({ 'ordinal-probit-cuts': gridRow })
    for (const [y, cavities] of [
      [0, [cavity(0.3, 1), cavity(0, 1), cavity(-0.4, 0.5)]],
      [1, [cavity(0.3, 1), cavity(0.2, 1.5), cavity(0.1, 1.5)]],
      [1, [cavity(-0.5, 0.6), cavity(-1, 0.3), cavity(0.8, 0.4)]],
      [2, [cavity(0.3, 1), cavity(0.6, 0.7), cavity(0, 1)]],
    ] as const) {
      const params = { likelihood: 'ordinal-probit-cuts', y, categories: 3 }
      for (const to of [0, 1, 2]) {
        if (y === 0 && to === 1) continue
        if (y === 2 && to === 2) continue
        const exact = op.tilted?.({ mode: 'ep', to, args: [...cavities], params, power: 1 })
        const grid = byGrid.tilted?.({ mode: 'ep', to, args: [...cavities], params, power: 1 })
        for (const i of [0, 1]) {
          expect(
            Math.abs((exact?.moments[i] as number) - (grid?.moments[i] as number)),
            `y=${y} port ${to} moment ${i}`,
          ).toBeLessThan(3e-3 * (1 + Math.abs(grid?.moments[i] as number)))
        }
        expect(Math.abs((exact?.logZ as number) - (grid?.logZ as number)), `y=${y} port ${to} evidence`).toBeLessThan(
          1e-2,
        )
      }
    }
  })
  it('the column form of the Student-t density is the density', () => {
    const row = tripleLikelihoods()['student-t-full']
    const flat = row?.density(1.3, { y: 1.3 }) as (a: number, b: number, c: number) => number
    const column = row?.column?.(1.3, { y: 1.3 }) as (b: number, c: number) => (a: number) => number
    for (const [a, b, c] of [
      [0.3, 0.9, 1.1],
      [-1.2, -0.4, 1.7],
      [0.8, 0.6, -0.4],
    ] as const)
      expect(column(b, c)(a), `at ${a}, ${b}, ${c}`).toBeCloseTo(flat(a, b, c), 10)
  })
  it('the ordinal density by the index of the nodes is the density at the nodes, in the tails and where the cuts are out of order', () => {
    const u = Float64Array.of(-30, -2.5, -0.3, 0.4, 3, 28)
    const lo = Float64Array.of(-31, -1, 0.2, 26, 40)
    const hi = Float64Array.of(-5, 0.1, 0.9, 27, 41)
    for (const name of ['ordinal-probit-cuts', 'ordinal-logit-cuts']) {
      const row = tripleLikelihoods()[name] as TripleLikelihood
      for (const y of [0, 1, 2]) {
        const params = { categories: 3 }
        const nodes = [u, y === 0 ? Float64Array.of(0) : lo, y === 2 ? Float64Array.of(0) : hi]
        const byIndex = (row.indexed as NonNullable<TripleLikelihood['indexed']>)(y, params)(nodes)
        const flat = row.density(y, params)
        for (let a = 0; a < u.length; a++)
          for (let b = 0; b < (nodes[1] as Float64Array).length; b++)
            for (let c = 0; c < (nodes[2] as Float64Array).length; c++) {
              const want = flat(
                u[a] as number,
                (nodes[1] as Float64Array)[b] as number,
                (nodes[2] as Float64Array)[c] as number,
              )
              const got = byIndex(a, b, c)
              if (want === Number.NEGATIVE_INFINITY) expect(got, `${name} ${y} at ${a},${b},${c}`).toBe(want)
              else
                expect(Math.abs(got - want), `${name} ${y} at ${a},${b},${c}: ${got} against ${want}`).toBeLessThan(
                  1e-9 * (1 + Math.abs(want)),
                )
            }
      }
    }
  })
  it('the ordinal density is the mass between the cuts, zero where they are out of order', () => {
    const row = tripleLikelihoods()['ordinal-probit-cuts']
    const middle = row?.density(1, { categories: 3 }) as (a: number, b: number, c: number) => number
    expect(middle(0.2, -0.5, 0.9)).toBeCloseTo(logDiffPhi(-0.7, 0.7), 12)
    expect(middle(0.2, 0.9, -0.5)).toBe(Number.NEGATIVE_INFINITY)
    expect(middle(0.2, 0.3, 0.3)).toBe(Number.NEGATIVE_INFINITY)
    // an interval far in the tail keeps its precision: the log mass of a unit interval at 30 sds is about -454
    expect(middle(0, 30, 31)).toBeCloseTo(logDiffPhi(30, 31), 8)
    expect(Number.isFinite(middle(0, 30, 31))).toBe(true)
    const first = row?.density(0, { categories: 3 }) as (a: number, b: number, c: number) => number
    expect(first(0.2, 99, 0.5)).toBeCloseTo(logDiffPhi(Number.NEGATIVE_INFINITY, 0.3), 12)
    // the categories of one predictor and the cuts -0.4 and 0.8 have the masses of the three intervals: they sum to one
    const at = (k: number, lower: number, upper: number): number =>
      Math.exp((row as TripleLikelihood).density(k, { categories: 3 })(0.3, lower, upper))
    expect(at(0, 99, -0.4) + at(1, -0.4, 0.8) + at(2, 0.8, 99)).toBeCloseTo(1, 12)
  })
  it('is in the default registry with its parameters declared, carries a log evidence and refuses what it cannot take', () => {
    const registry = defaultOperators()
    expect(registry['likelihood-triple']?.logEvidence).toBeDefined()
    expect(op.check?.({ likelihood: 'student-t-full', y: 1 }, [gaussian, gamma, gamma])).toBeUndefined()
    expect(op.check?.({ likelihood: 'student-t-full', y: 1 }, [gamma, gaussian, gaussian])).toMatch(
      /not a gamma variable/,
    )
    expect(op.check?.({ likelihood: 'ordinal-probit-cuts', y: 1, categories: 3 }, [gaussian, gamma, gaussian])).toMatch(
      /lower cut .* not a gamma variable/,
    )
    expect(op.check?.({ likelihood: 'ordinal-probit-cuts', y: 3, categories: 3 }, [])).toMatch(
      /from 0 to categories - 1/,
    )
    expect(op.check?.({ likelihood: 'ordinal-probit-cuts', y: 1, categories: 1 }, [])).toMatch(/at least 2/)
    expect(op.check?.({ likelihood: 'nope', y: 1 }, [])).toMatch(/unknown likelihood/)
    const declared = op.params as (p: Record<string, unknown>) => Record<string, unknown>
    expect(Object.keys(declared({ likelihood: 'ordinal-logit-cuts', y: 1, categories: 3 }))).toEqual([
      'likelihood',
      'y',
      'categories',
    ])
  })
  it('the mean field of the Student-t sends a finite message to each port, the ordinal does for its first and last category and says why it has none between two cuts', () => {
    const params = { likelihood: 'student-t-full', y: 1.2 }
    const args = [cavity(0.3, 0.8), cavity(-0.6, 0.4), cavity(0.8, 0.5)]
    for (const to of [0, 1, 2]) {
      const message = op.vmp?.({ mode: 'vmp', to, args, params, power: 1 }).message ?? []
      expect(message.length).toBe(2)
      expect(message.every(Number.isFinite)).toBe(true)
    }
    expect(Number.isFinite(op.logEvidence?.({ mode: 'vmp', to: 0, args, params, power: 1 }))).toBe(true)
    expect(op.unsupported?.('vmp', { likelihood: 'ordinal-probit-cuts', y: 1, categories: 3 })).toMatch(/out of order/)
    expect(op.unsupported?.('vmp', params)).toBeUndefined()
    // The first and the last category have one cut and a smooth density: they run (a message to each port, the unused cut included).
    for (const y of [0, 2]) {
      const ordinal = { likelihood: 'ordinal-probit-cuts', y, categories: 3 }
      expect(op.unsupported?.('vmp', ordinal)).toBeUndefined()
      for (const to of [0, 1, 2]) {
        const message = op.vmp?.({ mode: 'vmp', to, args, params: ordinal, power: 1 }).message ?? []
        expect(message.every(Number.isFinite)).toBe(true)
      }
    }
    expect(op.unsupported?.('ep', { likelihood: 'ordinal-probit-cuts', y: 1, categories: 3 })).toBeUndefined()
  })
})
