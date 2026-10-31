import { describe, expect, it } from 'vitest'
import { runAlgorithm } from '../trace/index.ts'
import {
  belief,
  certaintyInference,
  combineCertainty,
  combineDempster,
  combineYager,
  defuzzify,
  evidenceAlgorithm,
  linspaceUniverse,
  mamdani,
  massFunction,
  pignistic,
  plausibility,
  possibilityMeasures,
  sugeno,
  tConorm,
  tNorm,
  trapezoid,
  triangular,
} from './index.ts'

describe('certainty factors', () => {
  it('combine as MYCIN defines, commutatively and associatively', () => {
    expect(combineCertainty(0.6, 0.4)).toBeCloseTo(0.76, 12)
    expect(combineCertainty(-0.6, -0.4)).toBeCloseTo(-0.76, 12)
    expect(combineCertainty(0.8, -0.4)).toBeCloseTo(0.4 / 0.6, 12)
    expect(combineCertainty(0.3, 0.5)).toBeCloseTo(combineCertainty(0.5, 0.3), 14)
    expect(combineCertainty(combineCertainty(0.3, 0.5), -0.2)).toBeCloseTo(
      combineCertainty(0.3, combineCertainty(0.5, -0.2)),
      12,
    )
    expect(combineCertainty(1, 0.5)).toBe(1)
    expect(() => combineCertainty(1, -1)).toThrow()
    expect(() => combineCertainty(2, 0)).toThrow()
  })

  it('chain through rules: min of the conditions, times the rule factor, combined across rules', () => {
    const rules = [
      { conditions: ['fever', 'cough'], conclusion: 'flu', factor: 0.8 },
      { conditions: ['aches'], conclusion: 'flu', factor: 0.5 },
      { conditions: ['flu'], conclusion: 'rest', factor: 0.9 },
    ]
    const cf = certaintyInference(rules, { fever: 0.9, cough: 0.6, aches: 1 })
    expect(cf['flu']).toBeCloseTo(0.48 + 0.5 * 0.52, 12)
    expect(cf['rest']).toBeCloseTo(0.9 * (cf['flu'] ?? 0), 12)
  })

  it('a weak condition (at most 0.2) does not fire a rule', () => {
    expect(certaintyInference([{ conditions: ['a'], conclusion: 'b', factor: 1 }], { a: 0.2 })['b']).toBeUndefined()
  })
})

describe('Dempster-Shafer', () => {
  const f = ['a', 'b', 'c']
  it('belief and plausibility bracket the mass, and plausibility is one minus belief in the complement', () => {
    const m = massFunction(f, [
      [['a'], 0.4],
      [['a', 'b'], 0.3],
      [['a', 'b', 'c'], 0.3],
    ])
    expect(belief(m, ['a', 'b'])).toBeCloseTo(0.7, 12)
    expect(plausibility(m, ['c'])).toBeCloseTo(0.3, 12)
    for (const h of [['a'], ['b'], ['b', 'c']])
      expect(plausibility(m, h)).toBeCloseTo(
        1 -
          belief(
            m,
            f.filter((x) => !h.includes(x)),
          ),
        12,
      )
  })

  it("Zadeh's counterexample: Dempster gives certainty to the 1% hypothesis, Yager keeps the doubt", () => {
    const m1 = massFunction(f, [
      [['a'], 0.99],
      [['b'], 0.01],
    ])
    const m2 = massFunction(f, [
      [['c'], 0.99],
      [['b'], 0.01],
    ])
    const d = combineDempster(m1, m2)
    expect(d.conflict).toBeCloseTo(0.9999, 12)
    expect(d.mass['b']).toBeCloseTo(1, 12)
    const y = combineYager(f, m1, m2)
    expect(y.mass['a|b|c']).toBeCloseTo(0.9999, 12)
    expect(y.mass['b']).toBeCloseTo(0.0001, 12)
  })

  it("Dempster's rule is commutative and associative, with the vacuous mass as identity", () => {
    const m1 = massFunction(f, [
      [['a'], 0.5],
      [['a', 'b'], 0.5],
    ])
    const m2 = massFunction(f, [
      [['b'], 0.3],
      [['a', 'b', 'c'], 0.7],
    ])
    const m3 = massFunction(f, [
      [['a', 'c'], 0.6],
      [['b', 'c'], 0.4],
    ])
    const close = (x: Record<string, number>, y: Record<string, number>): void => {
      expect(Object.keys(x).sort()).toEqual(Object.keys(y).sort())
      for (const k of Object.keys(x)) expect(x[k]).toBeCloseTo(y[k] as number, 12)
    }
    close(combineDempster(m1, m2).mass, combineDempster(m2, m1).mass)
    close(
      combineDempster(combineDempster(m1, m2).mass, m3).mass,
      combineDempster(m1, combineDempster(m2, m3).mass).mass,
    )
    close(combineDempster(m1, { 'a|b|c': 1 }).mass, m1)
  })

  it('the pignistic transform shares mass equally and sums to one', () => {
    const p = pignistic(
      massFunction(f, [
        [['a'], 0.4],
        [['a', 'b'], 0.3],
        [['a', 'b', 'c'], 0.3],
      ]),
    )
    expect(p['a']).toBeCloseTo(0.4 + 0.15 + 0.1, 12)
    expect(Object.values(p).reduce((s, v) => s + v, 0)).toBeCloseTo(1, 12)
  })

  it('checks its inputs', () => {
    expect(() => massFunction(f, [[['a'], 0.5]])).toThrow()
    expect(() => massFunction(f, [[['z'], 1]])).toThrow()
    expect(() => combineDempster({ a: 1 }, { b: 1 })).toThrow()
  })

  it('adds evidence one body per step and the belief interval narrows', () => {
    const e = [
      massFunction(f, [
        [['a'], 0.5],
        [['a', 'b', 'c'], 0.5],
      ]),
      massFunction(f, [
        [['a'], 0.5],
        [['a', 'b', 'c'], 0.5],
      ]),
    ]
    const { state, steps } = runAlgorithm(evidenceAlgorithm(f, e), { maxSteps: 10, seed: 0 })
    expect(steps).toBe(2)
    expect(state.intervals['a']?.[0]).toBeCloseTo(0.75, 12)
    expect(state.intervals['a']?.[1]).toBe(1)
    expect(state.totalConflict).toBe(0)
  })
})

describe('fuzzy logic', () => {
  it('membership functions and t-norms', () => {
    expect(triangular(0, 5, 10)(7.5)).toBeCloseTo(0.5, 12)
    expect(trapezoid(0, 2, 4, 6)(5)).toBeCloseTo(0.5, 12)
    expect(tNorm('min', 0.6, 0.7)).toBe(0.6)
    expect(tNorm('product', 0.6, 0.7)).toBeCloseTo(0.42, 12)
    expect(tNorm('lukasiewicz', 0.6, 0.7)).toBeCloseTo(0.3, 12)
    // De Morgan: the conorm is the dual of the norm under 1 - x.
    for (const k of ['min', 'product', 'lukasiewicz'] as const)
      expect(tConorm(k, 0.3, 0.8)).toBeCloseTo(1 - tNorm(k, 0.7, 0.2), 12)
  })

  it('defuzzifiers of a clipped triangle', () => {
    const u = linspaceUniverse(0, 10, 1001)
    const clipped = u.map((x) => Math.min(0.5, triangular(0, 5, 10)(x)))
    expect(defuzzify(u, clipped, 'centroid')).toBeCloseTo(5, 9)
    expect(defuzzify(u, clipped, 'bisector')).toBeCloseTo(5, 1)
    expect(defuzzify(u, clipped, 'mom')).toBeCloseTo(5, 9)
    expect(defuzzify(u, clipped, 'som')).toBeCloseTo(2.5, 9)
    expect(defuzzify(u, clipped, 'lom')).toBeCloseTo(7.5, 9)
  })

  // The tipping problem of the scikit-fuzzy tutorial: quality and service from 0 to 10, a tip from 0 to 25.
  const q = linspaceUniverse(0, 10, 11)
  const quality = {
    universe: q,
    sets: { poor: triangular(0, 0, 5), average: triangular(0, 5, 10), good: triangular(5, 10, 10) },
  }
  const service = {
    universe: q,
    sets: { poor: triangular(0, 0, 5), average: triangular(0, 5, 10), good: triangular(5, 10, 10) },
  }
  const tipSets = { low: triangular(0, 0, 13), medium: triangular(0, 13, 25), high: triangular(13, 25, 25) }
  const rules = [
    {
      conditions: [
        ['quality', 'poor'],
        ['service', 'poor'],
      ] as const,
      connective: 'or' as const,
      consequent: 'low',
    },
    { conditions: [['service', 'average']] as const, connective: 'and' as const, consequent: 'medium' },
    {
      conditions: [
        ['service', 'good'],
        ['quality', 'good'],
      ] as const,
      connective: 'or' as const,
      consequent: 'high',
    },
  ]

  it('Mamdani inference: rule strengths by hand, and the centroid of the aggregated curve by direct integration', () => {
    const tip = { universe: linspaceUniverse(0, 25, 2501), sets: tipSets }
    const r = mamdani({ quality, service }, tip, rules, { quality: 6.5, service: 9.8 })
    // poor or poor = max(0, 0); average = (10 - 9.8) / 5; good or good = max((9.8 - 5) / 5, (6.5 - 5) / 5)
    expect(r.strengths[0]).toBeCloseTo(0, 12)
    expect(r.strengths[1]).toBeCloseTo(0.04, 12)
    expect(r.strengths[2]).toBeCloseTo(0.96, 12)
    // The aggregate is the maximum of the clipped sets; integrate x mu and mu on a much finer grid.
    let moment = 0
    let area = 0
    const n = 250000
    for (let i = 0; i <= n; i++) {
      const x = (25 * i) / n
      const mu = Math.max(Math.min(0.04, tipSets.medium(x)), Math.min(0.96, tipSets.high(x)))
      moment += x * mu
      area += mu
    }
    expect(r.output).toBeCloseTo(moment / area, 3)
    // On the coarse integer grid the answer is near 19.9: mostly the clipped "high" set.
    const coarse = mamdani(
      { quality, service },
      { universe: Array.from({ length: 26 }, (_, i) => i), sets: tipSets },
      rules,
      { quality: 6.5, service: 9.8 },
    )
    expect(coarse.output).toBeGreaterThan(19.5)
    expect(coarse.output).toBeLessThan(20.5)
  })

  it('Sugeno inference is the strength-weighted mean of constants and of linear outputs', () => {
    const x = { universe: q, sets: { low: triangular(0, 0, 10), high: triangular(0, 10, 10) } }
    const r = [
      { conditions: [['x', 'low']] as const, connective: 'and' as const, constant: 0 },
      { conditions: [['x', 'high']] as const, connective: 'and' as const, constant: 10 },
    ]
    expect(sugeno({ x }, r, { x: 2.5 }).output).toBeCloseTo(2.5, 12)
    const lin = [
      { conditions: [['x', 'low']] as const, connective: 'and' as const, linear: { bias: 1, coefficients: { x: 2 } } },
      { conditions: [['x', 'high']] as const, connective: 'and' as const, linear: { bias: 0, coefficients: { x: 1 } } },
    ]
    // strengths 0.75 and 0.25 at x = 2.5: (0.75 * 6 + 0.25 * 2.5) / 1
    expect(sugeno({ x }, lin, { x: 2.5 }).output).toBeCloseTo(0.75 * 6 + 0.25 * 2.5, 12)
  })

  it('possibility and necessity', () => {
    const r = possibilityMeasures({ a: 1, b: 0.7, c: 0.2 }, ['a', 'b'])
    expect(r.possibility).toBe(1)
    expect(r.necessity).toBeCloseTo(0.8, 12)
    expect(() => possibilityMeasures({ a: 0.5 }, ['a'])).toThrow()
  })
})
