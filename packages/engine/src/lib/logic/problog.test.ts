import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { runAlgorithm } from '../trace/index.ts'
import {
  conditionalProbability,
  disjunctionProbability,
  groundProbabilistic,
  learnProbabilities,
  type ProbabilisticProgram,
  probLogEmAlgorithm,
  proofSets,
  successProbability,
} from './index.ts'

const PATH_RULES = ['path(X,Y) :- edge(X,Y)', 'path(X,Y) :- edge(X,Z), path(Z,Y)']

describe('success probability', () => {
  it('two routes: the probability of the disjunction of the proofs', () => {
    const program: ProbabilisticProgram = {
      facts: [
        [0.8, 'edge(a,b)'],
        [0.6, 'edge(b,c)'],
        [0.3, 'edge(a,c)'],
      ],
      certain: [],
      rules: PATH_RULES,
    }
    const r = successProbability(program, 'path(a,c)')
    expect(r.probability).toBeCloseTo(1 - (1 - 0.48) * (1 - 0.3), 12)
    expect(r.proofs.map((p) => p.facts.join('+')).sort()).toEqual(['edge(a,b)+edge(b,c)', 'edge(a,c)'])
    expect(r.proofs.map((p) => p.probability).sort()).toEqual([0.3, 0.48].sort())
  })

  it('the burglar alarm', () => {
    const alarm: ProbabilisticProgram = {
      facts: [
        [0.1, 'burglary'],
        [0.2, 'earthquake'],
      ],
      certain: [],
      rules: ['alarm :- burglary', 'alarm :- earthquake'],
    }
    expect(successProbability(alarm, 'alarm').probability).toBeCloseTo(0.28, 12)
    expect(conditionalProbability(alarm, 'burglary', [['alarm', true]])).toBeCloseTo(0.1 / 0.28, 12)
    expect(conditionalProbability(alarm, 'burglary', [['alarm', false]])).toBe(0)
    expect(() =>
      conditionalProbability(
        {
          ...alarm,
          facts: [
            [0, 'burglary'],
            [0, 'earthquake'],
          ],
        },
        'burglary',
        [['alarm', true]],
      ),
    ).toThrow()
  })

  it('an underivable atom has probability 0 and a certain fact probability 1', () => {
    const program: ProbabilisticProgram = { facts: [[0.5, 'p']], certain: ['q'], rules: ['r :- p, q'] }
    expect(successProbability(program, 'z').probability).toBe(0)
    expect(successProbability(program, 'q').probability).toBe(1)
    expect(successProbability(program, 'r').probability).toBe(0.5)
  })

  it('cyclic graphs terminate and match enumeration of all possible worlds', () => {
    const edges = fc.uniqueArray(fc.tuple(fc.constantFrom('a', 'b', 'c', 'd'), fc.constantFrom('a', 'b', 'c', 'd')), {
      selector: ([x, y]) => `${x}${y}`,
      minLength: 1,
      maxLength: 6,
    })
    fc.assert(
      fc.property(
        edges,
        fc.array(fc.double({ min: 0.05, max: 0.95, noNaN: true }), { minLength: 6, maxLength: 6 }),
        (es, ps) => {
          const program: ProbabilisticProgram = {
            facts: es.map(([x, y], i) => [ps[i] as number, `edge(${x},${y})`] as const),
            certain: [],
            rules: PATH_RULES,
          }
          // Brute force: sum the probability of every world in which the transitive closure reaches d from a.
          let expected = 0
          for (let mask = 0; mask < 1 << es.length; mask++) {
            let weight = 1
            const reach = new Set<string>(['a'])
            es.forEach((_, i) => {
              weight *= (mask >> i) & 1 ? (ps[i] as number) : 1 - (ps[i] as number)
            })
            for (let changed = true; changed; ) {
              changed = false
              es.forEach(([x, y], i) => {
                if ((mask >> i) & 1 && reach.has(x) && !reach.has(y)) {
                  reach.add(y)
                  changed = true
                }
              })
            }
            // path(a, d) holds when d is reached by a path of at least one edge from a.
            const viaEdge = es.some(([x, y], i) => (mask >> i) & 1 && reach.has(x) && y === 'd')
            if (viaEdge) expected += weight
          }
          expect(successProbability(program, 'path(a,d)').probability).toBeCloseTo(expected, 10)
        },
      ),
      { numRuns: 100 },
    )
  })

  it('rejects a bad probability and a head variable missing from the body', () => {
    expect(() => groundProbabilistic({ facts: [[1.5, 'p']], certain: [], rules: [] })).toThrow()
    expect(() => groundProbabilistic({ facts: [[0.5, 'p(a)']], certain: [], rules: ['q(X,Y) :- p(X)'] })).toThrow()
  })
})

describe('proofs and counting', () => {
  it('proofs are minimal: a superset of a proof is absorbed', () => {
    const g = groundProbabilistic({
      facts: [
        [0.5, 'a'],
        [0.5, 'b'],
      ],
      certain: [],
      rules: ['c :- a', 'c :- a, b'],
    })
    expect(proofSets(g, [])['c']).toEqual([[0]])
  })

  it('weighted model counting equals inclusion-exclusion', () => {
    const p = [0.3, 0.5, 0.7]
    const proofs = [
      [0, 1],
      [1, 2],
      [0, 2],
    ]
    const [a, b, c] = p as [number, number, number]
    const ie = a * b + b * c + a * c - 3 * a * b * c + a * b * c
    expect(disjunctionProbability(proofs, p)).toBeCloseTo(ie, 12)
  })
})

describe('EM', () => {
  it("learns a coin's bias as the frequency of heads in one step", () => {
    const obs = [true, true, true, false, true, false, true, true].map((t) => [['heads', t]] as const)
    const r = learnProbabilities({ facts: [[0.2, 'heads']], certain: [], rules: [] }, obs)
    expect(r.probabilities[0]).toBeCloseTo(6 / 8, 10)
  })

  it('the likelihood never falls and it recovers the parameters behind a noisy-or', () => {
    // alarm :- burglary ; alarm :- earthquake; only the alarm and the earthquake are observed.
    const program: ProbabilisticProgram = {
      facts: [
        [0.5, 'burglary'],
        [0.5, 'earthquake'],
      ],
      certain: [],
      rules: ['alarm :- burglary', 'alarm :- earthquake'],
    }
    const rows: (readonly [string, boolean][])[] = []
    for (let i = 0; i < 100; i++) {
      const e = i % 4 === 0
      const alarm = e || i % 5 === 0
      rows.push([
        ['earthquake', e],
        ['alarm', alarm],
      ])
    }
    const alg = probLogEmAlgorithm(program, rows)
    let s = alg.init({ words: [0, 0] } as never)
    const lls = [s.logLikelihood]
    for (let t = 0; t < 30; t++) {
      s = alg.step(s, {} as never)
      lls.push(s.logLikelihood)
    }
    for (let i = 1; i < lls.length; i++) expect(lls[i]).toBeGreaterThanOrEqual((lls[i - 1] ?? 0) - 1e-9)
    expect(s.probabilities[1]).toBeCloseTo(0.25, 6)
    const { state } = runAlgorithm(alg, { maxSteps: 500, seed: 0 })
    expect(alg.deserialise?.(JSON.parse(JSON.stringify(alg.serialise?.(state))))).toEqual(state)
  })
})
