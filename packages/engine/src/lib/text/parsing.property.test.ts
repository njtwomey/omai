import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { metricSeries, recorder, runAlgorithm } from '../trace/index.ts'
import {
  applyArcStandard,
  arcStandardOracle,
  cyk,
  earley,
  eisner,
  grammar,
  insideOutside,
  isProjective,
  type ParseTree,
  pcfgEmAlgorithm,
  trainPcfg,
} from './index.ts'

const cnf = grammar(`
S -> A B [0.4] | B A [0.3] | S S [0.3]
A -> A A [0.2] | 'a' [0.5] | 'b' [0.3]
B -> B B [0.4] | 'b' [0.6]
`)

/** Count each production in a tree (by lhs and rhs). */
function productions(t: ParseTree, out: Map<string, number>): void {
  const rhs = t.children.map((c) => (typeof c === 'string' ? `'${c}'` : c.label)).join(' ')
  const key = `${t.label} -> ${rhs}`
  out.set(key, (out.get(key) ?? 0) + 1)
  for (const c of t.children) if (typeof c !== 'string') productions(c, out)
}

const probOf = new Map(
  cnf.productions.map((p) => [
    `${p.lhs} -> ${p.rhs.map((s) => ('terminal' in s ? `'${s.terminal}'` : s.nonterminal)).join(' ')}`,
    p.probability,
  ]),
)

describe('probabilistic parsing agrees with enumerating every parse', () => {
  it('inside-outside expected counts are posterior-weighted production counts', () => {
    fc.assert(
      fc.property(fc.array(fc.constantFrom('a', 'b'), { minLength: 1, maxLength: 5 }), (tokens) => {
        const trees = earley(cnf, tokens, { maxTrees: 100000 }).trees
        expect(cyk(cnf, tokens).count).toBe(trees.length)
        const io = insideOutside(cnf, tokens)
        let z = 0
        const expected = new Map<string, number>()
        for (const t of trees) {
          const c = new Map<string, number>()
          productions(t, c)
          let p = 1
          for (const [k, n] of c) p *= (probOf.get(k) ?? 0) ** n
          z += p
          for (const [k, n] of c) expected.set(k, (expected.get(k) ?? 0) + p * n)
        }
        expect(io.probability).toBeCloseTo(z, 14)
        for (const [i, p] of cnf.productions.entries()) {
          const key = `${p.lhs} -> ${p.rhs.map((s) => ('terminal' in s ? `'${s.terminal}'` : s.nonterminal)).join(' ')}`
          expect(io.expectedCounts[i] ?? 0).toBeCloseTo(z > 0 ? (expected.get(key) ?? 0) / z : 0, 10)
        }
      }),
    )
  })

  it('EM never decreases the log-likelihood and keeps each left-hand side normalised', () => {
    const rec = recorder()
    const data = [
      ['a', 'b'],
      ['b', 'a', 'b'],
      ['a', 'b', 'b', 'a'],
      ['b', 'b'],
    ]
    const { grammar: g } = trainPcfg(cnf, data, { maxIter: 30, tracer: rec })
    const ll = metricSeries(rec.run(), 'logLikelihood').values
    for (let i = 1; i < ll.length; i++) expect(ll[i] ?? 0).toBeGreaterThanOrEqual((ll[i - 1] ?? 0) - 1e-12)
    const sums = new Map<string, number>()
    for (const p of g.productions) sums.set(p.lhs, (sums.get(p.lhs) ?? 0) + p.probability)
    for (const s of sums.values()) expect(s).toBeCloseTo(1, 12)
    expect(() => runAlgorithm(pcfgEmAlgorithm(cnf, [['c']]), { maxSteps: 2 })).toThrow(/no parse/)
  })
})

/** Every projective tree over n tokens, by brute force over head assignments. */
function projectiveTrees(n: number): number[][] {
  const out: number[][] = []
  const heads = new Array<number>(n + 1).fill(-1)
  const rec = (d: number): void => {
    if (d > n) {
      // A tree: every token reaches the root without cycles.
      for (let i = 1; i <= n; i++) {
        let x = i
        for (let steps = 0; x !== 0; steps++) {
          if (steps > n) return
          x = heads[x] ?? 0
        }
      }
      if (isProjective(heads)) out.push([...heads])
      return
    }
    for (let h = 0; h <= n; h++) {
      if (h === d) continue
      heads[d] = h
      rec(d + 1)
    }
  }
  rec(1)
  return out
}

describe('dependency parsing', () => {
  it("Eisner's algorithm finds the best projective tree", () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 1, max: 4 }),
        fc.array(fc.integer({ min: -20, max: 20 }), { minLength: 25, maxLength: 25 }),
        (n, xs) => {
          const scores = Array.from({ length: n + 1 }, (_, h) =>
            Array.from({ length: n + 1 }, (_, d) => (xs[h * 5 + d] ?? 0) + 0.001 * (h * 7 + d)),
          )
          const all = projectiveTrees(n)
          const score = (hs: readonly number[]): number =>
            hs.slice(1).reduce((s, h, i) => s + (scores[h]?.[i + 1] ?? 0), 0)
          let best = Number.NEGATIVE_INFINITY
          for (const t of all) best = Math.max(best, score(t))
          const e = eisner(scores)
          expect(e.score).toBeCloseTo(best, 9)
          expect(score(e.heads)).toBeCloseTo(e.score, 9)
          expect(isProjective(e.heads)).toBe(true)
          const single = eisner(scores, { singleRoot: true })
          let bestSingle = Number.NEGATIVE_INFINITY
          for (const t of all) if (t.filter((h) => h === 0).length === 1) bestSingle = Math.max(bestSingle, score(t))
          expect(single.score).toBeCloseTo(bestSingle, 9)
          expect(single.heads.filter((h) => h === 0).length).toBe(1)
        },
      ),
      { numRuns: 60 },
    )
  })

  it('the arc-standard oracle rebuilds every projective tree', () => {
    for (let n = 1; n <= 4; n++)
      for (const t of projectiveTrees(n)) {
        const seq = arcStandardOracle(t)
        expect(seq.length).toBe(2 * n)
        expect(applyArcStandard(n, seq)).toEqual(t)
      }
    expect(() => arcStandardOracle([-1, 3, 4, 0, 3])).toThrow(/not projective/)
  })
})
