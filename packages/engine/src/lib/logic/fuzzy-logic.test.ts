import { describe, expect, it } from 'vitest'
import { gradCheck } from '../autodiff/index.ts'
import { flatObjective, smoothSolverConfig } from '../optim/index.ts'
import { key } from '../random/index.ts'
import { asTensor, item, sliceAxis, squeeze, toFlat, type Value } from '../tensor/index.ts'
import { recorder, runAlgorithm } from '../trace/index.ts'
import {
  aggregators,
  formulaSatisfaction,
  fuzzySemantics,
  fuzzyTheory,
  fuzzyTheoryAlgorithm,
  hamacherFamily,
  type NormName,
  networkPredicate,
  normFamily,
  pMean,
  pMeanError,
  tablePredicate,
  tConorm,
  tNorm,
  trainFuzzyTheory,
} from './index.ts'

const FAMILIES: NormName[] = ['product', 'lukasiewicz', 'goedel', 'hamacher']
const GRID = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
const num = (x: Value): number => item(x)

describe('t-norm families', () => {
  it.each(FAMILIES)('%s: the t-norm axioms hold on a grid', (name) => {
    const f = normFamily(name)
    for (const a of GRID) {
      expect(num(f.and(a, 1))).toBeCloseTo(a, 9)
      expect(num(f.and(a, 0))).toBeCloseTo(0, 9)
      expect(num(f.or(a, 0))).toBeCloseTo(a, 9)
      expect(num(f.or(a, 1))).toBeCloseTo(1, 9)
      expect(num(f.not(f.not(a)))).toBeCloseTo(a, 12)
      for (const b of GRID) {
        expect(num(f.and(a, b))).toBeCloseTo(num(f.and(b, a)), 12)
        // The conorm is the De Morgan dual and bounds the norm from above.
        expect(num(f.or(a, b))).toBeCloseTo(1 - num(f.and(1 - a, 1 - b)), 12)
        expect(num(f.or(a, b))).toBeGreaterThanOrEqual(num(f.and(a, b)) - 1e-12)
        for (const c of GRID) expect(num(f.and(f.and(a, b), c))).toBeCloseTo(num(f.and(a, f.and(b, c))), 9)
      }
    }
  })

  it('agrees with the plain-number t-norms of uncertain.ts', () => {
    const pairs: [NormName, 'min' | 'product' | 'lukasiewicz'][] = [
      ['goedel', 'min'],
      ['product', 'product'],
      ['lukasiewicz', 'lukasiewicz'],
    ]
    for (const [name, kind] of pairs) {
      const f = normFamily(name)
      for (const a of GRID)
        for (const b of GRID) {
          expect(num(f.and(a, b))).toBeCloseTo(tNorm(kind, a, b), 12)
          expect(num(f.or(a, b))).toBeCloseTo(tConorm(kind, a, b), 12)
        }
    }
  })

  it.each(['product', 'lukasiewicz', 'goedel'] as NormName[])('%s: the residuum is adjoint to the t-norm', (name) => {
    // T(a, x) <= b exactly when x <= R(a, b).
    const f = normFamily(name)
    for (const a of GRID)
      for (const b of GRID) {
        const r = num(f.residuum(a, b))
        for (const x of GRID) {
          const lhs = num(f.and(a, x)) <= b + 1e-9
          const rhs = x <= r + 1e-9
          expect(lhs, `${name} a=${a} b=${b} x=${x}`).toBe(rhs)
        }
      }
  })

  it('the material implication of the product family is Reichenbach’s 1 - a + ab', () => {
    expect(num(normFamily('product').implies(0.8, 0.5))).toBeCloseTo(1 - 0.8 + 0.4, 12)
  })

  it('Hamacher with lambda 1 is the product, and lambda 0 gives ab / (a + b - ab)', () => {
    expect(num(hamacherFamily(1).and(0.3, 0.8))).toBeCloseTo(0.24, 12)
    expect(num(hamacherFamily(0).and(0.5, 0.5))).toBeCloseTo(1 / 3, 12)
    expect(() => hamacherFamily(-1)).toThrow()
  })

  it.each(FAMILIES)('%s: reducing along an axis equals folding the binary connective', (name) => {
    const f = normFamily(name)
    const x = [0.9, 0.6, 0.75, 0.4]
    const and = x.reduce((acc, v) => num(f.and(acc, v)), 1)
    const or = x.reduce((acc, v) => num(f.or(acc, v)), 0)
    expect(num(f.reduceAnd(x, 0))).toBeCloseTo(and, 10)
    expect(num(f.reduceOr(x, 0))).toBeCloseTo(or, 10)
    const rows = [
      [0.9, 0.6],
      [0.2, 0.7],
    ]
    expect(toFlat(f.reduceAnd(rows, 1)).map((v) => Number(v.toFixed(10)))).toEqual(
      rows.map((r) => Number(r.reduce((acc, v) => num(f.and(acc, v)), 1).toFixed(10))),
    )
  })
})

describe('aggregators', () => {
  it('the power means are the mean at p = 1 and approach the extremes as p grows', () => {
    const x = [0.2, 0.9, 0.5, 0.7]
    expect(num(pMean(1)(x, 0))).toBeCloseTo(0.575, 12)
    expect(num(pMeanError(1)(x, 0))).toBeCloseTo(0.575, 12)
    expect(num(pMean(60)(x, 0))).toBeGreaterThan(0.85)
    expect(num(pMeanError(60)(x, 0))).toBeLessThan(0.3)
    expect(num(pMean(60)(x, 0))).toBeLessThanOrEqual(0.9)
    expect(num(pMeanError(2)(x, 0))).toBeLessThan(num(pMeanError(1)(x, 0)))
    expect(num(aggregators.min(x, 0))).toBe(0.2)
    expect(num(aggregators.max(x, 0))).toBe(0.9)
    expect(() => pMean(0.5)).toThrow()
  })

  it('quantifies along a chosen axis of a matrix', () => {
    const m = [
      [1, 0.5],
      [0.2, 0.4],
    ]
    expect(toFlat(aggregators.min(m, 1))).toEqual([0.5, 0.2])
    expect(toFlat(aggregators.tnorm(normFamily('product'))(m, 0)).map((v) => Number(v.toFixed(10)))).toEqual([0.2, 0.2])
  })
})

describe('propositional satisfaction', () => {
  it('evaluates formulas over truth tensors with the chosen family', () => {
    expect(num(formulaSatisfaction('a -> b', { a: 0.8, b: 0.5 }))).toBeCloseTo(0.6, 12)
    const sem = fuzzySemantics({ norm: 'lukasiewicz' })
    expect(num(formulaSatisfaction('a & b', { a: 0.8, b: 0.5 }, sem))).toBeCloseTo(0.3, 12)
    expect(num(formulaSatisfaction('a | b', { a: 0.8, b: 0.5 }, sem))).toBeCloseTo(1, 12)
    expect(
      num(
        formulaSatisfaction('a -> b', { a: 0.8, b: 0.5 }, fuzzySemantics({ norm: 'goedel', implication: 'residuum' })),
      ),
    ).toBe(0.5)
    expect(toFlat(formulaSatisfaction('~a', { a: [0.1, 0.9] }))).toEqual([0.9, 0.09999999999999998])
    expect(() => formulaSatisfaction('a & c', { a: 1 })).toThrow()
  })

  it('on crisp truth values it is classical logic, for every family', () => {
    const formula = '(a | b) & ~(a & c) -> (b <-> c)'
    for (const name of FAMILIES) {
      const sem = fuzzySemantics({ norm: name })
      for (let m = 0; m < 8; m++) {
        const a = m & 1
        const b = (m >> 1) & 1
        const c = (m >> 2) & 1
        const classical = !((a || b) && !(a && c)) || b === c
        expect(Math.round(num(formulaSatisfaction(formula, { a, b, c }, sem)))).toBe(classical ? 1 : 0)
      }
    }
  })

  it.each(FAMILIES)('%s: the gradient agrees with finite differences', (name) => {
    const sem = fuzzySemantics({ norm: name })
    const f = (x: Value): Value =>
      formulaSatisfaction('(a & b) | (~a -> c) <-> b', { a: sliceOf(x, 0), b: sliceOf(x, 1), c: sliceOf(x, 2) }, sem)
    const r = gradCheck(f, [0.35, 0.62, 0.81], { eps: 1e-6, rtol: 1e-4, atol: 1e-6 })
    expect(r.failures).toEqual([])
  })
})

function sliceOf(x: Value, i: number): Value {
  return squeeze(sliceAxis(x, 0, i, i + 1), 0)
}

// A friends-and-smokers theory: facts, and two axioms that derive the rest.
const ONE_HOT = [
  [1, 0, 0, 0],
  [0, 1, 0, 0],
  [0, 0, 1, 0],
  [0, 0, 0, 1],
]
const smokersTheory = (semantics?: Parameters<typeof fuzzyTheory>[0]['semantics']) =>
  fuzzyTheory({
    variables: { x: ONE_HOT, y: ONE_HOT },
    constants: {
      anna: ONE_HOT[0] as number[],
      bob: ONE_HOT[1] as number[],
      carl: ONE_HOT[2] as number[],
      dave: ONE_HOT[3] as number[],
    },
    predicates: { smokes: tablePredicate([4]), friends: tablePredicate([4, 4]), cancer: tablePredicate([4]) },
    axioms: [
      { formula: 'friends(anna,bob)' },
      { formula: 'friends(bob,carl)' },
      { formula: 'friends(carl,dave)' },
      { formula: 'smokes(anna)' },
      { formula: '-friends(anna,carl)' },
      { formula: '-friends(bob,dave)' },
      { formula: '-friends(dave,anna)' },
      { formula: 'all x y. (friends(x,y) & smokes(x) -> smokes(y))' },
      { formula: 'all x. (smokes(x) -> cancer(x))' },
    ],
    ...(semantics === undefined ? {} : { semantics }),
  })

describe('a fuzzy theory', () => {
  it('evaluates quantifiers over the domains, and crisp tables reproduce classical truth', () => {
    // A crisp world: p holds of constants 0 and 1, q of constant 1 only.
    const t = fuzzyTheory({
      variables: { x: ONE_HOT, y: ONE_HOT, X: ONE_HOT },
      predicates: { p: tablePredicate([4]), q: tablePredicate([4]), r: tablePredicate([4, 4]) },
      axioms: [{ formula: 'p(a)' }],
      constants: { a: ONE_HOT[0] as number[] },
      semantics: { norm: 'goedel', forall: aggregators.min, exists: aggregators.max },
    })
    const big = 40
    const theta = [
      ...[big, big, -big, -big], // p
      ...[-big, big, -big, -big], // q
      ...Array.from({ length: 16 }, (_, k) => (Math.floor(k / 4) === k % 4 ? big : -big)), // r: identity relation
    ]
    // Sorted predicate order: p, q, r.
    expect(t.predicateNames).toEqual(['p', 'q', 'r'])
    const truth = (f: string): number => Math.round(num(t.truth(theta, f)))
    expect(truth('all x. (q(x) -> p(x))')).toBe(1)
    expect(truth('all x. (p(x) -> q(x))')).toBe(0)
    expect(truth('exists x. (p(x) & -q(x))')).toBe(1)
    expect(truth('exists x. (q(x) & -p(x))')).toBe(0)
    expect(truth('all x. exists y. r(x,y)')).toBe(1)
    expect(truth('exists y. all x. r(x,y)')).toBe(0)
    expect(truth('all x y. (r(x,y) -> r(y,x))')).toBe(1)
    expect(truth('p(a) & -q(a)')).toBe(1)
    // A free variable is read as universally quantified.
    expect(truth('q(X) -> p(X)')).toBe(1)
  })

  it('trains the predicates to satisfy the axioms, deriving what they imply', () => {
    const t = smokersTheory()
    const before = trainFuzzyTheory(t, { maxSteps: 0 })
    const fit = trainFuzzyTheory(t)
    expect(before.satisfaction).toBeLessThan(0.75)
    expect(fit.satisfaction).toBeGreaterThan(0.93)
    // The chain of friends carries smoking from anna to dave, and smoking implies cancer, though neither was stated.
    for (const who of ['bob', 'carl', 'dave']) expect(num(t.truth(fit.theta, `smokes(${who})`))).toBeGreaterThan(0.85)
    for (const who of ['anna', 'bob', 'carl', 'dave'])
      expect(num(t.truth(fit.theta, `cancer(${who})`))).toBeGreaterThan(0.85)
    expect(num(t.truth(fit.theta, 'friends(anna,carl)'))).toBeLessThan(0.15)
  })

  it('the loss gradient agrees with finite differences', () => {
    const t = smokersTheory()
    const problem = t.problem()
    const flat = flatObjective(problem.objective, problem.x0)
    const x = Array.from({ length: t.size }, (_, i) => Math.sin(i + 1))
    const g = flat.valueAndGrad(x).grad
    for (const j of [0, 3, 5, 11, 18, 22]) {
      const up = [...x]
      const down = [...x]
      up[j] = (up[j] as number) + 1e-6
      down[j] = (down[j] as number) - 1e-6
      expect(g[j]).toBeCloseTo((flat.value(up) - flat.value(down)) / 2e-6, 6)
    }
  })

  it('the optimiser is a slot: Adam through the stochastic mode also satisfies the axioms', () => {
    const t = smokersTheory()
    const solver = smoothSolverConfig({
      kind: 'gradient',
      optimiser: { kind: 'adam' },
      learningRate: 0.2,
      gradTol: 1e-6,
    })
    const fit = trainFuzzyTheory(t, { solver, maxSteps: 600 })
    expect(fit.satisfaction).toBeGreaterThan(0.9)
    const quasi = trainFuzzyTheory(t, { solver: smoothSolverConfig({ kind: 'bfgs', gradTol: 1e-8 }) })
    expect(quasi.satisfaction).toBeGreaterThan(0.93)
  }, 60_000)

  it('other families satisfy the axioms too, and the satisfaction rises on the tape', () => {
    const t = smokersTheory({ norm: 'lukasiewicz' })
    const rec = recorder()
    const run = runAlgorithm(fuzzyTheoryAlgorithm(t), { seed: 0, maxSteps: 200, tracer: rec })
    expect(run.steps).toBeGreaterThan(5)
    const text = JSON.stringify(rec.run())
    for (const p of ['theory/satisfaction', 'theory/axioms', 'theory/parameters', 'solver/objective'])
      expect(text).toContain(p)
    expect(trainFuzzyTheory(t).satisfaction).toBeGreaterThan(0.9)
  })

  it('a network predicate on features learns facts about constants', () => {
    const features = [[-2], [-1], [1], [2]]
    const t = fuzzyTheory({
      variables: { x: features },
      constants: { c0: [-2], c1: [-1], c2: [1], c3: [2] },
      predicates: { big: networkPredicate([1], 3, key(2)) },
      axioms: [{ formula: '-big(c0)' }, { formula: '-big(c1)' }, { formula: 'big(c2)' }, { formula: 'big(c3)' }],
      aggregate: 'mean',
    })
    const fit = trainFuzzyTheory(t, { maxSteps: 400 })
    expect(fit.satisfaction).toBeGreaterThan(0.95)
    expect(num(t.truth(fit.theta, 'all x. (big(x) | -big(x))'))).toBeGreaterThan(0.5)
    const r = gradCheck((x: Value) => t.satisfaction(x).total, t.x0, { rtol: 1e-4, atol: 1e-6, vmap: false })
    expect(r.failures).toEqual([])
  })

  it('rejects undeclared names', () => {
    const t = fuzzyTheory({
      variables: { x: ONE_HOT },
      predicates: { p: tablePredicate([4]) },
      axioms: [{ formula: 'all x. q(x)' }],
    })
    expect(() => t.satisfaction(t.x0)).toThrow()
    expect(() => t.truth(t.x0, 'p(z)')).toThrow()
    expect(() => tablePredicate([4]).truth(asTensor([0, 0, 0, 0], 'float64'), [])).toThrow()
  })
})
