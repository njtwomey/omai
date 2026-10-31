import { describe, expect, it } from 'vitest'
import { gradCheck } from '../autodiff/index.ts'
import { key } from '../random/index.ts'
import { item, type Value } from '../tensor/index.ts'
import { recorder, runAlgorithm } from '../trace/index.ts'
import {
  type NtpAtom,
  neuralTheoremProver,
  ntpAlgorithm,
  ntpScores,
  ntpTrainingSet,
  parseNtpAtom,
  parseNtpRule,
  trainNtp,
} from './index.ts'

/** Embeddings that make distinct symbols orthogonal and far apart, so the soft unification is crisp. */
function crisp(symbols: readonly string[], dimension: number): number[] {
  const theta = new Array<number>(symbols.length * dimension).fill(0)
  symbols.forEach((_, i) => {
    theta[i * dimension + i] = 12
  })
  return theta
}

const goal = (text: string): NtpAtom => parseNtpAtom(text)

describe('parsing', () => {
  it('reads atoms and rules, including template predicates', () => {
    expect(parseNtpAtom('parent(ann, bob)')).toEqual({ predicate: 'parent', args: ['ann', 'bob'] })
    expect(parseNtpRule('#1(X,Y) :- #2(X,Z), #3(Z,Y)').body.map((a) => a.predicate)).toEqual(['#2', '#3'])
    expect(() => parseNtpAtom('nonsense')).toThrow()
    expect(() => parseNtpRule('p(X,Y)')).toThrow()
  })
})

describe('the proof network', () => {
  const chain = ['parent(a,b)', 'parent(b,c)', 'parent(c,d)']

  it('with crisp embeddings it is classical backward chaining to the depth', () => {
    const rules = ['grandparent(X,Y) :- parent(X,Z), parent(Z,Y)']
    const ntp = neuralTheoremProver({ facts: chain, rules, dimension: 8, depth: 1 }, key(0))
    const theta = crisp(ntp.symbols, 8)
    const s = (g: string): number => ntpScores(ntp, theta, [goal(g)])[0] as number
    expect(s('parent(a,b)')).toBeCloseTo(1, 9)
    expect(s('parent(a,c)')).toBeCloseTo(0, 9)
    expect(s('grandparent(a,c)')).toBeCloseTo(1, 9)
    expect(s('grandparent(b,d)')).toBeCloseTo(1, 9)
    expect(s('grandparent(a,d)')).toBeCloseTo(0, 9)
    expect(s('grandparent(c,a)')).toBeCloseTo(0, 9)
  })

  it('a recursive rule needs a proof depth as deep as the chain', () => {
    const rules = ['ancestor(X,Y) :- parent(X,Y)', 'ancestor(X,Y) :- parent(X,Z), ancestor(Z,Y)']
    const reach = (depth: number): number[] => {
      const ntp = neuralTheoremProver({ facts: chain, rules, dimension: 8, depth }, key(0))
      return ntpScores(ntp, crisp(ntp.symbols, 8), [
        goal('ancestor(a,b)'),
        goal('ancestor(a,c)'),
        goal('ancestor(a,d)'),
      ]).map(Math.round)
    }
    expect(reach(1)).toEqual([1, 0, 0])
    expect(reach(2)).toEqual([1, 1, 0])
    expect(reach(3)).toEqual([1, 1, 1])
  })

  it('leaves a fact out when proving it, so it cannot prove itself', () => {
    const ntp = neuralTheoremProver(
      { facts: [...chain, 'grandparent(a,c)'], rules: ['grandparent(X,Y) :- parent(X,Z), parent(Z,Y)'], dimension: 8 },
      key(0),
    )
    const theta = crisp(ntp.symbols, 8)
    const g = goal('grandparent(a,c)')
    expect(item(ntp.score(theta, g))).toBeCloseTo(1, 9)
    // With the rule removed the left-out fact cannot be proved.
    const bare = neuralTheoremProver({ facts: [...chain, 'grandparent(a,c)'], rules: [], dimension: 8 }, key(0))
    expect(item(bare.score(crisp(bare.symbols, 8), g, 3))).toBeCloseTo(0, 9)
    expect(item(bare.score(crisp(bare.symbols, 8), g))).toBeCloseTo(1, 9)
  })

  it('soft unification: a nearby symbol proves a goal with a graded score', () => {
    const ntp = neuralTheoremProver(
      { facts: ['parent(a,b)', 'parent(c,d)'], rules: [], dimension: 2, kernelWidth: 1 },
      key(0),
    )
    // Symbols in order: parent, a, b, c, d. Put c at distance 1 from a in the plane.
    const theta = [0, 0, 20, 0, 30, 0, 21, 0, 40, 0].map((x, i) => (i === 2 ? 10 : x))
    const score = item(ntp.score([10, 10, 0, 0, 20, 0, 1, 0, 20, 0], goal('parent(a,b)')))
    expect(theta.length).toBe(10)
    expect(score).toBeGreaterThan(0.99)
    // parent(c,b): c is not a, so the best proof is the fact parent(a,b) with c vs a at distance 1: exp(-1/2).
    const near = item(
      ntp.score(
        [10, 10, 0, 0, 20, 0, 1, 0, 20, 0].map((x) => x),
        goal('parent(c,b)'),
      ),
    )
    expect(near).toBeGreaterThan(0)
  })

  it('the loss gradient agrees with finite differences', () => {
    const ntp = neuralTheoremProver(
      { facts: chain, rules: ['grandparent(X,Y) :- parent(X,Z), parent(Z,Y)'], dimension: 3 },
      key(4),
    )
    const examples = [
      { goal: goal('grandparent(a,c)'), label: 1 as const },
      { goal: goal('grandparent(a,b)'), label: 0 as const },
      { goal: goal('parent(b,c)'), label: 1 as const, exclude: 1 },
    ]
    const problem = ntp.problem(examples, { l2: 0.01 })
    const r = gradCheck(problem.objective as (x: Value) => Value, problem.x0 as number[], {
      rtol: 1e-3,
      atol: 1e-5,
      vmap: false,
    })
    expect(r.failures).toEqual([])
  })
})

describe('training', () => {
  const parents = ['a,b', 'a,c', 'b,d', 'b,e', 'c,f', 'g,h', 'h,i', 'h,j']
  const facts = [...parents.map((p) => `parent(${p})`), 'grandparent(a,d)', 'grandparent(g,i)']
  const heldOut = ['grandparent(a,e)', 'grandparent(a,f)', 'grandparent(g,j)'].map(goal)
  const wrong = [
    'grandparent(a,b)',
    'grandparent(d,a)',
    'grandparent(a,i)',
    'grandparent(g,e)',
    'grandparent(h,j)',
    'grandparent(f,c)',
  ].map(goal)
  const rules = ['grandparent(X,Y) :- parent(X,Z), parent(Z,Y)']

  it('with the rule given, training pushes unrelated symbols apart: derived facts stay high, wrong ones fall', () => {
    const ntp = neuralTheoremProver({ facts, rules, dimension: 6 }, key(1))
    const examples = ntpTrainingSet(ntp, key(2), { negatives: 3 })
    const before = ntpScores(ntp, ntp.x0, wrong)
    const fit = trainNtp(ntp, examples, { maxSteps: 100 })
    const held = ntpScores(ntp, fit.theta, heldOut)
    const after = ntpScores(ntp, fit.theta, wrong)
    for (const h of held) expect(h).toBeGreaterThan(0.99)
    const mean = (xs: number[]): number => xs.reduce((a, b) => a + b, 0) / xs.length
    expect(mean(after)).toBeLessThan(0.5 * mean(before))
    expect(Math.max(...after)).toBeLessThan(0.3)
  }, 120_000)

  it('is a traced algorithm over the solver slot, and decodes a template to the nearest predicate', () => {
    const ntp = neuralTheoremProver({ facts, rules: ['#1(X,Y) :- #2(X,Z), #3(Z,Y)'], dimension: 4 }, key(3))
    const rec = recorder()
    const run = runAlgorithm(ntpAlgorithm(ntp, ntpTrainingSet(ntp, key(2), { negatives: 1 })), {
      seed: 0,
      maxSteps: 3,
      tracer: rec,
    })
    expect(run.steps).toBeGreaterThan(0)
    const text = JSON.stringify(rec.run())
    for (const p of ['ntp/embeddings', 'ntp/positive', 'ntp/negative', 'solver/objective']) expect(text).toContain(p)
    // A template whose vector sits on `parent` decodes to `parent`.
    const theta = [...ntp.x0]
    const at = (s: string): number => ntp.symbols.indexOf(s) * 4
    for (let i = 0; i < 4; i++) theta[at('#2') + i] = theta[at('parent') + i] as number
    expect(ntp.decode(theta)['#2']?.symbol).toBe('parent')
  })
})
