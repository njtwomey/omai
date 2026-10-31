import { describe, expect, it } from 'vitest'
import { flatObjective, smoothSolverConfig } from '../optim/index.ts'
import { key } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { recorder, runAlgorithm } from '../trace/index.ts'
import { type DilpClause, type DilpTask, diffIlp, diffIlpAlgorithm, dilpClauseText, learnProgram } from './index.ts'

const digits = (n: number): string[] => Array.from({ length: n }, (_, i) => String(i))

/** A clause as text with its body atoms sorted, so the order of the body does not matter. */
const canonical = (c: DilpClause): string =>
  `${c.head.predicate}(${c.head.args.join(',')}) :- ${c.body
    .map((a) => `${a.predicate}(${a.args.join(',')})`)
    .sort()
    .join(', ')}`

const predecessor: DilpTask = {
  constants: digits(5),
  extensional: [{ name: 'succ', arity: 2, facts: [0, 1, 2, 3].map((i) => [String(i), String(i + 1)]) }],
  intensional: [{ name: 'pred', arity: 2, templates: [{ maxBody: 1 }] }],
  examples: digits(5).flatMap((x) =>
    digits(5).map((y) => ({ predicate: 'pred', args: [x, y], label: (Number(x) === Number(y) + 1 ? 1 : 0) as 0 | 1 })),
  ),
}

// The paper’s even numbers: zero, successor, and an auxiliary "add two" the learner must invent the use of.
const even: DilpTask = {
  constants: digits(7),
  extensional: [
    { name: 'zero', arity: 1, facts: [['0']] },
    { name: 'succ', arity: 2, facts: [0, 1, 2, 3, 4, 5].map((i) => [String(i), String(i + 1)]) },
  ],
  intensional: [
    { name: 'inc2', arity: 2, templates: [{ existential: 1, maxBody: 2 }] },
    {
      name: 'even',
      arity: 1,
      templates: [{ maxBody: 1 }, { existential: 1, maxBody: 2, intensional: true }],
    },
  ],
  examples: digits(7).map((x) => ({ predicate: 'even', args: [x], label: (Number(x) % 2 === 0 ? 1 : 0) as 0 | 1 })),
  steps: 5,
}

const NODES = ['a', 'b', 'c', 'd', 'e']
const EDGES = [
  ['a', 'b'],
  ['b', 'c'],
  ['c', 'd'],
  ['e', 'a'],
]
const reaches = (x: string, y: string): boolean => {
  const seen = new Set<string>()
  const todo = [x]
  while (todo.length > 0) {
    const u = todo.pop() as string
    for (const [s, t] of EDGES)
      if (s === u && !seen.has(t as string)) {
        seen.add(t as string)
        todo.push(t as string)
      }
  }
  return seen.has(y)
}
const reachability: DilpTask = {
  constants: NODES,
  extensional: [{ name: 'edge', arity: 2, facts: EDGES }],
  intensional: [
    {
      name: 'connected',
      arity: 2,
      templates: [{ maxBody: 1 }, { existential: 1, maxBody: 2, intensional: true }],
    },
  ],
  examples: NODES.flatMap((x) =>
    NODES.map((y) => ({ predicate: 'connected', args: [x, y], label: (reaches(x, y) ? 1 : 0) as 0 | 1 })),
  ),
  steps: 4,
}

describe('the language', () => {
  it('generates the safe, non-trivial candidates of a template', () => {
    const ilp = diffIlp(predecessor, key(0))
    expect(ilp.candidates).toHaveLength(1)
    // Both head variables must appear in a single succ atom: succ(X,Y) and succ(Y,X).
    expect(ilp.candidates[0]?.clauses.map(dilpClauseText).sort()).toEqual([
      'pred(X,Y) :- succ(X,Y)',
      'pred(X,Y) :- succ(Y,X)',
    ])
    const e = diffIlp(even, key(0))
    for (const slot of e.candidates)
      for (const c of slot.clauses) {
        const used = new Set(c.body.flatMap((a) => a.args))
        for (const v of c.head.args) expect(used.has(v)).toBe(true)
        expect(c.body.some((a) => a.predicate === c.head.predicate && a.args.join() === c.head.args.join())).toBe(false)
      }
    expect(
      e.candidates
        .find((s) => s.predicate === 'inc2')
        ?.clauses.every((c) => c.body.every((a) => a.predicate !== 'inc2' && a.predicate !== 'even')),
    ).toBe(true)
    expect(() =>
      diffIlp({ ...predecessor, examples: [{ predicate: 'nope', args: ['0', '1'], label: 1 }] }, key(0)),
    ).toThrow()
  })

  it('with the planted clauses weighted, the valuation is the crisp least model', () => {
    const ilp = diffIlp(even, key(0))
    const planted = new Set([
      'even(X) :- zero(X)',
      'even(X) :- even(Z), inc2(Z,X)',
      'inc2(X,Y) :- succ(X,Z), succ(Z,Y)',
    ])
    let at = 0
    const theta: number[] = []
    for (const slot of ilp.candidates) {
      const texts = slot.clauses.map(canonical)
      const hit = texts.findIndex((t) =>
        [...planted].some((p) => {
          const c = p.split(' :- ')
          return t === `${c[0]} :- ${(c[1] as string).split(', ').sort().join(', ')}`
        }),
      )
      expect(hit).toBeGreaterThanOrEqual(0)
      for (let i = 0; i < slot.clauses.length; i++) theta.push(i === hit ? 40 : 0)
      at += slot.clauses.length
    }
    expect(theta.length).toBe(at)
    const program = ilp.extract(theta)
    expect(program.clauses.map(canonical).sort()).toEqual(
      [...planted]
        .map((p) =>
          canonical({
            head: parse(p.split(' :- ')[0] as string),
            body: (p.split(' :- ')[1] as string)
              .split(/\), ?/)
              .map((b, i, all) => parse(i < all.length - 1 ? `${b})` : b)),
          }),
        )
        .sort(),
    )
    expect(ilp.accuracy(program.clauses)).toBe(1)
    // The differentiable forward chaining agrees with the crisp one on every example.
    const scores = toFlat(ilp.scores(theta))
    for (const [i, e] of even.examples.entries()) expect(scores[i]).toBeCloseTo(e.label, 6)
  })

  it('the loss gradient agrees with finite differences', () => {
    const ilp = diffIlp(even, key(3))
    const problem = ilp.problem()
    const flat = flatObjective(problem.objective, problem.x0)
    const x = Array.from({ length: ilp.size }, (_, i) => 0.5 * Math.sin(1.7 * i))
    const g = flat.valueAndGrad(x).grad
    let checked = 0
    for (let j = 0; j < ilp.size && checked < 8; j += Math.ceil(ilp.size / 8)) {
      const up = [...x]
      const down = [...x]
      up[j] = (up[j] as number) + 1e-6
      down[j] = (down[j] as number) - 1e-6
      expect(g[j]).toBeCloseTo((flat.value(up) - flat.value(down)) / 2e-6, 5)
      checked++
    }
  })
})

function parse(text: string): { predicate: string; args: string[] } {
  const m = /^(\w+)\((.*)\)$/.exec(text.trim()) as RegExpExecArray
  return { predicate: m[1] as string, args: (m[2] as string).split(',') }
}

describe('learning programs', () => {
  it('learns the predecessor from successor facts', () => {
    const fit = learnProgram(diffIlp(predecessor, key(1)), { restarts: 3 })
    expect(fit.program.clauses.map(canonical)).toEqual(['pred(X,Y) :- succ(Y,X)'])
    expect(fit.program.clauses[0]?.probability).toBeGreaterThan(0.9)
    expect(fit.loss).toBeLessThan(0.05)
  })

  it('learns graph reachability, a recursive rule over an intensional atom', () => {
    const ilp = diffIlp(reachability, key(2))
    const fit = learnProgram(ilp, { restarts: 4, key: key(7) })
    expect(fit.program.clauses.map(canonical).sort()).toEqual(
      ['connected(X,Y) :- connected(Z,Y), edge(X,Z)', 'connected(X,Y) :- edge(X,Y)'].sort(),
    )
    expect(ilp.accuracy(fit.program.clauses)).toBe(1)
  }, 120_000)

  it('learns the even numbers with an invented auxiliary predicate: the extracted program is the planted one', () => {
    const ilp = diffIlp(even, key(4))
    const fit = learnProgram(ilp, { restarts: 12, key: key(11) })
    const texts = fit.program.clauses.map(canonical).sort()
    expect(texts).toEqual(
      ['even(X) :- zero(X)', 'even(X) :- even(Z), inc2(Z,X)', 'inc2(X,Y) :- succ(X,Z), succ(Z,Y)'].sort(),
    )
    expect(ilp.accuracy(fit.program.clauses)).toBe(1)
    // The program generalises beyond the training numbers: it derives exactly the even ones.
    const model = ilp.derive(fit.program.clauses)
    expect(digits(7).filter((x) => model.has(`even(${x})`))).toEqual(['0', '2', '4', '6'])
  }, 300_000)

  it('the optimiser is a slot: Adam also learns the predecessor, and the tape holds the weights', () => {
    const ilp = diffIlp(predecessor, key(5))
    const solver = smoothSolverConfig({
      kind: 'gradient',
      optimiser: { kind: 'adam' },
      learningRate: 0.3,
      gradTol: 1e-6,
    })
    const fit = learnProgram(ilp, { solver, maxSteps: 200 })
    expect(fit.program.clauses.map(canonical)).toEqual(['pred(X,Y) :- succ(Y,X)'])
    const rec = recorder()
    runAlgorithm(diffIlpAlgorithm(ilp), { seed: 0, maxSteps: 5, tracer: rec })
    const text = JSON.stringify(rec.run())
    for (const p of ['dilp/probabilities', 'dilp/accuracy', 'solver/objective']) expect(text).toContain(p)
  })
})
