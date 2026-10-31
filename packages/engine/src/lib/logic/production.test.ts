import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { runAlgorithm } from '../trace/index.ts'
import {
  type Activation,
  productionRule,
  productionSystemAlgorithm,
  ReteNetwork,
  resolveConflict,
  runProduction,
} from './index.ts'
import { parseTerm } from './parse.ts'

const FAMILY = [
  productionRule('grandparent', ['parent(X, Y)', 'parent(Y, Z)'], { add: ['grandparent(X, Z)'] }),
  productionRule('ancestor-base', ['parent(X, Y)'], { add: ['ancestor(X, Y)'] }),
  productionRule('ancestor-step', ['parent(X, Y)', 'ancestor(Y, Z)'], { add: ['ancestor(X, Z)'] }),
]
const PARENTS = ['parent(a, b)', 'parent(b, c)', 'parent(c, d)']

describe('forward chaining', () => {
  it('derives the least model of a monotone rule base', () => {
    const r = runProduction(FAMILY, PARENTS)
    const compact = (f: string): string => f.replace(/ /g, '')
    expect([...r.facts].sort()).toEqual(
      [
        ...PARENTS,
        'grandparent(a, c)',
        'grandparent(b, d)',
        'ancestor(a, b)',
        'ancestor(b, c)',
        'ancestor(c, d)',
        'ancestor(a, c)',
        'ancestor(b, d)',
        'ancestor(a, d)',
      ]
        .map(compact)
        .sort(),
    )
  })

  it('the conflict resolution strategy changes the order of firing, not a monotone result', () => {
    const order = (strategy: 'order' | 'recency' | 'specificity'): string[] =>
      runProduction(FAMILY, PARENTS, { strategy }).firings.map((f) => f.rule)
    expect(order('order')[0]).toBe('grandparent')
    expect(order('recency')).not.toEqual(order('order'))
    for (const s of ['order', 'recency', 'specificity'] as const) {
      expect(new Set(runProduction(FAMILY, PARENTS, { strategy: s }).facts)).toEqual(
        new Set(runProduction(FAMILY, PARENTS).facts),
      )
    }
  })

  it('with retraction the strategies reach different conclusions', () => {
    const rules = [
      productionRule('go-red', ['light(green)'], { add: ['light(red)'], remove: ['light(green)'] }),
      productionRule('go-amber', ['light(green)'], { add: ['light(amber)'], remove: ['light(green)'] }),
    ]
    expect(runProduction(rules, ['light(green)'], { strategy: 'order' }).facts).toEqual(['light(red)'])
    expect(runProduction(rules, ['light(green)'], { strategy: 'priority' }).facts).toEqual(['light(red)'])
    const prioritised = [rules[0] as never, { ...(rules[1] as object), priority: 5 } as never]
    expect(runProduction(prioritised, ['light(green)'], { strategy: 'priority' }).facts).toEqual(['light(amber)'])
  })

  it('negative conditions require absence and halt stops the system', () => {
    const rules = [
      productionRule('bird-flies', ['bird(X)', 'not(penguin(X))'], { add: ['flies(X)'] }),
      productionRule('stop', ['flies(tweety)'], { halt: true }),
    ]
    const r = runProduction(rules, ['bird(tweety)', 'bird(pingu)', 'penguin(pingu)'])
    expect(r.facts).toContain('flies(tweety)')
    expect(r.facts).not.toContain('flies(pingu)')
    expect(r.firings.at(-1)?.rule).toBe('stop')
  })

  it('refraction: an instantiation fires once', () => {
    const r = runProduction([productionRule('loop', ['p'], { add: ['p'] })], ['p'])
    expect(r.firings.length).toBe(1)
  })

  it('rejects a non-ground fact', () => {
    expect(() => productionSystemAlgorithm(FAMILY, ['parent(X, b)'])).toThrow()
  })
})

describe('the algorithm', () => {
  it('fires one rule per step, with the working memory and the rule fired in the state', () => {
    const alg = productionSystemAlgorithm(FAMILY, PARENTS)
    const { state, steps } = runAlgorithm(alg, { maxSteps: 100, seed: 0 })
    expect(steps).toBe(state.cycle + 1)
    expect(state.last).toBeNull()
    expect(alg.deserialise?.(JSON.parse(JSON.stringify(alg.serialise?.(state))))).toEqual(state)
  })
})

describe('Rete-lite', () => {
  it('gives the same conflict sets as recomputing from scratch, cycle by cycle', () => {
    for (const strategy of ['order', 'recency', 'specificity', 'priority'] as const) {
      const a = runProduction(FAMILY, PARENTS, { strategy, matcher: 'rete' })
      const b = runProduction(FAMILY, PARENTS, { strategy, matcher: 'naive' })
      expect(a).toEqual(b)
    }
  })

  it('agrees with the naive matcher on random rule bases, including retractions and negation', () => {
    const atom = fc.constantFrom('p(X)', 'q(X)', 'r(X, Y)', 'p(Y)', 'q(Y)', 's(X)')
    const cond = fc.oneof(
      atom,
      atom.map((a) => `not(${a})`),
    )
    const rule = fc.record({
      conditions: fc.array(cond, { minLength: 1, maxLength: 3 }),
      add: fc.array(fc.constantFrom('p(X)', 'q(X)', 's(X)', 'q(Y)'), { maxLength: 2 }),
      remove: fc.array(fc.constantFrom('p(X)', 'q(X)'), { maxLength: 1 }),
    })
    const factsArb = fc.uniqueArray(fc.constantFrom('p(a)', 'p(b)', 'q(a)', 'q(c)', 'r(a, b)', 'r(b, c)', 's(a)'), {
      minLength: 1,
      maxLength: 5,
    })
    fc.assert(
      fc.property(
        fc.array(rule, { minLength: 1, maxLength: 3 }),
        factsArb,
        fc.constantFrom('order', 'recency', 'specificity'),
        (rs, facts, strategy) => {
          // Only rules whose actions use variables bound by positive conditions.
          const rules = rs
            .filter((r) => {
              const bound = new Set(
                r.conditions
                  .filter((c) => !c.startsWith('not'))
                  .join(' ')
                  .match(/[XY]/g) ?? [],
              )
              return [...r.add, ...r.remove].every((a) => (a.match(/[XY]/g) ?? []).every((v) => bound.has(v)))
            })
            .map((r, i) => productionRule(`r${i}`, r.conditions, { add: r.add, remove: r.remove }))
          if (rules.length === 0) return
          const a = runProduction(rules, facts, { strategy, matcher: 'rete', maxFirings: 30 })
          const b = runProduction(rules, facts, { strategy, matcher: 'naive', maxFirings: 30 })
          expect(a).toEqual(b)
        },
      ),
      { numRuns: 150 },
    )
  })

  it('joins incrementally: adding one fact does far less work than rebuilding', () => {
    const net = new ReteNetwork(FAMILY)
    let id = 0
    for (let i = 0; i < 30; i++) net.add(id++, parseTerm(`parent(n${i}, n${i + 1})`))
    const before = net.joins
    net.add(id++, parseTerm('parent(n30, n31)'))
    expect(net.joins - before).toBeLessThan(before / 5)
    net.remove(0)
    expect(net.activations().every((a) => !a.facts.includes(0))).toBe(true)
  })
})

describe('conflict resolution', () => {
  const rules = [productionRule('a', ['p']), productionRule('b', ['p', 'q'])]
  const conflict: Activation[] = [
    { rule: 0, facts: [5], bindings: {} },
    { rule: 1, facts: [0, 1], bindings: {} },
  ]
  it('picks by order, recency and specificity', () => {
    expect(resolveConflict(conflict, rules, 'order')?.rule).toBe(0)
    expect(resolveConflict(conflict, rules, 'recency')?.rule).toBe(0)
    expect(resolveConflict(conflict, rules, 'specificity')?.rule).toBe(1)
    expect(resolveConflict([], rules, 'order')).toBeUndefined()
  })
})
