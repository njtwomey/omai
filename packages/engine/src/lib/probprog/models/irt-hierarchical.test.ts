import { describe, expect, it } from 'vitest'
import { key } from '../../random/index.ts'
import { defaults } from '../../space/index.ts'
import { session } from '../../trace/index.ts'
import { inferenceMethodFor, modelRegistry } from '../inference/index.ts'
import { generateIrt, hierarchicalIrtDefinition } from './irt-hierarchical.ts'

describe('generateIrt', () => {
  it('makes a prefix of one table: the first students and questions are the same whatever the size', () => {
    const small = generateIrt(key(3), { students: 5, questions: 3 })
    const large = generateIrt(key(3), { students: 30, questions: 9 })
    expect(small.theta).toEqual(large.theta.slice(0, 5))
    expect(small.b).toEqual(large.b.slice(0, 3))
    expect(small.y).toEqual(large.y.slice(0, 5).map((r) => r.slice(0, 3)))
  })

  it('answers more often right the more able the student', () => {
    const g = generateIrt(key(1), { students: 200, questions: 30, abilitySd: 1.5 })
    const score = g.y.map((r) => r.reduce((a, b) => a + b, 0))
    const order = g.theta.map((_, i) => i).sort((a, b) => (g.theta[a] as number) - (g.theta[b] as number))
    const low = order.slice(0, 50).reduce((a, i) => a + (score[i] as number), 0)
    const high = order.slice(-50).reduce((a, i) => a + (score[i] as number), 0)
    expect(high).toBeGreaterThan(low)
  })
})

describe('hierarchicalIrtDefinition', () => {
  it('binds to answers with the students and questions as plates', () => {
    for (const model of ['rasch', '2pl'] as const)
      for (const form of ['centred', 'non-centred'] as const)
        expect(
          hierarchicalIrtDefinition(model, form).bind({
            y: [
              [1, 0, 1],
              [0, 0, 1],
            ],
          }).sizes,
        ).toEqual({
          student: 2,
          question: 3,
        })
  })
})

describe('the Rasch model with Gamma precisions under message passing', () => {
  const g = generateIrt(key(2), { students: 24, questions: 8 })
  const registry = (spreads: 'precision' | 'fixed') =>
    modelRegistry({
      irt: { definition: hierarchicalIrtDefinition('rasch', 'centred', { spreads }), data: { y: g.y } },
    })
  const fit = (kind: string, spreads: 'precision' | 'fixed', config: Record<string, unknown> = {}) => {
    const method = inferenceMethodFor(registry(spreads), kind, 'irt')
    const cfg = { ...(defaults(method.config) as Record<string, unknown>), ...config }
    const s = session(method as never, { config: cfg as never, seed: 3, steps: 400, mode: 'eager' })
    return (
      method as never as {
        posterior: (state: unknown, c: unknown) => { summary: () => { name: string; mean: number; sd: number }[] }
      }
    ).posterior(s.state, cfg)
  }

  it('expectation propagation matches a long NUTS run: abilities, difficulties and the precisions', () => {
    const ep = fit('ep', 'precision').summary()
    const nuts = fit('nuts', 'precision', { numChains: 2, numWarmup: 150, numDraws: 250 }).summary()
    const by = new Map(nuts.map((r) => [r.name, r]))
    const shared = ep.filter((r) => by.has(r.name) && (/^(theta|b)\[/.test(r.name) || /^prec_/.test(r.name)))
    expect(shared.length).toBeGreaterThan(30)
    const errors = shared.map((r) => Math.abs(r.mean - (by.get(r.name)?.mean ?? 0)) / (by.get(r.name)?.sd ?? 1))
    expect(Math.max(...errors)).toBeLessThan(0.5)
    const ratios = shared.map((r) => r.sd / (by.get(r.name)?.sd ?? 1)).sort((a, b) => a - b)
    expect(ratios[Math.floor(ratios.length / 2)]).toBeGreaterThan(0.85)
    expect(ratios[Math.floor(ratios.length / 2)]).toBeLessThan(1.15)
  }, 120000)

  it('variational message passing runs it and understates the spreads', () => {
    const vmp = fit('vmp', 'precision').summary()
    const ep = fit('ep', 'precision').summary()
    const sd = (rows: { name: string; sd: number }[]) => rows.filter((r) => /^theta\[/.test(r.name)).map((r) => r.sd)
    const mean = (v: number[]) => v.reduce((a, b) => a + b, 0) / v.length
    expect(mean(sd(vmp))).toBeLessThan(mean(sd(ep)))
  }, 60000)

  it('runs with known spreads too (a plain logistic regression with a Gaussian per student and question)', () => {
    expect(
      fit('ep', 'fixed')
        .summary()
        .some((r) => r.name === 'theta[0]'),
    ).toBe(true)
  })
})
