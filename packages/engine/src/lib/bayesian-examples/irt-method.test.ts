import { describe, expect, it } from 'vitest'
import { type DictSpace, defaults, type UnionSpace } from '../space/index.ts'
import { type Config, session } from '../trace/index.ts'
import { exampleSteps } from './example-method.ts'
import { irtExample, irtProblem } from './irt-method.ts'

const base = { ...(defaults(irtExample.method.config) as Config), students: 20, questions: 6 }
const options = (irtExample.method.config.spaces['inference'] as UnionSpace).options
const run = (extra: Config, inference: Config) => {
  const kind = String(inference['method'])
  const config = {
    ...base,
    ...extra,
    inference: { method: kind, ...(defaults(options[kind] as DictSpace) as Config), ...inference },
    budgetMs: 0,
  }
  return session(irtExample.method, { config, steps: exampleSteps(config), mode: 'eager' })
}
const nuts = { method: 'nuts', numChains: 2, numWarmup: 40, numDraws: 60 }

describe('irtProblem', () => {
  it('puts the truth on the scale the model fixes: mean ability 0, and for 2PL ability sd 1', () => {
    const rasch = irtProblem(base)
    expect(rasch.truth.theta).toHaveLength(20)
    expect(rasch.truth.b).toHaveLength(6)
    expect(rasch.truth.a).toBeNull()
    expect(rasch.truth.theta.reduce((a, b) => a + b, 0)).toBeCloseTo(0, 10)
    const two = irtProblem({ ...base, model: '2pl' })
    const sd = Math.sqrt(two.truth.theta.reduce((a, t) => a + t * t, 0) / 20)
    expect(sd).toBeCloseTo(1, 10)
    expect(two.truth.a).toHaveLength(6)
  })

  it('keeps the first students and questions when more are asked for', () => {
    const small = irtProblem({ ...base, students: 10, questions: 4 })
    const large = irtProblem({ ...base, students: 20, questions: 6 })
    const rows = (p: ReturnType<typeof irtProblem>) =>
      (p.data['y'] as number[][]).slice(0, 10).map((r) => r.slice(0, 4))
    expect(rows(small)).toEqual(rows(large))
  })
})

describe('irtExample', () => {
  it('offers many methods, with message passing on the Gamma-precision form, and says why it does not apply elsewhere', () => {
    const offer = irtExample.offer(base)
    for (const kind of ['nuts', 'svi', 'laplace', 'tempered-smc', 'ep', 'vmp']) expect(offer.applicable).toContain(kind)
    expect(offer.unavailable['gibbs']).toBeDefined()
    // A half-normal sd is projected once onto a Gaussian on its log and the log-normal discriminations are exp of a Gaussian
    // (the apply factor), so EP applies to both; the product of two random values has no variational rule.
    expect(irtExample.offer({ ...base, spreads: 'half-normal' }).applicable).toContain('ep')
    const normal2pl = irtExample.offer({ ...base, model: '2pl', discriminationPrior: 'normal' })
    expect(normal2pl.applicable).toContain('ep')
    expect(normal2pl.unavailable['vmp']).toMatch(/product/)
    const lognormal2pl = irtExample.offer({ ...base, model: '2pl' })
    expect(lognormal2pl.applicable).toContain('ep')
    expect(lognormal2pl.unavailable['vmp']).toMatch(/product|exp|apply/)
  })

  for (const model of ['rasch', '2pl'])
    for (const parameterisation of ['non-centred', 'centred'])
      it(`${model}, ${parameterisation}: NUTS runs and every probe reads its posterior`, () => {
        const s = run({ model, parameterisation }, nuts)
        const r = s.probe(s.t, 'irt/recovery') as { abilities: { truth: number[] }; discriminations: unknown }
        expect(r.abilities.truth).toHaveLength(20)
        expect(r.discriminations === null).toBe(model === 'rasch')
        const cal = s.probe(s.t, 'irt/calibration') as { abilities: number; difficulties: number }
        expect(cal.abilities).toBeGreaterThanOrEqual(0)
        expect(cal.abilities).toBeLessThanOrEqual(1)
        expect((s.probe(s.t, 'irt/pairs') as { kind: string }).kind).toBe('posteriorPairs')
        expect((s.probe(s.t, 'irt/diagnostics') as { kind: string }).kind).toBe('chainDiagnostics')
        const summary = s.probe(s.t, 'irt/summary') as { rows: { name: string }[] }
        expect(summary.rows.some((x) => x.name === 'theta[19]')).toBe(true)
      }, 60000)

  it('refits on the first students and questions seen: fewer questions, wider intervals', () => {
    const s = run({}, { method: 'laplace' })
    const all = s.probe(s.t, 'irt/recovery') as { abilities: { width: number } }
    const few = s.probe(s.t, 'irt/recovery', { students: 20, questions: 3 }) as {
      questions: number
      abilities: { width: number }
    }
    expect(few.questions).toBe(3)
    expect(few.abilities.width).toBeGreaterThan(all.abilities.width)
  }, 60000)

  it('a variational and a Laplace fit have the derived sites (the abilities, the difficulties)', () => {
    for (const method of ['svi', 'laplace']) {
      const s = run({}, { method })
      expect(s.probe(s.t, 'irt/recovery')).not.toBeNull()
    }
  }, 60000)

  it('checks a run against a reference run of the same answers', () => {
    const a = run({}, nuts)
    const reference = run({}, { ...nuts, numWarmup: 60, numDraws: 100 })
    const check = a.probe(a.t, 'irt/check', { summary: reference.probe(reference.t, 'irt/summary') }) as {
      compared: number
      maxMeanError: number
    }
    expect(check.compared).toBeGreaterThan(20)
    expect(check.maxMeanError).toBeLessThan(3)
  }, 60000)

  it('keeps the model but writes the population priors non-centred for the reference', () => {
    expect(irtExample.reference({ parameterisation: 'centred' }, { method: 'nuts' })['parameterisation']).toBe(
      'non-centred',
    )
  })
})
