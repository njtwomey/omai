import { conceptScores, ruleSet, ruleText } from 'mdml-engine/exposition'
import { describe, expect, it } from 'vitest'
import { stillOf, viewFor } from './registry.ts'
import { conceptView, ruleListView } from './rules.ts'

type Opt = { series: { name?: string; type?: string }[]; yAxis?: { data?: string[] }[] }

describe('rule and concept views', () => {
  it('lists rules by importance or an anchor by precision and coverage', () => {
    const fit = ruleSet({
      method: 'RuleFit',
      features: ['age', 'bmi'],
      output: 'y',
      rules: [
        { clauses: [{ feature: 0, op: '>', threshold: 40 }], weight: 2, support: 0.3, importance: 0.9 },
        { clauses: [], linear: 1, weight: -0.5, support: 1, importance: 0.4 },
      ],
    })
    expect((ruleListView(fit) as unknown as Opt).series.some((s) => s.name === '__importance')).toBe(true)
    expect(ruleText(fit.rules[1] ?? { clauses: [] }, fit.features)).toBe('bmi')
    const anchor = ruleSet({
      method: 'anchors',
      features: ['age'],
      output: 'y',
      rules: [{ clauses: [{ feature: 0, op: 'in', lower: 1, upper: 2 }], precision: 0.96, coverage: 0.2 }],
    })
    expect((ruleListView(anchor) as unknown as Opt).series.map((s) => s.name)).toContain('precision')
    expect(viewFor('rules').component).toBe('RuleListView')
  })

  it('draws TCAV scores with the random baseline and every run', () => {
    const r = conceptScores({
      method: 'TCAV',
      target: '8',
      layer: 'hidden',
      concepts: [{ name: 'zeros', scores: [0.9, 0.8], randomScores: [0.5, 0.4], mean: 0.85, pValue: 0.01 }],
    })
    const names = (conceptView(r) as unknown as Opt).series.map((s) => s.name)
    expect(names).toContain('random')
    expect(names).toContain('__runs')
    expect(stillOf(r)).toBeDefined()
  })
})
