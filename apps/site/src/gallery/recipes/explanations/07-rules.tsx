import type { RuleEntry } from 'omai-engine/exposition'
import { ruleSet, ruleText } from 'omai-engine/exposition'
import { Choice, ControlPanel, ExpositionView, NumberField, SeedInput } from 'omai-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'
import { CREDIT_FEATURES, creditRows, creditScore } from '../../toys/synthetic.ts'

export const recipe: Recipe = {
  title: 'Rules',
  question: 'How do I show a model as a short list of if-then rules, with how often each one is right?',
  explain:
    'A `rules` record lists rules (clauses on features), each with its precision, coverage and, for a rule ensemble, a weight; `RuleListView` writes each rule out with bars for its precision and coverage. Without precisions (a rule ensemble, as RuleFit) the bars are each rule’s importance, red where its weight raises the output and blue where it lowers it. Here the rules are a fixed handful over the toy credit data, and their precision and coverage are measured on the sample you draw: more rows, steadier numbers. Raise the minimum precision to drop the weak ones.',
  shows: ['rules', 'RuleListView'],
}

const [INCOME, DEBT, , TENURE, LATE, SAVINGS] = [0, 1, 2, 3, 4, 5]
type Clause = RuleEntry['clauses'][number]
const RULES: readonly (readonly Clause[])[] = [
  [{ feature: LATE, op: '>', threshold: 0.8 }],
  [
    { feature: DEBT, op: '>', threshold: 0.5 },
    { feature: INCOME, op: '<=', threshold: 0 },
  ],
  [
    { feature: INCOME, op: '>', threshold: 0.3 },
    { feature: SAVINGS, op: '>', threshold: 0 },
  ],
  [
    { feature: TENURE, op: 'in', lower: 0.5, upper: 2.5 },
    { feature: LATE, op: '<=', threshold: 0 },
  ],
  [{ feature: DEBT, op: '<=', threshold: -0.5 }],
]
const fires = (r: readonly number[], clauses: readonly Clause[]): boolean =>
  clauses.every((c) => {
    const v = r[c.feature] ?? 0
    if (c.op === '<=') return v <= (c.threshold ?? 0)
    if (c.op === '>') return v > (c.threshold ?? 0)
    return v >= (c.lower ?? -Infinity) && v <= (c.upper ?? Infinity)
  })

export default function RulesRecipe(): ReactNode {
  const [rows, setRows] = useState(400)
  const [minPrecision, setMinPrecision] = useState(0.3)
  const [seed, setSeed] = useState(3)
  const [kind, setKind] = useState<'anchors' | 'rule ensemble'>('anchors')
  const record = useMemo(() => {
    const x = creditRows(seed, rows)
    const bad = x.map((r) => creditScore(r) < 0)
    const entries = RULES.map((clauses): RuleEntry => {
      const hit = x.flatMap((r, i) => (fires(r, clauses) ? [i] : []))
      const right = hit.filter((i) => bad[i]).length
      return {
        clauses,
        precision: right / Math.max(1, hit.length),
        coverage: hit.length / x.length,
        support: hit.length,
      }
    })
    const kept = entries.filter((r) => (r.precision ?? 0) >= minPrecision)
    return ruleSet({
      method: 'hand-written rule list',
      features: CREDIT_FEATURES,
      output: 'P(default)',
      rules:
        kind === 'anchors'
          ? kept
          : kept.map(({ clauses, coverage }): RuleEntry => {
              const weight = 3 * ((entries.find((e) => e.clauses === clauses)?.precision ?? 0.5) - 0.5)
              const c = coverage ?? 0
              return { clauses, weight, coverage: c, importance: Math.abs(weight) * Math.sqrt(c * (1 - c)) }
            }),
      ...(kind === 'rule ensemble' ? { intercept: 0 } : {}),
    })
  }, [rows, minPrecision, seed, kind])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={4}>
        <NumberField label="rows to measure on" value={rows} onChange={setRows} min={50} max={2000} step={50} integer />
        <NumberField
          label="minimum precision"
          value={minPrecision}
          onChange={setMinPrecision}
          min={0}
          max={1}
          step={0.05}
        />
        <SeedInput value={seed} onChange={setSeed} />
        <Choice label="as" value={kind} options={['anchors', 'rule ensemble'] as const} onChange={setKind} />
      </ControlPanel>
      <ExpositionView
        record={record}
        title={`${record.rules.length} rules, e.g. ${record.rules[0] ? ruleText(record.rules[0], CREDIT_FEATURES) : 'none'}`}
      />
    </div>
  )
}
