import { groupReport, groupReportFromTables } from 'mdml-engine/exposition'
import { rocCurve } from 'mdml-engine/metrics'
import { ControlPanel, ExpositionView, GroupSweepView, NumberField, SeedInput, Toggle } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'
import { type ScoredRows, scoredGroups } from '../../toys/synthetic.ts'

export const recipe: Recipe = {
  title: 'Group report',
  question: 'How do I show whether a classifier treats two groups alike, and what a threshold per group would change?',
  explain:
    'A `groupReport` record holds, per group, the contingency table, the rates read from it (selection, true and false positive rates, precision), the disparities between groups, each group’s ROC curve and, from the scored rows, what any other threshold would do. `GroupReportView` draws the rates and gaps with each group’s operating point on its ROC curve; drag a group’s point and its table and the gaps follow. `GroupSweepView` plots the rates over every threshold. The made-up scorer is pushed down for the minority group by the gap you set: at one threshold it selects them less and misses more of their positives, and moving their threshold repairs one gap at the price of another.',
  shows: ['groupReport', 'GroupReportView', 'GroupSweepView'],
}

const cell = (rows: ScoredRows, g: number, threshold: number): number[][] => {
  const t = [
    [0, 0],
    [0, 0],
  ]
  rows.groups.forEach((gr, i) => {
    if (gr !== g) return
    const row = t[rows.labels[i] ?? 0] as number[]
    const p = (rows.scores[i] ?? 0) >= threshold ? 1 : 0
    row[p] = (row[p] ?? 0) + 1
  })
  return t
}

export default function GroupReportRecipe(): ReactNode {
  const [gap, setGap] = useState(1)
  const [threshold, setThreshold] = useState(0.5)
  const [seed, setSeed] = useState(1)
  const [sweep, setSweep] = useState(false)
  const record = useMemo(() => {
    const rows = scoredGroups(seed, 3000, gap)
    const grid = Array.from({ length: 21 }, (_, i) => i / 20)
    const rate = (g: number, t: number, name: 'sel' | 'tpr' | 'fpr'): number => {
      const [[tn = 0, fp = 0] = [], [fn = 0, tp = 0] = []] = cell(rows, g, t)
      const ratio = (a: number, b: number): number => (b === 0 ? 0 : a / b)
      return name === 'sel'
        ? ratio(tp + fp, tn + fp + fn + tp)
        : name === 'tpr'
          ? ratio(tp, tp + fn)
          : ratio(fp, fp + tn)
    }
    return groupReport(
      groupReportFromTables({
        attribute: 'group',
        groups: ['majority', 'minority'],
        classes: ['rejected', 'approved'],
        thresholds: [threshold, threshold],
        tables: [0, 1].map((g) => cell(rows, g, threshold)),
        scored: rows,
        roc: [0, 1].map((g) => {
          const idx = rows.groups.flatMap((gr, i) => (gr === g ? [i] : []))
          return rocCurve(
            idx.map((i) => rows.labels[i] ?? 0),
            idx.map((i) => rows.scores[i] ?? 0),
          )
        }),
        sweep: {
          thresholds: grid,
          selectionRate: [0, 1].map((g) => grid.map((t) => rate(g, t, 'sel'))),
          truePositiveRate: [0, 1].map((g) => grid.map((t) => rate(g, t, 'tpr'))),
          falsePositiveRate: [0, 1].map((g) => grid.map((t) => rate(g, t, 'fpr'))),
        },
      }),
    )
  }, [gap, threshold, seed])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={4}>
        <NumberField label="score gap against the minority" value={gap} onChange={setGap} min={0} max={3} step={0.25} />
        <NumberField label="threshold" value={threshold} onChange={setThreshold} min={0.05} max={0.95} step={0.05} />
        <SeedInput value={seed} onChange={setSeed} />
        <Toggle label="rates over every threshold" value={sweep} onChange={setSweep} />
      </ControlPanel>
      {sweep ? (
        <GroupSweepView record={record} />
      ) : (
        <ExpositionView record={record} title="Two groups, one threshold" />
      )}
    </div>
  )
}
