import { type GroupRateName, subgroupReport, tableRates } from 'omai-engine/exposition'
import { ControlPanel, ExpositionView, NumberField, SeedInput } from 'omai-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'
import { uniforms } from '../../toys/synthetic.ts'

export const recipe: Recipe = {
  title: 'Subgroup report',
  question:
    'How do I find the intersection of attributes where a classifier does worst, when each attribute alone looks fine?',
  explain:
    'A `subgroupReport` record has the contingency table of every combination of attributes (here sex and age band), each combination’s rates and counts, the largest gap of each rate from the overall one and the subgroup it belongs to. `SubgroupView` draws a heatmap of one rate with the cell counts, the worst cell outlined, and cells below the minimum size greyed. The classifier here is made to fail one combination (older women), so the two attributes’ marginals stay close while the cell is far off. Raise the minimum size and small cells drop out of the ranking.',
  shows: ['subgroupReport', 'SubgroupView'],
}

const SEX = ['women', 'men']
const AGE = ['young', 'middle', 'older']
const RATES: readonly GroupRateName[] = [
  'selectionRate',
  'truePositiveRate',
  'falsePositiveRate',
  'precision',
  'accuracy',
  'baseRate',
]

export default function SubgroupReportRecipe(): ReactNode {
  const [penalty, setPenalty] = useState(0.35)
  const [minSize, setMinSize] = useState(30)
  const [seed, setSeed] = useState(1)
  const [rows, setRows] = useState(3000)
  const record = useMemo(() => {
    const u = uniforms(seed, 4 * rows)
    const subgroups = SEX.flatMap((s) => AGE.map((a) => [s, a]))
    const tables = subgroups.map(() => [
      [0, 0],
      [0, 0],
    ])
    for (let i = 0; i < rows; i++) {
      const s = (u[i] ?? 0) < 0.5 ? 0 : 1
      const a = Math.min(2, Math.floor((u[rows + i] ?? 0) * 3))
      const y = (u[2 * rows + i] ?? 0) < 0.4 ? 1 : 0
      // The classifier is right 85% of the time, and for older women it also rejects a share of the people it should accept.
      const right = (u[3 * rows + i] ?? 0) < 0.85
      const missed = s === 0 && a === 2 && y === 1 && (u[3 * rows + i] ?? 0) > 1 - penalty
      const p = missed ? 0 : right ? y : 1 - y
      const t = tables[s * AGE.length + a] as number[][]
      const r = t[y] as number[]
      r[p] = (r[p] ?? 0) + 1
    }
    const counts = tables.map((t) => t.flat().reduce((x, y) => x + y, 0))
    const rates = tables.map(tableRates)
    const total = [0, 1].map((a) => [0, 1].map((p) => tables.reduce((s, t) => s + (t[a]?.[p] ?? 0), 0)))
    const overall = tableRates(total)
    const eligible = counts.flatMap((c, i) => (c >= minSize ? [i] : []))
    const gaps = {} as Record<GroupRateName, number>
    const worst = {} as Record<GroupRateName, { subgroup: number; gap: number }>
    for (const r of RATES) {
      let best = eligible[0] ?? 0
      for (const i of eligible)
        if (Math.abs((rates[i]?.[r] ?? 0) - overall[r]) > Math.abs((rates[best]?.[r] ?? 0) - overall[r])) best = i
      gaps[r] = Math.abs((rates[best]?.[r] ?? 0) - overall[r])
      worst[r] = { subgroup: best, gap: (rates[best]?.[r] ?? 0) - overall[r] }
    }
    return subgroupReport({
      attributes: ['sex', 'age'],
      values: [SEX, AGE],
      subgroups,
      counts,
      tables,
      rates,
      overall,
      minSize,
      gaps,
      worst,
    })
  }, [penalty, minSize, seed, rows])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={4}>
        <NumberField
          label="older women wrongly rejected"
          value={penalty}
          onChange={setPenalty}
          min={0}
          max={0.6}
          step={0.05}
        />
        <NumberField
          label="minimum subgroup size"
          value={minSize}
          onChange={setMinSize}
          min={1}
          max={500}
          step={10}
          integer
        />
        <NumberField label="rows" value={rows} onChange={setRows} min={500} max={20000} step={500} integer />
        <SeedInput value={seed} onChange={setSeed} />
      </ControlPanel>
      <ExpositionView record={record} title="Rates by sex and age" />
    </div>
  )
}
