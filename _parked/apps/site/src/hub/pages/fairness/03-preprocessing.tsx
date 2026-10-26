import { generate } from 'mdml-engine/data'
import { biasedCredit } from 'mdml-engine/datasets'
import { recordsOf } from 'mdml-engine/exposition'
import {
  disparateImpactRemover,
  type FairnessData,
  fairnessAudit,
  reweighedFit,
  withoutColumns,
} from 'mdml-engine/methods/fairness'
import { logisticRegression } from 'mdml-engine/methods/linear'
import { key } from 'mdml-engine/random'
import { figure } from 'mdml-render/figure'
import { Choice, CoefficientTableView, ControlPanel, CurvesView, GroupReportView, NumberField } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../../gallery/recipe.ts'

export const recipe: Recipe = {
  title: 'Preprocessing: reweighing and repair',
  question: 'How can the training data itself be changed so a model learns less of a group difference?',
  explain:
    'Two fixes before training. `reweighing(data)` (Kamiran and Calders) weights each row by P(group) P(label) / P(group, label), so in the weighted data the label is independent of the group: the minority’s good outcomes, made rare by the label bias, count for more. `disparateImpactRemover(data, { repairLevel })` (Feldman et al.) moves each feature towards the groups’ median distribution while keeping every row’s rank within its group (both models here are fitted without the group column); the figure shows the most group-dependent feature’s distribution per group before (solid) and after (dashed). Raise the label bias, or the repair level, and compare the group reports of the model trained on each.',
}

const SMALL = figure({ width: 460, height: 320 })
const REPORT = figure({ width: 940, height: 480 })

type Method = 'reweighing' | 'repair'

export default function PreprocessingRecipe(): ReactNode {
  const [method, setMethod] = useState<Method>('reweighing')
  const [labelBias, setLabelBias] = useState(0.4)
  const [repairLevel, setRepairLevel] = useState(1)
  const data = useMemo(() => generate(biasedCredit, { key: key(12), knobs: { n: 1200, labelBias } }).data, [labelBias])
  const reweighed = useMemo(() => (method === 'reweighing' ? reweighedFit(data) : undefined), [data, method])
  const repaired = useMemo(() => {
    if (method !== 'repair') return undefined
    const r = disparateImpactRemover(data, { repairLevel })
    // Neither model reads the group itself: both are fitted and audited without it.
    const unaware = (d: FairnessData) => logisticRegression().fit(withoutColumns(d, ['group']))
    const before = fairnessAudit(unaware(data), data, { dropSensitive: true, label: 'original features' })
    const after = fairnessAudit(unaware(r.data), r.data, {
      dropSensitive: true,
      label: `repaired (level ${repairLevel})`,
    })
    return { r, before, after }
  }, [data, method, repairLevel])
  const reports =
    method === 'reweighing'
      ? recordsOf(reweighed?.exposition ?? [], 'groupReport')
      : [
          ...recordsOf(repaired?.before.exposition ?? [], 'groupReport'),
          ...recordsOf(repaired?.after.exposition ?? [], 'groupReport'),
        ]
  const table = recordsOf(reweighed?.exposition ?? [], 'coefficientTable')[0]
  const curves = recordsOf(repaired?.r.exposition ?? [], 'curves')[0]

  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={3}>
        <Choice
          label="method"
          value={method}
          onChange={setMethod}
          options={[
            { value: 'reweighing', label: 'reweighing' },
            { value: 'repair', label: 'disparate-impact repair' },
          ]}
        />
        <NumberField label="label bias" value={labelBias} onChange={setLabelBias} min={0} max={0.9} step={0.05} />
        <NumberField
          label="repair level"
          value={repairLevel}
          onChange={setRepairLevel}
          min={0}
          max={1}
          step={0.05}
          disabled={method !== 'repair'}
        />
      </ControlPanel>
      {table !== undefined && (
        <CoefficientTableView record={table} figure={SMALL} title="weight of each group and label" />
      )}
      {curves !== undefined && <CurvesView record={curves} figure={SMALL} play={false} />}
      {reports.map((r) => (
        <GroupReportView key={r.label} record={r} figure={REPORT} />
      ))}
    </div>
  )
}
