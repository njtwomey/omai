import { generate } from 'mdml-engine/data'
import { biasedCredit } from 'mdml-engine/datasets'
import { recordsOf } from 'mdml-engine/exposition'
import { fairnessAudit, withoutColumns } from 'mdml-engine/methods/fairness'
import { logisticRegression } from 'mdml-engine/methods/linear'
import { key } from 'mdml-engine/random'
import { figure } from 'mdml-render/figure'
import { ControlPanel, GroupReportView, GroupThresholds, NumberField, Toggle } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Group report',
  question: 'How do I show whether a classifier treats two groups alike, and what a per-group threshold changes?',
  explain:
    'A logistic regression is trained on `biasedCredit` data, whose recorded labels drop a share of the minority’s good outcomes (label bias) and whose postcode is a proxy for the group. `fairnessAudit(model, data)` scores every row and returns a `groupReport`: each group’s rates side by side, its gap to the reference group, its ROC curve with its operating point, and its contingency table. `GroupThresholds` gives each group a threshold explorer over its own scores (`analyseThresholds` of its rows): drag either group’s threshold (or use the controls). A lower minority threshold closes the selection-rate gap (demographic parity) but moves its false-positive rate; equal true- and false-positive rates (equalised odds) need the two operating points to meet on the ROC panel. Withhold the group from the model and the proxy carries it anyway.',
}

const FIGURE = figure({ width: 460, height: 260 })
const REPORT = figure({ width: 940, height: 500 })

export default function GroupReportRecipe(): ReactNode {
  const [labelBias, setLabelBias] = useState(0.3)
  const [proxy, setProxy] = useState(1.5)
  const [aware, setAware] = useState(true)
  const [thresholds, setThresholds] = useState<readonly number[]>([0.5, 0.5])
  const data = useMemo(
    () => generate(biasedCredit, { key: key(3), knobs: { n: 1200, labelBias, proxyStrength: proxy } }).data,
    [labelBias, proxy],
  )
  const audit = useMemo(() => {
    const train = aware ? data : withoutColumns(data, ['group'])
    const model = logisticRegression().fit(train)
    return fairnessAudit(model, data, { dropSensitive: !aware })
  }, [data, aware])
  const record = recordsOf(audit.exposition, 'groupReport')[0]
  const setGroup = (g: number) => (v: number) => setThresholds((t) => t.map((x, i) => (i === g ? v : x)))

  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={3}>
        <NumberField label="label bias" value={labelBias} onChange={setLabelBias} min={0} max={0.9} step={0.05} />
        <NumberField label="proxy strength" value={proxy} onChange={setProxy} min={0} max={4} step={0.25} />
        <Toggle label="model sees the group" value={aware} onChange={setAware} />
        <NumberField
          label="majority threshold"
          value={thresholds[0] ?? 0.5}
          onChange={setGroup(0)}
          min={0}
          max={1}
          step={0.01}
        />
        <NumberField
          label="minority threshold"
          value={thresholds[1] ?? 0.5}
          onChange={setGroup(1)}
          min={0}
          max={1}
          step={0.01}
        />
      </ControlPanel>
      {record !== undefined && (
        <>
          <GroupThresholds record={record} thresholds={thresholds} onThresholds={setThresholds} figure={FIGURE} />
          <GroupReportView record={record} thresholds={thresholds} figure={REPORT} />
        </>
      )}
    </div>
  )
}
