import { type ReactNode, useMemo, useState } from 'react'
import { generate } from 'whitebox-engine/data'
import { biasedCredit } from 'whitebox-engine/datasets'
import { recordsOf } from 'whitebox-engine/exposition'
import { fairnessAudit, withoutColumns } from 'whitebox-engine/methods/fairness'
import { logisticRegression } from 'whitebox-engine/methods/linear'
import { key } from 'whitebox-engine/random'
import { cartesian, histogram } from 'whitebox-render/charts'
import { figure } from 'whitebox-render/figure'
import {
  Chart,
  ControlPanel,
  DraggableLine,
  FigureFrame,
  GroupReportView,
  InteractionOverlay,
  NumberField,
  Overlaid,
  Toggle,
  useChartProjection,
  useColourMode,
} from 'whitebox-render/react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Group report',
  question: 'How do I show whether a classifier treats two groups alike, and what a per-group threshold changes?',
  explain:
    'A logistic regression is trained on `biasedCredit` data, whose recorded labels drop a share of the minority’s good outcomes (label bias) and whose postcode is a proxy for the group. `fairnessAudit(model, data)` scores every row and returns a `groupReport`: each group’s rates side by side, its gap to the reference group, its ROC curve with its operating point, and its contingency table. Drag either group’s threshold on its scores (or use the controls): a lower minority threshold closes the selection-rate gap (demographic parity) but moves its false-positive rate; equal true- and false-positive rates (equalised odds) need the two operating points to meet on the ROC panel. Withhold the group from the model and the proxy carries it anyway.',
}

const FIGURE = figure({ width: 460, height: 260 })
const REPORT = figure({ width: 940, height: 500 })
const GROUPS = ['majority', 'minority'] as const

export default function GroupReportRecipe(): ReactNode {
  const mode = useColourMode()
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
  const projections = [useChartProjection(), useChartProjection()]
  const y = useMemo(() => Array.from(data.y.data), [data])
  // Each group's scores by outcome, on fixed axes so the figure stays still while a threshold is dragged.
  const panels = useMemo(
    () =>
      GROUPS.map((name, g) => {
        const rows = audit.groupIndex.flatMap((a, i) => (a === g ? [i] : []))
        const scoresOf = (c: number) => rows.filter((i) => y[i] === c).map((i) => audit.scores[i] ?? 0)
        const counts = [scoresOf(0), scoresOf(1)].map((s) => {
          const bins = new Array<number>(25).fill(0)
          for (const v of s) bins[Math.min(24, Math.floor(v * 25))] = (bins[Math.min(24, Math.floor(v * 25))] ?? 0) + 1
          return Math.max(...bins)
        })
        const top = Math.ceil((Math.max(...counts) * 1.15) / 5) * 5
        const option = cartesian(
          [
            histogram(scoresOf(0), { bins: 25, range: [0, 1], name: 'default', opacity: 0.6, mode }),
            histogram(scoresOf(1), { bins: 25, range: [0, 1], name: 'repaid', opacity: 0.6, colourOffset: 1, mode }),
          ],
          {
            mode,
            figure: FIGURE,
            title: `${name} (${rows.length} applicants)`,
            x: { label: 'predicted probability of repaying', limits: [0, 1] },
            y: { label: 'applicants', limits: [0, top] },
          },
        )
        return { option, top }
      }),
    [audit, y, mode],
  )

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
      <div className="grid gap-3 md:grid-cols-2">
        {panels.map((p, g) => {
          const projection = projections[g]
          return (
            <FigureFrame key={GROUPS[g]} figure={FIGURE} filename={`scores-${GROUPS[g]}`}>
              <Overlaid
                width={FIGURE.width}
                height={FIGURE.height}
                overlay={
                  projection === undefined || projection.projection === null ? null : (
                    <InteractionOverlay projection={projection.projection} width={FIGURE.width} height={FIGURE.height}>
                      <DraggableLine
                        axis="x"
                        value={thresholds[g] ?? 0.5}
                        onChange={setGroup(g)}
                        span={[0, p.top]}
                        min={0}
                        max={1}
                        step={0.01}
                        label="threshold"
                      />
                    </InteractionOverlay>
                  )
                }
              >
                <Chart
                  option={p.option}
                  figure={FIGURE}
                  {...(projection === undefined ? {} : { onInit: projection.onInit })}
                />
              </Overlaid>
            </FigureFrame>
          )
        })}
      </div>
      {record !== undefined && <GroupReportView record={record} thresholds={thresholds} figure={REPORT} />}
    </div>
  )
}
