import { generate } from 'mdml-engine/data'
import { biasedCredit } from 'mdml-engine/datasets'
import { type ThresholdConstraint, thresholdDecisions } from 'mdml-engine/fairness'
import { buildGroupReport, positiveScores, sensitiveGroups, thresholdOptimizer } from 'mdml-engine/methods/fairness'
import { logisticRegression } from 'mdml-engine/methods/linear'
import { key } from 'mdml-engine/random'
import { cartesian, lines } from 'mdml-render/charts'
import { figure } from 'mdml-render/figure'
import {
  Chart,
  Choice,
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
} from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../../gallery/recipe.ts'

export const recipe: Recipe = {
  title: 'Threshold optimiser',
  question: 'How do per-group thresholds make a trained classifier meet a fairness constraint, and at what cost?',
  explain:
    '`thresholdOptimizer(model, data, { constraint })` (fairlearn’s `ThresholdOptimizer`, matched to it in the golden tests) leaves the model alone and picks a randomised threshold for each group. For demographic parity it traces each group’s accuracy against its selection rate over every threshold, keeps the upper convex hull (mixing two thresholds reaches any point between them), and chooses the shared selection rate where the groups’ weighted accuracy peaks. For equalised odds the hulls are ROC curves: the shared true-positive rate is the lowest group’s, and a group above it mixes in a coin flip. Drag the shared rate along the left figure to see what other choices would cost; the report shows the expected decisions, so the operating points meet on the ROC panel.',
}

const FIGURE = figure({ width: 460, height: 340 })
const REPORT = figure({ width: 940, height: 500 })

const CONSTRAINTS = [
  { value: 'demographicParity', label: 'demographic parity' },
  { value: 'equalisedOdds', label: 'equalised odds' },
  { value: 'truePositiveRateParity', label: 'equal opportunity' },
] as const

export default function ThresholdOptimiserRecipe(): ReactNode {
  const mode = useColourMode()
  const [constraint, setConstraint] = useState<ThresholdConstraint>('demographicParity')
  const [labelBias, setLabelBias] = useState(0.3)
  const [follow, setFollow] = useState(true)
  const [manual, setManual] = useState(0.4)
  const { data, model } = useMemo(() => {
    const data = generate(biasedCredit, { key: key(5), knobs: { n: 1500, labelBias } }).data
    return { data, model: logisticRegression().fit(data) }
  }, [labelBias])
  const fitted = useMemo(() => thresholdOptimizer(model, data, { constraint }), [model, data, constraint])
  const x = follow ? fitted.xBest : manual
  const report = useMemo(() => {
    const groups = sensitiveGroups(data)
    const { scores } = positiveScores(model, data.x)
    const labels = Array.from(data.y.data, Number)
    const rules = fitted.rulesAt(x)
    return buildGroupReport({
      scores,
      labels,
      groups,
      thresholds: rules
        .map((r) => (r.p0 >= r.p1 ? r.rule0 : r.rule1).threshold)
        .map((t) => Math.min(1, Math.max(0, t))),
      decisions: thresholdDecisions(rules, scores, groups.index),
      classes: ['default', 'repaid'],
      label: follow ? 'optimised' : 'chosen by hand',
      sweep: 0,
    })
  }, [data, model, fitted, x, follow])
  const eo = constraint === 'equalisedOdds'
  const { projection, onInit } = useChartProjection()
  const shared = useMemo(() => fitted.grid.map((_, j) => Math.min(...fitted.curves.map((c) => c[j] ?? 0))), [fitted])
  const option = useMemo(() => {
    const step = (v: readonly number[]): number[] => v.filter((_, j) => j % 5 === 0)
    const gx = step(fitted.grid)
    return cartesian(
      [
        lines(
          fitted.groups.map((name, g) => ({ x: gx, y: step(fitted.curves[g] ?? []), name })),
          { mode },
        ),
        lines(
          [
            {
              x: gx,
              y: step(eo ? shared : fitted.overall),
              name: eo ? 'shared TPR' : 'overall',
              colour: 'fgSecondary',
              dash: 'dashed',
            },
          ],
          {
            mode,
          },
        ),
      ],
      {
        mode,
        figure: FIGURE,
        x: {
          label: eo
            ? 'false-positive rate'
            : constraint === 'demographicParity'
              ? 'selection rate'
              : 'true-positive rate',
          limits: [0, 1],
        },
        y: { label: eo ? 'true-positive rate (ROC hull)' : 'accuracy', limits: [0, 1] },
      },
    )
  }, [fitted, shared, eo, constraint, mode])
  const accuracy = report.overall.accuracy

  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <Choice label="constraint" value={constraint} onChange={setConstraint} options={CONSTRAINTS} />
        <NumberField label="label bias" value={labelBias} onChange={setLabelBias} min={0} max={0.9} step={0.05} />
        <Toggle label="use the best shared rate" value={follow} onChange={setFollow} />
        <NumberField
          label="shared rate"
          value={x}
          onChange={(v) => {
            setFollow(false)
            setManual(v)
          }}
          min={0}
          max={1}
          step={0.01}
        />
      </ControlPanel>
      <div className="flex flex-wrap items-start gap-4">
        <FigureFrame figure={FIGURE} filename="threshold-optimiser-tradeoff">
          <Overlaid
            width={FIGURE.width}
            height={FIGURE.height}
            overlay={
              projection === null ? null : (
                <InteractionOverlay projection={projection} width={FIGURE.width} height={FIGURE.height}>
                  <DraggableLine
                    axis="x"
                    value={x}
                    onChange={(v) => {
                      setFollow(false)
                      setManual(v)
                    }}
                    span={[0, 1]}
                    min={0}
                    max={1}
                    step={0.001}
                    label={follow ? 'best' : 'shared rate'}
                  />
                </InteractionOverlay>
              )
            }
          >
            <Chart option={option} figure={FIGURE} onInit={onInit} />
          </Overlaid>
        </FigureFrame>
        <div className="flex max-w-sm flex-col gap-2 text-sm">
          <p>
            Accuracy with these decisions: <strong>{accuracy.toFixed(3)}</strong> (best for the constraint{' '}
            {(eo ? (fitted.overall[Math.round(fitted.xBest * 1000)] ?? 0) : fitted.yBest).toFixed(3)}).
          </p>
          <p>
            The coloured curves are each group’s upper hull; the dashed one is what the groups can share. The line is
            the shared {eo ? 'false-positive rate' : 'rate'} every group is moved to.
          </p>
        </div>
      </div>
      <GroupReportView record={report} figure={REPORT} />
    </div>
  )
}
