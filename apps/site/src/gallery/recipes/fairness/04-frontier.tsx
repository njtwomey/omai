import { type ReactNode, useMemo, useState } from 'react'
import { generate } from 'whitebox-engine/data'
import { biasedCredit } from 'whitebox-engine/datasets'
import { recordsOf } from 'whitebox-engine/exposition'
import { fairnessFrontier, type ReductionConstraint } from 'whitebox-engine/methods/fairness'
import { key } from 'whitebox-engine/random'
import { figure } from 'whitebox-render/figure'
import {
  Chart,
  Choice,
  ControlPanel,
  DraggableLine,
  FigureFrame,
  InteractionOverlay,
  NumberField,
  Overlaid,
  useChartProjection,
  useColourMode,
} from 'whitebox-render/react'
import { curveSetView } from 'whitebox-render/views'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Accuracy–fairness frontier',
  question: 'How much accuracy does each step towards parity cost?',
  explain:
    '`fairnessFrontier(data, { constraint })` traces the best accuracy available at each level of unfairness two ways: in training, a grid of prices on the violation (`gridSearchReduction`), and after training, the unconstrained model’s scores with the disadvantaged group’s threshold moved. Both are reduced to their Pareto fronts; the mark is the exact threshold optimiser. Drag the allowed violation to read the best accuracy each family offers there. With more label bias the frontier steepens: parity costs more measured accuracy, because the recorded labels themselves are biased.',
}

const FIGURE = figure({ width: 640, height: 400 })

/** The best accuracy of a front at a violation of at most `v`. */
function bestWithin(x: readonly number[], y: readonly number[], v: number): number | undefined {
  const ok = y.filter((_, i) => (x[i] ?? 1) <= v)
  return ok.length === 0 ? undefined : Math.max(...ok)
}

export default function FrontierRecipe(): ReactNode {
  const mode = useColourMode()
  const [constraint, setConstraint] = useState<ReductionConstraint>('demographicParity')
  const [labelBias, setLabelBias] = useState(0.35)
  const [allowed, setAllowed] = useState(0.05)
  const data = useMemo(() => generate(biasedCredit, { key: key(21), knobs: { n: 1000, labelBias } }).data, [labelBias])
  const frontier = useMemo(() => fairnessFrontier(data, { constraint, key: key(2) }), [data, constraint])
  const record = recordsOf(frontier.exposition, 'curves')[0]
  const option = useMemo(
    () => (record === undefined ? undefined : curveSetView(record, { figure: { ...FIGURE, mode } })),
    [record, mode],
  )
  const { projection, onInit } = useChartProjection()
  const read = (f: readonly { accuracy: number; violation: number }[]): string => {
    const v = bestWithin(
      f.map((p) => p.violation),
      f.map((p) => p.accuracy),
      allowed,
    )
    return v === undefined ? 'none reaches it' : v.toFixed(3)
  }

  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={3}>
        <Choice
          label="constraint"
          value={constraint}
          onChange={setConstraint}
          options={[
            { value: 'demographicParity', label: 'demographic parity' },
            { value: 'equalisedOdds', label: 'equalised odds' },
          ]}
        />
        <NumberField label="label bias" value={labelBias} onChange={setLabelBias} min={0} max={0.9} step={0.05} />
        <NumberField label="allowed violation" value={allowed} onChange={setAllowed} min={0} max={0.5} step={0.005} />
      </ControlPanel>
      {option !== undefined && (
        <FigureFrame figure={FIGURE} filename="fairness-frontier">
          <Overlaid
            width={FIGURE.width}
            height={FIGURE.height}
            overlay={
              projection === null ? null : (
                <InteractionOverlay projection={projection} width={FIGURE.width} height={FIGURE.height}>
                  <DraggableLine
                    axis="x"
                    value={allowed}
                    onChange={setAllowed}
                    min={0}
                    max={0.5}
                    step={0.005}
                    label="allowed violation"
                  />
                </InteractionOverlay>
              )
            }
          >
            <Chart option={option} figure={FIGURE} onInit={onInit} />
          </Overlaid>
        </FigureFrame>
      )}
      <p className="text-sm">
        Best accuracy within a violation of {allowed.toFixed(3)}: in training{' '}
        <strong>{read(frontier.reduction)}</strong>, after training <strong>{read(frontier.thresholds)}</strong>.
      </p>
    </div>
  )
}
