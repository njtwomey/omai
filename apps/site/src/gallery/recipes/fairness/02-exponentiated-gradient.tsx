import { generate } from 'mdml-engine/data'
import { biasedCredit } from 'mdml-engine/datasets'
import { recordsOf } from 'mdml-engine/exposition'
import type { ReductionConstraint } from 'mdml-engine/fairness'
import { exponentiatedGradient } from 'mdml-engine/methods/fairness'
import { key } from 'mdml-engine/random'
import { figure } from 'mdml-render/figure'
import {
  Chart,
  Choice,
  ControlPanel,
  CurvesView,
  DraggableLine,
  FigureFrame,
  GroupReportView,
  InteractionOverlay,
  NumberField,
  Overlaid,
  useChartProjection,
  useColourMode,
} from 'mdml-render/react'
import { curveSetView } from 'mdml-render/views'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Exponentiated gradient',
  question: 'How does a fairness constraint enter training, and how does the trade-off move as it does?',
  explain:
    '`exponentiatedGradient(data, { constraint, epsilon })` (Agarwal et al., 2018; fairlearn’s reduction) plays a game: an auditor prices every violated constraint (a group’s mean decision straying more than ε from everyone’s), and a learner refits logistic regression to the priced problem, relabelling and reweighting rows. The average of the learner’s classifiers is a randomised classifier. Play the first figure to watch the violation of the average fall while its error rises a little and the prices settle; the second traces the same average in (violation, error) space from the unconstrained model. Drag ε on it (or use the control) to loosen or tighten the constraint.',
}

const PATH = figure({ width: 460, height: 340 })
const REPORT = figure({ width: 940, height: 500 })

export default function ExponentiatedGradientRecipe(): ReactNode {
  const mode = useColourMode()
  const [constraint, setConstraint] = useState<ReductionConstraint>('demographicParity')
  const [epsilon, setEpsilon] = useState(0.02)
  const [iterations, setIterations] = useState(25)
  const data = useMemo(() => generate(biasedCredit, { key: key(9), knobs: { n: 1000, labelBias: 0.35 } }).data, [])
  const result = useMemo(
    () => exponentiatedGradient(data, { constraint, epsilon, iterations, key: key(1) }),
    [data, constraint, epsilon, iterations],
  )
  const [trace, path] = recordsOf(result.exposition, 'curves')
  const after = recordsOf(result.exposition, 'groupReport')[1]
  const { projection, onInit } = useChartProjection()
  const pathOption = useMemo(
    () => (path === undefined ? undefined : curveSetView(path, { figure: { ...PATH, mode } })),
    [path, mode],
  )

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
        <NumberField
          label="ε (allowed violation)"
          value={epsilon}
          onChange={setEpsilon}
          min={0}
          max={0.3}
          step={0.005}
        />
        <NumberField label="iterations" value={iterations} onChange={setIterations} min={2} max={60} step={1} />
      </ControlPanel>
      <div className="flex flex-wrap items-start gap-4">
        {trace !== undefined && <CurvesView record={trace} figure={PATH} />}
        {pathOption !== undefined && (
          <FigureFrame figure={PATH} filename="exponentiated-gradient-path">
            <Overlaid
              width={PATH.width}
              height={PATH.height}
              overlay={
                projection === null ? null : (
                  <InteractionOverlay projection={projection} width={PATH.width} height={PATH.height}>
                    <DraggableLine
                      axis="x"
                      value={epsilon}
                      onChange={setEpsilon}
                      min={0}
                      max={0.3}
                      step={0.005}
                      label="ε"
                    />
                  </InteractionOverlay>
                )
              }
            >
              <Chart option={pathOption} figure={PATH} onInit={onInit} />
            </Overlaid>
          </FigureFrame>
        )}
      </div>
      {after !== undefined && <GroupReportView record={after} figure={REPORT} />}
    </div>
  )
}
