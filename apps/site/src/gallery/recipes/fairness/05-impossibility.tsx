import { recordsOf } from 'mdml-engine/exposition'
import { impossibilityStudy } from 'mdml-engine/methods/fairness'
import { key } from 'mdml-engine/random'
import { figure } from 'mdml-render/figure'
import {
  Chart,
  ControlPanel,
  DraggableLine,
  FigureFrame,
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
  title: 'Calibration against equalised odds',
  question: 'Why can a classifier not be calibrated and have equal error rates when groups’ base rates differ?',
  explain:
    'Kleinberg et al. and Chouldechova showed that when groups’ base rates differ, calibrated scores (a score of 0.7 means 70% for everyone) cut at one threshold give unequal false-positive and false-negative rates; and equalising those rates makes a positive decision mean different things for each group (unequal predictive value). `impossibilityStudy()` uses the exact posterior of `biasedCredit` data with a growing real base-rate gap and no label bias, so the scores are calibrated by construction (the calibration gap stays near zero). Solid: those scores at 0.5; dashed: after the equalised-odds threshold optimiser. Only one of the two disparities can be zero, unless the gap is. Drag to read a gap.',
}

const FIGURE = figure({ width: 640, height: 400 })

export default function ImpossibilityRecipe(): ReactNode {
  const mode = useColourMode()
  const [n, setN] = useState(3000)
  const [at, setAt] = useState(0.2)
  const study = useMemo(() => impossibilityStudy({ n, key: key(4) }), [n])
  const record = recordsOf(study.exposition, 'curves')[0]
  const option = useMemo(
    () => (record === undefined ? undefined : curveSetView(record, { figure: { ...FIGURE, mode } })),
    [record, mode],
  )
  const { projection, onInit } = useChartProjection()
  const xs = record?.x ?? []
  const nearest = xs.reduce((best, x, i) => (Math.abs(x - at) < Math.abs((xs[best] ?? 0) - at) ? i : best), 0)
  const p = study.points[nearest]

  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <NumberField label="applicants per dataset" value={n} onChange={setN} min={500} max={8000} step={500} />
        <NumberField label="base-rate gap" value={at} onChange={setAt} min={0} max={0.6} step={0.01} />
      </ControlPanel>
      {option !== undefined && (
        <FigureFrame figure={FIGURE} filename="impossibility">
          <Overlaid
            width={FIGURE.width}
            height={FIGURE.height}
            overlay={
              projection === null ? null : (
                <InteractionOverlay projection={projection} width={FIGURE.width} height={FIGURE.height}>
                  <DraggableLine axis="x" value={at} onChange={setAt} min={0} max={0.6} step={0.01} label="gap" />
                </InteractionOverlay>
              )
            }
          >
            <Chart option={option} figure={FIGURE} onInit={onInit} />
          </Overlaid>
        </FigureFrame>
      )}
      {p !== undefined && (
        <p className="text-sm">
          Nearest gap {(xs[nearest] ?? 0).toFixed(3)} (base rates {p.baseRates.map((r) => r.toFixed(2)).join(' and ')}):
          calibrated scores have equalised-odds difference <strong>{p.equalisedOdds.toFixed(3)}</strong> and
          predictive-parity difference <strong>{p.predictiveParity.toFixed(3)}</strong>; after equalising odds,{' '}
          <strong>{p.equalisedOddsAfter.toFixed(3)}</strong> and <strong>{p.predictiveParityAfter.toFixed(3)}</strong>.
        </p>
      )}
    </div>
  )
}
