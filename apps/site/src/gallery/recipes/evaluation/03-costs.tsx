import { type ReactNode, useMemo, useState } from 'react'
import { thresholdConfusion, thresholdCosts } from 'whitebox-engine/metrics'
import { key, normal } from 'whitebox-engine/random'
import { cartesian, histogram, lines, referenceLines } from 'whitebox-render/charts'
import { figure } from 'whitebox-render/figure'
import {
  Chart,
  ContingencyTable,
  ControlPanel,
  DraggableLine,
  FigureFrame,
  InteractionOverlay,
  NumberField,
  Overlaid,
  Toggle,
  useChartProjection,
  useColourMode,
} from 'whitebox-render/react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Misclassification costs',
  question: 'How do I choose a threshold when the two kinds of error cost different amounts?',
  explain:
    'Give each decision a cost — `costs[actual][predicted]`, here a missed case (false negative) and a false alarm (false positive), with correct decisions free — and `ContingencyTable` shows each cell’s cost and the total and per-case cost. `thresholdCosts(yTrue, scores, costs)` sweeps every threshold in one sorted pass and returns the total cost at each and the cheapest. Make misses dearer and the cheapest threshold drops, buying sensitivity with false alarms; rare positives (low prevalence) push the other way. Drag the threshold on either figure (it leaves the cheapest and sets it by hand). For calibrated probabilities the cheapest cut is cost(FP) / (cost(FP) + cost(FN)); for raw scores, sweep.',
}

const n = 600
const noise = Array.from(normal(key(21), [n]).data)
const order = Array.from(normal(key(22), [n]).data)
const separation = 2
const FIGURE = figure({ width: 480, height: 360 })
const X: [number, number] = [-3, 5.5]

export default function CostsRecipe(): ReactNode {
  const mode = useColourMode()
  const [missCost, setMissCost] = useState(20)
  const [alarmCost, setAlarmCost] = useState(1)
  const [prevalence, setPrevalence] = useState(0.1)
  const [follow, setFollow] = useState(true)
  const [manual, setManual] = useState(1)
  const { y, scores } = useMemo(() => {
    const cut = [...order].sort((a, b) => a - b)[Math.floor((1 - prevalence) * n)] ?? 0
    const y = order.map((v) => (v >= cut ? 1 : 0))
    return { y, scores: noise.map((e, i) => (y[i] === 1 ? separation : 0) + e) }
  }, [prevalence])
  const costs = useMemo(
    () => [
      [0, alarmCost],
      [missCost, 0],
    ],
    [missCost, alarmCost],
  )
  const sweep = useMemo(() => thresholdCosts(y, scores, costs), [y, scores, costs])
  const cheapest = Number.isFinite(sweep.best.threshold) ? sweep.best.threshold : 6
  const threshold = follow ? cheapest : manual
  const table = useMemo(() => thresholdConfusion(y, scores, threshold), [y, scores, threshold])
  const setThreshold = (v: number): void => {
    setFollow(false)
    setManual(v)
  }
  const costProjection = useChartProjection()
  const scoresProjection = useChartProjection()
  // Fixed axes, so neither figure moves under the hand while the threshold is dragged.
  const costMax = useMemo(() => {
    const seen = sweep.thresholds.flatMap((t, i) => (t >= X[0] && t <= X[1] ? [sweep.totals[i] ?? 0] : []))
    return Math.max(1, ...seen) * 1.08
  }, [sweep])
  const countMax = useMemo(() => {
    const counts = [new Array<number>(30).fill(0), new Array<number>(30).fill(0)]
    scores.forEach((s, i) => {
      const bin = Math.min(29, Math.max(0, Math.floor(((s - X[0]) / (X[1] - X[0])) * 30)))
      const row = counts[y[i] ?? 0]
      if (row !== undefined) row[bin] = (row[bin] ?? 0) + 1
    })
    return Math.ceil((Math.max(...counts.flat()) * 1.15) / 5) * 5
  }, [y, scores])

  const costOption = useMemo(() => {
    const finite = sweep.thresholds
      .map((t, i) => [t, sweep.totals[i] ?? 0] as const)
      .filter(([t]) => Number.isFinite(t))
    return cartesian(
      [
        lines([{ x: finite.map(([t]) => t), y: finite.map(([, c]) => c), name: 'total cost', step: 'end' }], { mode }),
        ...(follow
          ? []
          : [
              referenceLines(
                { x: [{ value: cheapest, label: 'cheapest' }] },
                { colour: 'success', dash: 'dotted', mode },
              ),
            ]),
      ],
      { mode, figure: FIGURE, x: { label: 'threshold', limits: X }, y: { label: 'total cost', limits: [0, costMax] } },
    )
  }, [sweep, cheapest, follow, costMax, mode])
  const scoresOption = useMemo(() => {
    const range = X
    return cartesian(
      [
        histogram(
          scores.filter((_, i) => y[i] === 0),
          { bins: 30, range, name: 'negative', opacity: 0.6, mode },
        ),
        histogram(
          scores.filter((_, i) => y[i] === 1),
          { bins: 30, range, name: 'positive', opacity: 0.6, colourOffset: 1, mode },
        ),
      ],
      { mode, figure: FIGURE, x: { label: 'score', limits: X }, y: { label: 'count', limits: [0, countMax] } },
    )
  }, [scores, y, countMax, mode])
  const draggable = (projection: ReturnType<typeof useChartProjection>['projection'], top: number): ReactNode =>
    projection === null ? null : (
      <InteractionOverlay projection={projection} width={FIGURE.width} height={FIGURE.height}>
        <DraggableLine
          axis="x"
          value={threshold}
          onChange={setThreshold}
          span={[0, top]}
          min={-3}
          max={5}
          step={0.05}
          label={follow ? 'threshold (cheapest)' : 'threshold'}
        />
      </InteractionOverlay>
    )

  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={3}>
        <NumberField label="cost of a miss (FN)" value={missCost} onChange={setMissCost} min={0} max={100} step={1} />
        <NumberField
          label="cost of a false alarm (FP)"
          value={alarmCost}
          onChange={setAlarmCost}
          min={0}
          max={100}
          step={1}
        />
        <NumberField label="prevalence" value={prevalence} onChange={setPrevalence} min={0.02} max={0.5} step={0.02} />
        <Toggle label="use the cheapest threshold" value={follow} onChange={setFollow} />
        <NumberField label="threshold" value={threshold} onChange={setThreshold} min={-3} max={5} step={0.05} />
      </ControlPanel>
      <div className="grid gap-3 md:grid-cols-2">
        <FigureFrame figure={FIGURE} filename="costs-by-threshold">
          <Overlaid width={FIGURE.width} height={FIGURE.height} overlay={draggable(costProjection.projection, costMax)}>
            <Chart option={costOption} figure={FIGURE} onInit={costProjection.onInit} />
          </Overlaid>
        </FigureFrame>
        <FigureFrame figure={FIGURE} filename="costs-scores">
          <Overlaid
            width={FIGURE.width}
            height={FIGURE.height}
            overlay={draggable(scoresProjection.projection, countMax)}
          >
            <Chart option={scoresOption} figure={FIGURE} onInit={scoresProjection.onInit} />
          </Overlaid>
        </FigureFrame>
      </div>
      <ContingencyTable table={table} classNames={['negative', 'positive']} positive={1} costs={costs} />
    </div>
  )
}
