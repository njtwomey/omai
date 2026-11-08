import { normal } from 'omai-engine/dists'
import { averagePrecisionScore, precisionRecallCurve, prGainAuc, prGainCurve } from 'omai-engine/metrics'
import { key, uniform } from 'omai-engine/random'
import { toArray } from 'omai-engine/tensor'
import { precisionRecallChart, prGainChart } from 'omai-render/charts'
import { figure } from 'omai-render/figure'
import { Chart, ControlPanel, FigureFrame, NumberField, SeedInput, useColourMode } from 'omai-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Precision-recall and PR-gain curves',
  question: 'How do I draw precision-recall curves, and their precision-recall-gain form?',
  explain:
    "`precisionRecallChart` draws precision against recall with the no-skill line at the prevalence; `prGainChart` draws Flach and Kull's precision and recall gains, which put the always-positive classifier at the origin, so areas compare across prevalences and the F-gain isometrics are straight. Lower the share of positives: the precision-recall curve sinks with it, the gain curve barely moves.",
}

const FIGURE = figure({ width: 440, height: 440 })

// Toy scores for a binary task: negatives ~ N(0, 1), positives ~ N(separation, 1), positives a share of the n rows.
function toyScores(n: number, share: number, separation: number, seed: number): { y: number[]; s: number[] } {
  const z = toArray(normal(0, 1).sample(key(seed), [n])) as number[]
  const u = toArray(uniform(key(seed + 1000), [n])) as number[]
  const y = u.map((v) => (v < share ? 1 : 0))
  return { y, s: z.map((v, i) => v + (y[i] === 1 ? separation : 0)) }
}

export default function PrecisionRecallGain(): ReactNode {
  const mode = useColourMode()
  const [separation, setSeparation] = useState(1.5)
  const [share, setShare] = useState(0.2)
  const [seed, setSeed] = useState(0)
  const { pr, prg } = useMemo(() => {
    const { y, s } = toyScores(1500, share, separation, seed)
    const ap = averagePrecisionScore(y, s).estimate.value
    const auprg = prGainAuc(y, s).estimate.value
    return {
      pr: precisionRecallChart([{ name: `AP ${ap.toFixed(3)}`, curve: precisionRecallCurve(y, s) }], {
        mode,
        prevalence: share,
        title: `precision-recall (average precision ${ap.toFixed(3)})`,
      }),
      prg: prGainChart([{ name: `AUPRG ${auprg.toFixed(3)}`, curve: prGainCurve(y, s) }], {
        mode,
        title: `precision-recall gain (area ${auprg.toFixed(3)})`,
      }),
    }
  }, [separation, share, seed, mode])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={3}>
        <NumberField label="separation" value={separation} onChange={setSeparation} min={0} max={4} step={0.1} />
        <NumberField label="share of positives" value={share} onChange={setShare} min={0.02} max={0.9} step={0.02} />
        <SeedInput value={seed} onChange={setSeed} />
      </ControlPanel>
      <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
        <FigureFrame figure={FIGURE} filename="precision-recall">
          <Chart option={pr} figure={FIGURE} />
        </FigureFrame>
        <FigureFrame figure={FIGURE} filename="precision-recall-gain">
          <Chart option={prg} figure={FIGURE} />
        </FigureFrame>
      </div>
    </div>
  )
}
