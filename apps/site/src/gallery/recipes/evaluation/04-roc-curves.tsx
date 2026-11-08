import { normal } from 'omai-engine/dists'
import { rocAucScore, rocCurve } from 'omai-engine/metrics'
import { key, uniform } from 'omai-engine/random'
import { toArray } from 'omai-engine/tensor'
import { rocChart } from 'omai-render/charts'
import { figure } from 'omai-render/figure'
import { Chart, ControlPanel, FigureFrame, NumberField, SeedInput, useColourMode } from 'omai-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'ROC curves',
  question: 'How do I compare classifiers by their ROC curves?',
  explain:
    '`rocCurve(y, scores)` gives the true and false positive rates at every threshold and `rocChart` draws them with the chance diagonal; name each curve with its area (`rocAucScore`). The curves here come from two toy scorers whose positives sit further from the negatives; the share of positives changes nothing, because both rates are conditioned on the true class.',
}

const FIGURE = figure({ width: 520, height: 520 })

// Toy scores for a binary task: negatives ~ N(0, 1), positives ~ N(separation, 1), positives a share of the n rows.
function toyScores(n: number, share: number, separation: number, seed: number): { y: number[]; s: number[] } {
  const z = toArray(normal(0, 1).sample(key(seed), [n])) as number[]
  const u = toArray(uniform(key(seed + 1000), [n])) as number[]
  const y = u.map((v) => (v < share ? 1 : 0))
  return { y, s: z.map((v, i) => v + (y[i] === 1 ? separation : 0)) }
}

export default function RocCurves(): ReactNode {
  const mode = useColourMode()
  const [weak, setWeak] = useState(0.8)
  const [strong, setStrong] = useState(2)
  const [share, setShare] = useState(0.3)
  const [seed, setSeed] = useState(0)
  const option = useMemo(() => {
    const curves = [
      { name: 'weak', d: weak, seed },
      { name: 'strong', d: strong, seed: seed + 1 },
    ].map(({ name, d, seed: s }) => {
      const { y, s: scores } = toyScores(600, share, d, s)
      const area = rocAucScore(y, scores).estimate.value
      return { name: `${name} (AUC ${area.toFixed(3)})`, curve: rocCurve(y, scores) }
    })
    return rocChart(curves, { mode })
  }, [weak, strong, share, seed, mode])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <NumberField label="weak separation" value={weak} onChange={setWeak} min={0} max={4} step={0.1} />
        <NumberField label="strong separation" value={strong} onChange={setStrong} min={0} max={4} step={0.1} />
        <NumberField label="share of positives" value={share} onChange={setShare} min={0.02} max={0.98} step={0.02} />
        <SeedInput value={seed} onChange={setSeed} />
      </ControlPanel>
      <FigureFrame figure={FIGURE} filename="roc-curves">
        <Chart option={option} figure={FIGURE} />
      </FigureFrame>
    </div>
  )
}
