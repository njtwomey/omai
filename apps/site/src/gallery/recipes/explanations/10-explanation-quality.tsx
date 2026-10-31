import { explanationQuality } from 'mdml-engine/exposition'
import { ControlPanel, ExpositionView, NumberField, SeedInput } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'
import { normals } from '../../toys/synthetic.ts'

export const recipe: Recipe = {
  title: 'Explanation quality',
  question: 'How do I compare explainers on faithfulness, robustness and the rest, and see where they agree?',
  explain:
    'An `explanationQuality` record scores each explainer on named metrics in categories (faithfulness, robustness, localisation, complexity, randomisation), says for each whether higher or lower is better, and may carry a matrix of how much the explainers agree. `ExplanationQualityView` draws one small bar chart per metric with each explainer’s spread, the title giving the direction and the winner, and the agreement matrix after them. The scores here are made up from each explainer’s assumed strengths plus noise, so move the noise to see which rankings survive it.',
  shows: ['explanationQuality', 'ExplanationQualityView'],
}

const EXPLAINERS = ['SHAP', 'LIME', 'Saliency', 'IntGrad', 'Random']
// Strength of each explainer on each metric, in [0, 1] where 1 is best.
const METRICS = [
  { name: 'deletion AUC', category: 'faithfulness', better: 'lower', strengths: [0.9, 0.7, 0.5, 0.8, 0.1] },
  { name: 'insertion AUC', category: 'faithfulness', better: 'higher', strengths: [0.85, 0.7, 0.45, 0.8, 0.1] },
  { name: 'max sensitivity', category: 'robustness', better: 'lower', strengths: [0.7, 0.4, 0.3, 0.6, 0.5] },
  { name: 'mass accuracy', category: 'localisation', better: 'higher', strengths: [0.6, 0.5, 0.7, 0.75, 0.15] },
  { name: 'sparseness', category: 'complexity', better: 'higher', strengths: [0.4, 0.8, 0.3, 0.45, 0.2] },
  { name: 'randomisation', category: 'randomisation', better: 'lower', strengths: [0.8, 0.7, 0.3, 0.75, 0.9] },
] as const

export default function ExplanationQualityRecipe(): ReactNode {
  const [noise, setNoise] = useState(0.08)
  const [seed, setSeed] = useState(5)
  const record = useMemo(() => {
    const eps = normals(seed, METRICS.length * EXPLAINERS.length, noise)
    const metrics = METRICS.map((m, i) => ({
      name: m.name,
      category: m.category,
      better: m.better,
      values: m.strengths.map((s, e) => {
        const v = Math.min(1, Math.max(0, s + (eps[i * EXPLAINERS.length + e] ?? 0)))
        return m.better === 'higher' ? v : 1 - v
      }),
      spread: m.strengths.map(() => noise * 0.8),
    }))
    const agreement = EXPLAINERS.map((_, a) =>
      EXPLAINERS.map((_, b) => {
        if (a === b) return 1
        const close = 1 - Math.abs((METRICS[0].strengths[a] ?? 0) - (METRICS[0].strengths[b] ?? 0))
        return Math.max(-1, Math.min(1, close * 0.9 - 0.2 + 0.1 * (eps[a + b] ?? 0)))
      }),
    )
    return explanationQuality({
      explainers: EXPLAINERS,
      metrics,
      agreement: { metric: 'rank correlation', matrix: agreement },
      nRows: 100,
    })
  }, [noise, seed])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <NumberField label="noise" value={noise} onChange={setNoise} min={0} max={0.4} step={0.02} />
        <SeedInput value={seed} onChange={setSeed} />
      </ControlPanel>
      <ExpositionView record={record} title="Five explainers, six metrics" />
    </div>
  )
}
