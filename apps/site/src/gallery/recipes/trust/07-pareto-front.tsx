import { paretoFront } from 'omai-engine/exposition'
import { paretoFrontier, selectFairModel } from 'omai-engine/fairness'
import { ControlPanel, ExpositionView, NumberField, SeedInput } from 'omai-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'
import { normals } from '../../toys/synthetic.ts'

export const recipe: Recipe = {
  title: 'Pareto front',
  question: 'How do I show the price of fairness: the best accuracy reachable at each level of unfairness?',
  explain:
    'A `paretoFront` record holds families of candidate models, each with a violation (a fairness gap, to minimise) and an accuracy (to maximise), the indices on each family’s frontier, and optionally the model a rule chose. `ParetoFrontView` draws every candidate faintly, each family’s frontier as a line and the chosen model marked. The candidates are made-up trade-offs for two ways of fixing a model; raise the steepness and fairness costs more accuracy. The chosen model is the most accurate within the violation you allow.',
  shows: ['paretoFront', 'ParetoFrontView'],
}

export default function ParetoFrontRecipe(): ReactNode {
  const [steepness, setSteepness] = useState(1)
  const [allowed, setAllowed] = useState(0.05)
  const [seed, setSeed] = useState(1)
  const record = useMemo(() => {
    const n = 60
    const noise = normals(seed, 2 * n, 0.012)
    const family = (name: string, offset: number, bend: number, from: number) => {
      const violation = Array.from({ length: n }, (_, i) =>
        Math.min(0.4, Math.max(0, ((i + 0.5) / n) * 0.4 + (noise[from + i] ?? 0) * 0.5)),
      )
      const accuracy = violation.map(
        (v, i) => 0.9 - offset - steepness * bend * (0.4 - v) ** 2 - Math.abs(noise[from + i] ?? 0),
      )
      return {
        name,
        violation,
        accuracy,
        points: violation.map((v, i) => ({ violation: v, accuracy: accuracy[i] ?? 0 })),
      }
    }
    const raw = [family('shift thresholds', 0, 0.55, 0), family('reweigh the data', 0.015, 0.4, n)]
    const families = raw.map(({ name, violation, accuracy, points }) => ({
      name,
      violation,
      accuracy,
      front: paretoFrontier(points),
    }))
    const chosen = raw.map((f) => selectFairModel(f.points, { maxViolation: allowed }))
    const best = chosen.reduce(
      (b, i, f) => ((raw[f]?.points[i]?.accuracy ?? 0) > (raw[b]?.points[chosen[b] ?? 0]?.accuracy ?? 0) ? f : b),
      0,
    )
    return paretoFront({
      violationName: 'demographic parity gap',
      families,
      chosen: { family: best, index: chosen[best] ?? 0, rule: `most accurate with a gap of at most ${allowed}` },
    })
  }, [steepness, allowed, seed])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={3}>
        <NumberField
          label="steepness of the trade"
          value={steepness}
          onChange={setSteepness}
          min={0.2}
          max={3}
          step={0.1}
        />
        <NumberField label="gap allowed" value={allowed} onChange={setAllowed} min={0.01} max={0.3} step={0.01} />
        <SeedInput value={seed} onChange={setSeed} />
      </ControlPanel>
      <ExpositionView record={record} title="Accuracy against unfairness" />
    </div>
  )
}
