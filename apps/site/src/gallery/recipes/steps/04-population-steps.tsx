import { populationSteps } from 'mdml-engine/exposition'
import { ControlPanel, ExpositionView, NumberField, SeedInput } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'
import { normals } from '../../toys/synthetic.ts'

export const recipe: Recipe = {
  title: 'Population steps',
  question: 'How do I show an evolutionary search: a population sampled, selected and moved over generations?',
  explain:
    'A `populationSteps` record has, per generation, each member’s fitness and position, the mean and covariance the search is sampling from, and the best member; with the surface the search runs on. `PopulationPlayer` plays the generations as points on the surface with the sampling ellipse and the best-so-far curve beside it. The search is an evolution strategy on a bumpy bowl: it samples around a mean, keeps the best fifth and moves the mean to them, shrinking its steps as it closes in. A small population or a big first step gets caught in a side valley.',
  shows: ['populationSteps', 'PopulationPlayer'],
}

const f = (x: number, y: number): number =>
  (x - 1) ** 2 + (y + 0.5) ** 2 + 1.2 * (2 - Math.cos(3 * x) - Math.cos(3 * y))

export default function PopulationStepsRecipe(): ReactNode {
  const [size, setSize] = useState(24)
  const [sigma, setSigma] = useState(1.2)
  const [seed, setSeed] = useState(1)
  const record = useMemo(() => {
    const noise = normals(seed, 2 * size * 30)
    let mean: [number, number] = [-2.5, 2.2]
    let s = sigma
    const generations = Array.from({ length: 30 }, (_, g) => {
      const pts = Array.from({ length: size }, (_, i): [number, number] => [
        mean[0] + s * (noise[2 * (g * size + i)] ?? 0),
        mean[1] + s * (noise[2 * (g * size + i) + 1] ?? 0),
      ])
      const fitness = pts.map(([x, y]) => f(x, y))
      const order = fitness.map((_, i) => i).sort((a, b) => (fitness[a] ?? 0) - (fitness[b] ?? 0))
      const elite = order.slice(0, Math.max(2, Math.floor(size / 5)))
      const best = elite[0] ?? 0
      const out = {
        fitness,
        points: pts,
        mean: [...mean] as [number, number],
        covariance: [s * s, 0, s * s] as [number, number, number],
        best: { fitness: fitness[best] ?? 0, point: pts[best] as [number, number] },
      }
      mean = [
        elite.reduce((a, i) => a + (pts[i]?.[0] ?? 0), 0) / elite.length,
        elite.reduce((a, i) => a + (pts[i]?.[1] ?? 0), 0) / elite.length,
      ]
      s *= 0.88
      return out
    })
    const axis = Array.from({ length: 60 }, (_, i) => -3.5 + (7 * i) / 59)
    return populationSteps({
      method: 'evolution strategy',
      lowerIsBetter: true,
      surface: {
        x: axis,
        y: axis,
        values: axis.map((y) => axis.map((x) => f(x, y))),
        names: ['x', 'y'],
        optimum: [1, -0.5],
      },
      generations,
    })
  }, [size, sigma, seed])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={3}>
        <NumberField label="population" value={size} onChange={setSize} min={5} max={100} integer />
        <NumberField label="first step size" value={sigma} onChange={setSigma} min={0.2} max={3} step={0.1} />
        <SeedInput value={seed} onChange={setSeed} />
      </ControlPanel>
      <ExpositionView record={record} title="A population climbing a bumpy bowl" />
    </div>
  )
}
