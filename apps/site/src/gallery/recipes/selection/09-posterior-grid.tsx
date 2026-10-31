import { posteriorGrid } from 'mdml-engine/exposition'
import { inv } from 'mdml-engine/numerics/linalg'
import { tensor, toArray } from 'mdml-engine/tensor'
import { ControlPanel, ExpositionView, NumberField, SeedInput } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'
import { normals } from '../../toys/synthetic.ts'

export const recipe: Recipe = {
  title: 'Posterior over a grid',
  question: 'How do I show what a model believes about a function, including where it is unsure?',
  explain:
    'A `posteriorGrid` record gives a mean and a variance at each input of a grid, the observed points, and optionally sampled functions. `PosteriorView` draws the mean with a band of two standard deviations, the observations and any samples drawn through it; the band pinches at data and opens between and beyond it. The posterior here is a Bayesian polynomial regression solved in closed form (`inv` from the linear algebra library): few points or a high noise and the band is wide, and a polynomial of too low degree is confident and wrong.',
  shows: ['posteriorGrid', 'PosteriorView'],
}

export default function PosteriorGridRecipe(): ReactNode {
  const [points, setPoints] = useState(8)
  const [noise, setNoise] = useState(0.2)
  const [degree, setDegree] = useState(5)
  const [seed, setSeed] = useState(1)
  const record = useMemo(() => {
    const truth = (v: number): number => Math.sin(1.5 * v) + 0.3 * v
    const e = normals(seed, points, noise)
    const xs = Array.from({ length: points }, (_, i) => -2.5 + (3.5 * (i + 0.5 + 0.3 * Math.sin(5 * i))) / points)
    const ys = xs.map((v, i) => truth(v) + (e[i] ?? 0))
    const phi = (v: number): number[] => Array.from({ length: degree + 1 }, (_, k) => (v / 3) ** k)
    const beta = 1 / noise ** 2
    const alpha = 1
    const design = xs.map(phi)
    const gram = Array.from({ length: degree + 1 }, (_, i) =>
      Array.from(
        { length: degree + 1 },
        (_, j) => (i === j ? alpha : 0) + beta * design.reduce((s, r) => s + (r[i] ?? 0) * (r[j] ?? 0), 0),
      ),
    )
    const cov = toArray(inv(tensor(gram))) as number[][]
    const mean = Array.from(
      { length: degree + 1 },
      (_, i) =>
        beta * cov[i]!.reduce((s, c, j) => s + c * design.reduce((a, r, n) => a + (r[j] ?? 0) * (ys[n] ?? 0), 0), 0),
    )
    const grid = Array.from({ length: 120 }, (_, i) => -3 + (6 * i) / 119)
    const mu = grid.map((v) => phi(v).reduce((s, p, k) => s + p * (mean[k] ?? 0), 0))
    const variance = grid.map((v) => {
      const p = phi(v)
      return 1 / beta + p.reduce((s, pi, i) => s + pi * p.reduce((t, pj, j) => t + (cov[i]?.[j] ?? 0) * pj, 0), 0)
    })
    return posteriorGrid({
      inputs: grid.map((v) => [v]),
      mean: mu,
      variance,
      observed: { inputs: xs.map((v) => [v]), targets: ys },
      labels: { input: 'x', output: 'y' },
    })
  }, [points, noise, degree, seed])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={4}>
        <NumberField label="observations" value={points} onChange={setPoints} min={2} max={40} integer />
        <NumberField label="noise" value={noise} onChange={setNoise} min={0.05} max={1} step={0.05} />
        <NumberField label="polynomial degree" value={degree} onChange={setDegree} min={1} max={9} integer />
        <SeedInput value={seed} onChange={setSeed} />
      </ControlPanel>
      <ExpositionView record={record} title="What the model believes between the points" />
    </div>
  )
}
