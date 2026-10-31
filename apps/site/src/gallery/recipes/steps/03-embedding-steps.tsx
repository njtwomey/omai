import { embeddingSteps } from 'mdml-engine/exposition'
import { ControlPanel, ExpositionView, NumberField, SeedInput } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'
import { gaussianRows, normals } from '../../toys/synthetic.ts'

export const recipe: Recipe = {
  title: 'Embedding steps',
  question: 'How do I show a low-dimensional map taking shape as an embedding is optimised?',
  explain:
    'An `embeddingSteps` record holds the position of every row at a series of iterations, the objective, and optionally the neighbour graph the method works from and the colours of the rows. `EmbeddingPlayer` plays the points moving, coloured by group, with the objective falling beside them. Here three clusters in ten dimensions are laid out in the plane by gradient descent on the stress between the distances in the plane and those in ten dimensions, starting from a random scatter; a small step size is slow and a large one overshoots and wobbles.',
  shows: ['embeddingSteps', 'EmbeddingPlayer'],
}

export default function EmbeddingStepsRecipe(): ReactNode {
  const [rate, setRate] = useState(0.03)
  const [dims, setDims] = useState(10)
  const [seed, setSeed] = useState(1)
  const record = useMemo(() => {
    const per = 15
    const k = 3
    const n = per * k
    const centres = gaussianRows(seed + 3, k, dims).map((r) => r.map((v) => v * 3))
    const noise = gaussianRows(seed, n, dims)
    const x = noise.map((r, i) => r.map((v, j) => v * 0.7 + (centres[Math.floor(i / per)]?.[j] ?? 0)))
    const dist = x.map((a) => x.map((b) => Math.hypot(...a.map((v, j) => v - (b[j] ?? 0)))))
    const scale = (1 / Math.max(...dist.flat())) * 8
    const init = normals(seed + 11, 2 * n, 1)
    let y = Array.from({ length: n }, (_, i) => [init[2 * i] ?? 0, init[2 * i + 1] ?? 0])
    const positions: number[][][] = []
    const iterations: number[] = []
    const objective: number[] = []
    for (let t = 0; t <= 80; t++) {
      let stress = 0
      const grad = y.map(() => [0, 0])
      for (let i = 0; i < n; i++)
        for (let j = 0; j < n; j++) {
          if (i === j) continue
          const dx = (y[i]?.[0] ?? 0) - (y[j]?.[0] ?? 0)
          const dy = (y[i]?.[1] ?? 0) - (y[j]?.[1] ?? 0)
          const dd = Math.hypot(dx, dy) || 1e-9
          const target = (dist[i]?.[j] ?? 0) * scale
          stress += (dd - target) ** 2
          const g = (dd - target) / dd
          const gi = grad[i] as number[]
          gi[0] = (gi[0] ?? 0) + g * dx
          gi[1] = (gi[1] ?? 0) + g * dy
        }
      if (t % 2 === 0) {
        positions.push(y.map((p) => [...p]))
        iterations.push(t)
        objective.push(stress / 2)
      }
      y = y.map((p, i) => [(p[0] ?? 0) - rate * (grad[i]?.[0] ?? 0), (p[1] ?? 0) - rate * (grad[i]?.[1] ?? 0)])
    }
    return embeddingSteps({
      method: 'stress minimisation',
      positions,
      iterations,
      objective,
      colourBy: {
        kind: 'classes',
        labels: x.map((_, i) => Math.floor(i / per)),
        classes: ['group A', 'group B', 'group C'],
      },
    })
  }, [rate, dims, seed])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={3}>
        <NumberField label="step size" value={rate} onChange={setRate} min={0.002} max={0.08} step={0.002} log />
        <NumberField label="input dimensions" value={dims} onChange={setDims} min={2} max={30} integer />
        <SeedInput value={seed} onChange={setSeed} />
      </ControlPanel>
      <ExpositionView record={record} title="A map unfolding" />
    </div>
  )
}
