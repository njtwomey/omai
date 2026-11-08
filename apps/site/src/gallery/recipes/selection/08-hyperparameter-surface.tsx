import { hyperparameterSurface } from 'omai-engine/exposition'
import { ControlPanel, ExpositionView, NumberField, SeedInput } from 'omai-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'
import { normals } from '../../toys/synthetic.ts'

export const recipe: Recipe = {
  title: 'Hyperparameter surface',
  question: 'How do I show how a score depends on two hyperparameters, and where a search went?',
  explain:
    'A `hyperparameterSurface` record is a score over a grid of two parameters (here on log scales), with the points a search visited in order and the direction that is better. `HyperparameterSurfaceView` draws the surface as a heatmap with contours, the visited points joined by a path and the best marked. The surface is a made-up validation loss with a curved valley (ridge strength against kernel bandwidth); the search is coordinate descent from a corner, which zig-zags along a curved valley instead of following it. Roughen the surface to give it false minima.',
  shows: ['hyperparameterSurface', 'HyperparameterSurfaceView'],
}

export default function HyperparameterSurfaceRecipe(): ReactNode {
  const [bumps, setBumps] = useState(0.02)
  const [curve, setCurve] = useState(1)
  const [seed, setSeed] = useState(1)
  const record = useMemo(() => {
    const m = 40
    const lo = -4
    const hi = 2
    const axis = Array.from({ length: m }, (_, i) => lo + ((hi - lo) * i) / (m - 1))
    const z = normals(seed, 12 * 12)
    const bump = (a: number, b: number): number => {
      const ia = Math.min(11, Math.max(0, Math.round(((a - lo) / (hi - lo)) * 11)))
      const ib = Math.min(11, Math.max(0, Math.round(((b - lo) / (hi - lo)) * 11)))
      return bumps * (z[ia * 12 + ib] ?? 0)
    }
    const loss = (a: number, b: number): number =>
      0.2 + 0.08 * (b + 1 - curve * 0.6 * (a + 1)) ** 2 + 0.03 * (a + 1) ** 2 + bump(a, b)
    const values = axis.map((b) => axis.map((a) => loss(a, b)))
    // Coordinate descent: move along one axis to its best value, then the other, from a corner.
    let [a, b] = [lo + 0.3, hi - 0.3]
    const path: [number, number][] = [[a, b]]
    for (let k = 0; k < 8; k++) {
      a = axis.reduce((best, v) => (loss(v, b) < loss(best, b) ? v : best), a)
      path.push([a, b])
      b = axis.reduce((best, v) => (loss(a, v) < loss(a, best) ? v : best), b)
      path.push([a, b])
    }
    return hyperparameterSurface({
      parameters: ['ridge strength (log10)', 'bandwidth (log10)'],
      x: axis,
      y: axis,
      values,
      path,
      objective: 'validation loss',
      higherIsBetter: false,
    })
  }, [bumps, curve, seed])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={3}>
        <NumberField label="roughness" value={bumps} onChange={setBumps} min={0} max={0.3} step={0.01} />
        <NumberField label="curvature of the valley" value={curve} onChange={setCurve} min={0} max={2} step={0.1} />
        <SeedInput value={seed} onChange={setSeed} />
      </ControlPanel>
      <ExpositionView record={record} title="Validation loss over two hyperparameters" />
    </div>
  )
}
