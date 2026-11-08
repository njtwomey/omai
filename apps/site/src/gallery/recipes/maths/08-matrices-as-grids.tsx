import { matrix, type Panel, subplots } from 'omai-render/charts'
import { figure, gridSpec } from 'omai-render/figure'
import { Chart, ControlPanel, FigureFrame, NumberField, useColourMode } from 'omai-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Matrices as grids',
  question: 'How do I draw a matrix product, A times B, as grids of numbers?',
  explain:
    'Each matrix is a `matrix` layer (cells coloured on a diverging ramp centred on zero, values written in) and `subplots` sets the three side by side: entry (i, j) of AB is row i of A against column j of B. Turn the rotation and watch the product rotate the columns of B.',
}

const FIGURE = figure({ width: 860, height: 320 })

export default function MatricesAsGrids(): ReactNode {
  const mode = useColourMode()
  const [angle, setAngle] = useState(30)
  const option = useMemo(() => {
    const a = (angle * Math.PI) / 180
    const A = [
      [Math.cos(a), -Math.sin(a)],
      [Math.sin(a), Math.cos(a)],
    ]
    const B = [
      [1, 0, 2],
      [0, 1, 1],
    ]
    const AB = A.map((row) => B[0]?.map((_, j) => row.reduce((s, v, k) => s + v * (B[k]?.[j] ?? 0), 0)) ?? [])
    const cell = (z: number[][], title: string): Panel => ({
      layers: [matrix(z, { mode, centre: 0, range: [-2.5, 2.5], format: (v) => v.toFixed(2), colourBar: false })],
      options: { title },
    })
    const grid = gridSpec(1, 3, { widthRatios: [2, 3, 3], wspace: 40 })
    return subplots(
      grid,
      { r0c0: cell(A, 'A (rotation)'), r0c1: cell(B, 'B'), r0c2: cell(AB, 'AB') },
      { figure: { ...FIGURE, mode }, legend: false },
    )
  }, [angle, mode])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <NumberField label="rotation (degrees)" value={angle} onChange={setAngle} min={-180} max={180} integer />
      </ControlPanel>
      <FigureFrame figure={FIGURE} filename="matrices-as-grids">
        <Chart option={option} figure={FIGURE} />
      </FigureFrame>
    </div>
  )
}
