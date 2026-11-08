import { decisionSurface, marginRecord } from 'omai-engine/exposition'
import { ControlPanel, ExpositionView, NumberField, SeedInput } from 'omai-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'
import { gaussianRows } from '../../toys/synthetic.ts'

export const recipe: Recipe = {
  title: 'Margins',
  question: 'How do I show how far each row is from the boundary, and which rows hold it in place?',
  explain:
    'A `margin` record gives each row’s signed margin (positive: on the right side), the support rows with their dual coefficients, and the intercept; `MarginView` draws the rows with the boundary and its two margin lines as contours, support rows ringed, and a histogram of the margins with the boundary (0) and the margin (1) marked. The boundary here is set by hand, with its angle and offset as the knobs, not fitted: move it and the rows that lie on or inside the margin become the support, and a poor boundary has negative margins. Widen the classes’ overlap and no boundary gets every row.',
  shows: ['margin', 'MarginView'],
}

export default function MarginsRecipe(): ReactNode {
  const [angle, setAngle] = useState(0.3)
  const [offset, setOffset] = useState(0)
  const [overlap, setOverlap] = useState(0.7)
  const [seed, setSeed] = useState(1)
  const record = useMemo(() => {
    const noise = gaussianRows(seed, 60, 2)
    const labels = noise.map((_, i) => (i % 2 === 0 ? 1 : -1))
    const points = noise.map((r, i) => [(labels[i] ?? 1) * 1.4 + overlap * (r[0] ?? 0), overlap * 1.6 * (r[1] ?? 0)])
    const w = [Math.cos(angle), Math.sin(angle)]
    const margins = points.map(
      (p, i) => (labels[i] ?? 1) * ((p[0] ?? 0) * (w[0] ?? 0) + (p[1] ?? 0) * (w[1] ?? 0) + offset),
    )
    const support = margins.flatMap((m, i) => (m <= 1 ? [i] : []))
    const axis = Array.from({ length: 40 }, (_, i) => -4 + (8 * i) / 39)
    const surface = decisionSurface({
      features: ['x', 'y'],
      x: axis,
      y: axis,
      decision: axis.map((y) => axis.map((x) => x * (w[0] ?? 0) + y * (w[1] ?? 0) + offset)),
      levels: [-1, 0, 1],
    })
    return marginRecord({
      surface,
      support,
      dualCoefficients: support.map((i) => (labels[i] ?? 1) * Math.min(1, 1 - (margins[i] ?? 1) + 0.1)),
      intercept: offset,
      margins,
      rows: {
        points,
        features: ['x', 'y'],
        labels: labels.map((l) => (l > 0 ? 1 : 0)),
        classes: ['negative', 'positive'],
      },
    })
  }, [angle, offset, overlap, seed])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={4}>
        <NumberField label="boundary angle" value={angle} onChange={setAngle} min={-1.6} max={1.6} step={0.1} />
        <NumberField label="offset" value={offset} onChange={setOffset} min={-2} max={2} step={0.1} />
        <NumberField label="overlap" value={overlap} onChange={setOverlap} min={0.2} max={1.5} step={0.1} />
        <SeedInput value={seed} onChange={setSeed} />
      </ControlPanel>
      <ExpositionView record={record} title="Rows against a hand-set boundary" />
    </div>
  )
}
