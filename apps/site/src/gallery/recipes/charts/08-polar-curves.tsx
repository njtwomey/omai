import { cartesian, polar } from 'omai-render/charts'
import { figure } from 'omai-render/figure'
import { Chart, Choice, ControlPanel, FigureFrame, NumberField, useColourMode } from 'omai-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Polar curves',
  question: 'How do I plot a curve given in polar form, r as a function of θ?',
  explain:
    '`polar(r, [θ0, θ1])` traces r(θ) as a curve in the plane; draw it at equal aspect. A rose r = cos(kθ) has k petals for odd k and 2k for even k; a fractional k needs more turns to close (set the turns up and watch it fill in).',
}

const FIGURE = figure({ width: 520, height: 520 })
type Curve = 'rose' | 'cardioid' | 'spiral' | 'limaçon'
const CURVES: Record<Curve, (k: number) => (t: number) => number> = {
  rose: (k) => (t) => Math.cos(k * t),
  cardioid: () => (t) => 1 - Math.cos(t),
  spiral: (k) => (t) => (k * t) / (2 * Math.PI) / 6,
  limaçon: (k) => (t) => 0.5 + k * Math.cos(t),
}

export default function PolarCurves(): ReactNode {
  const mode = useColourMode()
  const [curve, setCurve] = useState<Curve>('rose')
  const [k, setK] = useState(5)
  const [turns, setTurns] = useState(1)
  const option = useMemo(
    () =>
      cartesian([polar(CURVES[curve](k), [0, 2 * Math.PI * turns], { mode, n: 2000 * turns, name: curve })], {
        mode,
        aspect: 'equal',
        x: { label: 'x' },
        y: { label: 'y' },
      }),
    [curve, k, turns, mode],
  )
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={3}>
        <Choice label="curve" value={curve} onChange={setCurve} options={Object.keys(CURVES) as Curve[]} />
        <NumberField label="k" value={k} onChange={setK} min={0.25} max={12} step={0.25} />
        <NumberField label="turns" value={turns} onChange={setTurns} min={1} max={12} integer />
      </ControlPanel>
      <FigureFrame figure={FIGURE} filename="polar-curves">
        <Chart option={option} figure={FIGURE} />
      </FigureFrame>
    </div>
  )
}
