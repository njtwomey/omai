import { cartesian, implicit } from 'omai-render/charts'
import { figure } from 'omai-render/figure'
import { Chart, Choice, ControlPanel, FigureFrame, NumberField, useColourMode } from 'omai-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Implicit curves',
  question: 'How do I draw the curve where f(x, y) = c, without solving for y?',
  explain:
    "`implicit(f, xRange, yRange, { level })` traces the level set f(x, y) = level by marching squares on a grid. Move the level and the curve deforms: the lemniscate splits into two ovals and merges again, the folium's loop opens.",
}

const FIGURE = figure({ width: 520, height: 520 })
type Name = 'lemniscate' | 'folium' | 'heart' | 'cassini ovals'
const CURVES: Record<Name, (x: number, y: number) => number> = {
  lemniscate: (x, y) => (x * x + y * y) ** 2 - 2 * (x * x - y * y),
  folium: (x, y) => x ** 3 + y ** 3 - 3 * x * y,
  heart: (x, y) => (x * x + y * y - 1) ** 3 - x * x * y ** 3,
  'cassini ovals': (x, y) => ((x - 1) ** 2 + y * y) * ((x + 1) ** 2 + y * y),
}

export default function ImplicitCurves(): ReactNode {
  const mode = useColourMode()
  const [name, setName] = useState<Name>('lemniscate')
  const [level, setLevel] = useState(0)
  const option = useMemo(
    () =>
      cartesian(
        [implicit(CURVES[name], [-2.5, 2.5], [-2.5, 2.5], { mode, n: 241, level, name: `${name} = ${level}` })],
        {
          mode,
          aspect: 'equal',
          x: { label: 'x', limits: [-2.5, 2.5] },
          y: { label: 'y', limits: [-2.5, 2.5] },
        },
      ),
    [name, level, mode],
  )
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <Choice label="curve" value={name} onChange={setName} options={Object.keys(CURVES) as Name[]} />
        <NumberField label="level" value={level} onChange={setLevel} min={-1} max={3} step={0.05} />
      </ControlPanel>
      <FigureFrame figure={FIGURE} filename="implicit-curves">
        <Chart option={option} figure={FIGURE} />
      </FigureFrame>
    </div>
  )
}
