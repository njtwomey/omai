import { type ReactNode, useState } from 'react'
import { Choice, ControlPanel, Select } from 'whitebox-render/react'
import { Line, Plot, plotFrame, sampleCurve } from '../../demo/Plot.tsx'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Select and choice',
  question: 'How do I let the reader pick from a list, and step through it without opening it?',
  explain:
    '`Select` keeps its options in a menu, with previous and next arrows either side that cycle through them in place, wrapping at the ends; good for datasets and models. `Choice` shows two to five short options side by side; `cycle` gives it the arrows too. Options are values or `{ value, label }`.',
}

const curves = {
  sin: { label: 'sine', f: Math.sin },
  cos: { label: 'cosine', f: Math.cos },
  tanh: { label: 'hyperbolic tangent', f: Math.tanh },
  sinc: { label: 'sinc', f: (x: number) => (x === 0 ? 1 : Math.sin(x) / x) },
  gauss: { label: 'Gaussian bump', f: (x: number) => Math.exp((-x * x) / 2) },
} as const
type CurveId = keyof typeof curves

const frame = plotFrame({ x: [-6, 6], y: [-3, 3] })

export default function SelectAndChoice(): ReactNode {
  const [curve, setCurve] = useState<CurveId>('sin')
  const [scale, setScale] = useState(1)
  const f = curves[curve].f
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <Select
          label="curve"
          value={curve}
          onChange={setCurve}
          options={Object.entries(curves).map(([value, c]) => ({ value: value as CurveId, label: c.label }))}
        />
        <Choice label="amplitude" value={scale} onChange={setScale} options={[0.5, 1, 2, 3]} cycle />
      </ControlPanel>
      <Plot frame={frame} xLabel="x" yLabel="y">
        <Line frame={frame} points={sampleCurve((x) => scale * f(x), -6, 6, 300)} />
      </Plot>
    </div>
  )
}
