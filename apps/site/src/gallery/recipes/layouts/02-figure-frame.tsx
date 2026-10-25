import { cartesian, fn, referenceLines } from 'mdml-render/charts'
import { figure } from 'mdml-render/figure'
import { Chart, Choice, ControlPanel, FigureFrame, useColourMode } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Figure frame',
  question: 'How do I give a figure a fixed size, a full-screen view and an export menu?',
  explain:
    'A `figure` has a width and box aspect per medium (page, blog, slide), as a matplotlib figure does; content never resizes it. `FigureFrame` holds it at that size, its expand button opens the same chart in a modal at the same aspect, and the export menu saves PNG or SVG. `fn` breaks curves at asymptotes.',
}

const fig = figure({ medium: 'page', aspect: '2:1' })
const functions = {
  'tan x': { f: Math.tan, asymptotes: [-1.5, -0.5, 0.5, 1.5].map((k) => k * Math.PI) },
  '1 / x': { f: (x: number) => 1 / x, asymptotes: [0] },
  'sec x': { f: (x: number) => 1 / Math.cos(x), asymptotes: [-1.5, -0.5, 0.5, 1.5].map((k) => k * Math.PI) },
  'x / (x² − 4)': { f: (x: number) => x / (x * x - 4), asymptotes: [-2, 2] },
} as const
type Name = keyof typeof functions

export default function FrameRecipe(): ReactNode {
  const mode = useColourMode()
  const [name, setName] = useState<Name>('tan x')
  const option = useMemo(() => {
    const { f, asymptotes } = functions[name]
    return cartesian(
      [
        fn(f, [-2 * Math.PI, 2 * Math.PI], { name, clip: [-6, 6], mode, n: 800 }),
        referenceLines({ x: asymptotes }, { mode }),
      ],
      { mode, figure: fig, y: { limits: [-6, 6] } },
    )
  }, [mode, name])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={1}>
        <Choice label="function" value={name} options={Object.keys(functions) as Name[]} onChange={setName} />
      </ControlPanel>
      <FigureFrame
        figure={fig}
        title={`${name}, with gaps at its asymptotes`}
        caption="Expand for a full-screen view."
        filename="function"
      >
        <Chart option={option} />
      </FigureFrame>
    </div>
  )
}
