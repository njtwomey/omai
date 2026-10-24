import { type ReactNode, useMemo } from 'react'
import { cartesian, fn, referenceLines } from 'whitebox-render/charts'
import { figure } from 'whitebox-render/figure'
import { Chart, FigureFrame, useColourMode } from 'whitebox-render/react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Figure frame',
  question: 'How do I give a figure a fixed size, a full-screen view and an export menu?',
  explain:
    'A `figure` has a width and box aspect per medium (page, blog, slide), as a matplotlib figure does; content never resizes it. `FigureFrame` holds it at that size, its expand button opens the same chart in a modal at the same aspect, and the export menu saves PNG or SVG.',
}

const fig = figure({ medium: 'page', aspect: '2:1' })

export default function FrameRecipe(): ReactNode {
  const mode = useColourMode()
  const option = useMemo(
    () =>
      cartesian(
        [
          fn(Math.tan, [-2 * Math.PI, 2 * Math.PI], { name: 'tan x', clip: [-6, 6], mode }),
          referenceLines({ x: [-1.5 * Math.PI, -Math.PI / 2, Math.PI / 2, 1.5 * Math.PI] }, { mode }),
        ],
        { mode, figure: fig, y: { limits: [-6, 6] } },
      ),
    [mode],
  )
  return (
    <FigureFrame
      figure={fig}
      title="tan x, with gaps at its asymptotes"
      caption="Press the expand button for a full-screen view."
      filename="tan"
    >
      <Chart option={option} />
    </FigureFrame>
  )
}
