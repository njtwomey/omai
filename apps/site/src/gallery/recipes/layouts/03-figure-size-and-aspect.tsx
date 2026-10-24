import { type ReactNode, useMemo, useState } from 'react'
import { cartesian, lines } from 'whitebox-render/charts'
import { figure, type Medium } from 'whitebox-render/figure'
import { Chart, Choice, ControlPanel, FigureFrame, useColourMode } from 'whitebox-render/react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Figure size and box aspect',
  question: 'How do I set a figure’s size and shape?',
  explain:
    "A `figure()` has a default width and box aspect per medium (`page` 640 at 16:10, `blog`, `slide` 960 at 16:9), as a matplotlib figure does. Content never resizes it: only `width`, `aspect` (a number or `'2:1'`) or `height` do.",
}

const aspects = ['default', '2:1', '1:1', '16:9'] as const
const x = Array.from({ length: 200 }, (_, i) => (i / 199) * 4 * Math.PI)

export default function FigureSize(): ReactNode {
  const mode = useColourMode()
  const [medium, setMedium] = useState<Medium>('page')
  const [aspect, setAspect] = useState<(typeof aspects)[number]>('default')
  const fig = useMemo(
    () =>
      figure({
        medium,
        mode,
        ...(aspect === 'default' ? {} : { aspect }),
        ...(medium === 'slide' ? { width: 760 } : {}),
      }),
    [medium, aspect, mode],
  )
  const option = useMemo(
    () => cartesian([lines([{ x, y: x.map((v) => Math.sin(v) * Math.exp(-v / 8)) }], { mode })], { mode, figure: fig }),
    [fig, mode],
  )
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <Choice
          label="medium"
          value={medium}
          options={['page', 'blog', 'slide', 'thumbnail'] as const}
          onChange={setMedium}
        />
        <Choice label="aspect" value={aspect} options={aspects} onChange={setAspect} />
      </ControlPanel>
      <FigureFrame figure={fig} caption={`${fig.width} × ${fig.height} px, box aspect ${fig.aspect.toFixed(3)}`}>
        <Chart option={option} />
      </FigureFrame>
    </div>
  )
}
