import { type ReactNode, useMemo } from 'react'
import { normal } from 'whitebox-engine/dists'
import { toFlat } from 'whitebox-engine/tensor'
import { cartesian, density, histogram, kde, rug } from 'whitebox-render/charts'
import { Chart, FigureFrame, useColourMode } from 'whitebox-render/react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Histogram, KDE and density',
  question: 'How do I compare a sample with a density?',
  explain:
    '`histogram(values, { bins, density: true })` bins with the engine’s `stats.histogram` (counts, densities, rules such as `fd` or explicit edges); `kde` draws a Gaussian kernel density estimate, `density(dist)` any engine `Distribution`, and `rug` marks each observation.',
}

// A deterministic sample: normal quantiles at evenly spaced probabilities.
const sample = toFlat(normal(0, 1).ppf(Array.from({ length: 200 }, (_, i) => (i + 0.5) / 200)))

export default function HistogramDensity(): ReactNode {
  const mode = useColourMode()
  const option = useMemo(
    () =>
      cartesian(
        [
          histogram(sample, { bins: 'fd', density: true, mode, name: 'histogram' }),
          kde(sample, { mode, name: 'KDE', colourOffset: 1 }),
          density(normal(0, 1), { mode, name: 'N(0, 1)', colour: 'fg', dash: 'dashed' }),
          rug(sample, { mode }),
        ],
        { mode, x: { label: 'x' }, y: { label: 'density' } },
      ),
    [mode],
  )
  return (
    <FigureFrame filename="histogram">
      <Chart option={option} />
    </FigureFrame>
  )
}
