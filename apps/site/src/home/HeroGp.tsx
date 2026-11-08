import { gpRegression } from 'omai-engine/gp'
import { constantKernel, productKernel, rbfKernel } from 'omai-engine/kernels'
import type { Point2 } from 'omai-engine/numerics/geometry'
import { key } from 'omai-engine/random'
import { tensor, toFlat } from 'omai-engine/tensor'
import { band, cartesian, lines, points } from 'omai-render/charts'
import { figure } from 'omai-render/figure'
import {
  Chart,
  DraggablePoints,
  InteractionOverlay,
  Overlaid,
  useChartProjection,
  useColourMode,
} from 'omai-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import { useWidth } from './live.ts'
import { Slider } from './Slider.tsx'

const GRID = Array.from({ length: 121 }, (_, i) => -5 + (10 * i) / 120)
const START: Point2[] = [
  [-3.6, -0.8],
  [-2, 1.1],
  [-0.6, 0.4],
  [1.2, -1.2],
  [3, 0.9],
]

/** The exact GP posterior on the grid for the points: its mean, standard deviation and three draws. */
function posterior(pts: readonly Point2[], lengthScale: number, noiseSd: number) {
  const fit = gpRegression
    .with({
      kernel: productKernel(constantKernel({ value: 1 }), rbfKernel({ lengthScale })),
      alpha: noiseSd ** 2,
      optimise: false,
    })
    .fit({ x: tensor(pts.map((p) => [p[0]])), y: tensor(pts.map((p) => p[1])) })
  const at = tensor(GRID.map((v) => [v]))
  const { mean, std } = fit.predictWithStd(at)
  const draws = toFlat(fit.sample(at, 3, key(7)))
  return {
    mean: Array.from(toFlat(mean)),
    std: Array.from(toFlat(std)),
    draws: [0, 1, 2].map((k) => Array.from(draws.slice(k * GRID.length, (k + 1) * GRID.length))),
  }
}

/** The front page's live figure: a Gaussian process refitted on every move of a data point. */
export function HeroGp(): ReactNode {
  const mode = useColourMode()
  const [box, width] = useWidth<HTMLDivElement>()
  const [pts, setPts] = useState<Point2[]>(START)
  const [lengthScale, setLengthScale] = useState(1.4)
  const [noise, setNoise] = useState(0.3)
  const { projection, onInit } = useChartProjection()
  const size = useMemo(
    () => figure({ width: Math.max(width, 280), height: Math.round(Math.max(width, 280) * 0.78) }),
    [width],
  )
  const option = useMemo(() => {
    const { mean, std, draws } = posterior(pts, lengthScale, noise)
    return cartesian(
      [
        band(
          [
            {
              x: GRID,
              lower: mean.map((m, i) => m - 2 * (std[i] ?? 0)),
              upper: mean.map((m, i) => m + 2 * (std[i] ?? 0)),
              name: '± 2 sd',
              opacity: 0.18,
            },
          ],
          { mode },
        ),
        lines(
          [
            { x: GRID, y: mean, name: 'mean', width: 2.5 },
            ...draws.map((d, k) => ({ x: GRID, y: d, name: `draw ${k + 1}`, width: 1 })),
          ],
          { mode },
        ),
        points([{ x: pts.map((p) => p[0]), y: pts.map((p) => p[1]), name: 'data', size: 1, opacity: 0 }], { mode }),
      ],
      {
        mode,
        figure: size,
        x: { label: 'x', limits: [-5, 5] },
        y: { label: 'f(x)', limits: [-3, 3] },
        legend: false,
        tooltip: false,
      },
    )
  }, [pts, lengthScale, noise, mode, size])
  return (
    <div ref={box} className="space-y-3">
      {width === 0 ? null : (
        <Overlaid
          width={size.width}
          height={size.height}
          overlay={
            projection === null ? null : (
              <InteractionOverlay projection={projection} width={size.width} height={size.height}>
                <DraggablePoints points={pts} onChange={setPts} />
              </InteractionOverlay>
            )
          }
        >
          <Chart option={option} figure={size} onInit={onInit} />
        </Overlaid>
      )}
      <div className="space-y-1 px-1">
        <Slider
          label="length scale"
          value={lengthScale}
          min={0.3}
          max={3}
          step={0.01}
          onChange={setLengthScale}
          format={(v) => v.toFixed(2)}
        />
        <Slider
          label="noise sd"
          value={noise}
          min={0.01}
          max={0.5}
          step={0.01}
          onChange={setNoise}
          format={(v) => v.toFixed(2)}
        />
      </div>
      <p className="px-1 text-xs text-muted">
        Drag any point: the mean, the ± 2 sd band and three posterior draws follow.
      </p>
    </div>
  )
}
