import { type ReactNode, useMemo } from 'react'
import { iris } from 'whitebox-engine/datasets'
import { toArray } from 'whitebox-engine/tensor'
import { cartesian, points } from 'whitebox-render/charts'
import { brushSelection, figure, linkStore } from 'whitebox-render/figure'
import {
  Brush,
  Chart,
  InteractionOverlay,
  NearestHover,
  Overlaid,
  type Point2,
  useChartProjection,
  useColourMode,
  useLinkState,
} from 'whitebox-render/react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Linked figures',
  question: 'How do I link figures, so brushing or hovering in one shows up in the other?',
  explain:
    'A `linkStore` holds what linked figures share (hovered item, selection, brush); `useLinkState` reads it in React. Here the iris flowers are drawn twice, by their sepals and by their petals: brush a region of the sepals and the same flowers light up among the petals; hover a petal point (`NearestHover`) and its flower is ringed in both.',
}

const FIGURE = figure({ width: 420, aspect: 1.15 })
const data = iris()
const rows = toArray(data.x) as number[][]
const sepals: Point2[] = rows.map((r) => [r[0] ?? 0, r[1] ?? 0])
const petals: Point2[] = rows.map((r) => [r[2] ?? 0, r[3] ?? 0])
const SEPAL_AXES = {
  x: { label: 'sepal length', limits: [4, 8] as const },
  y: { label: 'sepal width', limits: [1.8, 4.6] as const },
}
const PETAL_AXES = {
  x: { label: 'petal length', limits: [0.5, 7.5] as const },
  y: { label: 'petal width', limits: [0, 2.8] as const },
}

function view(
  xy: readonly Point2[],
  selected: ReadonlySet<number> | null,
  hover: number | undefined,
  mode: 'light' | 'dark',
  axes: typeof SEPAL_AXES | typeof PETAL_AXES,
): ReturnType<typeof cartesian> {
  const pick = (keep: (i: number) => boolean): number[][] => xy.flatMap((p, i) => (keep(i) ? [[...p]] : []))
  const inside = pick((i) => selected === null || selected.has(i))
  const outside = pick((i) => selected !== null && !selected.has(i))
  const ring = hover === undefined ? undefined : xy[Number(hover)]
  return cartesian(
    [
      points(
        [
          ...(outside.length > 0 ? [{ x: outside, name: 'not selected', colour: 'muted', opacity: 0.5 }] : []),
          ...(inside.length > 0 ? [{ x: inside, name: 'selected' }] : []),
          ...(ring === undefined ? [] : [{ x: [[...ring]], name: 'hovered', colour: 'fg', size: 16, opacity: 0.35 }]),
        ],
        { mode },
      ),
    ],
    { mode, figure: FIGURE, legend: false, tooltip: false, ...axes },
  )
}

export default function LinkedFigures(): ReactNode {
  const mode = useColourMode()
  const link = useMemo(() => linkStore(), [])
  const { brush, hover } = useLinkState(link)
  const left = useChartProjection()
  const right = useChartProjection()
  const rect = brush?.x !== undefined && brush.y !== undefined ? { x: brush.x, y: brush.y } : null
  const selected = useMemo(() => (brush === undefined ? null : new Set(brushSelection(sepals, brush))), [brush])
  const hovered = typeof hover === 'number' ? hover : undefined
  const sepalOption = useMemo(() => view(sepals, selected, hovered, mode, SEPAL_AXES), [selected, hovered, mode])
  const petalOption = useMemo(() => view(petals, selected, hovered, mode, PETAL_AXES), [selected, hovered, mode])
  const size = { width: FIGURE.width, height: FIGURE.height }
  return (
    <div className="flex flex-wrap gap-4">
      <div className="flex flex-col gap-1">
        <span className="text-xs text-muted">
          Sepals: drag to brush ({selected?.size ?? rows.length} of {rows.length})
        </span>
        <Overlaid
          {...size}
          overlay={
            left.projection === null ? null : (
              <InteractionOverlay projection={left.projection} {...size}>
                <Brush
                  value={rect}
                  onChange={(r) => link.set(({ brush: _, ...rest }) => (r === null ? rest : { ...rest, brush: r }))}
                />
              </InteractionOverlay>
            )
          }
        >
          <Chart option={sepalOption} figure={FIGURE} onInit={left.onInit} />
        </Overlaid>
      </div>
      <div className="flex flex-col gap-1">
        <span className="text-xs text-muted">Petals: hover a flower</span>
        <Overlaid
          {...size}
          overlay={
            right.projection === null ? null : (
              <InteractionOverlay projection={right.projection} {...size}>
                <NearestHover
                  points={petals}
                  onHover={(i) => {
                    if ((link.get().hover ?? -1) !== i)
                      link.set(({ hover: _, ...rest }) => (i < 0 ? rest : { ...rest, hover: i }))
                  }}
                />
              </InteractionOverlay>
            )
          }
        >
          <Chart option={petalOption} figure={FIGURE} onInit={right.onInit} />
        </Overlaid>
      </div>
    </div>
  )
}
