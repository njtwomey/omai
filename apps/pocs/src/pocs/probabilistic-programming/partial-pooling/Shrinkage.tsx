import { band, type Curve, cartesian, lines, type PointSet, points } from 'omai-render/charts'
import type { Figure } from 'omai-render/figure'
import { ResponsiveChart, useColourMode } from 'omai-render/react'
import { type ReactNode, useCallback, useMemo } from 'react'
import { classicalEstimates, type GroupData, type PoolingFrame } from './compute.ts'

/** Each group's interval as a vertical stroke at `x`, one curve with gaps between the strokes. */
function strokes(
  x: readonly number[],
  lower: readonly number[],
  upper: readonly number[],
): { x: number[]; y: number[] } {
  const out = { x: [] as number[], y: [] as number[] }
  x.forEach((v, j) => {
    if (!Number.isFinite(lower[j]) || !Number.isFinite(upper[j])) return
    out.x.push(v, v, Number.NaN)
    out.y.push(lower[j] as number, upper[j] as number, Number.NaN)
  })
  return out
}

/**
 * The shrinkage plot: every group's estimate with its 90% interval against the group's size (log scale), the group
 * alone (left of each size) and by the hierarchical posterior so far (right of it) joined by a line, the common mean as
 * a dashed line with its band, and the effect each group's mean was drawn around as a diamond. The axes stay put while the
 * run goes on, so the intervals visibly settle.
 *
 * @param props - The groups, the posterior's intervals at the playhead (none while it is warming up) and the form's name.
 * @returns The chart.
 */
export function Shrinkage(props: {
  readonly data: GroupData
  readonly frame: PoolingFrame | undefined
  readonly pooledLabel: string
}): ReactNode {
  const { data, frame, pooledLabel } = props
  const mode = useColourMode()
  const { alone, common } = useMemo(() => classicalEstimates(data), [data])
  const ranges = useMemo(() => {
    const lo = Math.min(...alone.map((a) => a.lower), ...data.truth)
    const hi = Math.max(...alone.map((a) => a.upper), ...data.truth)
    const pad = 0.05 * (hi - lo)
    return {
      x: [Math.min(...data.size) / 1.5, Math.max(...data.size) * 1.5] as [number, number],
      y: [lo - pad, hi + pad] as [number, number],
    }
  }, [alone, data])

  const layers = useMemo(() => {
    const left = data.size.map((n) => n / 1.06)
    const right = data.size.map((n) => n * 1.06)
    const [x0, x1] = ranges.x
    const noPool = strokes(
      left,
      alone.map((a) => a.lower),
      alone.map((a) => a.upper),
    )
    const curves: Curve[] = [
      {
        x: [x0, x1],
        y: [common.mean, common.mean],
        name: 'complete pooling',
        colour: 'tab7',
        dash: 'dashed',
        width: 1.5,
      },
      { ...noPool, name: 'no pooling', colour: 'tab0', width: 2 },
    ]
    const dots: PointSet[] = [
      { x: left, y: alone.map((a) => a.mean), name: 'no pooling', colour: 'tab0', size: 6 },
      {
        x: data.size,
        y: data.truth,
        name: 'generating effect',
        colour: 'tab2',
        symbol: 'diamond',
        size: 7,
        opacity: 0.8,
      },
    ]
    if (frame !== undefined) {
      const mean = right.flatMap((_, j) => (Number.isFinite(frame.mean[j]) ? [j] : []))
      const partial = strokes(right, frame.lower, frame.upper)
      const arrows = { x: [] as number[], y: [] as number[] }
      for (const j of mean) {
        arrows.x.push(left[j] as number, right[j] as number, Number.NaN)
        arrows.y.push(alone[j]?.mean ?? Number.NaN, frame.mean[j] as number, Number.NaN)
      }
      curves.push({ ...arrows, name: 'shrinkage', colour: 'tab7', width: 1, opacity: 0.8 })
      curves.push({ ...partial, name: 'partial pooling', colour: 'tab1', width: 2 })
      dots.push({
        x: mean.map((j) => right[j] as number),
        y: mean.map((j) => frame.mean[j] as number),
        name: 'partial pooling',
        colour: 'tab1',
        size: 6,
      })
    }
    return { curves, dots }
  }, [alone, common, data, frame, ranges])

  const build = useCallback(
    (figure: Figure) =>
      cartesian(
        [
          band(
            [
              {
                x: ranges.x,
                lower: [common.lower, common.lower],
                upper: [common.upper, common.upper],
                colour: 'tab7',
                opacity: 0.12,
              },
            ],
            { mode },
          ),
          lines(layers.curves, { mode }),
          points(layers.dots, { mode }),
        ],
        {
          mode,
          figure,
          legend: false,
          x: { label: 'group size n (log scale)', scale: 'log', limits: ranges.x },
          y: { label: 'group effect', limits: ranges.y },
        },
      ),
    [layers, ranges, common, mode],
  )
  return (
    <div className="flex flex-col gap-2">
      <ResponsiveChart aspect={1.35} narrowAspect={1.1} build={build} />
      <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted">
        <Key colour="--wb-tab0">no pooling, 90%</Key>
        <Key colour="--wb-tab1">{pooledLabel}</Key>
        <Key colour="--wb-tab7">common mean and its band</Key>
        <Key colour="--wb-tab2">generating effect</Key>
      </div>
    </div>
  )
}

function Key(props: { readonly colour: string; readonly children: ReactNode }): ReactNode {
  return (
    <span className="inline-flex items-center gap-1.5">
      <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: `var(${props.colour})` }} />
      {props.children}
    </span>
  )
}
