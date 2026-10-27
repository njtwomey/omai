// Linked scrubbing on curve charts: a click on the plot area resolves to the nearest recorded step of the clicked x,
// and a thin vertical marker (a layer named with a `__` prefix, so out of the legend) shows the viewed step.
import { metricSeries, type Run } from 'mdml-engine/trace'
import { type Layer, lines } from 'mdml-render/charts'
import type { ChartProps } from 'mdml-render/react'
import { useCallback, useRef } from 'react'

type ECharts = Parameters<NonNullable<ChartProps['onInit']>>[0]

/** The distinct recorded steps of the named metrics, ascending. */
export function recordedSteps(run: Run, names: readonly string[]): number[] {
  return [...new Set(names.flatMap((n) => metricSeries(run, n).steps))].sort((a, b) => a - b)
}

/** The nearest recorded step to `x`, or `undefined` when nothing is recorded. */
export function nearestStep(steps: readonly number[], x: number): number | undefined {
  let best: number | undefined
  for (const s of steps) if (best === undefined || Math.abs(s - x) < Math.abs(best - x)) best = s
  return best
}

/** A vertical marker at `step` spanning the y range of `curves` (nothing when there is no range). */
export function stepMarker(curves: Layer, step: number, mode: 'light' | 'dark'): Layer[] {
  const y = curves.extent.y
  if (y === undefined || !Number.isFinite(step)) return []
  const [lo, hi] = y
  return [
    lines([{ x: [step, step], y: [lo, hi === lo ? lo + 1 : hi], name: '__step', colour: 'muted', width: 1 }], { mode }),
  ]
}

/**
 * The `onInit` of a curve chart that scrubs: a click anywhere on the plot area calls `onScrub` with the nearest
 * recorded step of the clicked x (`null` for the head). The latest steps, head and callback are read at click time.
 */
export function useScrubClick(
  steps: readonly number[],
  head: number,
  onScrub: ((t: number | null) => void) | undefined,
): (chart: ECharts) => void {
  const latest = useRef({ steps, head, onScrub })
  latest.current = { steps, head, onScrub }
  return useCallback((chart: ECharts): void => {
    chart.getZr().on('click', (e: { offsetX: number; offsetY: number }) => {
      const { steps: ss, head: h, onScrub: scrub } = latest.current
      if (scrub === undefined) return
      const pixel = [e.offsetX, e.offsetY]
      if (!chart.containPixel({ gridIndex: 0 }, pixel)) return
      const x = (chart.convertFromPixel({ gridIndex: 0 }, pixel) as number[])[0]
      if (x === undefined || !Number.isFinite(x)) return
      const step = nearestStep(ss, x)
      if (step === undefined) return
      scrub(step >= h ? null : step)
    })
  }, [])
}
