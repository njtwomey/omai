/**
 * Linked scrubbing on curve charts: a click on the plot area resolves to the nearest recorded step of the clicked x
 * and moves the shared step cursor, and a thin vertical marker (a layer named with a `__` prefix, so out of the
 * legend) shows the viewed step.
 */

import { metricSeries, type Run } from 'mdml-engine/trace'
import { useCallback, useRef } from 'react'
import { lines } from '../../charts/lines.ts'
import type { Layer } from '../../charts/plot.ts'
import type { ChartProps } from '../Chart.tsx'
import { useStepCursor } from './cursor.tsx'

type ECharts = Parameters<NonNullable<ChartProps['onInit']>>[0]

/**
 * The distinct recorded steps of the named metrics, ascending.
 *
 * @param run - The tape.
 * @param names - The metric paths.
 * @returns The steps, ascending and distinct.
 */
export function recordedSteps(run: Run, names: readonly string[]): number[] {
  return [...new Set(names.flatMap((n) => metricSeries(run, n).steps))].sort((a, b) => a - b)
}

/**
 * The recorded step nearest to `x`, or `undefined` when nothing is recorded. With `xs` (the x position of each step,
 * such as the time an ODE reached), the nearest is by those positions instead of by step number.
 *
 * @param steps - The recorded steps.
 * @param x - The position to find the nearest step to.
 * @param xs - The x position of each step (default the step numbers).
 * @returns The nearest step, or `undefined` when nothing is recorded.
 */
export function nearestStep(steps: readonly number[], x: number, xs?: readonly number[]): number | undefined {
  let best: number | undefined
  let bestDistance = Number.POSITIVE_INFINITY
  steps.forEach((s, i) => {
    const distance = Math.abs((xs?.[i] ?? s) - x)
    if (distance < bestDistance) {
      best = s
      bestDistance = distance
    }
  })
  return best
}

/**
 * A vertical marker at `step` spanning the y range of `curves` (nothing when there is no range).
 *
 * @param curves - The curves whose y range the marker spans.
 * @param step - The x position to mark.
 * @param mode - The colour mode.
 * @returns The marker layers (none when the curves have no range).
 */
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
 * recorded step of the clicked x (`null` for the head). With `xs`, the chart's x axis is those positions (one per
 * step) rather than the step itself. Without a head or callback, the enclosing step cursor's are used (a chart
 * inside `SessionPanels`). The latest steps, head and callback are read at click time.
 *
 * @param steps - The recorded steps the chart's x axis covers.
 * @param head - The head of the run (a step at or past it follows the head).
 * @param onScrub - The setter of the viewed step.
 * @param xs - The x position of each step, when the axis is not the step.
 * @returns The chart's `onInit`.
 */
export function useScrubClick(
  steps: readonly number[],
  head?: number,
  onScrub?: (t: number | null) => void,
  xs?: readonly number[],
): (chart: ECharts) => void {
  const cursor = useStepCursor()
  const latest = useRef({ steps, head: head ?? cursor?.head ?? 0, onScrub: onScrub ?? cursor?.scrub, xs })
  latest.current = { steps, head: head ?? cursor?.head ?? 0, onScrub: onScrub ?? cursor?.scrub, xs }
  return useCallback((chart: ECharts): void => {
    chart.getZr().on('click', (e: { offsetX: number; offsetY: number }) => {
      const { steps: ss, head: h, onScrub: scrub, xs: positions } = latest.current
      if (scrub === undefined) return
      const pixel = [e.offsetX, e.offsetY]
      if (!chart.containPixel({ gridIndex: 0 }, pixel)) return
      const x = (chart.convertFromPixel({ gridIndex: 0 }, pixel) as number[])[0]
      if (x === undefined || !Number.isFinite(x)) return
      const step = nearestStep(ss, x, positions)
      if (step === undefined) return
      scrub(step >= h ? null : step)
    })
  }, [])
}
