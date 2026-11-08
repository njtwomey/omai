/** Small helpers for figures that draw a field on a grid with paths or points over it. */
import { contours, heatmap, type Layer } from 'omai-render/charts'
import { sequentialRamp } from 'omai-render/theme'

/** The window a map covers. */
export interface Window {
  readonly x: readonly [number, number]
  readonly y: readonly [number, number]
}

/** `n` evenly spaced values from `lo` to `hi`, the end points included. */
export const linspace = (lo: number, hi: number, n: number): number[] =>
  Array.from({ length: n }, (_, i) => lo + ((hi - lo) * i) / (n - 1))

/** The values of a function on a grid, rows are y and columns x. */
export function sampleField(f: (x: number, y: number) => number, window: Window, n: number): number[][] {
  const xs = linspace(window.x[0], window.x[1], n)
  return linspace(window.y[0], window.y[1], n).map((y) => xs.map((x) => f(x, y)))
}

/** The layers of a field on one pale ramp with its own contours, to draw lines or points over. */
export function fieldLayers(
  window: Window,
  field: readonly (readonly number[])[],
  label: string,
  mode: 'light' | 'dark',
  levels = 10,
): Layer[] {
  const flat = field.flat()
  const n = field.length
  return [
    heatmap(field, {
      x: window.x,
      y: window.y,
      ramp: sequentialRamp('blues', mode).slice(0, 5),
      range: [Math.min(...flat), Math.max(...flat)],
      resolution: 96,
      label,
      colourBar: false,
    }),
    contours(field, {
      x: linspace(window.x[0], window.x[1], n),
      y: linspace(window.y[0], window.y[1], n),
      levels,
      colour: 'fg',
      width: 1,
      opacity: 0.35,
      labels: false,
    }),
  ]
}
