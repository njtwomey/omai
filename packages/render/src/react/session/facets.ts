/**
 * The pure parts of small multiples: which cells a facet makes from a panel spec, the vertical limits curves share
 * across them, and the number of columns to lay them in.
 */

import { metricSeries, type Run } from 'omai-engine/trace'
import type { Panel } from './layout.ts'

/** The small-multiples panel of a layout. */
export type MultiplesSpec = Extract<Panel, { kind: 'multiples' }>

/** One cell: a title, the panel it draws and, for a variant facet, the variant it draws it for. */
export interface Cell {
  readonly label: string
  readonly panel: MultiplesSpec['panel']
  /** The index of the variant of a comparison this cell shows alone (`undefined`: all of them). */
  readonly variant?: number
}

/**
 * The cells a facet makes of a panel spec: one per variant, per path of a curves panel, or per value of a probe
 * argument. Each cell's panel is the spec with its title set to the cell's label, its paths cut to the one path, or its
 * arguments extended by the value.
 *
 * @param spec - The small-multiples panel.
 * @param variants - The labels of the comparison's variants (one empty label for a session on its own).
 * @returns The cells, in order; none when the facet does not apply to the panel (paths of a panel without paths).
 * @example
 * ```ts
 * import { facetCells } from 'omai-render/react'
 * const cells = facetCells({ kind: 'multiples', panel: { kind: 'curves', paths: ['a', 'b'] }, over: { by: 'paths' } }, [''])
 * cells.map((c) => c.label) // => ['a', 'b']
 * ```
 */
export function facetCells(spec: MultiplesSpec, variants: readonly string[]): Cell[] {
  const { panel, over } = spec
  if (over.by === 'variants')
    return variants.map((label, variant) => ({
      label: label || 'run',
      panel: { ...panel, title: label || 'run' },
      variant,
    }))
  if (over.by === 'paths')
    return panel.kind === 'curves'
      ? panel.paths.map((path) => ({ label: path, panel: { ...panel, paths: [path], title: path } }))
      : []
  if (!('args' in panel) && !('path' in panel)) return []
  return over.values.map((value, i) => {
    const label = over.labels?.[i] ?? `${over.key} = ${typeof value === 'string' ? value : JSON.stringify(value)}`
    const base =
      'args' in panel && typeof panel.args === 'object' && panel.args !== null && !Array.isArray(panel.args)
        ? panel.args
        : {}
    return { label, panel: { ...panel, args: { ...base, [over.key]: value }, title: label } as MultiplesSpec['panel'] }
  })
}

/**
 * The vertical limits shared by the curves of several cells: the range of every named scalar path on every run, padded
 * by a twentieth of its span so that curves do not touch the frame.
 *
 * @param runs - The runs of the variants.
 * @param paths - The scalar paths drawn.
 * @returns `[low, high]`, or `undefined` when nothing is recorded.
 * @example
 * ```ts
 * import { sharedLimits } from 'omai-render/react'
 * sharedLimits([], ['loss']) // => undefined
 * ```
 */
export function sharedLimits(runs: readonly Run[], paths: readonly string[]): [number, number] | undefined {
  let lo = Number.POSITIVE_INFINITY
  let hi = Number.NEGATIVE_INFINITY
  for (const run of runs)
    for (const path of paths)
      for (const v of metricSeries(run, path).values)
        if (Number.isFinite(v)) {
          lo = Math.min(lo, v)
          hi = Math.max(hi, v)
        }
  if (!(lo <= hi)) return undefined
  const pad = hi > lo ? (hi - lo) / 20 : Math.max(1e-9, Math.abs(lo) / 20)
  return [lo - pad, hi + pad]
}

/**
 * How many columns to lay `n` cells in: the requested number (default 3), no more than there are cells.
 *
 * @param n - The number of cells.
 * @param requested - The columns asked for.
 * @returns At least one.
 * @example
 * ```ts
 * import { multiplesColumns } from 'omai-render/react'
 * multiplesColumns(2) // => 2
 * multiplesColumns(7, 4) // => 4
 * ```
 */
export function multiplesColumns(n: number, requested?: number): number {
  return Math.max(1, Math.min(n, Math.floor(requested ?? 3)))
}
