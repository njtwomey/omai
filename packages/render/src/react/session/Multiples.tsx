/**
 * Small multiples of a session layout: one panel spec repeated over a facet in a grid of cells with their own titles,
 * under the step cursor the enclosing panels share, curves on one shared vertical scale. The cell itself is drawn by
 * the host (`renderCell`), so every panel kind works in a cell.
 */

import type { Run } from 'mdml-engine/trace'
import { type ReactNode, useMemo } from 'react'
import type { Cell, MultiplesSpec } from './multiples.ts'
import { facetCells, multiplesColumns, sharedLimits } from './multiples.ts'
import type { LiveVariant } from './useSession.ts'

/**
 * The grid of cells of a small-multiples panel.
 *
 * @param props - The panel, the variants of the comparison, and how to draw one cell given the shared limits.
 * @returns The grid.
 */
export function Multiples(props: {
  readonly spec: MultiplesSpec
  // biome-ignore lint/suspicious/noExplicitAny: the host is generic over methods
  readonly variants: readonly LiveVariant<any, any>[]
  readonly renderCell: (cell: Cell, limits: [number, number] | undefined) => ReactNode
}): ReactNode {
  const { spec, variants } = props
  const labelKey = variants.map((v) => v.label).join('\u0000')
  // biome-ignore lint/correctness/useExhaustiveDependencies: the labels are keyed by their joined text
  const cells = useMemo(() => facetCells(spec, labelKey.split('\u0000')), [spec, labelKey])
  const version = variants.reduce((v, x) => Math.max(v, x.live.version), 0)
  const first = variants[0]?.live.session
  // The tapes re-read as they grow, so the shared scale follows the run.
  // biome-ignore lint/correctness/useExhaustiveDependencies: the tapes change with the version
  const runs = useMemo(
    (): readonly Run[] => variants.map((v) => v.live.session.run()),
    [first, version, variants.length],
  )
  const paths = spec.panel.kind === 'curves' ? spec.panel.paths : []
  const limits = spec.shared === false || paths.length === 0 ? undefined : sharedLimits(runs, paths)
  const columns = multiplesColumns(cells.length, spec.columns)
  return (
    <div className="flex flex-col gap-2" data-testid="multiples">
      {spec.title === undefined ? null : <div className="text-sm font-semibold text-(--wb-fg)">{spec.title}</div>}
      {cells.length === 0 ? (
        <div className="text-xs text-muted">Nothing to repeat: the facet does not apply to this panel.</div>
      ) : null}
      <div className="grid gap-3" style={{ gridTemplateColumns: `repeat(${columns}, minmax(0, max-content))` }}>
        {cells.map((cell) => (
          <div key={cell.label} className="min-w-0">
            {props.renderCell(cell, limits)}
          </div>
        ))}
      </div>
    </div>
  )
}
