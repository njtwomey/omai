/**
 * The React view of a stability region: the filled region with its boundary and the marked eigenvalues, with a caption
 * that says in words whether the method is stable for every one of them.
 */

import type { StabilityRegion } from 'mdml-engine/exposition'
import { type ReactElement, useCallback } from 'react'
import type { Figure } from '../../figure/figure.ts'
import { stabilityRegionView, stabilityVerdict } from '../../views/stability-region.ts'
import { RecordFigure, type RecordViewProps, useViewFigure } from './_frame.tsx'

const DEFAULTS = { aspect: 1.3 } as const

/**
 * A region of absolute stability with the eigenvalues times the step size on it. The caption reads the verdict: stable
 * when every mark is inside the region, with the count outside otherwise.
 *
 * @param props - The record and the figure and frame settings.
 * @returns The figure.
 */
export function StabilityRegionView(props: RecordViewProps<StabilityRegion>): ReactElement {
  const { record } = props
  const figure = useViewFigure(props.figure, DEFAULTS)
  const build = useCallback((fig: Figure) => stabilityRegionView(record, { figure: fig }), [record])
  const v = stabilityVerdict(record)
  const total = v.stable + v.unstable + v.off
  const verdict =
    total === 0
      ? `${record.method}: no eigenvalues marked`
      : v.unstable + v.off === 0
        ? `${record.method} is stable: all ${total} values of hλ are inside the region`
        : `${record.method} is unstable: ${v.unstable + v.off} of ${total} values of hλ are outside the region`
  const caption = props.caption ?? (record.caption === undefined ? verdict : `${record.caption}: ${verdict}`)
  return (
    <RecordFigure {...props} figure={figure} build={build} caption={caption} filename={props.filename ?? 'stability'} />
  )
}
