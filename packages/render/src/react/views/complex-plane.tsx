/**
 * The React view of a complex plane: poles as crosses, zeros as circles, the locus and the stability boundary on equal
 * axes.
 */

import type { ComplexPlane } from 'omai-engine/exposition'
import { isStablePole } from 'omai-engine/exposition'
import { type ReactElement, useCallback } from 'react'
import type { Figure } from '../../figure/figure.ts'
import { complexPlaneView } from '../../views/complex-plane.ts'
import { RecordFigure, type RecordViewProps, useViewFigure } from './_frame.tsx'

const DEFAULTS = { aspect: 1.3 } as const

/**
 * A complex plane with its poles and zeros. The caption says in words whether the system is stable (every pole in the
 * stable region), so the verdict does not rest on colour.
 *
 * @param props - The record and the figure and frame settings.
 * @returns The figure.
 */
export function ComplexPlaneView(props: RecordViewProps<ComplexPlane>): ReactElement {
  const { record } = props
  const figure = useViewFigure(props.figure, DEFAULTS)
  const build = useCallback((fig: Figure) => complexPlaneView(record, { figure: fig }), [record])
  const unstable = record.poles.filter((p) => !isStablePole(record.domain, p)).length
  const boundary = record.domain === 'z' ? 'the unit circle' : 'the imaginary axis'
  const where = unstable === 0 ? 'every pole is inside' : `${unstable} of ${record.poles.length} poles are outside`
  const verdict = `${unstable === 0 ? 'stable' : 'unstable'}: ${where} the dashed line, ${boundary}`
  const caption = props.caption ?? (record.caption === undefined ? verdict : `${record.caption}: ${verdict}`)
  return (
    <RecordFigure
      {...props}
      figure={figure}
      build={build}
      caption={caption}
      filename={props.filename ?? 'complex-plane'}
    />
  )
}
