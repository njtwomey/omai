/**
 * The React view of explanation-quality records: a panel per metric and the explainers' agreement.
 */

import { type ReactElement, useCallback, useMemo } from 'react'
import type { ExplanationQuality } from 'whitebox-engine/exposition'
import type { Figure } from '../../figure/figure.ts'
import { explanationQualityView, qualityLayout } from '../../views/explanation-quality.ts'
import { RecordFigure, type RecordViewProps, useViewFigure } from './_frame.tsx'

/**
 * Explainers compared on quality metrics, a panel per metric, and their agreement as a matrix.
 *
 * @param props - The record and the figure settings.
 * @returns The figure.
 */
export function ExplanationQualityView(props: RecordViewProps<ExplanationQuality>): ReactElement {
  const { record } = props
  const defaults = useMemo(
    () => ({ width: 720, height: Math.max(240, 40 + qualityLayout(record).rows * 230) }),
    [record],
  )
  const figure = useViewFigure(props.figure, defaults)
  const build = useCallback((fig: Figure) => explanationQualityView(record, { figure: fig }), [record])
  return <RecordFigure {...props} figure={figure} build={build} filename={props.filename ?? 'explanation-quality'} />
}
