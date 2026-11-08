/**
 * The React view of an equation against its truth: the fitted expression beside the true one typeset with KaTeX (each
 * coefficient coloured by its error), a table of the coefficients with their status in words, the fit against the
 * truth and the residual strip beneath.
 */

import { coefficientErrors, type EquationFit } from 'omai-engine/exposition'
import { type ReactElement, useMemo } from 'react'
import {
  equationFitView,
  fittedTex,
  residualStripView,
  STATUS_WORDS,
  termCounts,
  truthTex,
} from '../../views/equation-fit.ts'
import { Chart } from '../Chart.tsx'
import { FigureFrame } from '../FigureFrame.tsx'
import { Tex } from '../MathView.tsx'
import { useColourMode } from '../theme.tsx'
import { type RecordViewProps, useViewFigure } from './_frame.tsx'

const DEFAULTS = { width: 440, height: 480 } as const
const CURVE = { width: 440, height: 210 } as const
const STRIP = { width: 440, height: 90 } as const

/**
 * A fitted equation against the truth. The expressions are typeset with the coefficients coloured by how far they are
 * from the truth; beneath are the table of coefficients (the status in words), the fit against the truth and the
 * residual strip.
 *
 * @param props - The record and the figure and frame settings.
 * @returns The figure.
 */
export function EquationFitView(props: RecordViewProps<EquationFit>): ReactElement {
  const { record } = props
  const figure = useViewFigure(props.figure, DEFAULTS)
  const mode = useColourMode()
  const fitted = useMemo(() => fittedTex(record, mode), [record, mode])
  const truth = useMemo(() => truthTex(record), [record])
  const errors = useMemo(() => coefficientErrors(record), [record])
  const counts = termCounts(record)
  const curve = useMemo(
    () => equationFitView(record, { mode, figure: { ...figure, ...CURVE } }),
    [record, mode, figure],
  )
  const strip = useMemo(
    () => residualStripView(record, { mode, figure: { ...figure, ...STRIP } }),
    [record, mode, figure],
  )
  const summary =
    record.terms === undefined || errors.every((e) => e.status === 'unknown')
      ? ''
      : `${counts.right} of ${record.terms.length} terms match the truth${counts.extra + counts.missing > 0 ? `, ${counts.extra} extra, ${counts.missing} missing` : ''}`
  return (
    <FigureFrame
      figure={figure}
      filename={props.filename ?? 'equation'}
      {...(props.title === undefined ? {} : { title: props.title })}
      caption={props.caption ?? [record.caption, summary].filter((x) => x !== undefined && x !== '').join(': ')}
    >
      <div className="flex h-full flex-col gap-1 overflow-y-auto">
        <div className="flex flex-col gap-0.5 text-sm" data-testid="equations">
          <div>
            <span className="mr-2 text-xs text-muted">fit</span>
            <Tex>{fitted}</Tex>
          </div>
          {truth === undefined ? null : (
            <div>
              <span className="mr-2 text-xs text-muted">truth</span>
              <Tex>{truth}</Tex>
            </div>
          )}
        </div>
        {record.terms === undefined || errors.every((e) => e.status === 'unknown') ? null : (
          <table className="w-full text-xs tabular-nums">
            <thead>
              <tr className="text-left text-muted">
                <th className="font-normal">term</th>
                <th className="text-right font-normal">fit</th>
                <th className="text-right font-normal">truth</th>
                <th className="pl-2 font-normal">status</th>
              </tr>
            </thead>
            <tbody>
              {record.terms.map((t, k) => (
                <tr key={t.tex}>
                  <td>
                    <Tex>{t.tex}</Tex>
                  </td>
                  <td className="text-right">{t.estimate.toPrecision(3)}</td>
                  <td className="text-right">{t.truth === undefined ? '' : t.truth.toPrecision(3)}</td>
                  <td className="pl-2">{STATUS_WORDS[errors[k]?.status ?? 'unknown']}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        <Chart option={curve} width={CURVE.width} height={CURVE.height} mode={mode} />
        <Chart option={strip} width={STRIP.width} height={STRIP.height} mode={mode} />
      </div>
    </FigureFrame>
  )
}
