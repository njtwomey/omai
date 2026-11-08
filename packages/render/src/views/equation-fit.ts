/**
 * The views of an `EquationFit` record: the fitted expression and the true one as TeX, with each fitted coefficient
 * coloured by how far it is from its truth (and listed with its status in words, so colour is never the only code), the
 * fitted function drawn against the true one and the observations, and the residual in a strip beneath.
 */

import type { EChartsOption } from 'echarts'
import { maxOf } from 'omai-engine/base'
import { coefficientErrors, type EquationFit, fitResidual, type TermStatus } from 'omai-engine/exposition'
import { lines } from '../charts/lines.ts'
import { cartesian } from '../charts/plot.ts'
import { points } from '../charts/points.ts'
import { type ColourMode, tokens } from '../theme/tokens.ts'
import { ariaOf, type ViewOptions, viewFigure } from './_view.ts'

/** What each status of a coefficient says, in words. */
export const STATUS_WORDS: Readonly<Record<TermStatus, string>> = {
  right: 'matches the truth',
  off: 'differs from the truth',
  missing: 'dropped, but in the law',
  extra: 'kept, but not in the law',
  unknown: 'no truth to compare with',
}

// A coefficient as TeX: three significant digits, no trailing zeros, and a minus sign as TeX's.
function coefficientTex(c: number): string {
  const a = Math.abs(c)
  const text =
    a === 0
      ? '0'
      : a >= 1e4 || a < 1e-3
        ? a
            .toExponential(2)
            .replace(/e([+-])(\d+)/, (_, s: string, e: string) => `\\times 10^{${s === '-' ? '-' : ''}${Number(e)}}`)
        : String(Number(a.toPrecision(3)))
  return text
}

/**
 * The fitted expression as TeX: the terms with a coefficient other than zero, each coefficient wrapped in a colour
 * that says how it compares with the truth (green right, amber off, red kept without being in the law).
 *
 * @param record - The record.
 * @param mode - The colour mode, for the colours of the statuses.
 * @returns The TeX, or the record's own `expression` when it has no terms.
 * @example
 * ```ts
 * import { equationFit } from 'omai-engine/exposition'
 * import { fittedTex } from 'omai-render/views'
 * const r = equationFit({ terms: [{ tex: 'x', estimate: 2 }, { tex: '1', estimate: -0.5 }], x: [0, 1], fit: [0, 1] })
 * fittedTex(r, 'light', false) // => 'y = 2\\,x - 0.5'
 * ```
 */
export function fittedTex(record: EquationFit, mode: ColourMode = 'light', colour = true): string {
  const target = record.target ?? 'y'
  if (record.terms === undefined) return `${target} = ${record.expression ?? ''}`
  const t = tokens(mode)
  const errors = coefficientErrors(record)
  const paint: Record<TermStatus, string> = {
    right: t.success,
    off: t.warning,
    missing: t.danger,
    extra: t.danger,
    unknown: t.fg,
  }
  const parts: string[] = []
  for (const [k, term] of record.terms.entries()) {
    if (term.estimate === 0) continue
    const status = errors[k]?.status ?? 'unknown'
    const number = coefficientTex(term.estimate)
    const body = term.tex === '1' ? number : `${number}\\,${term.tex}`
    const painted = colour ? `\\textcolor{${paint[status]}}{${body}}` : body
    const sign = term.estimate < 0 ? '-' : '+'
    parts.push(parts.length === 0 ? (term.estimate < 0 ? `-${painted}` : painted) : ` ${sign} ${painted}`)
  }
  return `${target} = ${parts.length === 0 ? '0' : parts.join('')}`
}

/**
 * The true expression as TeX: the terms whose true coefficient is not zero, or the record's `truthExpression`.
 *
 * @param record - The record.
 * @returns The TeX, or `undefined` when the truth is not known.
 * @example
 * ```ts
 * import { equationFit } from 'omai-engine/exposition'
 * import { truthTex } from 'omai-render/views'
 * const r = equationFit({ terms: [{ tex: 'x', estimate: 2, truth: 3 }, { tex: '1', estimate: 0.1, truth: 0 }], x: [0, 1], fit: [0, 1] })
 * truthTex(r) // => 'y = 3\\,x'
 * ```
 */
export function truthTex(record: EquationFit): string | undefined {
  const target = record.target ?? 'y'
  if (record.truthExpression !== undefined) return `${target} = ${record.truthExpression}`
  const terms = (record.terms ?? []).filter((t) => t.truth !== undefined && t.truth !== 0)
  if (terms.length === 0 || (record.terms ?? []).every((t) => t.truth === undefined)) return undefined
  const parts = terms.map((t, k) => {
    const c = t.truth ?? 0
    const body = t.tex === '1' ? coefficientTex(c) : `${coefficientTex(c)}\\,${t.tex}`
    return k === 0 ? (c < 0 ? `-${body}` : body) : ` ${c < 0 ? '-' : '+'} ${body}`
  })
  return `${target} = ${parts.join('')}`
}

/**
 * How many terms of the fit are right, off, missing, extra or unknown.
 *
 * @param record - The record.
 * @returns The counts by status.
 * @example
 * ```ts
 * import { equationFit } from 'omai-engine/exposition'
 * import { termCounts } from 'omai-render/views'
 * const r = equationFit({ terms: [{ tex: 'x', estimate: 2, truth: 2 }, { tex: '1', estimate: 1, truth: 0 }], x: [0, 1], fit: [0, 1] })
 * termCounts(r) // => { right: 1, off: 0, missing: 0, extra: 1, unknown: 0 }
 * ```
 */
export function termCounts(record: EquationFit): Record<TermStatus, number> {
  const c: Record<TermStatus, number> = { right: 0, off: 0, missing: 0, extra: 0, unknown: 0 }
  for (const e of coefficientErrors(record)) c[e.status]++
  return c
}

/**
 * The fitted function against the true one, with the observations as points.
 *
 * @param record - The record.
 * @param options - Mode and figure.
 * @returns The ECharts option.
 * @example
 * ```ts
 * import { equationFit } from 'omai-engine/exposition'
 * import { equationFitView } from 'omai-render/views'
 * const r = equationFit({ terms: [], x: [0, 1, 2], fit: [0, 1, 2], truth: [0, 1, 3] })
 * equationFitView(r).series.length // => 2
 * ```
 */
export function equationFitView(record: EquationFit, options: ViewOptions = {}): EChartsOption {
  const fig = viewFigure(options, { aspect: 2 })
  const mode = fig.mode
  const [xn, yn] = record.axes ?? ['x', 'y']
  const layers = [
    ...(record.observed === undefined
      ? []
      : [
          points(
            [
              {
                x: [...record.observed.x],
                y: [...record.observed.y],
                name: 'observed',
                colour: 'muted',
                size: 5,
                opacity: 0.6,
                outline: false,
              },
            ],
            { mode },
          ),
        ]),
    ...(record.truth === undefined
      ? []
      : [
          lines([{ x: [...record.x], y: [...record.truth], name: 'truth', colour: 'fg', width: 2, dash: 'dashed' }], {
            mode,
          }),
        ]),
    lines([{ x: [...record.x], y: [...record.fit], name: 'fit', colour: 'tab0', width: 2 }], { mode }),
  ]
  const r = fitResidual(record)
  const worst = Math.max(0, maxOf(r.values.map(Math.abs)))
  return {
    ...cartesian(layers, { mode, figure: fig, x: { label: xn }, y: { label: yn }, legend: true }),
    animation: false,
    ...ariaOf(
      `The fitted function against ${record.truth === undefined ? 'the observations' : 'the true one'}; the largest residual is ${worst.toPrecision(3)}`,
    ),
  }
}

/**
 * The residual of the fit (the fit minus the truth, or minus the observations) in a strip, with the zero line.
 *
 * @param record - The record.
 * @param options - Mode and figure (default a wide, short one).
 * @returns The ECharts option.
 * @example
 * ```ts
 * import { equationFit } from 'omai-engine/exposition'
 * import { residualStripView } from 'omai-render/views'
 * const r = equationFit({ terms: [], x: [0, 1, 2], fit: [0, 1, 2], truth: [0, 1, 3] })
 * residualStripView(r).series.length // => 2
 * ```
 */
export function residualStripView(record: EquationFit, options: ViewOptions = {}): EChartsOption {
  const fig = viewFigure(options, { aspect: 5 })
  const mode = fig.mode
  const r = fitResidual(record)
  const xs = [...r.x]
  return {
    ...cartesian(
      [
        lines(
          [
            {
              x: [xs[0] ?? 0, xs[xs.length - 1] ?? 1],
              y: [0, 0],
              name: '__zero',
              colour: 'muted',
              width: 1,
              dash: 'dotted',
            },
          ],
          { mode },
        ),
        lines([{ x: xs, y: [...r.values], name: 'residual', colour: 'tab3', width: 1.6, area: true, opacity: 0.9 }], {
          mode,
        }),
      ],
      { mode, figure: fig, x: { label: record.axes?.[0] ?? 'x' }, y: { label: 'fit − truth' }, legend: false },
    ),
    animation: false,
    ...ariaOf('The residual of the fit across the grid'),
  }
}
