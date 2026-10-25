/**
 * A contingency table (confusion matrix) with full margins: actual classes as rows and predicted as columns, named
 * cells (TP, FN, FP, TN), row and column totals with their rates (recall, precision, false-positive rate and the rest),
 * prevalence and accuracy; hovering a rate outlines its numerator and denominator.
 */

import { type ReactNode, useMemo, useState } from 'react'
import { type ConfusionMatrix, confusionMargins, tableCost } from 'whitebox-engine/metrics'
import { formatNumber, formatPercent } from '../theme/format.ts'
import { mixColours } from '../theme/palettes.ts'
import { useTokens } from './theme.tsx'

/** Props of {@link ContingencyTable}. */
export interface ContingencyTableProps {
  /** A confusion matrix of counts (`confusionMatrix` or `thresholdConfusion` from `whitebox-engine/metrics`): rows actual, columns predicted. */
  readonly table: ConfusionMatrix
  /** A name per class, in the table's order (default the labels). */
  readonly classNames?: readonly string[]
  /**
   * The index of the positive class of a two-class table: names the cells TP, FN, FP and TN and the margin rates
   * sensitivity (recall), miss rate, specificity, false-alarm rate, precision, false discovery, NPV and false omission;
   * hovering a rate shows its acronym (TPR, FNR, TNR, FPR, PPV, FDR, NPV, FOR) and formula. Omitted, or with more than
   * two classes, the rates are each class's recall and precision.
   */
  readonly positive?: number
  /** A share under each count: of every case (`all`, default), of its row, of its column, or none. */
  readonly share?: 'all' | 'row' | 'column' | 'none'
  /** A colour per class, shown as a dot by each class name (e.g. to match a scatter). */
  readonly classColours?: readonly string[]
  /**
   * Misclassification costs, `costs[actual][predicted]` in the table's class order (correct decisions usually 0, and
   * negative to count as benefits): each cell then shows its count times its cost, and the corner the total cost and
   * the cost per case (`tableCost` from `whitebox-engine/metrics`).
   */
  readonly costs?: readonly (readonly number[])[]
  /** The names of the two axes (default "actual" and "predicted"). */
  readonly rowLabel?: string
  readonly columnLabel?: string
  readonly className?: string
}

type Cell = readonly [number, number]
/** What a margin rate divides: the cells summed on top and the cells summed underneath. */
interface Focus {
  readonly numerator: readonly Cell[]
  readonly denominator: readonly Cell[]
}
interface Rate {
  readonly label: string
  /** The rate's other common names (acronyms), shown in its tooltip. */
  readonly alias?: string
  readonly name: string
  readonly value: number
  readonly formula: string
  readonly focus: Focus
  /** How to show the value (default three significant digits). */
  readonly format?: (v: number) => string
}

const count = (v: number): string => formatNumber(v, { largeExponent: 1e12 })
const rate = (v: number): string => (Number.isFinite(v) ? formatNumber(v, { digits: 3 }) : '–')
const share = (v: number): string => (Number.isFinite(v) ? formatPercent(v) : '–')
const join = (...parts: (string | false | undefined)[]): string => parts.filter(Boolean).join(' ')

/**
 * A contingency table (confusion matrix) with full margins: actual classes as rows and predicted classes as columns,
 * each cell a count with an optional share, shaded lightly by count. Row totals carry the row rates (recall and miss
 * rate; for two classes sensitivity, miss rate, specificity and false-alarm rate), column totals the column rates
 * (precision and false discovery; for two classes also NPV and false omission), and the corner the grand total with
 * the prevalence and the accuracy. Hovering or focusing a rate outlines the cells it divides (the numerator solid, the
 * rest of the denominator dashed) and shows its acronym and formula. Static: give it any
 * table; make it reactive by recomputing the table (e.g. `thresholdConfusion` at a threshold from a control).
 *
 * @param props - The table and display options.
 * @returns The table.
 * @example
 * ```tsx
 * import { ContingencyTable } from 'whitebox-render/react'
 * import { thresholdConfusion } from 'whitebox-engine/metrics'
 * <ContingencyTable table={thresholdConfusion(y, scores, 0.5)} classNames={['benign', 'malignant']} positive={1} />
 * ```
 */
export function ContingencyTable(props: ContingencyTableProps): ReactNode {
  const { table, classNames, positive, classColours, rowLabel = 'actual', columnLabel = 'predicted' } = props
  const mode = props.share ?? 'all'
  const t = useTokens()
  const [focus, setFocus] = useState<Focus | null>(null)
  const k = table.labels.length
  const m = useMemo(() => confusionMargins(table), [table])
  const costs = props.costs
  const cost = useMemo(() => (costs === undefined ? undefined : tableCost(table, costs)), [table, costs])
  const binary = k === 2 && (positive === 0 || positive === 1)
  const names = classNames ?? table.labels.map(String)
  const at = (j: number, i: number): number => table.matrix[j]?.[i] ?? 0
  const row = (j: number): Cell[] => Array.from({ length: k }, (_, i): Cell => [j, i])
  const col = (i: number): Cell[] => Array.from({ length: k }, (_, j): Cell => [j, i])
  const every: Cell[] = Array.from({ length: k * k }, (_, c): Cell => [Math.floor(c / k), c % k])
  const v = (xs: readonly number[], i: number): number => xs[i] ?? Number.NaN

  const cellName = (j: number, i: number): string =>
    binary ? `${j === i ? 'T' : 'F'}${i === positive ? 'P' : 'N'}` : j === i ? 'correct' : ''
  const rowRates = (j: number): Rate[] => {
    const pos = binary && j === positive
    return [
      {
        label: binary ? (pos ? 'sensitivity' : 'specificity') : 'recall',
        ...(binary ? { alias: pos ? 'recall, TPR' : 'TNR' } : {}),
        name: binary ? (pos ? 'recall, sensitivity' : 'specificity') : `recall of ${names[j]}`,
        value: v(m.rowRate, j),
        formula: binary ? (pos ? 'TP / (TP + FN)' : 'TN / (TN + FP)') : 'diagonal / row total',
        focus: { numerator: [[j, j]], denominator: row(j) },
      },
      {
        label: binary ? (pos ? 'miss rate' : 'false-alarm rate') : 'miss rate',
        ...(binary ? { alias: pos ? 'FNR' : 'FPR' } : {}),
        name: binary ? (pos ? 'false-negative rate' : 'false-positive rate') : `miss rate of ${names[j]}`,
        value: v(m.rowMiss, j),
        formula: binary ? (pos ? 'FN / (TP + FN)' : 'FP / (TN + FP)') : '(row total − diagonal) / row total',
        focus: { numerator: row(j).filter(([, i]) => i !== j), denominator: row(j) },
      },
    ]
  }
  const colRates = (i: number): Rate[] => {
    const pos = binary && i === positive
    return [
      {
        label: binary ? (pos ? 'precision' : 'NPV') : 'precision',
        ...(binary ? { alias: pos ? 'PPV' : 'negative predictive value' } : {}),
        name: binary ? (pos ? 'precision' : 'negative predictive value') : `precision of ${names[i]}`,
        value: v(m.columnRate, i),
        formula: binary ? (pos ? 'TP / (TP + FP)' : 'TN / (TN + FN)') : 'diagonal / column total',
        focus: { numerator: [[i, i]], denominator: col(i) },
      },
      {
        label: binary ? (pos ? 'false discovery' : 'false omission') : 'false discovery',
        ...(binary ? { alias: pos ? 'FDR' : 'FOR' } : {}),
        name: binary ? (pos ? 'false discovery rate' : 'false omission rate') : `false discovery rate of ${names[i]}`,
        value: v(m.columnMiss, i),
        formula: binary ? (pos ? 'FP / (TP + FP)' : 'FN / (TN + FN)') : '(column total − diagonal) / column total',
        focus: { numerator: col(i).filter(([j]) => j !== i), denominator: col(i) },
      },
    ]
  }
  const p = positive ?? 0
  const corner: Rate[] = [
    ...(binary
      ? [
          {
            label: 'prevalence',
            name: 'prevalence',
            value: v(m.prevalence, p),
            formula: '(TP + FN) / n',
            focus: { numerator: row(p), denominator: every },
          },
        ]
      : []),
    {
      label: 'accuracy',
      name: 'accuracy',
      value: m.accuracy,
      formula: binary ? '(TP + TN) / n' : 'diagonal / n',
      focus: { numerator: Array.from({ length: k }, (_, j): Cell => [j, j]), denominator: every },
    },
    ...(cost === undefined
      ? []
      : [
          {
            label: 'total cost',
            name: 'total misclassification cost',
            value: cost.total,
            formula: 'Σ count × cost',
            focus: { numerator: every.filter(([a, b]) => (costs?.[a]?.[b] ?? 0) !== 0), denominator: [] },
            format: count,
          },
          {
            label: 'cost per case',
            name: 'expected cost of one case',
            value: cost.perCase,
            formula: 'total cost / n',
            focus: { numerator: every.filter(([a, b]) => (costs?.[a]?.[b] ?? 0) !== 0), denominator: every },
          },
        ]),
  ]

  // Light shading by count, from the surface towards the accent, so text stays legible in both themes.
  const max = Math.max(1, ...table.matrix.flat())
  const shade = (c: number): string => mixColours(t.surface, t.accent, 0.06 + 0.42 * (c / max))
  const inFocus = (cells: readonly Cell[] | undefined, j: number, i: number): boolean =>
    cells?.some(([a, b]) => a === j && b === i) ?? false
  const shareOf = (j: number, i: number): number =>
    mode === 'all' ? at(j, i) / m.n : mode === 'row' ? at(j, i) / v(m.actual, j) : at(j, i) / v(m.predicted, i)

  const rates = (list: Rate[]): ReactNode => (
    <div className="flex flex-col gap-0.5">
      {list.map((r) => (
        <button
          key={r.label}
          type="button"
          title={`${r.name}${r.alias === undefined ? '' : ` (${r.alias})`} = ${r.formula}`}
          onPointerEnter={() => setFocus(r.focus)}
          onPointerLeave={() => setFocus(null)}
          onFocus={() => setFocus(r.focus)}
          onBlur={() => setFocus(null)}
          className="flex items-baseline justify-between gap-x-1.5 rounded px-1 text-left hover:bg-(--wb-surface) focus-visible:bg-(--wb-surface) focus-visible:outline-none"
        >
          <span className="whitespace-nowrap text-(--wb-muted)">{r.label}</span>
          <span className="font-mono text-(--wb-fg) tabular-nums">{(r.format ?? rate)(r.value)}</span>
        </button>
      ))}
    </div>
  )
  const label = (j: number): ReactNode => (
    <span className="line-clamp-2">
      {classColours?.[j] !== undefined && (
        <span
          aria-hidden
          className="mr-1 inline-block size-2 rounded-full align-middle"
          style={{ background: classColours[j] }}
        />
      )}
      {names[j]}
    </span>
  )
  const head = (key: string, children: ReactNode): ReactNode => (
    <th key={key} scope="col" className="px-2 py-1 text-center align-bottom font-normal text-(--wb-muted)">
      {children}
    </th>
  )
  const total = (tag: string, value: number, strong = false): ReactNode => (
    <>
      <div className="text-[10px] text-(--wb-muted)">{tag}</div>
      <div className={join('font-mono text-(--wb-fg) tabular-nums', strong && 'font-medium')}>{count(value)}</div>
      {mode !== 'none' && !strong && (
        <div className="font-mono text-[10px] text-(--wb-muted) tabular-nums">{share(value / m.n)}</div>
      )}
    </>
  )
  const axis = 'text-[11px] font-medium tracking-wide text-(--wb-muted) uppercase'

  return (
    <div className={join('w-full max-w-3xl overflow-auto text-xs', props.className)}>
      <table className="w-full table-fixed border-separate border-spacing-1">
        <colgroup>
          <col className="w-[18%]" />
          {names.map((n) => (
            <col key={n} />
          ))}
          <col className="w-[13%]" />
          <col className="w-[24%]" />
        </colgroup>
        <thead>
          <tr>
            <th />
            <th scope="colgroup" colSpan={k} className={join('border-b border-(--wb-grid) pb-0.5 text-center', axis)}>
              {columnLabel}
            </th>
            <th />
            <th />
          </tr>
          <tr>
            <th className={join('text-left align-bottom', axis)}>{rowLabel}</th>
            {names.map((n, i) => head(n, <span className="text-(--wb-fg)">{label(i)}</span>))}
            {head('total', 'total')}
            {head('rates', 'row rates')}
          </tr>
        </thead>
        <tbody>
          {names.map((name, j) => (
            <tr key={name}>
              <th scope="row" className="px-1 text-left font-normal text-(--wb-fg)">
                {label(j)}
              </th>
              {names.map((_, i) => {
                const num = inFocus(focus?.numerator, j, i)
                const den = !num && inFocus(focus?.denominator, j, i)
                return (
                  <td
                    key={names[i]}
                    className={join(
                      'rounded-md px-2 py-1 text-center align-middle transition-[box-shadow,opacity]',
                      focus !== null && !num && !den && 'opacity-45',
                    )}
                    style={{
                      background: shade(at(j, i)),
                      boxShadow: num ? `inset 0 0 0 2px ${t.fg}` : undefined,
                      outline: den ? `1.5px dashed ${t.fg}` : undefined,
                      outlineOffset: den ? -3 : undefined,
                    }}
                  >
                    {cellName(j, i) !== '' && <div className="text-[10px] text-(--wb-muted)">{cellName(j, i)}</div>}
                    <div className="font-mono text-sm text-(--wb-fg) tabular-nums">{count(at(j, i))}</div>
                    {mode !== 'none' && (
                      <div className="font-mono text-[10px] text-(--wb-muted) tabular-nums">{share(shareOf(j, i))}</div>
                    )}
                    {cost !== undefined && (costs?.[j]?.[i] ?? 0) !== 0 && (
                      <div
                        className="font-mono text-[10px] tabular-nums"
                        style={{ color: (costs?.[j]?.[i] ?? 0) > 0 ? t.danger : t.success }}
                        title={`${count(at(j, i))} × ${formatNumber(costs?.[j]?.[i] ?? 0)}`}
                      >
                        cost {count(cost.cells[j]?.[i] ?? 0)}
                      </div>
                    )}
                  </td>
                )
              })}
              <td className="rounded-md border border-(--wb-grid) px-2 py-1 text-center align-middle">
                {total(binary ? (j === positive ? 'P' : 'N') : '', v(m.actual, j))}
              </td>
              <td className="px-0.5 align-middle">{rates(rowRates(j))}</td>
            </tr>
          ))}
          <tr>
            <th scope="row" className="px-1 text-left font-normal text-(--wb-muted)">
              total
            </th>
            {names.map((n, i) => (
              <td key={n} className="rounded-md border border-(--wb-grid) px-2 py-1 text-center align-middle">
                {total(binary ? (i === positive ? 'PP' : 'PN') : '', v(m.predicted, i))}
              </td>
            ))}
            <td className="rounded-md border border-(--wb-grid) bg-(--wb-surface) px-2 py-1 text-center align-middle">
              {total('n', m.n, true)}
            </td>
            <td />
          </tr>
          <tr>
            <th scope="row" className="px-1 text-left align-top font-normal text-(--wb-muted)">
              column rates
            </th>
            {names.map((n, i) => (
              <td key={n} className="px-0.5 align-top">
                {rates(colRates(i))}
              </td>
            ))}
            <td colSpan={2} className="px-0.5 align-top">
              {rates(corner)}
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  )
}
