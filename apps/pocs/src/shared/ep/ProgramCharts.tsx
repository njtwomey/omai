import { add, arange, mul, toFlat } from 'omai-engine/tensor'
import type { ReactNode } from 'react'

const W = 680
const H = 250
const PAD = { left: 44, right: 12, top: 12, bottom: 36 }

/** A few round tick values between `lo` and `hi`. */
function ticksOf(lo: number, hi: number, count = 5): number[] {
  const raw = (hi - lo) / count
  const mag = 10 ** Math.floor(Math.log10(raw || 1))
  const step = [1, 2, 2.5, 5, 10].map((s) => s * mag).find((s) => s >= raw) ?? raw
  const first = Math.ceil(lo / step) * step
  const n = Math.max(0, Math.floor((hi + 1e-9 - first) / step) + 1)
  return toFlat(add(mul(arange(n), step), first)).map((v) => Number(v.toPrecision(10)))
}

const colour = {
  truth: 'var(--wb-tab3, #e45756)',
  ep: 'var(--wb-tab0, #4c78a8)',
  reference: 'var(--wb-fg, #222)',
  muted: 'var(--wb-muted, #888)',
}

/** One series of dots with intervals, per category. */
export interface IntervalSeries {
  readonly name: string
  readonly mean: readonly number[]
  readonly lower?: readonly number[]
  readonly upper?: readonly number[]
}

/**
 * Posterior means with their credible intervals per quantity beside the truth the data were simulated from (red crosses),
 * and optionally a reference (hollow circles, with intervals when it has them): the picture of "did it recover the truth".
 *
 * @param props - The labels, the truth, EP's means and intervals, the reference, a y-axis label.
 * @returns The chart as an SVG that fills its box.
 */
export function IntervalPlot(props: {
  readonly labels: readonly string[]
  readonly truth: readonly number[]
  readonly ep: IntervalSeries
  readonly reference?: IntervalSeries
  readonly yLabel?: string
}): ReactNode {
  const { labels, truth, ep, reference, yLabel } = props
  const n = labels.length
  const values = [
    ...truth,
    ...ep.mean,
    ...(ep.lower ?? []),
    ...(ep.upper ?? []),
    ...(reference?.mean ?? []),
    ...(reference?.lower ?? []),
    ...(reference?.upper ?? []),
  ].filter(Number.isFinite)
  const lo0 = Math.min(...values)
  const hi0 = Math.max(...values)
  const pad = 0.08 * (hi0 - lo0 || 1)
  const [lo, hi] = [lo0 - pad, hi0 + pad]
  const x = (i: number): number => PAD.left + ((i + 0.5) / n) * (W - PAD.left - PAD.right)
  const y = (v: number): number => PAD.top + (1 - (v - lo) / (hi - lo)) * (H - PAD.top - PAD.bottom)
  const step = (W - PAD.left - PAD.right) / n
  return (
    <figure className="m-0 flex flex-col gap-1">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={yLabel ?? 'recovered values'} className="w-full">
        {ticksOf(lo, hi).map((t) => (
          <g key={t}>
            <line
              x1={PAD.left}
              x2={W - PAD.right}
              y1={y(t)}
              y2={y(t)}
              stroke="var(--wb-line, #ddd)"
              strokeWidth={t === 0 ? 1.2 : 0.5}
            />
            <text x={PAD.left - 6} y={y(t) + 3.5} textAnchor="end" fontSize={10} fill={colour.muted}>
              {t}
            </text>
          </g>
        ))}
        {labels.map((l, i) => (
          <text key={l} x={x(i)} y={H - PAD.bottom + 14} textAnchor="middle" fontSize={10} fill={colour.muted}>
            {l}
          </text>
        ))}
        {reference?.lower !== undefined && reference.upper !== undefined
          ? labels.map((l, i) => (
              <line
                key={`r${l}`}
                x1={x(i) + 0.18 * step}
                x2={x(i) + 0.18 * step}
                y1={y(reference.lower?.[i] ?? 0)}
                y2={y(reference.upper?.[i] ?? 0)}
                stroke={colour.reference}
                strokeWidth={1.4}
              />
            ))
          : null}
        {labels.map((l, i) => (
          <g key={`e${l}`}>
            {ep.lower !== undefined && ep.upper !== undefined ? (
              <line
                x1={x(i)}
                x2={x(i)}
                y1={y(ep.lower[i] ?? 0)}
                y2={y(ep.upper[i] ?? 0)}
                stroke={colour.ep}
                strokeWidth={3}
                strokeLinecap="round"
                opacity={0.55}
              />
            ) : null}
            <circle cx={x(i)} cy={y(ep.mean[i] ?? 0)} r={3.6} fill={colour.ep} />
            {reference === undefined ? null : (
              <circle
                cx={x(i) + 0.18 * step}
                cy={y(reference.mean[i] ?? 0)}
                r={3}
                fill="none"
                stroke={colour.reference}
                strokeWidth={1.4}
              />
            )}
            <path
              d={`M${x(i) - 4.5},${y(truth[i] ?? 0) - 4.5} l9,9 m0,-9 l-9,9`}
              stroke={colour.truth}
              strokeWidth={2}
              fill="none"
            />
          </g>
        ))}
      </svg>
      <figcaption className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted">
        <span style={{ color: colour.truth }}>x true value</span>
        <span style={{ color: colour.ep }}>● {ep.name} (bar: its 90% interval)</span>
        {reference === undefined ? null : <span>○ {reference.name}</span>}
        {yLabel === undefined ? null : <span>y axis: {yLabel}</span>}
      </figcaption>
    </figure>
  )
}

/**
 * Probabilities per category as bars (EP's), the exact ones as hollow circles beside them, and the truth (0 or 1) as red
 * crosses: the inclusion probability of each coefficient against which ones are truly active.
 *
 * @param props - The labels, EP's probabilities, the exact ones and which categories are true.
 * @returns The chart.
 */
export function ProbabilityPlot(props: {
  readonly labels: readonly string[]
  readonly ep: readonly number[]
  readonly exact?: readonly number[]
  readonly truth: readonly boolean[]
  readonly epName?: string
  readonly exactName?: string
}): ReactNode {
  const { labels, ep, exact, truth, epName = 'EP', exactName = 'exact' } = props
  const n = labels.length
  const x = (i: number): number => PAD.left + ((i + 0.5) / n) * (W - PAD.left - PAD.right)
  const y = (v: number): number => PAD.top + (1 - v) * (H - PAD.top - PAD.bottom)
  const stride = Math.max(1, Math.ceil(n / 24)) // a label for about every one of 24 bars
  const bar = Math.min(26, 0.6 * ((W - PAD.left - PAD.right) / n))
  return (
    <figure className="m-0 flex flex-col gap-1">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="inclusion probabilities" className="w-full">
        {[0, 0.5, 1].map((t) => (
          <g key={t}>
            <line
              x1={PAD.left}
              x2={W - PAD.right}
              y1={y(t)}
              y2={y(t)}
              stroke="var(--wb-line, #ddd)"
              strokeWidth={t === 0.5 ? 0.8 : 0.5}
              strokeDasharray={t === 0.5 ? '4 3' : undefined}
            />
            <text x={PAD.left - 6} y={y(t) + 3.5} textAnchor="end" fontSize={10} fill={colour.muted}>
              {t}
            </text>
          </g>
        ))}
        {labels.map((l, i) => (
          <g key={l}>
            <rect
              x={x(i) - bar / 2}
              y={y(ep[i] ?? 0)}
              width={bar}
              height={Math.max(0, y(0) - y(ep[i] ?? 0))}
              fill={colour.ep}
              opacity={0.75}
            />
            {exact === undefined ? null : (
              <circle
                cx={x(i)}
                cy={y(exact[i] ?? 0)}
                r={3.4}
                fill="var(--wb-bg, #fff)"
                stroke={colour.reference}
                strokeWidth={1.6}
              />
            )}
            <path
              d={`M${x(i) - 4.5},${y(truth[i] ? 1 : 0) - 4.5} l9,9 m0,-9 l-9,9`}
              stroke={colour.truth}
              strokeWidth={2}
              fill="none"
              transform={`translate(0 ${truth[i] ? 8 : -8})`}
            />
            {i % stride === 0 ? (
              <text x={x(i)} y={H - PAD.bottom + 14} textAnchor="middle" fontSize={10} fill={colour.muted}>
                {l}
              </text>
            ) : null}
          </g>
        ))}
      </svg>
      <figcaption className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted">
        <span style={{ color: colour.ep }}>bar: {epName}</span>
        {exact === undefined ? null : <span>○ {exactName}</span>}
        <span style={{ color: colour.truth }}>x the truth (top: active, bottom: not)</span>
      </figcaption>
    </figure>
  )
}
