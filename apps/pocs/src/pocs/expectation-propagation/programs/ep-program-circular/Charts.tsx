import { logSumExpOf } from 'omai-engine/base'
import { add, arange, bincount, div, floor, minimum, mul, tensor, toFlat } from 'omai-engine/tensor'
import { type ReactNode, useMemo, useState } from 'react'
import { sig } from '../../../../shared/ep/program-check.ts'
import { fitLogDensity, type Mixture, truthLogDensity } from './compute.ts'
import { PERIOD, type TruthName, type Variant } from './program.ts'

const colour = {
  truth: 'var(--wb-fg, #222)',
  fit: 'var(--wb-tab0, #4c78a8)',
  bar: 'var(--wb-muted, #888)',
  band: 'var(--wb-tab2, #54a24b)',
  grid: 'var(--wb-line, #ddd)',
  muted: 'var(--wb-muted, #888)',
  accent: 'var(--wb-tab3, #e45756)',
}

/** Densities of a histogram with one bin per hour, in readings per hour per reading. */
export function histogram(hours: readonly number[], bins = 24): number[] {
  const bin = minimum(floor(mul(div(tensor(hours), PERIOD), bins)), bins - 1)
  return toFlat(div(bincount(bin, bins), hours.length * (PERIOD / bins)))
}

const grid = (n: number): number[] => toFlat(div(mul(arange(n + 1), PERIOD), n))

/** The fitted density of each component as a curve (its weight times its own density), for the model's way of treating the circle. */
function componentLogs(variant: Variant, mix: Mixture): ((x: number) => number)[] {
  return mix.weights.map(
    (w, k) => (x: number) => fitLogDensity(variant, { weights: [w], means: [mix.means[k] as number], sd: mix.sd }, x),
  )
}

/** The pieces of [m - sd, m + sd] that lie on the day, with the part that runs past an edge brought round to the other (when the model wraps). */
function bandSegments(m: number, sd: number, wrap: boolean): [number, number][] {
  const lo = m - sd
  const hi = m + sd
  const out: [number, number][] = [[Math.max(lo, 0), Math.min(hi, PERIOD)]]
  if (wrap && lo < 0) out.push([PERIOD + lo, PERIOD])
  if (wrap && hi > PERIOD) out.push([0, hi - PERIOD])
  return out
}

/**
 * The day, from 0 to 24 hours: the readings as a histogram, the truth's exact wrapped density dashed, the fit's density solid, each
 * component's weighted density thin and a band of one sd under it.
 *
 * @param props - The readings, the truth, the model's way with the circle and the fit.
 * @returns The chart.
 */
export function DayChart(props: {
  readonly hours: readonly number[]
  readonly truth: TruthName
  readonly variant: Variant
  readonly mix: Mixture
}): ReactNode {
  const { hours, truth, variant, mix } = props
  const W = 680
  const H = 270
  const pad = { left: 44, right: 12, top: 12, bottom: 34 }
  const shown = useMemo(() => {
    const xs = grid(240)
    const bars = histogram(hours)
    const truthY = xs.map((x) => Math.exp(truthLogDensity(truth, x)))
    const fitY = xs.map((x) => Math.exp(fitLogDensity(variant, mix, x)))
    const parts = componentLogs(variant, mix).map((f) => xs.map((x) => Math.exp(f(x))))
    const top = 1.12 * Math.max(...bars, ...truthY, ...fitY)
    return { xs, bars, truthY, fitY, parts, top }
  }, [hours, truth, variant, mix])
  const x = (v: number): number => pad.left + (v / PERIOD) * (W - pad.left - pad.right)
  const y = (v: number): number => pad.top + (1 - v / shown.top) * (H - pad.top - pad.bottom)
  const path = (ys: readonly number[]): string =>
    shown.xs.map((v, i) => `${i === 0 ? 'M' : 'L'}${x(v)},${y(ys[i] ?? 0)}`).join('')
  const wrap = variant === 'means' || variant === 'data'
  const barW = (W - pad.left - pad.right) / shown.bars.length
  return (
    <figure className="m-0 flex flex-col gap-1">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label="the readings, the truth and the fit over the day"
        className="w-full"
      >
        {[0, 6, 12, 18, 24].map((h) => (
          <g key={h}>
            <line x1={x(h)} x2={x(h)} y1={pad.top} y2={H - pad.bottom} stroke={colour.grid} strokeWidth={0.6} />
            <text x={x(h)} y={H - pad.bottom + 14} textAnchor="middle" fontSize={10} fill={colour.muted}>
              {`${String(h).padStart(2, '0')}:00`}
            </text>
          </g>
        ))}
        <line x1={pad.left} x2={W - pad.right} y1={y(0)} y2={y(0)} stroke={colour.muted} strokeWidth={1} />
        {shown.bars.map((b, i) => (
          <rect
            // biome-ignore lint/suspicious/noArrayIndexKey: the bins are fixed
            key={i}
            x={pad.left + i * barW + 0.5}
            y={y(b)}
            width={barW - 1}
            height={y(0) - y(b)}
            fill={colour.bar}
            opacity={0.35}
          />
        ))}
        {mix.means.flatMap((m) =>
          bandSegments(m, mix.sd, wrap).map(([a, b]) => (
            <rect
              key={`band-${m}-${a}`}
              x={x(a)}
              y={y(0) - 9}
              width={Math.max(0, x(b) - x(a))}
              height={9}
              fill={colour.band}
              opacity={0.55}
            />
          )),
        )}
        {shown.parts.map((p, k) => (
          // biome-ignore lint/suspicious/noArrayIndexKey: one per component
          <path key={k} d={path(p)} fill="none" stroke={colour.fit} strokeWidth={1} opacity={0.5} />
        ))}
        <path d={path(shown.fitY)} fill="none" stroke={colour.fit} strokeWidth={2.4} />
        <path d={path(shown.truthY)} fill="none" stroke={colour.truth} strokeWidth={2} strokeDasharray="6 4" />
      </svg>
      <figcaption className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted">
        <span>grey bars: the readings (one bin per hour)</span>
        <span style={{ color: colour.truth }}>- - the truth, the exact wrapped density</span>
        <span style={{ color: colour.fit }}>solid: the fit (thin: each component)</span>
        <span style={{ color: colour.band }}>green: one sd either side of each mean</span>
      </figcaption>
    </figure>
  )
}

/**
 * The same day as a clock face: midnight at the top, the readings as wedges, the truth dashed and the fit solid, so that 23:50 and
 * 00:10 sit side by side.
 *
 * @param props - The readings, the truth, the model's way with the circle and the fit.
 * @returns The chart.
 */
export function ClockChart(props: {
  readonly hours: readonly number[]
  readonly truth: TruthName
  readonly variant: Variant
  readonly mix: Mixture
}): ReactNode {
  const { hours, truth, variant, mix } = props
  const size = 320
  const c = size / 2
  const inner = 62
  const shown = useMemo(() => {
    const xs = grid(240)
    const bars = histogram(hours)
    const truthY = xs.map((v) => Math.exp(truthLogDensity(truth, v)))
    const fitY = xs.map((v) => Math.exp(fitLogDensity(variant, mix, v)))
    return { xs, bars, truthY, fitY, top: 1.05 * Math.max(...bars, ...truthY, ...fitY) }
  }, [hours, truth, variant, mix])
  const reach = c - inner - 30
  const polar = (hour: number, r: number): [number, number] => {
    const angle = (2 * Math.PI * hour) / PERIOD
    return [c + r * Math.sin(angle), c - r * Math.cos(angle)]
  }
  const at = (hour: number, density: number): [number, number] => polar(hour, inner + (density / shown.top) * reach)
  const ring = (ys: readonly number[]): string =>
    `${shown.xs.map((v, i) => `${i === 0 ? 'M' : 'L'}${at(v, ys[i] ?? 0).join(',')}`).join('')}Z`
  const wedge = (i: number, d: number): string => {
    const [a, b, e, f] = [at(i, 0), at(i, d), at(i + 1, d), at(i + 1, 0)]
    return `M${a.join(',')}L${b.join(',')}L${e.join(',')}L${f.join(',')}Z`
  }
  return (
    <figure className="m-0 flex flex-col items-center gap-1">
      <svg
        viewBox={`-24 0 ${size + 48} ${size}`}
        role="img"
        aria-label="the readings, the truth and the fit on a clock face"
        className="w-full max-w-xs"
      >
        <circle cx={c} cy={c} r={inner} fill="none" stroke={colour.muted} strokeWidth={1} />
        {[0, 6, 12, 18].map((h) => {
          const [lx, ly] = polar(h, c - 10)
          return (
            <text key={h} x={lx} y={ly + 3} textAnchor="middle" fontSize={10} fill={colour.muted}>
              {`${String(h).padStart(2, '0')}:00`}
            </text>
          )
        })}
        {shown.bars.map((b, i) => (
          // biome-ignore lint/suspicious/noArrayIndexKey: the bins are fixed
          <path key={i} d={wedge(i, b)} fill={colour.bar} opacity={0.35} />
        ))}
        <path d={ring(shown.fitY)} fill="none" stroke={colour.fit} strokeWidth={2.2} />
        <path d={ring(shown.truthY)} fill="none" stroke={colour.truth} strokeWidth={1.8} strokeDasharray="5 3" />
      </svg>
    </figure>
  )
}

const logNormal = (x: number, mu: number, s: number): number =>
  -0.5 * ((x - mu) / s) ** 2 - Math.log(s) - 0.5 * Math.log(2 * Math.PI)

const lse = logSumExpOf

/**
 * The post's picture: drag the mean. In "shift the means" the data stay and the normal is copied a period early and late (three bumps);
 * in "shift the data" the normal stays and each reading is copied (three rows of ticks). The sum underneath, the log likelihood, is
 * the same number, since N(x; mu + k period, s) is N(x - k period; mu, s) term by term.
 *
 * @param props - The readings (a few are drawn), the sd and the starting mean.
 * @returns The picture with its slider.
 */
export function ShiftPicture(props: {
  readonly hours: readonly number[]
  readonly sd: number
  readonly start: number
}): ReactNode {
  const { hours, sd, start } = props
  const [mu, setMu] = useState(start)
  const shown = hours.slice(0, 40)
  const W = 330
  const H = 150
  const lo = -PERIOD
  const hi = 2 * PERIOD
  const x = (v: number): number => 8 + ((v - lo) / (hi - lo)) * (W - 16)
  const bump = (m: number): string => {
    const peak = Math.exp(logNormal(m, m, sd))
    const vs = toFlat(add(div(mul(arange(181), hi - lo), 180), lo))
    return vs
      .map((v, i) => `${i === 0 ? 'M' : 'L'}${x(v)},${70 - (Math.exp(logNormal(v, m, sd)) / peak) * 52}`)
      .join('')
  }
  const sum = (total: (v: number) => number): number => hours.reduce((s, v) => s + total(v), 0)
  const means = sum((v) => lse([-1, 0, 1].map((k) => Math.log(1 / 3) + logNormal(v, mu + k * PERIOD, sd))))
  const data = sum((v) => lse([-1, 0, 1].map((k) => Math.log(1 / 3) + logNormal(v - k * PERIOD, mu, sd))))
  const day = (
    <rect
      x={x(0)}
      y={4}
      width={x(PERIOD) - x(0)}
      height={H - 8}
      fill="none"
      stroke={colour.muted}
      strokeWidth={0.8}
      strokeDasharray="3 3"
    />
  )
  return (
    <div className="flex flex-col gap-2">
      <label className="flex items-center gap-3 text-xs">
        <span className="w-20 text-muted">mean {sig(mu, 3)} h</span>
        <input
          type="range"
          min={0}
          max={PERIOD}
          step={0.1}
          value={mu}
          onChange={(e) => setMu(Number(e.target.value))}
          className="w-full max-w-sm"
        />
      </label>
      <div className="grid gap-4 md:grid-cols-2">
        <figure className="m-0 flex flex-col gap-1">
          <svg
            viewBox={`0 0 ${W} ${H}`}
            className="w-full"
            role="img"
            aria-label="shift the means: three bumps, the data fixed"
          >
            {day}
            {[-1, 0, 1].map((k) => (
              <path
                key={k}
                d={bump(mu + k * PERIOD)}
                fill="none"
                stroke={colour.fit}
                strokeWidth={k === 0 ? 2 : 1.3}
                opacity={k === 0 ? 1 : 0.7}
              />
            ))}
            {shown.map((v, i) => (
              // biome-ignore lint/suspicious/noArrayIndexKey: ticks
              <line key={i} x1={x(v)} x2={x(v)} y1={86} y2={104} stroke={colour.bar} strokeWidth={1.4} />
            ))}
          </svg>
          <figcaption className="text-xs text-muted">
            shift the means: the data stay (ticks), the normal is copied at mu - 24, mu and mu + 24. log likelihood{' '}
            {sig(means, 12)}
          </figcaption>
        </figure>
        <figure className="m-0 flex flex-col gap-1">
          <svg
            viewBox={`0 0 ${W} ${H}`}
            className="w-full"
            role="img"
            aria-label="shift the data: one bump, each reading copied"
          >
            {day}
            <path d={bump(mu)} fill="none" stroke={colour.fit} strokeWidth={2} />
            {[-1, 0, 1].map((k) =>
              shown.map((v, i) => (
                <line
                  // biome-ignore lint/suspicious/noArrayIndexKey: ticks
                  key={`${k}-${i}`}
                  x1={x(v + k * PERIOD)}
                  x2={x(v + k * PERIOD)}
                  y1={80 + (k + 1) * 9}
                  y2={86 + (k + 1) * 9}
                  stroke={k === 0 ? colour.bar : colour.accent}
                  strokeWidth={1.4}
                />
              )),
            )}
          </svg>
          <figcaption className="text-xs text-muted">
            shift the data: the normal stays, each reading is copied a day early and late (the three rows). log
            likelihood {sig(data, 12)}
          </figcaption>
        </figure>
      </div>
      <p className="max-w-3xl text-xs text-muted">
        The two log likelihoods agree to {sig(Math.abs(means - data), 2)} (rounding) at every mean: wrapping stops being
        something the distribution does and becomes something the loss does. The dashed box is the day, 0 to 24 hours.
      </p>
    </div>
  )
}
