import type { Recovery } from 'omai-engine/bayesian-examples'
import { cartesian, lines, points } from 'omai-render/charts'
import { figure } from 'omai-render/figure'
import { Chart, useColourMode } from 'omai-render/react'
import { type ReactNode, useMemo } from 'react'

const FIG = figure({ width: 480, height: 360 })

/**
 * Estimated values against the true ones: each parameter is a dot at its posterior mean with its 90% interval as a vertical
 * line (drawn in a warning colour when the truth is outside it), and the dashed diagonal is a perfect recovery. Intervals that
 * straddle the diagonal are right; the dots hug it when the data say a lot.
 */
export function RecoveryChart(props: { readonly record: Recovery; readonly title: string }): ReactNode {
  const { record: r, title } = props
  const mode = useColourMode()
  const option = useMemo(() => {
    const inside = r.truth.map((t, i) => t >= (r.lo[i] as number) && t <= (r.hi[i] as number))
    const segment = (keep: boolean): { x: number[]; y: number[] } => {
      const x: number[] = []
      const y: number[] = []
      r.truth.forEach((t, i) => {
        if (inside[i] !== keep) return
        x.push(t, t, Number.NaN)
        y.push(r.lo[i] as number, r.hi[i] as number, Number.NaN)
      })
      return { x, y }
    }
    const lo = Math.min(...r.truth, ...r.lo)
    const hi = Math.max(...r.truth, ...r.hi)
    const pad = 0.05 * (hi - lo || 1)
    const hit = segment(true)
    const miss = segment(false)
    return cartesian(
      [
        lines([{ x: [lo, hi], y: [lo, hi], name: 'perfect recovery', dash: 'dashed', width: 1.5, colour: 'muted' }], {
          mode,
        }),
        lines([{ ...hit, name: 'interval holds the truth', width: 2, opacity: 0.6, colour: 'tab0' }], { mode }),
        ...(miss.x.length === 0
          ? []
          : [
              lines([{ ...miss, name: 'interval misses the truth', width: 2, opacity: 0.9, colour: 'tab3' }], { mode }),
            ]),
        points([{ x: [...r.truth], y: [...r.mean], name: 'posterior mean', size: 5, colour: 'tab0' }], { mode }),
      ],
      {
        mode,
        figure: FIG,
        x: { label: `true ${r.label}`, limits: [lo - pad, hi + pad] },
        y: { label: `estimated ${r.label}`, limits: [lo - pad, hi + pad] },
        legend: true,
      },
    )
  }, [r, mode])
  return (
    <div>
      <h3 className="text-xs font-semibold text-muted">{title}</h3>
      <Chart option={option} figure={FIG} />
    </div>
  )
}
