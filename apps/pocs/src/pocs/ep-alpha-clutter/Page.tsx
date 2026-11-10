import { Choice, ControlPanel, SeedInput, Select } from 'omai-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import { CurveChart } from '../../shared/CurveChart.tsx'
import { RunBar } from '../../shared/RunBar.tsx'
import { useAnytimeRuns } from '../../shared/useAnytimeRuns.ts'
import { ALPHAS, clutterRuns, clutterSetup, type Fit } from './compute.ts'

const gaussian = (x: number, mean: number, sd: number): number =>
  Math.exp(-0.5 * ((x - mean) / sd) ** 2) / (sd * Math.sqrt(2 * Math.PI))

export default function EpAlphaClutter(): ReactNode {
  const [seed, setSeed] = useState(5)
  const [n, setN] = useState(5)
  const [alpha, setAlpha] = useState(0.1)
  const [budget, setBudget] = useState(15)
  const setup = useMemo(() => clutterSetup(seed, n), [seed, n])
  const specs = useMemo(() => clutterRuns(setup.graph, setup.exact), [setup])
  const runs = useAnytimeRuns(specs, budget * 1000)
  const fit = (a: number): Fit | undefined => runs.runs.find((r) => r.id === String(a))?.value
  const ep = fit(1)
  const chosen = fit(alpha)
  const { exact, problem } = setup

  // The picture: the exact posterior, expectation propagation's Gaussian and the chosen alpha's, with the points below.
  const view = useMemo(() => {
    const spread = Math.max(3 * exact.sd, 3)
    const lo = Math.min(exact.mean - spread, ...problem.ys) - 0.5
    const hi = Math.max(exact.mean + spread, ...problem.ys) + 0.5
    const keep = exact.x.map((_, i) => i).filter((i) => (exact.x[i] ?? 0) >= lo && (exact.x[i] ?? 0) <= hi)
    const xs = keep.map((i) => exact.x[i] ?? 0)
    const curve = (name: string, colour: string, f: Fit | undefined, dash?: 'dashed') =>
      f === undefined
        ? []
        : [
            {
              name,
              colour,
              width: 2.2,
              ...(dash === undefined ? {} : { dash }),
              x: xs,
              y: xs.map((v) => gaussian(v, f.mean, f.sd)),
            },
          ]
    const top = Math.max(...exact.density) * 1.1
    return {
      curves: [
        { name: 'exact posterior', colour: 'fg', width: 3, x: xs, y: keep.map((i) => exact.density[i] ?? 0) },
        ...curve('EP (alpha 1)', 'tab0', ep),
        ...(alpha === 1 ? [] : curve(`alpha ${alpha}`, 'tab1', chosen)),
      ],
      dots: [
        {
          x: problem.ys,
          y: problem.ys.map(() => -0.04 * top),
          groups: problem.isClutter.map((c) => (c ? 1 : 0)),
          groupNames: { 0: 'good point', 1: 'clutter' },
          size: 8,
        },
      ],
      limits: [lo, hi] as [number, number],
      top,
    }
  }, [exact, problem, ep, chosen, alpha])

  // The divergences along the ladder, as far as the runs have got.
  const along = useMemo(() => {
    const got = ALPHAS.flatMap((a) => {
      const f = runs.runs.find((r) => r.id === String(a))?.value
      return f === undefined ? [] : [{ a, f }]
    }).sort((p, q) => p.a - q.a)
    const mark = Math.max(1, ...got.flatMap((g) => [g.f.forward, g.f.reverse]))
    return {
      curves: [
        {
          name: 'KL(exact ‖ Gaussian): misses mass',
          colour: 'tab3',
          width: 2,
          markers: true,
          x: got.map((g) => g.a),
          y: got.map((g) => g.f.forward),
        },
        {
          name: 'KL(Gaussian ‖ exact): spills over',
          colour: 'tab2',
          width: 2,
          markers: true,
          x: got.map((g) => g.a),
          y: got.map((g) => g.f.reverse),
        },
        { name: 'chosen alpha', colour: 'muted', dash: 'dashed' as const, width: 1.2, x: [alpha, alpha], y: [0, mark] },
      ],
    }
  }, [runs.runs, alpha])

  return (
    <div className="flex flex-col gap-4">
      <RunBar runs={runs} budget={budget} onBudget={setBudget} budgets={[5, 15, 45]} unit="sweeps" />
      <ControlPanel columns={3}>
        <Select
          label="alpha shown against EP"
          value={alpha}
          options={ALPHAS.map((a) => ({ value: a, label: a === 1 ? '1 (EP)' : String(a) }))}
          onChange={setAlpha}
        />
        <Choice label="points" value={n} options={[5, 8, 12]} onChange={setN} />
        <SeedInput value={seed} onChange={setSeed} />
      </ControlPanel>
      <div className="grid items-start gap-4 lg:grid-cols-2">
        <div className="flex min-w-0 flex-col gap-1">
          <h2 className="text-sm font-medium text-fg">The posterior of the mean and two Gaussians fitted to it</h2>
          <CurveChart
            curves={view.curves}
            dots={view.dots}
            x={{ label: 'mean', limits: view.limits }}
            y={{ label: 'density', limits: [-0.07 * view.top, view.top] }}
          />
        </div>
        <div className="flex min-w-0 flex-col gap-1">
          <h2 className="text-sm font-medium text-fg">What each alpha gets wrong</h2>
          <CurveChart
            curves={along.curves}
            x={{ label: 'alpha', limits: [0, 1.05] }}
            y={{ label: 'divergence (nats)' }}
          />
        </div>
      </div>
      <table className="w-full max-w-2xl text-left text-sm tabular-nums">
        <thead className="text-xs text-muted">
          <tr>
            <th className="py-1 font-normal">alpha</th>
            <th className="font-normal">mean</th>
            <th className="font-normal">sd</th>
            <th className="font-normal">KL(exact ‖ q)</th>
            <th className="font-normal">KL(q ‖ exact)</th>
          </tr>
        </thead>
        <tbody>
          <tr className="border-t border-line text-muted">
            <td className="py-1">exact</td>
            <td>{exact.mean.toFixed(2)}</td>
            <td>{exact.sd.toFixed(2)}</td>
            <td />
            <td />
          </tr>
          {ALPHAS.map((a) => {
            const f = fit(a)
            return (
              <tr key={a} className="border-t border-line">
                <td className="py-1">{a}</td>
                <td>{f?.mean.toFixed(2) ?? '…'}</td>
                <td>{f?.sd.toFixed(2) ?? '…'}</td>
                <td>{f?.forward.toFixed(3) ?? '…'}</td>
                <td>{f?.reverse.toFixed(3) ?? '…'}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
