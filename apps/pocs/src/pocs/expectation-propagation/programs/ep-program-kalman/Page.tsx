import type { PointSet } from 'omai-render/charts'
import { ControlPanel, NumberField, SeedInput } from 'omai-render/react'
import { type ReactNode, useCallback, useMemo, useState } from 'react'
import { CheckLines, endedText, ProgramFrame, RecoveryChart, sig } from '../../../../shared/ep/ProgramParts.tsx'
import { agreement, recovery } from '../../../../shared/ep/recovery.ts'
import { epHooks } from '../../../../shared/ep/sweeps.ts'
import { RunProgress } from '../../../../shared/RunProgress.tsx'
import type { HostBase } from '../../../../shared/run.ts'
import { useRun } from '../../../../shared/useRun.ts'
import { infer, type Setup, simulate } from './program.ts'
import source from './program.ts?raw'
import { kalmanReference } from './reference.ts'

type Made = Awaited<ReturnType<typeof fit>>

/** Simulates the data, runs EP on the chain, then the exact Kalman smoother beside it, reporting to the host. */
async function fit(setup: Setup, host: HostBase) {
  const sim = simulate(setup)
  const t0 = performance.now()
  const run = await infer(sim.y, setup, epHooks(host, 'EP'))
  const ms = performance.now() - t0
  return { sim, ...run, ms, exact: kalmanReference(sim.y, setup) }
}

export default function Page(): ReactNode {
  const [seed, setSeed] = useState(3)
  const [readingSd, setReadingSd] = useState(0.6)
  const [push, setPush] = useState(0.4)
  const [steps, setSteps] = useState(60)
  const task = useCallback(
    (host: HostBase) => fit({ seed, readingSd, push, steps }, host),
    [seed, readingSd, push, steps],
  )
  const ep = useRun<Made>(task)
  const made = ep.value
  return (
    <ProgramFrame
      id="kalman"
      intro={
        <>
          <p>
            A target moves in a line with a position and a velocity that is pushed around a little at every step; each
            step gives a noisy reading of the position (sd {readingSd}), never of the velocity. The model is a scan over
            a vector state: each step is a two-dimensional Gaussian around the matrix times the step before, and each
            reading a linear function of the state. EP reads the chain as one Gaussian vector per step, a dense
            transition factor between neighbours and a reading on the position, and recovers the position and, from the
            readings of the position alone, the velocity.
          </p>
          <p className="text-xs text-muted">
            This is a tree of linear Gaussian factors, so EP is exact: it must agree with the Kalman filter and the
            Rauch-Tung-Striebel smoother of the same model to rounding, and the check below holds it to that.
          </p>
        </>
      }
      controls={
        <ControlPanel title="change the data" columns={4}>
          <SeedInput value={seed} onChange={setSeed} />
          <NumberField label="reading sd" value={readingSd} onChange={setReadingSd} min={0.2} max={2} step={0.1} />
          <NumberField label="push on the velocity" value={push} onChange={setPush} min={0.05} max={1} step={0.05} />
          <NumberField
            label="steps"
            value={steps}
            onChange={(v) => setSteps(Math.round(v))}
            min={20}
            max={200}
            step={10}
          />
        </ControlPanel>
      }
      status={made === undefined ? 'Simulating and running EP...' : endedText(made.run, made.ms)}
      progress={<RunProgress run={ep} label="EP" />}
      figures={made === undefined ? null : <Figures made={made} />}
      check={made === undefined ? null : <Check made={made} />}
      bound={made?.bound}
      run={made?.run}
      source={source}
      file="ep-program-kalman/program.ts"
    />
  )
}

/** One band with its centre: EP's belief about a coordinate along the path. */
const bandOf = (t: readonly number[], rows: Made['position'], name: string) => [
  {
    name,
    x: t,
    mean: rows.map((r) => r.mean),
    lower: rows.map((r) => r.lower),
    upper: rows.map((r) => r.upper),
    colour: 'tab0',
  },
]

function Figures(props: { readonly made: Made }): ReactNode {
  const { sim, position, velocity } = props.made
  const t = useMemo(() => sim.y.map((_, i) => i), [sim])
  const positionBands = useMemo(() => bandOf(t, position, 'EP, position'), [position, t])
  const velocityBands = useMemo(() => bandOf(t, velocity, 'EP, velocity'), [velocity, t])
  const positionTruth = useMemo(() => ({ x: t, y: sim.position, name: 'true position' }), [t, sim])
  const velocityTruth = useMemo(() => ({ x: t, y: sim.velocity, name: 'true velocity' }), [t, sim])
  const dots = useMemo(() => [{ x: t, y: sim.y, name: 'reading' }] satisfies PointSet[], [t, sim])
  return (
    <div className="flex flex-col gap-3">
      <RecoveryChart
        bands={positionBands}
        truth={positionTruth}
        dots={dots}
        dotOffset={4}
        xLabel="step"
        yLabel="position"
        aspect={2.4}
      />
      <RecoveryChart
        bands={velocityBands}
        truth={velocityTruth}
        xLabel="step"
        yLabel="velocity (never read)"
        aspect={2.4}
      />
    </div>
  )
}

function Check(props: { readonly made: Made }): ReactNode {
  const { made } = props
  const { sim, position, velocity, exact } = made
  const recPosition = useMemo(() => recovery(position, sim.position), [position, sim])
  const recVelocity = useMemo(() => recovery(velocity, sim.velocity), [velocity, sim])
  const agreePosition = useMemo(() => agreement(position, exact.position), [position, exact])
  const agreeVelocity = useMemo(() => agreement(velocity, exact.velocity), [velocity, exact])
  const converged = made.run.status === 'converged'
  const worstMean = Math.max(agreePosition.meanZ, agreeVelocity.meanZ)
  const lowSd = Math.min(agreePosition.sdRatioRange[0], agreeVelocity.sdRatioRange[0])
  const highSd = Math.max(agreePosition.sdRatioRange[1], agreeVelocity.sdRatioRange[1])
  return (
    <CheckLines
      lines={[
        {
          label: 'EP converged',
          value: converged ? `yes, ${made.run.t} sweeps` : `no, stopped at ${made.run.t}`,
          rule: 'converged',
          ok: converged,
        },
        {
          label: 'the true position inside the 90% intervals',
          value: `${sig(100 * recPosition.coverage, 3)}% of steps`,
          rule: 'at least 80%',
          ok: recPosition.coverage >= 0.8,
        },
        {
          label: 'the true velocity inside the 90% intervals',
          value: `${sig(100 * recVelocity.coverage, 3)}% of steps`,
          rule: 'at least 80%',
          ok: recVelocity.coverage >= 0.8,
        },
        {
          label: 'error in posterior sds (root mean square), position and velocity',
          value: `${sig(recPosition.zrms, 3)} and ${sig(recVelocity.zrms, 3)}`,
          rule: 'at most 1.6',
          ok: recPosition.zrms <= 1.6 && recVelocity.zrms <= 1.6,
        },
        {
          label: 'EP against the Kalman smoother: the largest gap in the mean',
          value: `${worstMean.toExponential(1)} smoother sds`,
          rule: 'at most 1e-4',
          ok: worstMean <= 1e-4,
        },
        {
          label: 'EP against the Kalman smoother: the spread (sd ratio, smallest to largest)',
          value: `${sig(lowSd, 6)} to ${sig(highSd, 6)}`,
          rule: 'between 0.9999 and 1.0001',
          ok: lowSd >= 0.9999 && highSd <= 1.0001,
        },
      ]}
      note={
        <p>
          The velocity is never read, only inferred from how the position moves: its band is wide at the ends of the
          path (nothing reads before the first or after the last step) and narrow in the middle. The reference is the
          exact smoother of the same model.
        </p>
      }
    />
  )
}
