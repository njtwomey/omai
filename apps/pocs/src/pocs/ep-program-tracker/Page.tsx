import type { PointSet } from 'omai-render/charts'
import { ControlPanel, NumberField, SeedInput } from 'omai-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import { CheckLines, endedText, ProgramFrame, RecoveryChart, sig } from '../../shared/ep/ProgramParts.tsx'
import { agreement, recovery } from '../../shared/ep/recovery.ts'
import { infer, simulate } from './program.ts'
import source from './program.ts?raw'
import { gibbsReference } from './reference.ts'

export default function Page(): ReactNode {
  const [seed, setSeed] = useState(1)
  const [sd, setSd] = useState(0.5)
  const [outliers, setOutliers] = useState(0.1)
  const [steps, setSteps] = useState(100)
  const [contrast, setContrast] = useState(true)
  const made = useMemo(() => {
    const noise = { sd, outliers }
    const sim = simulate(seed, noise, steps)
    const t0 = performance.now()
    const run = infer(sim.y, noise)
    const ms = performance.now() - t0
    return { sim, ...run, ms, plain: infer(sim.y, noise, false).rows, gibbs: gibbsReference(sim.y, noise) }
  }, [seed, sd, outliers, steps])
  const { sim, rows, plain, gibbs } = made
  const t = useMemo(() => sim.y.map((_, i) => i), [sim])
  const bands = useMemo(
    () => [
      {
        name: 'EP, outlier model',
        x: t,
        mean: rows.map((r) => r.mean),
        lower: rows.map((r) => r.lower),
        upper: rows.map((r) => r.upper),
        colour: 'tab0',
      },
      ...(contrast
        ? [
            {
              name: 'EP, plain Gaussian noise',
              x: t,
              mean: plain.map((r) => r.mean),
              lower: plain.map((r) => r.lower),
              upper: plain.map((r) => r.upper),
              colour: 'tab1',
              dash: 'dashed' as const,
            },
          ]
        : []),
    ],
    [t, rows, plain, contrast],
  )
  const truth = useMemo(() => ({ x: t, y: sim.x, name: 'true path' }), [t, sim])
  const dots = useMemo(
    () =>
      [
        {
          x: t,
          y: sim.y,
          groups: sim.wild.map((w) => (w ? 1 : 0)),
          groupNames: { 0: 'reading', 1: 'wild reading' },
        },
      ] satisfies PointSet[],
    [t, sim],
  )
  const rec = useMemo(() => recovery(rows, sim.x), [rows, sim])
  const recPlain = useMemo(() => recovery(plain, sim.x), [plain, sim])
  const agree = useMemo(() => agreement(rows, gibbs), [rows, gibbs])
  const converged = made.run.status === 'converged'
  const wild = sim.wild.filter(Boolean).length

  return (
    <ProgramFrame
      id="tracker"
      intro={
        <>
          <p>
            A path x moves as an AR(1) process, and each step&apos;s reading is x plus noise of sd {sd}; {wild} of the{' '}
            {steps} readings are wild (noise 8 times as wide). The model is a scan over the state with an
            outlier-mixture observation (with probability {outliers} the reading is wild); EP infers the whole path. The
            same model with plain Gaussian noise is run beside it for contrast.
          </p>
          <p className="text-xs text-muted">
            The noise level and the outlier fraction are taken as known. With the noise level unknown, EP runs with
            plain Gaussian noise (a gamma precision), but not yet with the outlier or Student-t observation inside a
            scan.
          </p>
        </>
      }
      controls={
        <ControlPanel title="change the data" columns={4}>
          <SeedInput value={seed} onChange={setSeed} />
          <NumberField label="noise sd" value={sd} onChange={setSd} min={0.2} max={1.5} step={0.1} />
          <NumberField
            label="fraction of wild readings"
            value={outliers}
            onChange={setOutliers}
            min={0}
            max={0.3}
            step={0.05}
          />
          <NumberField
            label="steps"
            value={steps}
            onChange={(v) => setSteps(Math.round(v))}
            min={40}
            max={200}
            step={20}
          />
        </ControlPanel>
      }
      status={`${endedText(made.run, made.ms)}`}
      figures={
        <div className="flex flex-col gap-2">
          <RecoveryChart
            bands={bands}
            truth={truth}
            dots={dots}
            dotOffset={4}
            xLabel="step"
            yLabel="position"
            aspect={2.4}
          />
          <label className="flex items-center gap-2 text-xs text-muted">
            <input type="checkbox" checked={contrast} onChange={(e) => setContrast(e.target.checked)} />
            show the plain-Gaussian run (it follows the wild readings)
          </label>
        </div>
      }
      check={
        <CheckLines
          lines={[
            {
              label: 'EP converged',
              value: converged ? `yes, ${made.run.t} sweeps` : `no, stopped at ${made.run.t}`,
              rule: 'converged',
              ok: converged,
            },
            {
              label: 'the true path inside the 90% intervals',
              value: `${sig(100 * rec.coverage, 3)}% of steps`,
              rule: 'at least 80%',
              ok: rec.coverage >= 0.8,
            },
            {
              label: 'error in posterior sds (root mean square)',
              value: `${sig(rec.zrms, 3)} (rmse ${sig(rec.rmse, 3)}, mean sd ${sig(rec.meanSd, 3)})`,
              rule: 'at most 1.6',
              ok: rec.zrms <= 1.6,
            },
            {
              label: 'EP against Gibbs: the largest gap in the mean',
              value: `${sig(agree.meanZ, 2)} Gibbs sds`,
              rule: 'at most 0.3',
              ok: agree.meanZ <= 0.3,
            },
            {
              label: 'EP against Gibbs: the spread (sd ratio, smallest to largest)',
              value: `${sig(agree.sdRatioRange[0], 3)} to ${sig(agree.sdRatioRange[1], 3)}`,
              rule: 'between 0.8 and 1.25',
              ok: agree.sdRatioRange[0] >= 0.8 && agree.sdRatioRange[1] <= 1.25,
            },
          ]}
          note={
            <p>
              For contrast, the plain-Gaussian model has the truth inside its intervals at{' '}
              {sig(100 * recPlain.coverage, 3)}% of steps and errors of {sig(recPlain.zrms, 3)} posterior sds: its
              intervals are too narrow, because it believes every reading, and it is dragged toward each wild one. The
              reference is Gibbs sampling of the outlier model ({gibbs.sweeps} sweeps: which readings are wild given the
              path, the path given them by forward filtering and backward sampling).
            </p>
          }
        />
      }
      bound={made.bound}
      run={made.run}
      source={source}
      file="ep-program-tracker/program.ts"
    />
  )
}
