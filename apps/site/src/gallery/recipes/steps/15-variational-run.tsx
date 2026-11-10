import { gamma } from 'omai-engine/dists'
import { type ElboRun, elboRun, posteriorComparison, trailingMean } from 'omai-engine/exposition'
import { distributionOracle } from 'omai-engine/probprog/validation'
import { ControlPanel, ElboRunView, NumberField, SeedInput } from 'omai-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'
import { normals } from '../../toys/synthetic.ts'

export const recipe: Recipe = {
  title: 'Variational run',
  question:
    'How do I show a variational fit as it goes: the bound, the gradient, the guide and the posterior it has so far?',
  explain:
    'An `elboRun` record holds the evidence lower bound over the steps (raw and smoothed), the norm and variance of the gradient, the guide’s parameters, the posterior’s mean and sd where a snapshot was taken and, when the truth is known, the final posterior over its marginal. The same record serves a run in progress: drag “steps so far” back and every chart shows what an anytime view would have had at that moment. The toy run fits a normal to a skewed target, so the bound levels off below the evidence and the final bell visibly misses the skew.',
  shows: ['elboRun', 'ElboRunView'],
}

const STEPS = 400
// The target: a gamma with mean 2 and sd 1, which a normal can only approximate.
const TRUTH = { rate: distributionOracle('rate', gamma(4, 2)) }
// The best normal for it (the reverse KL pulls it a little inside the target's own mean and sd), and how the fit
// approaches it: the guide starts at 0.3 ± 0.35 and closes in with a time constant of 70 steps.
const BEST = { mean: 1.92, sd: 0.86 }
const decay = (t: number): number => Math.exp(-t / 70)
const meanAt = (t: number): number => 0.3 + (BEST.mean - 0.3) * (1 - decay(t))
const sdAt = (t: number): number => 0.35 + (BEST.sd - 0.35) * (1 - decay(t))

export default function VariationalRun(): ReactNode {
  const [shown, setShown] = useState(STEPS)
  const [noise, setNoise] = useState(0.6)
  const [seed, setSeed] = useState(1)
  const full = useMemo(() => {
    const e = normals(seed, STEPS, noise)
    const g = normals(seed + 1, STEPS, 1)
    const steps = Array.from({ length: STEPS }, (_, i) => i + 1)
    return {
      steps,
      elbo: steps.map((t, i) => -2.9 - 9 * decay(t) + (e[i] ?? 0) * (0.4 + 2.5 * decay(t))),
      gradNorm: steps.map((t, i) => 0.25 + 40 * decay(t) * Math.exp(0.2 * (g[i] ?? 0))),
      // A snapshot of the posterior at the first step and every twentieth.
      snapshots: steps.filter((t) => t === 1 || t % 20 === 0),
    }
  }, [noise, seed])
  const record = useMemo(() => {
    const keep = full.steps.filter((t) => t <= shown)
    const n = keep.length
    const upTo = <T,>(v: readonly T[]): T[] => v.slice(0, n)
    const window = Math.max(1, Math.ceil(n / 20))
    const snaps = full.snapshots.filter((t) => t <= shown)
    const last = snaps.at(-1)
    let final: ElboRun['final']
    if (last !== undefined) {
      const c = posteriorComparison([{ name: 'svi', normal: { rate: { mean: meanAt(last), sd: sdAt(last) } } }], {
        truth: TRUTH,
        oracle: 'truth',
        gridPoints: 160,
      })
      final = { oracle: c.oracle, method: 'svi', quantities: c.quantities }
    }
    return elboRun({
      method: 'svi',
      status: shown >= STEPS ? 'converged' : 'running',
      elbo: { steps: keep, values: upTo(full.elbo) },
      ...(n >= 2
        ? { smoothed: { steps: keep, values: trailingMean(upTo(full.elbo), window) }, smoothing: window }
        : {}),
      gradNorm: { steps: keep, values: upTo(full.gradNorm) },
      gradVariance: {
        steps: keep.filter((t) => t % 5 === 0),
        values: keep.flatMap((t, i) => (t % 5 === 0 ? [(full.gradNorm[i] ?? 0) ** 2 * 0.8 + 0.02] : [])),
      },
      diagnostics: [],
      paths: [],
      parameters: [
        { name: 'rate_auto_loc', site: 'rate', role: 'loc', steps: keep, values: keep.map(meanAt) },
        {
          name: 'rate_auto_scale',
          site: 'rate',
          role: 'scale',
          steps: keep,
          values: keep.map((t) => Math.log(sdAt(t))),
        },
      ],
      beliefs:
        snaps.length === 0 ? [] : [{ name: 'rate', steps: snaps, values: snaps.map(meanAt), sd: snaps.map(sdAt) }],
      ...(final === undefined ? {} : { final }),
    })
  }, [full, shown])
  return (
    <div className="flex w-fit max-w-full flex-col gap-3">
      <ControlPanel columns={3}>
        <NumberField label="steps so far" value={shown} onChange={setShown} min={20} max={STEPS} step={20} integer />
        <NumberField label="noise of the bound" value={noise} onChange={setNoise} min={0.1} max={2} step={0.1} />
        <SeedInput value={seed} onChange={setSeed} />
      </ControlPanel>
      <ElboRunView record={record} title="A variational fit" filename="variational-run" />
    </div>
  )
}
