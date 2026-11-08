import { mixtureSteps } from 'omai-engine/exposition'
import { ControlPanel, ExpositionView, NumberField, SeedInput } from 'omai-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'
import { blobs } from '../../toys/synthetic.ts'

export const recipe: Recipe = {
  title: 'Mixture steps',
  question: 'How do I show expectation-maximisation fitting a mixture, step by step?',
  explain:
    'A `mixtureSteps` record keeps, at every EM iteration, the mixing weights, the component means and covariances, the responsibilities of each row and the log likelihood. `MixturePlayer` plays the fit: each component as a pair of ellipses (one and two standard deviations) on the rows coloured by responsibility, the likelihood climbing beside it. The components start badly placed on three blobs and the likelihood only ever goes up; a fourth component on three blobs splits one, and a mismatched start can park two components on one blob.',
  shows: ['mixtureSteps', 'MixturePlayer'],
}

export default function MixtureStepsRecipe(): ReactNode {
  const [components, setComponents] = useState(3)
  const [spread, setSpread] = useState(0.7)
  const [seed, setSeed] = useState(1)
  const { record, points } = useMemo(() => {
    const { points: x } = blobs(seed, 3, 40, spread)
    const n = x.length
    let means = x.slice(0, components).map((p, j) => [(p[0] ?? 0) * 0.3 + j * 0.2, (p[1] ?? 0) * 0.3])
    let vars = means.map(() => 2)
    let weights = means.map(() => 1 / components)
    const out = {
      weights: [] as number[][],
      means: [] as number[][][],
      covariances: [] as number[][][],
      responsibilities: [] as number[][][],
      objective: [] as number[],
    }
    const dens = (p: number[], m: number[], v: number): number =>
      Math.exp(-(((p[0] ?? 0) - (m[0] ?? 0)) ** 2 + ((p[1] ?? 0) - (m[1] ?? 0)) ** 2) / (2 * v)) / (2 * Math.PI * v)
    for (let t = 0; t < 16; t++) {
      const joint = x.map((p) => means.map((m, j) => (weights[j] ?? 0) * dens(p, m, vars[j] ?? 1)))
      const resp = joint.map((r) => {
        const s = r.reduce((a, b) => a + b, 0) || 1e-300
        return r.map((v) => v / s)
      })
      out.weights.push([...weights])
      out.means.push(means.map((m) => [...m]))
      out.covariances.push(vars.map((v) => [v, 0, 0, v]))
      out.responsibilities.push(resp)
      out.objective.push(joint.reduce((s, r) => s + Math.log(r.reduce((a, b) => a + b, 0) || 1e-300), 0))
      const nk = means.map((_, j) => resp.reduce((s, r) => s + (r[j] ?? 0), 0))
      means = means.map((_, j) =>
        [0, 1].map((q) => resp.reduce((s, r, i) => s + (r[j] ?? 0) * (x[i]?.[q] ?? 0), 0) / (nk[j] || 1)),
      )
      vars = means.map((m, j) =>
        Math.max(
          0.02,
          resp.reduce(
            (s, r, i) =>
              s +
              (r[j] ?? 0) * ((x[i]?.[0] ?? 0) - (m[0] ?? 0)) ** 2 +
              (r[j] ?? 0) * ((x[i]?.[1] ?? 0) - (m[1] ?? 0)) ** 2,
            0,
          ) /
            (2 * (nk[j] || 1)),
        ),
      )
      weights = nk.map((v) => v / n)
    }
    return {
      points: x,
      record: mixtureSteps({ family: 'Gaussian', ...out, rows: { points: x, features: ['x', 'y'] } }),
    }
  }, [components, spread, seed])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={3}>
        <NumberField label="components" value={components} onChange={setComponents} min={1} max={6} integer />
        <NumberField label="blob spread" value={spread} onChange={setSpread} min={0.3} max={1.5} step={0.1} />
        <SeedInput value={seed} onChange={setSeed} />
      </ControlPanel>
      <ExpositionView record={record} points={points} title="EM fitting a Gaussian mixture" />
    </div>
  )
}
