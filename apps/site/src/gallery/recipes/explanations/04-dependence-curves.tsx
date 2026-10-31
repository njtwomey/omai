import { accumulatedLocalEffects, partialDependence } from 'mdml-engine/explain'
import { dependence } from 'mdml-engine/exposition'
import { tensor } from 'mdml-engine/tensor'
import {
  Choice,
  ControlPanel,
  DependenceSurfaceView,
  DependenceView,
  NumberField,
  SeedInput,
  Select,
  Toggle,
} from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'
import { normals, regressor } from '../../toys/synthetic.ts'

export const recipe: Recipe = {
  title: 'Dependence curves',
  question: 'How do I show what a model does as one feature moves: partial dependence, ICE curves and ALE?',
  explain:
    'A `dependence` record holds, per feature, a grid and the average output along it, with the individual (ICE) curves, a rug of where the data lies and, for ALE, the rows per bin. `DependenceView` draws one feature as a line with its ICE curves behind (or ALE bins, or a glass-box shape function as steps), `DependenceSurfaceView` a pair as a heatmap. The two features here are correlated, and the model is `sin(a) + b² + a·c`: partial dependence pushes `a` through values that never occur with the given `b`, while ALE stays on the data. Turn the correlation up and the two curves part.',
  shows: ['dependence', 'DependenceView', 'DependenceSurfaceView'],
}

const FEATURES = ['a', 'b', 'c']
const model = regressor(
  (x) => Math.sin(2 * (x[0] ?? 0)) + (x[1] ?? 0) ** 2 + 0.6 * (x[0] ?? 0) * (x[2] ?? 0) + 0.8 * (x[0] ?? 0),
  3,
)

type Method = 'partial dependence' | 'ale' | 'shape function'
// What a glass-box additive model would learn for each feature: the model's own one-feature terms.
const TERMS: readonly ((v: number) => number)[] = [(v) => Math.sin(2 * v) + 0.8 * v, (v) => v * v, () => 0]

export default function DependenceCurves(): ReactNode {
  const [method, setMethod] = useState<Method>('partial dependence')
  const [correlation, setCorrelation] = useState(0.8)
  const [seed, setSeed] = useState(5)
  const [ice, setIce] = useState(true)
  const [feature, setFeature] = useState('a')
  const [pair, setPair] = useState(false)
  const record = useMemo(() => {
    const n = 300
    const z = normals(seed, 3 * n)
    const rows = Array.from({ length: n }, (_, i) => {
      const a = z[3 * i] ?? 0
      const b = correlation * a + Math.sqrt(1 - correlation ** 2) * (z[3 * i + 1] ?? 0)
      return [a, b, z[3 * i + 2] ?? 0]
    })
    const x = tensor(rows)
    const curves = FEATURES.map((_, f) => {
      const rug = rows.map((r) => r[f] ?? 0)
      if (method === 'shape function') {
        const r = accumulatedLocalEffects(model, x, f, { bins: 12 })
        const term = TERMS[f] ?? (() => 0)
        const mids = r.edges.slice(1).map((e, i) => (e + (r.edges[i] ?? e)) / 2)
        const raw = mids.map(term)
        const mean = raw.reduce((a, b) => a + b, 0) / raw.length
        return { feature: f, grid: [...r.edges], average: raw.map((v) => v - mean), counts: [...r.counts], rug }
      }
      if (method === 'ale') {
        const r = accumulatedLocalEffects(model, x, f, { bins: 12 })
        return { feature: f, grid: [...r.edges], average: [...(r.effects[0] ?? [])], counts: [...r.counts], rug }
      }
      const r = partialDependence(model, x, [f], { gridResolution: 40 })
      return {
        feature: f,
        grid: [...(r.values[0] ?? [])],
        average: [...(r.average[0] ?? [])],
        individual: (r.individual[0] ?? []).slice(0, 25).map((c) => [...c]),
        rug,
      }
    })
    const pd = partialDependence(model, x, [0, 1], { gridResolution: 14 })
    const gridX = [...(pd.values[0] ?? [])]
    const gridY = [...(pd.values[1] ?? [])]
    const flat = pd.average[0] ?? []
    const surface = {
      features: [0, 1] as [number, number],
      gridX,
      gridY,
      values: gridX.map((_, i) => gridY.map((_, j) => flat[i * gridY.length + j] ?? 0)),
    }
    return dependence({
      method,
      features: FEATURES,
      output: 'prediction',
      curves,
      surfaces: [surface],
      ...(method === 'partial dependence'
        ? { indices: Array.from({ length: 25 }, (_, i) => i) }
        : { transform: 'centred' as const }),
    })
  }, [method, correlation, seed])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={4}>
        <Select
          label="method"
          value={method}
          options={['partial dependence', 'ale', 'shape function'] as const}
          onChange={setMethod}
        />
        <NumberField
          label="correlation of a and b"
          value={correlation}
          onChange={setCorrelation}
          min={0}
          max={0.95}
          step={0.05}
        />
        <SeedInput value={seed} onChange={setSeed} />
        <Toggle label="the pair a, b as a surface" value={pair} onChange={setPair} />
        {!pair && <Choice label="feature" value={feature} options={FEATURES} onChange={setFeature} />}
        {!pair && method === 'partial dependence' && <Toggle label="ICE curves" value={ice} onChange={setIce} />}
      </ControlPanel>
      {pair ? (
        <DependenceSurfaceView record={record} title="Partial dependence on a and b" />
      ) : (
        <DependenceView record={record} curve={FEATURES.indexOf(feature)} maxIce={ice ? 25 : 0} />
      )}
    </div>
  )
}
