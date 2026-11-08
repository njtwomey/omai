/**
 * Example "Principal components of the iris data" (in the index, not in the README): PCA fitted by power iteration as a
 * session, so the components are found one per step and each step is on the tape. The figure is the library's
 * components view of a record built from the session's `model/params` and `model/embedding` probes: the rows in the
 * plane of the first two components coloured by species (with the feature loadings as arrows), the share of the
 * variance each component explains, and the loadings.
 */

import { iris } from 'mdml-engine/datasets'
import { pca } from 'mdml-engine/decomposition'
import { componentsRecord, type Exposition } from 'mdml-engine/exposition'
import { key } from 'mdml-engine/random'
import { toFlat } from 'mdml-engine/tensor'
import { figure } from 'mdml-render/figure'
import { categoricalColour } from 'mdml-render/theme'
import { stillOf } from 'mdml-render/views'
import type { ExampleModule, Outcome } from '../_shared/contract.ts'
import { Figure, Panel } from '../_shared/figure.tsx'

export const meta: ExampleModule['meta'] = {
  title: 'Principal components of the iris data',
  summary:
    'PCA found one component per step: the flowers in the first two components, the variance each explains, the loadings.',
  readme: false,
  alt: 'PCA of the iris data: the rows in the plane of the first two components coloured by species, the variance explained and the loadings',
}

export function run(): Outcome<Exposition> {
  const data = iris()
  const fit = pca.with({ nComponents: 4, solver: 'power' }).session(data, { key: key(1) })
  fit.advance() // one component per step
  const params = fit.probe(fit.t, 'model/params') as {
    components: number[][]
    explainedVariance: number[]
    explainedVarianceRatio: number[]
  }
  const scores = fit.probe(fit.t, 'model/embedding') as number[][]
  const record = componentsRecord({
    method: 'pca',
    features: data.featureNames ?? [],
    components: params.components,
    explainedVariance: params.explainedVariance,
    explainedVarianceRatio: params.explainedVarianceRatio,
    scores,
    colourBy: { kind: 'classes', labels: Array.from(toFlat(data.y)), classes: data.classes },
  })
  const share = params.explainedVarianceRatio.map((r) => `${(100 * r).toFixed(1)}%`).join(', ')
  const notes = [`${fit.t} steps, ${fit.status}; variance explained by component: ${share}`]
  return { result: record, tapes: { session: fit }, notes }
}

const SPECIES = ['setosa', 'versicolor', 'virginica']

export function View({ result }: { result: Exposition }) {
  return (
    <Figure width={1000} height={476} title={meta.alt}>
      <Panel
        width={1000}
        height={420}
        title="Four measurements of 150 iris flowers, reduced to the two directions that keep most of the variance"
        still={stillOf(result, { mode: 'light', figure: figure({ width: 1000, height: 420 }) })}
      />
      {SPECIES.map((name, i) => (
        <g key={name} transform={`translate(${60 + i * 120}, 462)`}>
          <circle r={5} fill={categoricalColour(i)} />
          <text x={10} y={4} fontSize={12} fontFamily="system-ui, sans-serif" fill="#333333">
            {name}
          </text>
        </g>
      ))}
    </Figure>
  )
}
