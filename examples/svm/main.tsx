/**
 * Example "Support vectors and the margin" (in the index, not in the README): a support vector classifier with a
 * Gaussian kernel fitted twice to the same two-moons data, with a small and a large penalty C. Each panel is the
 * `model/support-vectors` record of the fitted session joined to its `model/decision-surface` record: the boundary, the
 * margins (dashed), the support vectors ringed, and the functional margin y·f(x) of every training row.
 */
import { generate } from 'mdml-engine/data'
import { moons } from 'mdml-engine/datasets'
import { type DecisionSurface, type Exposition, type MarginRecord, marginRecord } from 'mdml-engine/exposition'
import { key } from 'mdml-engine/random'
import { svc } from 'mdml-engine/svm'
import { stillOf } from 'mdml-render/views'
import type { ExampleModule, Outcome, Tape } from '../_shared/contract.ts'
import { Row } from '../_shared/figure.tsx'

export const meta: ExampleModule['meta'] = {
  title: 'Support vectors and the margin',
  summary:
    'A kernel support vector classifier with a small and a large C: the margins, and which rows are support vectors.',
  readme: false,
  alt: 'Two support vector classifiers on the two-moons data: boundary, margins and support vectors for a small and a large C',
}

interface Result {
  readonly fits: readonly { readonly title: string; readonly record: MarginRecord }[]
}

export function run(): Outcome<Result> {
  const { data, truth } = generate(moons, { knobs: { n: 100, noise: 0.25 }, key: key(0) })
  const test = generate(moons, { knobs: { n: 400, noise: 0.25 }, key: key(9) }).data
  const fits: { title: string; record: MarginRecord }[] = []
  const tapes: Record<string, Tape> = {}
  const notes = [`best possible accuracy ${(1 - (truth.bayesError ?? 0)).toFixed(3)}`]
  for (const [name, C] of [
    ['small-c', 0.2],
    ['large-c', 50],
  ] as const) {
    const fit = svc.with({ kernel: 'rbf', C }).session(data, { key: key(1) })
    fit.advance()
    // Two records of the finished fit, joined: the support vectors, and the surface with the rows they index.
    const support = fit.probe(fit.t, 'model/support-vectors') as MarginRecord
    const surface = fit.probe(fit.t, 'model/decision-surface', { resolution: 60 }) as DecisionSurface
    const { kind: _, ...rest } = support
    const record = marginRecord({ ...rest, ...(surface.rows === undefined ? {} : { rows: surface.rows }), surface })
    fits.push({ title: `C = ${C}`, record })
    tapes[name] = fit
    const model = fit.fittedAt(fit.t)
    const inside = support.margins.filter((m) => m < 1 - 1e-6).length
    notes.push(
      `C = ${C}: ${fit.t} steps, ${support.support.length} support vectors (${inside} inside the margin), accuracy ${model.score(data).toFixed(3)} on the training rows, ${model.score(test).toFixed(3)} on 400 new ones`,
    )
  }
  return { result: { fits }, tapes, notes }
}

export function View({ result }: { result: Result }) {
  return (
    <Row
      alt={meta.alt}
      width={560}
      height={340}
      panels={result.fits.map((f) => ({ title: f.title, still: stillOf(f.record as Exposition, { mode: 'light' }) }))}
    />
  )
}
