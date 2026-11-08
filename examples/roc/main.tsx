/**
 * Example "A metric curve" (in the index, not in the README): the ROC curve of a support vector classifier on 300 held-out points of the two-moons
 * dataset. The classifier is fitted as a session (its tape is kept), and the curve comes from `mdml-engine/metrics`.
 */
import { generate } from 'mdml-engine/data'
import { moons } from 'mdml-engine/datasets'
import { rocAucScore, rocCurve } from 'mdml-engine/metrics'
import { key } from 'mdml-engine/random'
import { svc } from 'mdml-engine/svm'
import { rocChart } from 'mdml-render/charts'
import type { ExampleModule, Outcome } from '../_shared/contract.ts'
import { Figure, Panel } from '../_shared/figure.tsx'

export const meta: ExampleModule['meta'] = {
  title: 'A metric curve',
  summary: 'The ROC curve and AUC of a support vector classifier on held-out data.',
  readme: false,
  alt: 'The ROC curve of the support vector classifier on 300 test points',
}

type Curve = ReturnType<typeof rocCurve>
interface Result {
  readonly curve: Curve
  readonly auc: number
}

export function run(): Outcome<Result> {
  const train = generate(moons, { knobs: { n: 120, noise: 0.25 }, key: key(0) }).data
  const test = generate(moons, { knobs: { n: 300, noise: 0.25 }, key: key(9) }).data
  const fit = svc.with({ kernel: 'rbf', C: 1 }).session(train, { key: key(1) })
  fit.advance()
  const scores = fit.fittedAt(fit.t).decisionFunction(test.x)
  const auc = rocAucScore(test.y, scores).estimate.value
  const curve = rocCurve(test.y, scores)
  return {
    result: { curve, auc },
    tapes: { session: fit },
    notes: [`AUC ${auc.toFixed(3)}, ${curve.fpr.shape[0]} points`],
  }
}

export function View({ result }: { result: Result }) {
  const chart = rocChart([{ name: `svc (AUC ${result.auc.toFixed(3)})`, curve: result.curve }])
  return (
    <Figure width={420} height={380} title={meta.alt}>
      <Panel width={420} height={380} still={chart} />
    </Figure>
  )
}
