import { generate } from 'mdml-engine/data'
import { sineCurve } from 'mdml-engine/datasets'
import { gpRegression } from 'mdml-engine/gp'
import { constantKernel, productKernel, rbfKernel } from 'mdml-engine/kernels'
import { key } from 'mdml-engine/random'
const { data } = generate(sineCurve, { knobs: { n: 14, noise: 0.15, design: 'sorted' }, key: key(0) })
const fitKernel = productKernel(constantKernel({ value: 1 }), rbfKernel({ lengthScale: 0.5 }))
const fit = gpRegression.with({ kernel: fitKernel, alpha: 0.03 }).session(data, { key: key(1) })
fit.advance()
for (const t of [0, fit.t]) {
  const f = fit.fittedAt(t)
  console.log(t, JSON.stringify(f.params))
  const post: any = fit.probe(t, 'model/posterior', { resolution: 5, samples: 1, seed: 3 })
  console.log(JSON.stringify(post).slice(0, 1500))
}
