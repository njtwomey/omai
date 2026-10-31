import { goldenSuite } from '../../testing/suite.ts'
import { type Tensor, tensor, toArray } from '../tensor/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import { barnesHutKl, jointProbabilities, sparseJointProbabilities, tsneAlgorithm, tsneGradient } from './tsne.ts'

const rowsOf = (t: Tensor | undefined): number[][] => toArray(t as Tensor) as number[][]

const sparseOf = (dense: number[][]) => {
  const indptr = [0]
  const indices: number[] = []
  const values: number[] = []
  for (const row of dense) {
    for (const [j, v] of row.entries())
      if (v !== 0) {
        indices.push(j)
        values.push(v)
      }
    indptr.push(indices.length)
  }
  return { n: dense.length, indptr, indices, values }
}

goldenSuite('t-SNE against scikit-learn', 'manifold/tsne', {
  joint: ([x], o) => tensor(jointProbabilities(x as Tensor, o['perplexity'] as number).p),
  jointNn: ([x], o) => {
    const { p } = sparseJointProbabilities(x as Tensor, o['perplexity'] as number)
    const dense = Array.from({ length: p.n }, () => new Array<number>(p.n).fill(0))
    for (let i = 0; i < p.n; i++)
      for (let e = p.indptr[i] ?? 0; e < (p.indptr[i + 1] ?? 0); e++)
        (dense[i] as number[])[p.indices[e] ?? 0] = p.values[e] ?? 0
    return tensor(dense)
  },
  kl: ([p, y]) => {
    const { kl, grad } = tsneGradient(rowsOf(p), rowsOf(y))
    return tensor([kl, ...grad.flat()])
  },
  klBh: ([p, y]) => {
    const rows = rowsOf(y)
    const grad = new Float64Array(rows.length * 2)
    const kl = barnesHutKl(sparseOf(rowsOf(p)), Float64Array.from(rows.flat()), 2, 0, 1, grad)
    return tensor([kl, ...Array.from(grad)])
  },
  descent: ([p, y], o) => {
    const out = runAlgorithm(
      tsneAlgorithm(rowsOf(p), rowsOf(y), {
        learningRate: o['learningRate'] as number,
        earlyExaggeration: o['exaggeration'] as number,
        exaggerationIter: o['exaggerationIter'] as number,
        minGradNorm: 0,
      }),
      { maxSteps: o['maxIter'] as number },
    )
    return tensor(out.state.y)
  },
  descentError: ([p, y], o) =>
    runAlgorithm(
      tsneAlgorithm(rowsOf(p), rowsOf(y), {
        learningRate: o['learningRate'] as number,
        earlyExaggeration: o['exaggeration'] as number,
        exaggerationIter: o['exaggerationIter'] as number,
        minGradNorm: 0,
      }),
      { maxSteps: o['maxIter'] as number },
    ).state.kl,
})
