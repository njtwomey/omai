import { goldenSuite } from '../../testing/suite.ts'
import { type Tensor, tensor, toArray } from '../tensor/index.ts'
import { dictLearning } from './dictionary-learning.ts'
import { dictLearningOnline } from './dictionary-learning-online.ts'
import { miniBatchDictionaryLearning } from './dictionary-learning-online.ts'
import { type SparseCodeAlgorithm, type SparseCodeOptions, sparseCode, splitSign } from './sparse-coding.ts'

const rows = (t: Tensor | undefined): number[][] => toArray(t as Tensor) as number[][]

const batch = (i: readonly (Tensor | undefined)[], p: Readonly<Record<string, unknown>>) =>
  dictLearning(rows(i[0]), 5, p['alpha'] as number, {
    maxIter: p['maxIter'] as number,
    tol: 0,
    fitAlgorithm: p['fitAlgorithm'] as SparseCodeAlgorithm,
  })

goldenSuite('sparse coding against SparseCoder of scikit-learn', 'decomposition/dictionary_learning', {
  sparseCode: (i, p) => sparseCode(i[0], i[1] as Tensor, { ...(p as SparseCodeOptions), tol: 1e-12 }),
  splitCoder: (i, p) =>
    splitSign(sparseCode(i[0], i[1] as Tensor, { algorithm: 'omp', nNonzeroCoefs: p['nNonzeroCoefs'] as number })),
  batchDictionary: (i, p) => tensor(batch(i, p).dictionary),
  batchCode: (i, p) => tensor(batch(i, p).code),
  batchObjective: (i, p) => tensor([...batch(i, p).errors]),
  onlineDictionary: (i, p) =>
    tensor(
      dictLearningOnline(rows(i[0]), 6, {
        alpha: p['alpha'] as number,
        maxIter: p['maxIter'] as number,
        batchSize: p['batchSize'] as number,
        shuffle: false,
        tol: 0,
        maxNoImprovement: null,
        dictInit: rows(i[1]),
      }).dictionary,
    ),
  onlineCode: (i, p) =>
    tensor(
      dictLearningOnline(rows(i[0]), 6, {
        alpha: p['alpha'] as number,
        maxIter: p['maxIter'] as number,
        batchSize: p['batchSize'] as number,
        shuffle: false,
        tol: 0,
        maxNoImprovement: null,
        dictInit: rows(i[1]),
      }).code,
    ),
  onlineSteps: (i, p) =>
    dictLearningOnline(rows(i[0]), 6, {
      alpha: p['alpha'] as number,
      maxIter: p['maxIter'] as number,
      batchSize: p['batchSize'] as number,
      shuffle: false,
      tol: 0,
      maxNoImprovement: null,
      dictInit: rows(i[1]),
    }).nSteps,
  onlineBand: (i, p) => {
    const x = rows(i[0])
    const fit = miniBatchDictionaryLearning
      .with({
        nComponents: p['components'] as number,
        alpha: p['alpha'] as number,
        maxIter: p['maxIter'] as number,
        batchSize: p['batchSize'] as number,
        tol: 0,
        maxNoImprovement: null,
        transformAlgorithm: 'lassoLars',
      })
      .fit({ x })
    const code = rows(fit.transform(x))
    return squaredError(x, code, fit.params.components)
  },
  onlineObjective: (i, p) => {
    const x = rows(i[0])
    const fit = miniBatchDictionaryLearning
      .with({
        nComponents: p['components'] as number,
        alpha: p['alpha'] as number,
        maxIter: p['maxIter'] as number,
        batchSize: p['batchSize'] as number,
        tol: 0,
        maxNoImprovement: null,
        transformAlgorithm: 'lassoLars',
      })
      .fit({ x })
    const code = rows(fit.transform(x))
    return 0.5 * squaredError(x, code, fit.params.components) + (p['alpha'] as number) * code.flat().reduce((a, v) => a + Math.abs(v), 0)
  },
})

function squaredError(x: number[][], code: number[][], components: number[][]): number {
  let s = 0
  for (const [r, row] of x.entries())
    for (const [q, v] of row.entries()) {
      let fit = 0
      for (const [j, c] of (code[r] ?? []).entries()) fit += c * (components[j]?.[q] ?? 0)
      s += (v - fit) ** 2
    }
  return s
}
