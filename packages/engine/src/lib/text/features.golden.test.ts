import { describe, expect, it } from 'vitest'
import { expectMatches, loadFixture } from '../../testing/golden.ts'
import { toDense } from '../numerics/sparse/index.ts'
import { tensor } from '../tensor/index.ts'
import {
  type Bm25Options,
  bm25,
  bm25Scores,
  type CountVectorizerOptions,
  countTransform,
  countVectorize,
  type HashingOptions,
  hashingVectorize,
  type TfidfOptions,
  tfidfTransform,
  tfidfVectorize,
} from './index.ts'

describe('text/features against scikit-learn and rank_bm25', () => {
  for (const c of loadFixture('text/features').cases) {
    it(c.name, () => {
      const docs = c.inputs[0] as string[]
      const options = c.inputs[1] as CountVectorizerOptions & TfidfOptions & HashingOptions
      switch (c.fn) {
        case 'countVectorize': {
          const { vectorizer, matrix } = countVectorize(docs, options)
          expect(vectorizer.featureNames).toEqual(c.params['names'])
          expectMatches(toDense(matrix), c.output, 0, 0, c.name)
          return
        }
        case 'countTransform':
          expectMatches(
            toDense(
              countTransform(countVectorize(docs, options).vectorizer, ['the fox and the dog', 'unseen words only']),
            ),
            c.output,
            0,
            0,
            c.name,
          )
          return
        case 'tfidfVectorize':
          expectMatches(toDense(tfidfVectorize(docs, options).matrix), c.output, c.rtol, c.atol, c.name)
          return
        case 'idf':
          expectMatches(tensor(tfidfVectorize(docs, options).vectorizer.idf), c.output, c.rtol, c.atol, c.name)
          return
        case 'tfidfTransform':
          expectMatches(
            toDense(tfidfTransform(tfidfVectorize(docs, options).vectorizer, ['the quick fox', 'unseen'])),
            c.output,
            c.rtol,
            c.atol,
            c.name,
          )
          return
        case 'hashingVectorize':
          expectMatches(toDense(hashingVectorize(docs, options)), c.output, c.rtol, c.atol, c.name)
          return
        default: {
          const corpus = c.inputs[0] as string[][]
          const query = c.inputs[1] as string[]
          const index = bm25(corpus, c.params as Bm25Options)
          expectMatches(tensor(bm25Scores(index, query)), c.output, 1e-12, 1e-14, c.name)
        }
      }
    })
  }
})
