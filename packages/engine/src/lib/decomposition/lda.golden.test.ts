import { goldenSuite } from '../../testing/suite.ts'
import { type Tensor, tensor, toArray } from '../tensor/index.ts'
import { type LdaLearningMethod, latentDirichletAllocation } from './lda.ts'

const fit = (inputs: readonly (Tensor | undefined)[], o: Readonly<Record<string, unknown>>) => {
  const [x, start] = inputs as Tensor[]
  const prior = (v: unknown): number | null => (typeof v === 'number' ? v : null)
  return latentDirichletAllocation
    .with({
      nComponents: (start as Tensor).shape[0] as number,
      componentsInit: toArray(start as Tensor) as number[][],
      docInit: 'constant',
      learningMethod: o['learning_method'] as LdaLearningMethod,
      maxIter: o['max_iter'] as number,
      batchSize: o['batch_size'] as number,
      learningOffset: (o['learning_offset'] as number | undefined) ?? 10,
      learningDecay: (o['learning_decay'] as number | undefined) ?? 0.7,
      docTopicPrior: prior(o['doc_topic_prior']),
      topicWordPrior: prior(o['topic_word_prior']),
      meanChangeTol: 1e-6,
      maxDocUpdateIter: 200,
    })
    .fit({ x: x as Tensor })
}

goldenSuite('LDA by variational Bayes against scikit-learn from fixed initial topics', 'decomposition/lda', {
  components: (i, o) => tensor(fit(i, o).params.components),
  docTopic: (i, o) => tensor(fit(i, o).params.docTopic),
  transformHeld: (i, o) => fit(i, o).transform(i[2] as Tensor),
  perplexity: (i, o) => fit(i, o).params.perplexity,
  perplexityHeld: (i, o) => fit(i, o).perplexity(i[2] as Tensor),
  score: (i, o) => fit(i, o).score({ x: i[0] as Tensor }),
  scoreHeld: (i, o) => fit(i, o).score({ x: i[2] as Tensor }),
})
