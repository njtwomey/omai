import { WhiteboxError } from 'whitebox-engine/base'
import type { CorpusData } from 'whitebox-engine/data'
import { documentTermCounts, latentDirichletAllocation } from 'whitebox-engine/methods/models'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

const inner = fitEstimator(latentDirichletAllocation({ nTopics: 5, nIter: 100 }))

/** Latent Dirichlet allocation by collapsed Gibbs sampling: topics as word distributions, documents as mixtures. */
export const lda: HubEntry = defineEntry({
  id: 'probabilistic/topic/lda',
  title: 'Latent Dirichlet allocation',
  summary: 'Every document mixes a few topics and every topic is a distribution over words; Gibbs sampling finds both.',
  collection: 'probabilistic/topic',
  tags: ['topic model', 'Gibbs sampling', 'text', 'admixture'],
  datasets: ['textCorpora'],
  params: inner.params,
  run: (input) => {
    const corpus = input.data as CorpusData | undefined
    if (corpus?.kind !== 'corpus') throw new WhiteboxError('value', 'lda: needs a corpus')
    return inner.run({ ...input, data: documentTermCounts(corpus) as never })
  },
  views: [
    { record: 'components', view: 'TopicWordsView', title: 'Topics by their most probable words' },
    { record: 'coefficientTable', title: 'Top words per topic', area: 'side' },
    { record: 'mixtureSteps', title: 'Gibbs sweeps', area: 'below' },
  ],
  methods: [{ file: 'methods/models/topic.ts', name: 'latentDirichletAllocation' }],
  references: ['Blei, Ng and Jordan (2003)', 'Griffiths and Steyvers (2004), Finding scientific topics'],
})
