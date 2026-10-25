/**
 * The text attribution entry: word-level ablation, Shapley values and LIME for a small sentiment scorer with negation.
 */
import { explainText, type TextMethod } from 'whitebox-engine/methods/explain'
import { choice, dict } from 'whitebox-engine/space'
import { defineEntry, type HubEntry } from '../../entry/entry.ts'

const LEXICON: Readonly<Record<string, number>> = {
  good: 1,
  great: 1.5,
  fine: 0.5,
  fun: 0.9,
  moving: 1,
  bad: -1,
  awful: -1.5,
  dull: -0.8,
  boring: -1,
}

/** A logistic score of the summed lexicon values of a text's words, each negated by a preceding "not". */
export function lexiconSentiment(texts: readonly string[]): number[] {
  return texts.map((t) => {
    const words = t.toLowerCase().split(/\s+/)
    let s = 0
    words.forEach((w, i) => {
      s += (LEXICON[w] ?? 0) * (words[i - 1] === 'not' ? -1 : 1)
    })
    return 1 / (1 + Math.exp(-2 * s))
  })
}

const TEXTS = [
  'The plot was not bad and the cast was great fun',
  'A dull film with awful dialogue but a fine score',
  'Not boring at all and genuinely moving',
  'Not good not great just fine',
]

/** Word attributions of a text classifier. */
export const textAttributions: HubEntry = defineEntry({
  id: 'explainability/text-attributions',
  title: 'Word attributions',
  summary:
    'Which words of a review raise or lower a sentiment score: ablation, Shapley values and LIME over its words.',
  collection: 'explainability',
  tags: ['text', 'tokens', 'SHAP', 'LIME', 'feature ablation', 'perturbation'],
  datasets: [],
  params: dict({
    text: choice(TEXTS, { default: TEXTS[0] ?? '', label: 'review' }),
    method: choice(['shap', 'ablation', 'lime'], { default: 'shap', label: 'explainer' }),
  }),
  run: ({ params, key }) =>
    explainText(lexiconSentiment, String(params['text']), {
      method: params['method'] as TextMethod,
      output: 'P(positive)',
      key,
    }),
  views: [{ record: 'featureAttributions', view: 'TokenStripView', title: 'Each word’s push' }],
  methods: [{ file: 'methods/explain/perturbation.ts', name: 'explainText' }],
  references: [
    'Ribeiro, Singh and Guestrin (2016), “Why should I trust you?”: explaining the predictions of any classifier.',
    'Lundberg and Lee (2017), A unified approach to interpreting model predictions.',
  ],
})
