import { type ReactNode, useMemo, useState } from 'react'
import { recordsOf } from 'whitebox-engine/exposition'
import { explainText, type TextMethod } from 'whitebox-engine/methods/explain'
import { key } from 'whitebox-engine/random'
import { Choice, ControlPanel, Select, TokenStripView } from 'whitebox-render/react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Token attributions over a text',
  question: 'How do I show which words of a text drove a classifier’s score?',
  explain:
    '`explainText` takes any function that scores texts and explains one text word by word: ablation drops each word in turn, SHAP gives the Shapley values of the words (every coalition of a short text), LIME fits a weighted linear model to random deletions. The record is laid out as tokens, so `TokenStripView` writes the text with each word on a background from blue (lowers the score) to red (raises it). The scorer here is a tiny sentiment lexicon with negation, so “not” flips the word after it: an interaction the attributions must share out.',
}

// region
const LEXICON: Readonly<Record<string, number>> = {
  good: 1,
  great: 1.5,
  fine: 0.5,
  bad: -1,
  awful: -1.5,
  dull: -0.8,
  fun: 0.9,
}
// A logistic score of the summed lexicon values, each negated by a preceding "not".
function sentiment(texts: readonly string[]): number[] {
  return texts.map((t) => {
    const words = t.toLowerCase().split(/\s+/)
    let s = 0
    words.forEach((w, i) => {
      s += (LEXICON[w] ?? 0) * (words[i - 1] === 'not' ? -1 : 1)
    })
    return 1 / (1 + Math.exp(-2 * s))
  })
}
const texts = [
  'The plot was not bad and the cast was great fun',
  'A dull film with awful dialogue but a fine score',
  'Not good not great just fine',
]
const methods: TextMethod[] = ['shap', 'ablation', 'lime']

export default function TokenStripRecipe(): ReactNode {
  const [text, setText] = useState(0)
  const [method, setMethod] = useState<TextMethod>('shap')
  const record = useMemo(
    () =>
      recordsOf(
        explainText(sentiment, texts[text] ?? '', { method, output: 'P(positive)', key: key(1) }).exposition,
        'featureAttributions',
      )[0],
    [text, method],
  )
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <Select label="text" value={text} onChange={setText} options={texts.map((t, i) => ({ value: i, label: t }))} />
        <Choice label="explainer" value={method} onChange={setMethod} options={methods} />
      </ControlPanel>
      {record !== undefined && <TokenStripView record={record} />}
    </div>
  )
}
// endregion
