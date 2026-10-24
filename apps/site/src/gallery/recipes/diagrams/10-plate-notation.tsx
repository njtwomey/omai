import { type ReactNode, useState } from 'react'
import { plateModel, toStructure } from 'whitebox-engine/pgm'
import { type Structure, structure } from 'whitebox-engine/scene'
import { Choice, ControlPanel, StructureView } from 'whitebox-render/react'
import { nestPlates } from 'whitebox-render/structures'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Plate notation',
  question: 'How do I draw nested plates, as in latent Dirichlet allocation?',
  explain:
    'A group with a `parent` is drawn inside its parent, so plates nest: in LDA the words of a document sit in a plate inside the documents’ plate, and the topics have a plate of their own. A plate model from whitebox-engine’s `pgm` (`toStructure`) names its plates by the index sets they combine; `nestPlates` turns those names into nested plates with their sizes.',
}

const lda = structure(
  'bayes-net',
  [
    { id: 'alpha', label: '$\\alpha$', role: 'parameter', detail: 'prior on topic proportions' },
    { id: 'theta', label: '$\\theta_d$', role: 'latent', group: 'D', detail: 'topic proportions of document d' },
    { id: 'z', label: '$z_{dn}$', role: 'latent', group: 'N', detail: 'topic of word n' },
    { id: 'w', label: '$w_{dn}$', role: 'observed', group: 'N', detail: 'word n of document d' },
    { id: 'phi', label: '$\\varphi_k$', role: 'latent', group: 'K', detail: 'word distribution of topic k' },
    { id: 'beta', label: '$\\beta$', role: 'parameter', detail: 'prior on topics' },
  ],
  [
    { from: 'alpha', to: 'theta' },
    { from: 'theta', to: 'z' },
    { from: 'z', to: 'w' },
    { from: 'phi', to: 'w' },
    { from: 'beta', to: 'phi' },
  ],
  {
    groups: [
      { id: 'D', label: 'documents', kind: 'plate', count: 'D' },
      { id: 'N', label: 'words', kind: 'plate', count: 'N', parent: 'D' },
      { id: 'K', label: 'topics', kind: 'plate', count: 'K' },
    ],
  },
)

// Students sit exams; each pass depends on the student's skill and the exam's difficulty.
const exams = nestPlates(
  toStructure(
    plateModel([
      { name: 'skill', states: ['low', 'high'], plates: ['student'], parents: [], cpt: [0.5, 0.5] },
      { name: 'difficulty', states: ['easy', 'hard'], plates: ['exam'], parents: [], cpt: [0.6, 0.4] },
      {
        name: 'pass',
        states: ['no', 'yes'],
        plates: ['student', 'exam'],
        parents: ['skill', 'difficulty'],
        cpt: [0.4, 0.6, 0.8, 0.2, 0.05, 0.95, 0.3, 0.7],
      },
    ]),
    ['pass'],
  ),
  { student: 'S', exam: 'E' },
)

const MODELS: Readonly<Record<string, Structure>> = { 'latent Dirichlet allocation': lda, 'students and exams': exams }

export default function PlateNotation(): ReactNode {
  const [name, setName] = useState('latent Dirichlet allocation')
  return (
    <div className="flex w-fit max-w-full flex-col gap-3">
      <ControlPanel>
        <Choice label="model" value={name} options={Object.keys(MODELS)} onChange={setName} />
      </ControlPanel>
      <StructureView structure={MODELS[name] ?? lda} frame={{ filename: 'plates' }} />
    </div>
  )
}
