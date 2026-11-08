import { macros, texError } from 'omai-render/maths'
import { Choice, ControlPanel, Tex } from 'omai-render/react'
import { type ReactNode, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'TeX and the shared macros',
  question: 'How do I typeset maths with the same symbols everywhere?',
  explain:
    '`Tex` typesets with KaTeX and the shared macro registry `macros`: bold vectors (`\\xvec`), matrices (`\\Sigmamat`), sets (`\\Dcal`), operators (`\\argmin`, `\\KL`), delimiters (`\\norm{}`, `\\inner{}{}`) and distributions (`\\Gauss`, `\\Dir`). `extendMacros` adds your own; `texError` reports what does not parse. Edit the TeX below.',
}

const examples = {
  'linear regression':
    '\\hat{\\betavec} = \\argmin_{\\betavec} \\norm{\\yvec - \\Xmat\\betavec}^2 = (\\Xmat\\transpose\\Xmat)^{-1}\\Xmat\\transpose\\yvec',
  Gaussian:
    '\\xvec \\sim \\Gauss(\\muvec, \\Sigmamat), \\quad p(\\xvec) \\propto \\exp\\!\\left(-\\tfrac12 (\\xvec-\\muvec)\\transpose \\Sigmamat^{-1} (\\xvec-\\muvec)\\right)',
  ELBO: '\\log p(\\Dcal) \\ge \\expect_{q(\\zvec)}\\LB \\log p(\\Dcal, \\zvec) \\RB - \\expect_{q}\\LB\\log q(\\zvec)\\RB',
  KL: '\\KL(q \\,\\|\\, p) = \\expect_{q}\\!\\left[\\log \\frac{q(\\xvec)}{p(\\xvec)}\\right] \\ge 0',
} as const
type Name = keyof typeof examples

export default function TexAndMacros(): ReactNode {
  const [name, setName] = useState<Name>('linear regression')
  const [source, setSource] = useState<string>(examples['linear regression'])
  const error = texError(source)
  const count = Object.keys(macros).length
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={1}>
        <Choice
          label="example"
          value={name}
          options={Object.keys(examples) as Name[]}
          onChange={(n) => {
            setName(n)
            setSource(examples[n])
          }}
        />
      </ControlPanel>
      <textarea
        value={source}
        onChange={(e) => setSource(e.target.value)}
        rows={3}
        spellCheck={false}
        className="w-full rounded border border-[var(--wb-grid)] bg-[var(--wb-surface)] p-2 font-mono text-sm"
      />
      <div className="rounded border border-[var(--wb-grid)] p-4 text-lg">
        <Tex display>{source}</Tex>
      </div>
      <p className="text-sm text-[var(--wb-muted)]">
        {error === undefined ? `Parses with the ${count} shared macros.` : error}
      </p>
    </div>
  )
}
