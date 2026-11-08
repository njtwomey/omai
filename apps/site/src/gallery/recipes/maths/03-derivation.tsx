import { ControlPanel, Derivation, NumberField } from 'omai-render/react'
import { type ReactNode, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Derivations',
  question: 'How do I show the steps of a derivation, aligned?',
  explain:
    '`Derivation` (or `derivationTex` for the TeX alone) sets steps as an `aligned` block: a step with `&=` starts a line, a bare right-hand side continues it, and each step may carry a note. Here, the steps of completing the square with live coefficients.',
}

const fmt = (v: number): string => String(Number(v.toPrecision(3)))

export default function DerivationRecipe(): ReactNode {
  const [a, setA] = useState(2)
  const [b, setB] = useState(4)
  const h = b / (2 * a)
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <NumberField label="a" value={a} onChange={setA} min={0.5} max={5} />
        <NumberField label="b" value={b} onChange={setB} min={-6} max={6} />
      </ControlPanel>
      <div className="text-lg">
        <Derivation
          steps={[
            { tex: `${fmt(a)}x^2 + ${fmt(b)}x &= ${fmt(a)}\\left(x^2 + ${fmt(b / a)}x\\right)`, note: 'factor out a' },
            {
              tex: `${fmt(a)}\\left(x^2 + ${fmt(b / a)}x + ${fmt(h * h)} - ${fmt(h * h)}\\right)`,
              note: 'add and subtract $(b/2a)^2$',
            },
            { tex: `${fmt(a)}\\left(x + ${fmt(h)}\\right)^2 - ${fmt(a * h * h)}`, note: 'complete the square' },
          ]}
        />
      </div>
    </div>
  )
}
