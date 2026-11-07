import { dict, int, real, union } from 'mdml-engine/space'
import { renderToString } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { ModelSelector } from './ModelSelector.tsx'

const models = union(
  { ridge: dict({ alpha: real(1e-4, 10, { log: true }) }), tree: dict({ depth: int(1, 12) }), baseline: dict({}) },
  { label: 'model' },
)

describe('ModelSelector', () => {
  it('lists the options with the chosen one checked and a settings button on those with knobs', () => {
    const html = renderToString(
      <ModelSelector space={models} value={{ kind: 'tree', depth: 3 }} onChange={() => undefined} label="model" />,
    )
    // The group is named by its visible label, and the chosen option is the one pressed (no radio roles without arrows).
    const labelId = /aria-labelledby="([^"]+)"/.exec(html)?.[1]
    expect(html).toContain(`id="${labelId}"`)
    expect(html).not.toContain('role="radio"')
    expect(html.match(/aria-pressed/g)).toHaveLength(3)
    expect(html).toMatch(/aria-pressed="true"[^>]*>(?:<span[^>]*>)?tree/)
    expect(html).toContain('aria-label="Settings of ridge"')
    expect(html).toContain('baseline has no settings')
  })

  it('draws the options a rule rules out disabled, with the reason as their title', () => {
    const html = renderToString(
      <ModelSelector
        space={models}
        value={{ kind: 'ridge', alpha: 1 }}
        onChange={() => undefined}
        admissible={(o) => o !== 'tree'}
        why={(o) => `${o} needs a different loss`}
      />,
    )
    expect(html).toContain('title="tree needs a different loss"')
    expect(html).toContain('disabled=""')
  })
})
