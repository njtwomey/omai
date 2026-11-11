// Every EP atom of the registry has a page of the group, with the atom's title: the pages are generated from the registry.
import { epAtoms } from 'omai-engine/bayesian-examples'
import { describe, expect, it } from 'vitest'
import { POCS, pageOf } from '../registry.ts'

describe('the pages of the EP atoms', () => {
  it.each(epAtoms.map((a) => [a.name, a.title] as const))(
    '%s has a page in the group, titled as the atom',
    (name, title) => {
      const entry = POCS.find((p) => p.slug === `ep-${name}`)
      expect(entry?.group).toBe('expectation-propagation')
      expect(entry?.title).toBe(title)
      expect(pageOf(`ep-${name}`)).toBeDefined()
    },
  )

  it('has a coverage page too', () => {
    expect(POCS.find((p) => p.slug === 'ep-coverage')?.group).toBe('expectation-propagation')
  })
})
