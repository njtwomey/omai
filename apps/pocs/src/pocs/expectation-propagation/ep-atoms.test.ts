// Every EP atom of the registry has a page in its directory (expectation-propagation, or composed-atoms for the ones that put pieces together), with the atom's title: the pages are generated from the registry.
import { epAtoms } from 'omai-engine/probprog/ep'
import { describe, expect, it } from 'vitest'
import { POCS, pageOf } from '../../registry.ts'
import { atomSource } from '../../shared/ep/atomSource.ts'
import { styleProblems } from '../../shared/styleScan.ts'

describe('the pages of the EP atoms', () => {
  it.each(epAtoms.map((a) => [a.name, a.title] as const))(
    '%s has a page in the group, titled as the atom',
    (name, title) => {
      const entry = POCS.find((p) => p.slug === `ep-${name}`)
      expect(['expectation-propagation', 'composed-atoms']).toContain(entry?.dir[0])
      expect(entry?.title).toBe(title)
      expect(pageOf(`ep-${name}`)).toBeDefined()
    },
  )

  it('has a coverage page too', () => {
    expect(POCS.find((p) => p.slug === 'ep-coverage')?.dir[0]).toBe('expectation-propagation')
  })
})

describe('the code shown on the atom pages', () => {
  it.each(epAtoms.map((a) => [a.name, a] as const))(
    '%s has a region of its own model, short, and the inference lines',
    (_, atom) => {
      const { code, file } = atomSource(atom, atom.problem(atom.defaults))
      expect(file).toMatch(/^[a-z-]+\.ts$/)
      expect(code).toMatch(/defineModel|trueskillModel/)
      expect(code).toMatch(/epRoute\(bound/)
      expect(code).toMatch(/session\(method/)
      expect(code.split('\n').length).toBeLessThan(60)
    },
  )
})

describe('the style of the code shown on the atom pages', () => {
  it.each(epAtoms.map((a) => [a.name, a] as const))(
    '%s: one statement per line, no latent inside a call, no trailing comments, short lines',
    (_, atom) => {
      expect(styleProblems(atomSource(atom, atom.problem(atom.defaults)).code)).toEqual([])
    },
  )
})
