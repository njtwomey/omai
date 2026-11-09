import { afterEach, describe, expect, it, vi } from 'vitest'

// A second loaded copy of omai-engine (two bundler chunks, a duplicate from a dependency optimiser) is modelled by
// resetting the module graph and importing again: the second import evaluates every module afresh.
afterEach(() => {
  vi.resetModules()
})

describe('a second copy of the engine', () => {
  it('shares the default context with the first, so a seed set by one is the seed of both', async () => {
    const first = await import('./default.ts')
    const before = first.defaultContext()
    vi.resetModules()
    const second = await import('./default.ts')
    expect(second).not.toBe(first)
    // Both copies see the one default, and replacing it through either is seen by the other.
    expect(second.defaultContext()).toBe(before)
    const { createContext } = await import('./context.ts')
    const mine = createContext({ seed: 7 })
    const previous = second.setDefaultContext(mine)
    expect(first.defaultContext()).toBe(mine)
    expect(first.setDefaultContext(previous)).toBe(mine)
    expect(second.defaultContext()).toBe(before)
  })

  it('loads without error, and its registries answer lookups, for the entries that build one at import', async () => {
    const load = async (): Promise<readonly [string, string]> => {
      const datasets = await import('../datasets/index.ts')
      const environments = await import('../environments/index.ts')
      const base = await import('../base/index.ts')
      const moons = base.lookup(datasets.datasetRegistry, 'moons')
      const taxi = base.lookup(environments.environmentRegistry, 'taxi')
      return [moons.name, taxi.name]
    }
    const first = await load()
    vi.resetModules()
    const second = await load()
    expect(second).toEqual(first)
    expect(second).toEqual(['moons', 'taxi'])
  }, 30_000)
})
