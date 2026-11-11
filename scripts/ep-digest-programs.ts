/**
 * The EP programs in the digest: the programs of the pocs (`apps/pocs/src/pocs/ep-program-*`), each run once on its default
 * simulated data as the page runs it (`infer` returns the bound model and the finished session), digested as an atom is: the
 * compiled graphs, the verdict, the beliefs and evidence. They live in the pocs, so this module imports them from there: it is
 * the one place that knows where they are, and the digest says `unavailable` for a program whose module cannot be imported
 * (moved into the library, renamed), which fails the check until this file points at the new place. Excluded: the programs'
 * reference samplers (Gibbs, NUTS, exact filters: they are the pocs' own validation, not EP) and the extended changepoint
 * calibration over seeds (`calibration.ts`).
 */

import { epSpecMethod } from '../packages/engine/src/lib/probprog/inference/index.ts'
import type { BoundModel } from '../packages/engine/src/lib/probprog/model/index.ts'
import type { Posterior } from '../packages/engine/src/lib/probprog/posterior.ts'
import {
  canonical,
  finish,
  type GraphStore,
  graphsOf,
  type Item,
  type J,
  posteriorOf,
  routeOf,
} from './ep-digest-core.ts'

/** What every program's `infer` returns that the digest reads. */
interface Inferred {
  readonly bound: BoundModel
  readonly run: { readonly state: unknown; readonly config: unknown; readonly status?: string; readonly t: number }
}

interface Case {
  readonly key: string
  readonly load: () => Promise<Inferred>
}

const pocs = (dir: string): string => `../apps/pocs/src/pocs/${dir}/program.ts`

// A dynamic import takes a path built at run time: it keeps a missing module an `unavailable` item instead of a crash.
// The modules are the pocs' own, untyped here on purpose: the digest reads only `simulate`, `infer` and the defaults.
// biome-ignore lint/suspicious/noExplicitAny: a module of another app, read by name
type Program = Record<string, any>
const load = async (dir: string): Promise<Program> => (await import(pocs(dir))) as Program

const CASES: readonly Case[] = [
  {
    key: 'changepoint',
    load: async () => {
      const m = await load('ep-program-changepoint')
      const sim = m['simulate'](1)
      return m['infer'](sim.y, sim.stay)
    },
  },
  {
    key: 'changepoint-extended',
    load: async () => {
      const m = await load('ep-program-changepoint-extended')
      return m['infer'](m['simulate'](1).y)
    },
  },
  ...(['ignore', 'means', 'data', 'rows'] as const).map(
    (variant): Case => ({
      key: `circular-${variant}`,
      load: async () => {
        const m = await load('ep-program-circular')
        return m['infer'](variant, 2, m['simulate'](1, 'two', 50))
      },
    }),
  ),
  {
    key: 'cox',
    load: async () => {
      const m = await load('ep-program-cox')
      const sim = m['simulate'](1, { amplitude: 0.8, lengthScale: 1.5 })
      return m['infer'](sim.counts, sim.prior)
    },
  },
  {
    key: 'kalman',
    load: async () => {
      const m = await load('ep-program-kalman')
      return m['infer'](m['simulate'](m['defaultSetup']).y, m['defaultSetup'])
    },
  },
  {
    key: 'skills',
    load: async () => {
      const m = await load('ep-program-skills')
      return m['infer'](m['simulate'](m['defaultSetup']))
    },
  },
  ...(['gaussian', 'student-t'] as const).map(
    (likelihood): Case => ({
      key: `sparse-${likelihood}`,
      load: async () => {
        const m = await load('ep-program-sparse')
        return m['infer'](m['sparseModel'](likelihood, m['defaultSetup'].noise), m['simulate'](m['defaultSetup']))
      },
    }),
  ),
  {
    key: 'sparse-mixture',
    load: async () => {
      const m = await load('ep-program-sparse')
      return m['infer'](m['sparseMixtureModel'](), m['simulate'](m['defaultSetup']))
    },
  },
  ...([true, false] as const).map(
    (robust): Case => ({
      key: `tracker-${robust ? 'robust' : 'plain'}`,
      load: async () => {
        const m = await load('ep-program-tracker')
        const noise = { sd: 0.5, outliers: 0.1 }
        return m['infer'](m['simulate'](1, noise).y, noise, robust)
      },
    }),
  ),
  {
    key: 'zero-inflated',
    load: async () => {
      const m = await load('ep-program-zero-inflated')
      return m['infer'](m['simulate']({ ...m['defaultSetup'], rows: 60 }))
    },
  },
]

/** The cases as digest items: `program/<name>`. */
export function programItems(): { readonly key: string; readonly run: (store: GraphStore) => Promise<Item> }[] {
  return CASES.map((c) => ({
    key: `program/${c.key}`,
    run: async (store: GraphStore): Promise<Item> => {
      const key = `program/${c.key}`
      let inferred: Inferred
      try {
        inferred = await c.load()
      } catch (e) {
        return finish('program', { unavailable: e instanceof Error ? e.message : String(e) } as J, {})
      }
      const { bound, run } = inferred
      const { route, verdict } = routeOf(bound)
      const method = epSpecMethod(bound)
      const post = method.posterior?.(run.state as never, run.config as never) as Posterior | undefined
      const p = post === undefined ? { exact: { noPosterior: true }, num: {} } : posteriorOf(post)
      const exact = {
        graphs: graphsOf(key, bound, store),
        route,
        verdict,
        posterior: p.exact,
        sessionStatus: run.status ?? null,
        steps: run.t,
      }
      return finish('program', JSON.parse(canonical(exact)) as J, JSON.parse(canonical({ posterior: p.num })) as J)
    },
  }))
}
