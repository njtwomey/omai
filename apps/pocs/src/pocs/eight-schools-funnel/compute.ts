import { type ChainDiagnostics, chainDiagnostics } from 'omai-engine/exposition'
import { posteriorChains } from 'omai-engine/probprog'
import { type MethodModel, nutsMethod } from 'omai-engine/probprog/inference'
import { applyCollapse, collapsePlan } from 'omai-engine/probprog/model'
import { eightSchoolsCentred, eightSchoolsNonCentred } from 'omai-engine/probprog/models'
import { chainEssCurves, ess, rhat } from 'omai-engine/sampling'
import type { AnytimeRunSpec } from '../../shared/useAnytimeRuns.ts'

/** The three forms of the model, in the order the page lists them. */
export const FORMS = ['centred', 'non-centred', 'collapsed'] as const
export type Form = (typeof FORMS)[number]

/** What a run keeps at a snapshot: the diagnostics record of its chains and the numbers the comparison table shows. */
export interface Kept {
  readonly record: ChainDiagnostics | undefined
  readonly divergences: number
  /** Draws per chain so far. */
  readonly draws: number
  /** Bulk ESS and R-hat of the spread of the effects, tau. */
  readonly essTau: number
  readonly rhatTau: number
}

const CENTRED = eightSchoolsCentred.bound()
const MODELS = {
  centred: CENTRED,
  'non-centred': eightSchoolsNonCentred.bound(),
  // The school effects integrated out in closed form: a program over mu and tau only.
  // (a program closed over its data: its type says `undefined` where a method takes `void`).
  collapsed: applyCollapse(CENTRED, collapsePlan(CENTRED)).model as unknown as MethodModel,
} as const

const WARMUP = 150

/** Reads the chains of a snapshot: the quantities the funnel shows (mu, log tau and the first school's effect), with their R-hat and ESS. */
function keep(posterior: ReturnType<NonNullable<ReturnType<typeof nutsMethod>['posterior']>>, final: boolean): Kept {
  const c = posteriorChains(posterior, { sites: ['mu', 'tau', 'theta'] })
  const tau = c.draws['tau']
  const draws = tau?.[0]?.length ?? 0
  if (tau === undefined || draws < 8)
    return { record: undefined, divergences: 0, draws, essTau: Number.NaN, rhatTau: Number.NaN }
  const quantities: Record<string, number[][]> = {
    mu: c.draws['mu'] ?? [],
    'log tau': tau.map((row) => row.map((v) => Math.log(Math.max(v, 1e-300)))),
    // The collapsed form has no school effects to show: its funnel is gone with them.
    ...(c.draws['theta[0]'] === undefined ? {} : { 'theta[0]': c.draws['theta[0]'] }),
  }
  const summary = Object.entries(quantities).map(([name, rows]) => ({
    site: name,
    name,
    rhat: rhat(rows),
    ess: ess(rows),
  }))
  const divergences = (c.divergent ?? []).reduce((a, row) => a + row.reduce((b, v) => b + (v > 0 ? 1 : 0), 0), 0)
  return {
    record: chainDiagnostics(quantities, {
      summary,
      ...(c.divergent === undefined ? {} : { divergent: c.divergent }),
      ...(c.energy === undefined ? {} : { energy: c.energy }),
      maxDraws: 400,
      // The ESS curves cost a little: made once, for the run's last state.
      ...(final ? { essCurves: chainEssCurves(quantities, { points: 6, probs: [0.1, 0.25, 0.5, 0.75, 0.9] }) } : {}),
    }),
    divergences,
    draws,
    essTau: ess(tau),
    rhatTau: rhat(tau),
  }
}

/** The three runs: NUTS on each form, four chains of the same warm-up, as many draws as the budget allows (up to 800 a chain). */
export function schoolsRuns(): AnytimeRunSpec<Kept>[] {
  return FORMS.map((form) => {
    const method = nutsMethod(MODELS[form])
    return {
      id: form,
      label: form,
      method,
      config: { numChains: 3, numWarmup: WARMUP, numDraws: 800, targetAcceptance: 0.9, maxTreeDepth: 6 },
      seed: 3,
      derive: (s, x) =>
        keep(method.posterior?.(x.state as never, s.config as never) as never, x.final || x.status !== undefined),
    }
  })
}
