import type { ReactNode } from 'react'
import { frameOf } from './shown.ts'
import type { MethodResult } from './types.ts'

const cell = 'px-2 py-1 text-right tabular-nums whitespace-nowrap'
const head = 'px-2 py-1 text-right font-medium text-muted whitespace-nowrap'

const fixed = (v: number | undefined, digits = 2): string =>
  v === undefined || !Number.isFinite(v) ? '-' : v.toFixed(digits)

const endingText = {
  converged: 'converged',
  completed: 'ran its steps',
  budget: 'budget spent',
  stopped: 'stopped',
  diverged: 'diverged',
  failed: 'failed',
} as const

/**
 * What a sampler says about its chains, and what an approximation says about its fit: the table of the methods. A sampler
 * is `converged` when its worst R-hat is under 1.05 and its smallest effective sample size over 100; a fit when its own
 * convergence test was met.
 *
 * @param props - The results.
 * @returns The table.
 */
export function MethodTable(props: { readonly results: readonly MethodResult[] }): ReactNode {
  return (
    <div className="overflow-x-auto rounded-lg border border-line bg-surface">
      <table className="w-full border-collapse text-xs">
        <thead>
          <tr className="border-b border-line">
            <th className={`${head} text-left`}>method</th>
            <th className={head}>steps</th>
            <th className={head}>time (s)</th>
            <th className={`${head} text-left`}>ended</th>
            <th className={head}>R-hat (max)</th>
            <th className={head}>ESS (min)</th>
            <th className={head}>divergent</th>
            <th className={head}>ELBO / objective</th>
            <th className={head}>test accuracy</th>
            <th className={head}>test log predictive</th>
            <th className={head}>train accuracy</th>
          </tr>
        </thead>
        <tbody>
          {props.results.map((r) => {
            const sampler = r.organisation === 'chain'
            const ok = sampler
              ? r.maxRhat !== undefined && r.minEss !== undefined && r.maxRhat < 1.05 && r.minEss > 100
              : r.ending === 'converged' || r.ending === 'completed'
            const last = r.curve === undefined ? undefined : r.curve.values[r.curve.values.length - 1]
            return (
              <tr key={r.id} className="border-b border-line last:border-b-0" title={r.message}>
                <td className="px-2 py-1 text-left text-fg whitespace-nowrap">{r.label}</td>
                <td className={cell}>{r.steps}</td>
                <td className={cell}>{(r.elapsedMs / 1000).toFixed(1)}</td>
                <td className="px-2 py-1 text-left whitespace-nowrap">
                  {endingText[r.ending]}
                  {r.draws === 0
                    ? ' in warm-up'
                    : sampler
                      ? ok
                        ? ', mixed'
                        : ', not mixed'
                      : ok
                        ? ''
                        : r.ending === 'budget'
                          ? ', not settled'
                          : ''}
                </td>
                <td className={cell}>{fixed(r.maxRhat, 3)}</td>
                <td className={cell}>{fixed(r.minEss, 0)}</td>
                <td className={cell}>{r.divergences === undefined ? '-' : r.divergences}</td>
                <td className={cell}>{fixed(last, 1)}</td>
                <td className={cell}>{fixed(r.test.accuracy, 3)}</td>
                <td className={cell}>{fixed(r.test.logPredictive, 3)}</td>
                <td className={cell}>{fixed(r.train.accuracy, 3)}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

/**
 * The mean and standard deviation of the chosen coefficients under each method, at the playhead (the last frame is the
 * method's final answer).
 *
 * @param props - The results, the coefficients and their labels, and the playhead in milliseconds.
 * @returns The table.
 */
export function CoefficientTable(props: {
  readonly results: readonly MethodResult[]
  readonly coordinates: readonly number[]
  readonly labels: readonly string[]
  readonly t: number
}): ReactNode {
  const { results, coordinates, labels, t } = props
  return (
    <div className="overflow-x-auto rounded-lg border border-line bg-surface">
      <table className="w-full border-collapse text-xs">
        <thead>
          <tr className="border-b border-line">
            <th className={`${head} text-left`}>coefficient</th>
            {results.map((r) => (
              <th key={r.id} className={head}>
                {r.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {coordinates.map((j) => (
            <tr key={j} className="border-b border-line last:border-b-0">
              <td className="px-2 py-1 text-left whitespace-nowrap text-fg">
                w{j} <span className="text-muted">{labels[j]}</span>
              </td>
              {results.map((r) => {
                const f = frameOf(r, t)
                return (
                  <td key={r.id} className={cell}>
                    {fixed(f?.mean[j], 2)} <span className="text-muted">± {fixed(f?.sd[j], 2)}</span>
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
