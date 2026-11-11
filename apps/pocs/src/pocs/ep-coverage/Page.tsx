import { type CoverageRow, epAtom, epCoverage, verifyAtom } from 'omai-engine/bayesian-examples'
import { type ReactNode, useCallback, useEffect, useRef, useState } from 'react'
import { Link } from '../../../../site/src/shell/Link.tsx'
import { useDeferred } from '../../shared/useDeferred.ts'

const sig = (v: number): string => (Number.isFinite(v) ? String(Number(v.toPrecision(2))) : '-')

/** The outcome of running an atom's check: pass or fail with the worst error in exact sds. */
type Outcome =
  | { readonly pass: boolean; readonly worstZ: number; readonly worstKs: number }
  | { readonly error: string }

function Status(props: { readonly row: CoverageRow }): ReactNode {
  const { row } = props
  const ok = row.status === 'runs'
  const off = row.declared !== undefined && row.declared !== row.status
  return (
    <span style={{ color: off ? 'var(--wb-danger)' : ok ? 'var(--wb-success)' : 'var(--wb-warning)' }}>
      {row.status}
      {off ? ` (declared ${row.declared})` : ''}
    </span>
  )
}

export default function EpCoverage(): ReactNode {
  const coverage = useDeferred(epCoverage)
  const [outcomes, setOutcomes] = useState<Readonly<Record<string, Outcome>>>({})
  const [running, setRunning] = useState(false)
  const stop = useRef(false)
  const runChecks = useCallback((names: readonly string[]) => {
    stop.current = false
    setRunning(true)
    let i = 0
    const next = (): void => {
      const name = names[i++]
      if (name === undefined || stop.current) {
        setRunning(false)
        return
      }
      let outcome: Outcome
      try {
        const v = verifyAtom(epAtom(name))
        outcome =
          v.check === undefined
            ? { error: 'EP did not run' }
            : {
                pass: v.check.pass,
                worstZ: Math.max(...v.check.rows.map((r) => r.meanZ)),
                worstKs: Math.max(...v.check.rows.map((r) => r.ks)),
              }
      } catch (e) {
        outcome = { error: e instanceof Error ? e.message : String(e) }
      }
      setOutcomes((o) => ({ ...o, [name]: outcome }))
      setTimeout(next, 20)
    }
    setTimeout(next, 20)
  }, [])
  useEffect(
    () => () => {
      stop.current = true
    },
    [],
  )
  if (coverage === undefined)
    return <p className="text-sm text-muted">Compiling every setup and every benchmark model...</p>
  const runnable = coverage.atoms.filter((r) => r.status === 'runs').map((r) => r.name)
  return (
    <div className="flex flex-col gap-6">
      <section className="flex flex-col gap-2">
        <h2 className="text-sm font-semibold">
          Setups: EP runs on {coverage.atomsRun} of {coverage.atoms.length}
        </h2>
        <div className="flex items-center gap-3">
          <button
            type="button"
            className="rounded border border-line px-3 py-1 text-xs hover:bg-surface disabled:opacity-50"
            disabled={running}
            onClick={() => runChecks(runnable)}
          >
            {running ? 'Running the checks...' : `Run the ${runnable.length} checks`}
          </button>
          <span className="text-xs text-muted">
            Each check runs EP and compares it with the exact answer; a few take a second or two.
          </span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead className="text-xs text-muted">
              <tr>
                <th className="py-1 font-normal">setup</th>
                <th className="font-normal">family</th>
                <th className="font-normal">status</th>
                <th className="font-normal">check</th>
                <th className="font-normal">the factors, or the library’s reason</th>
              </tr>
            </thead>
            <tbody>
              {coverage.atoms.map((r) => {
                const o = outcomes[r.name]
                return (
                  <tr key={r.name} className="border-t border-line align-top">
                    <td className="py-1 pr-3">
                      <Link to={`/poc/ep-${r.name}`} className="underline">
                        {r.title}
                      </Link>
                    </td>
                    <td className="pr-3 text-muted">{r.group}</td>
                    <td className="pr-3">
                      <Status row={r} />
                    </td>
                    <td className="pr-3 tabular-nums">
                      {o === undefined ? (
                        <span className="text-muted">-</span>
                      ) : 'error' in o ? (
                        <span style={{ color: 'var(--wb-danger)' }}>{o.error}</span>
                      ) : (
                        <span style={{ color: o.pass ? 'var(--wb-success)' : 'var(--wb-danger)' }}>
                          {o.pass ? 'pass' : 'FAIL'} (mean {sig(o.worstZ)} sd, KS {sig(o.worstKs)})
                        </span>
                      )}
                    </td>
                    <td className="max-w-xl text-xs text-muted">
                      {r.status === 'runs' ? r.factors.join('; ') : r.reason}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </section>
      <section className="flex flex-col gap-2">
        <h2 className="text-sm font-semibold">
          Benchmark models: EP runs on {coverage.benchmarksRun} of {coverage.benchmarks.length}
        </h2>
        <p className="max-w-3xl text-xs text-muted">
          Each benchmark model is compiled to a factor graph and the EP method is asked for its verdict (no run): the
          audit of docs/probprog/ep-plan.md section 1, now a tested figure that only goes up.
        </p>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead className="text-xs text-muted">
              <tr>
                <th className="py-1 font-normal">model</th>
                <th className="font-normal">status</th>
                <th className="font-normal">the library’s reason</th>
              </tr>
            </thead>
            <tbody>
              {coverage.benchmarks.map((r) => (
                <tr key={r.name} className="border-t border-line align-top">
                  <td className="py-1 pr-3">{r.title}</td>
                  <td className="pr-3">
                    <Status row={r} />
                  </td>
                  <td className="max-w-2xl text-xs text-muted">{r.reason}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  )
}
