import { atomBound, type EpAtom, marginalMean } from 'omai-engine/bayesian-examples'
import { messagePassingKnobs } from 'omai-engine/message-passing'
import { toStructure } from 'omai-engine/probprog/model'
import { type DictSpace, defaults, dict } from 'omai-engine/space'
import type { Config } from 'omai-engine/trace'
import {
  ControlPanel,
  FigurePanel,
  ModelRunPlayer,
  Player,
  SeedInput,
  SpaceControls,
  StructureView,
  usePanels,
  usePlayback,
} from 'omai-render/react'
import { modelGraph } from 'omai-render/structures'
import { type ReactNode, useEffect, useMemo, useRef, useState } from 'react'
import { BeliefChart, DataFigure, JointChart, ProbabilityBars } from './ep/AtomCharts.tsx'
import { type AtomState, useAtom } from './ep/useAtom.ts'

const EP_KNOBS = ['schedule', 'damping', 'improper', 'tol', 'maxSweeps']

/** A number to three significant digits, a dash when it is not one. */
const sig = (v: number, digits = 3): string => (Number.isFinite(v) ? String(Number(v.toPrecision(digits))) : '-')

/** The EP knobs a page offers: the library's own space for the method, without the budgets. */
const epSpace: DictSpace = dict(
  Object.fromEntries(Object.entries(messagePassingKnobs('ep').config.spaces).filter(([k]) => EP_KNOBS.includes(k))),
)

const tone = (ok: boolean): { color: string; borderColor: string } => ({
  color: ok ? 'var(--wb-success)' : 'var(--wb-danger)',
  borderColor: ok ? 'var(--wb-success)' : 'var(--wb-danger)',
})

function Badge(props: { readonly children: ReactNode; readonly color: string }): ReactNode {
  return (
    <span
      className="inline-block rounded border px-2 py-0.5 text-xs font-semibold uppercase tracking-wide"
      style={{ color: props.color, borderColor: props.color }}
    >
      {props.children}
    </span>
  )
}

/**
 * The page of one expectation propagation atom: the problem (the story, a figure of the data, the knobs, the model drawn as
 * a plate diagram with its factors named), the inference (EP's own knobs and how the run ended), what to observe (EP's
 * belief of each quantity over the exact one, the joint as contours, the beliefs moving sweep by sweep, the messages on the
 * model graph) and how we know it works (the numbers against the atom's tolerance, with a badge). An atom the library
 * refuses shows the library's verdict, where the missing rule is planned and the reference EP will be compared against,
 * and starts to run the day the rule lands, with no change to the page: the verdict is asked at run time.
 *
 * @param props - The atom.
 * @returns The page.
 */
export function EpAtomPage(props: { readonly atom: EpAtom }): ReactNode {
  const { atom } = props
  const [knobs, setKnobs] = useState<Config>(atom.defaults)
  const [ep, setEp] = useState<Config>({ ...(defaults(epSpace) as Config), ...atom.ep })
  const panels = usePanels(`ep-${atom.name}`, {
    graph: 'The model as a diagram',
    messages: 'The messages, step by step',
    numbers: 'The numbers',
  })
  const state = useAtom(atom, knobs, ep)
  const { problem, run, reference, check } = state
  const ran = run?.status === 'ran' ? run : undefined
  const frames = ran?.history ?? []
  const playback = usePlayback(Math.max(1, frames.length), { fps: 3 })
  const seek = useRef(playback.seek)
  seek.current = playback.seek
  // A new run opens on its last sweep, the answer.
  useEffect(() => {
    seek.current(Math.max(0, frames.length - 1))
  }, [frames])
  const frame = Math.min(playback.frame, Math.max(0, frames.length - 1))
  const at = frames[frame]?.beliefs
  const earlier = useMemo(() => {
    const before = frames.slice(0, frame)
    const step = Math.max(1, Math.ceil(before.length / 6))
    return before.filter((_, i) => i % step === 0)
  }, [frames, frame])
  const exactMeans = useMemo(() => (reference === undefined ? undefined : marginalMean(reference)), [reference])
  const epMeans = useMemo(
    () =>
      at === undefined
        ? undefined
        : Object.fromEntries(Object.entries(at).flatMap(([k, b]) => (b.proper ? [[k, b.mean] as const] : []))),
    [at],
  )
  const bound = useMemo(() => (problem === undefined ? undefined : atomBound(problem)), [problem])
  const graph = useMemo(() => (bound === undefined ? undefined : modelGraph(toStructure(bound))), [bound])
  const record = useMemo(
    () => (ran !== undefined && panels.active('messages') ? ran.record() : undefined),
    [ran, panels],
  )
  const qs = problem?.quantities ?? []
  const pair = problem?.pair
  const qa = pair === undefined ? undefined : qs.find((q) => q.name === pair[0])
  const qb = pair === undefined ? undefined : qs.find((q) => q.name === pair[1])
  const planned = run !== undefined && run.status !== 'ran'
  const hasKnobs = Object.keys(atom.knobs.spaces).length > 0
  const probabilities = qs.filter((q) => q.support === 'probability')
  return (
    <div className="flex flex-col gap-5">
      <Banner atom={atom} state={state} />
      <section className="flex flex-col gap-2">
        <h2 className="text-sm font-semibold">The setup</h2>
        {problem === undefined ? null : (
          <pre className="max-w-3xl overflow-x-auto rounded border border-line bg-surface px-3 py-2 font-mono text-xs leading-relaxed">
            {problem.equations.join('\n')}
          </pre>
        )}
        <div
          className={`grid grid-cols-1 items-start gap-4 ${hasKnobs && problem?.figure !== undefined ? 'lg:grid-cols-[minmax(0,20rem)_minmax(0,1fr)]' : ''}`}
        >
          {hasKnobs ? (
            <ControlPanel title="change the setup" columns={1}>
              <SpaceControls
                space={atom.knobs}
                value={knobs}
                onChange={(v) => setKnobs((k) => ({ ...k, ...v }))}
                segmentedUpTo={1}
              />
            </ControlPanel>
          ) : null}
          {problem?.figure === undefined ? null : (
            <div className="flex min-w-0 flex-col gap-1">
              <DataFigure figure={problem.figure} exact={exactMeans} ep={epMeans} />
              <p className="text-xs text-muted">
                {problem.figure.caption}
                {planned
                  ? ' Thick: the exact answer (EP does not run on this model yet).'
                  : ' Thick: the exact answer. Dashed: what EP believes at the sweep shown below.'}
              </p>
            </div>
          )}
        </div>
        <FigurePanel panels={panels} name="graph">
          {graph === undefined ? null : (
            <StructureView structure={graph} frame={{ filename: `ep-${atom.name}-model` }} />
          )}
          <p className="mt-1 text-xs text-muted">The factors this exercises: {atom.factors.join('; ')}.</p>
        </FigurePanel>
      </section>
      <section className="flex flex-col gap-2">
        <h2 className="text-sm font-semibold">The inference</h2>
        <ControlPanel title="expectation propagation: its knobs" columns={4}>
          <SpaceControls
            space={epSpace}
            value={ep}
            onChange={(v) => setEp((e) => ({ ...e, ...v }))}
            segmentedUpTo={0}
          />
        </ControlPanel>
        <p className="text-xs text-muted" role="status">
          {state.error !== undefined
            ? state.error
            : run === undefined
              ? ''
              : run.status !== 'ran'
                ? 'Expectation propagation does not run on this model (the library says why, above).'
                : `${run.converged ? 'Converged' : 'Stopped without converging'} after ${run.sweeps} sweeps (${sig(state.runMs, 2)} ms).`}
        </p>
      </section>
      <section className="flex flex-col gap-3">
        <h2 className="text-sm font-semibold">What to observe</h2>
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          {qs
            .filter((q) => q.support !== 'probability')
            .map((q) => (
              <div key={q.name} className="flex min-w-0 flex-col gap-1">
                <h3 className="text-xs font-semibold text-muted">
                  {q.label}
                  {planned ? ': the exact posterior' : ': EP over the exact posterior'}
                </h3>
                <BeliefChart
                  quantity={q}
                  reference={reference?.marginals[q.name]}
                  belief={at?.[q.name]}
                  earlier={earlier.map((f) => f.beliefs[q.name])}
                />
              </div>
            ))}
          {reference?.joint !== undefined && qa !== undefined && qb !== undefined ? (
            <div className="flex min-w-0 flex-col gap-1">
              <h3 className="text-xs font-semibold text-muted">
                {qa.label} and {qb.label} together
              </h3>
              <JointChart joint={reference.joint} a={qa} b={qb} epA={at?.[qa.name]} epB={at?.[qb.name]} />
              <p className="text-xs text-muted">
                Solid: the exact joint posterior. Dashed: EP's belief, which keeps one belief per quantity (the product
                of the two curves on the left), so it cannot lean.
              </p>
            </div>
          ) : null}
        </div>
        {probabilities.length === 0 ? null : (
          <ProbabilityBars quantities={probabilities} exact={reference?.marginals} ep={at} />
        )}
        {ran === undefined || frames.length < 2 ? null : (
          <Player controller={playback} stamp={(f) => `sweep ${frames[f]?.sweep ?? f}`} />
        )}
        <FigurePanel panels={panels} name="messages">
          {record === undefined ? (
            <p className="text-sm text-muted">
              {planned ? 'There are no messages: EP does not run on this model.' : 'Computing...'}
            </p>
          ) : (
            <ModelRunPlayer record={record} filename={`ep-${atom.name}-messages`} />
          )}
        </FigurePanel>
      </section>
      <section className="flex flex-col gap-2">
        <h2 className="text-sm font-semibold">How we know it works</h2>
        <FigurePanel panels={panels} name="numbers">
          <Check atom={atom} state={state} />
        </FigurePanel>
      </section>
    </div>
  )
}

/** The expectation, the library's verdict and where the missing rule is planned. */
function Banner(props: { readonly atom: EpAtom; readonly state: AtomState }): ReactNode {
  const { atom, state } = props
  const { run, check } = state
  const refused = run !== undefined && run.status !== 'ran'
  return (
    <div className="flex flex-col gap-1 rounded border border-line bg-surface px-3 py-2">
      <div className="flex flex-wrap items-center gap-2">
        {refused ? <Badge color="var(--wb-warning)">planned</Badge> : <Badge color="var(--wb-success)">runs</Badge>}
        {check === undefined ? null : (
          <Badge color={tone(check.pass).color}>{check.pass ? 'check passes' : 'check fails'}</Badge>
        )}
        {atom.expected === 'planned' && !refused ? (
          <Badge color="var(--wb-danger)">declared planned: flip it to runs</Badge>
        ) : null}
        {atom.expected === 'runs' && refused ? (
          <Badge color="var(--wb-danger)">declared runs, now refused</Badge>
        ) : null}
        <span className="text-xs text-muted">
          {atom.family}: {atom.factors.join('; ')}
        </span>
      </div>
      {refused && run !== undefined ? (
        <div className="flex flex-col gap-1 text-sm">
          <p>
            <b>
              {run.status === 'failed'
                ? 'EP is accepted by the compiler and the run breaks:'
                : "The library's verdict:"}
            </b>{' '}
            {run.reason}
          </p>
          {atom.plan === undefined ? null : (
            <p className="text-muted">
              <b>What is missing and where it is planned:</b> {atom.plan}
            </p>
          )}
          <p className="text-xs text-muted">
            The reference below (the exact posterior of the same model) is what EP will be compared against; when the
            rule lands this page starts to run with no change.
          </p>
        </div>
      ) : null}
    </div>
  )
}

/** The check: EP's mean and spread beside the reference's, with the tolerance and why it is that wide. */
function Check(props: { readonly atom: EpAtom; readonly state: AtomState }): ReactNode {
  const { atom, state } = props
  const { check, reference, run } = state
  if (run !== undefined && run.status !== 'ran')
    return (
      <p className="text-sm text-muted">
        There is nothing to check yet. The reference is {reference === undefined ? 'being computed' : reference.source}
        {reference?.logEvidence === undefined ? '' : ` (log evidence ${sig(reference.logEvidence, 5)})`}; when EP runs,
        each quantity is held to: mean within {atom.tolerance.meanZ} reference sd, sd ratio {atom.tolerance.sdRatio[0]}{' '}
        to {atom.tolerance.sdRatio[1]}, distribution distance (KS) at most {atom.tolerance.ks}.
      </p>
    )
  if (check === undefined) return <p className="text-sm text-muted">Computing the reference...</p>
  const t = check.tolerance
  return (
    <div className="flex flex-col gap-2">
      <div className="overflow-x-auto">
        <table className="w-full text-left text-sm tabular-nums">
          <thead className="text-xs text-muted">
            <tr>
              <th className="py-1 font-normal">quantity</th>
              <th className="font-normal">EP mean</th>
              <th className="font-normal">exact mean</th>
              <th className="font-normal">error (exact sds)</th>
              <th className="font-normal">EP sd</th>
              <th className="font-normal">exact sd</th>
              <th className="font-normal">sd ratio</th>
              <th className="font-normal">KS</th>
              <th className="font-normal" />
            </tr>
          </thead>
          <tbody>
            {check.rows.map((r) => (
              <tr key={r.name} className="border-t border-line">
                <td className="py-1">{r.label}</td>
                <td>{sig(r.ep.mean, 4)}</td>
                <td>{sig(r.refMean, 4)}</td>
                <td>{sig(r.meanZ, 2)}</td>
                <td>{sig(r.ep.sd, 4)}</td>
                <td>{sig(r.refSd, 4)}</td>
                <td>{sig(r.sdRatio, 4)}</td>
                <td>{sig(r.ks, 2)}</td>
                <td style={{ color: tone(r.pass).color }}>{r.pass ? 'pass' : 'FAIL'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="max-w-3xl text-xs text-muted">
        Tolerance: mean within {t.meanZ} exact sd, sd ratio between {t.sdRatio[0]} and {t.sdRatio[1]}, KS distance at
        most {t.ks}. {t.reason}. The reference is {reference?.source}.
      </p>
    </div>
  )
}
