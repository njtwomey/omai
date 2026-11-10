import { hyphenationTables } from 'omai-engine/datasets'
import { type CrfEstimator, type CrfFitted, type CrfParams, crf, crfData } from 'omai-engine/pgm'
import { bool, choice, defaults, dict, int, real } from 'omai-engine/space'
import type { TokenTable } from 'omai-engine/text'
import type { Config, Method, Session } from 'omai-engine/trace'
import { barChart } from 'omai-render/charts'
import { figure } from 'omai-render/figure'
import { Chart, ControlPanel, SessionPanels, SpaceControls, useColourMode } from 'omai-render/react'
import { type ReactNode, useMemo, useState, useSyncExternalStore } from 'react'
import { BakedSession } from '../../shared/BakedSession.tsx'
import {
  extractorOf,
  firstWords,
  gapEvidence,
  patternLabels,
  readWord,
  type Scores,
  scoreLabels,
  type TemplateSurvival,
  templateLines,
  templateSurvival,
  type Window,
  wordSplit,
} from './hyphenation.ts'

// Steps whose held-out scores and surviving templates are prepared for playing back.
const FRAMES = 30
const CURVES = figure({ width: 300, height: 170 })
const RESULT = figure({ width: 520, height: 230 })
const PATHS = ['solver/objective', 'solver/gradient', 'solver/sequenceError', 'solver/nonzeros']

const SPACE = dict({
  train: int(50, 1500, { default: 300, log: true, label: 'training words' }),
  radius: int(1, 3, { default: 2, label: 'template window: letters on each side' }),
  classes: bool({ default: true, label: 'vowel/consonant class templates' }),
  algorithm: choice(['lbfgs', 'l2sgd', 'ap', 'pa'], {
    default: 'lbfgs',
    label: 'algorithm',
    descriptions: {
      lbfgs: 'L-BFGS on the penalised log-likelihood; with c1 above 0 it is OWL-QN, which sets weights exactly to zero',
      l2sgd: 'stochastic gradient descent on the L2-penalised log-likelihood',
      ap: 'averaged structured perceptron: no probabilities, no penalty',
      pa: 'passive-aggressive: a perceptron that moves just far enough to fix its mistake',
    },
  }),
  c1: real(0, 2, { default: 0, label: 'L1 penalty c1 (above 0: OWL-QN prunes the model)' }),
  c2: real(0, 10, { default: 1, label: 'L2 penalty c2' }),
  iterations: int(10, 300, { default: 100, label: 'most iterations' }),
  epochs: int(2, 20, { default: 6, label: 'epochs (passes over the words)' }),
})
type Controls = {
  train: number
  radius: number
  classes: boolean
  algorithm: 'lbfgs' | 'l2sgd' | 'ap' | 'pa'
  c1: number
  c2: number
  iterations: number
  epochs: number
}
const START: Controls = {
  train: 300,
  radius: 2,
  classes: true,
  algorithm: 'lbfgs',
  c1: 0,
  c2: 1,
  iterations: 100,
  epochs: 6,
}
// What an algorithm has no use for: the controls left out.
const EXCLUDED: Readonly<Record<Controls['algorithm'], string[]>> = {
  lbfgs: ['epochs'],
  l2sgd: ['epochs', 'c1'],
  ap: ['iterations', 'c1', 'c2'],
  pa: ['iterations', 'c1', 'c2'],
}

// Four curves in a row where there is room for them, two by two on a phone.
const WIDE = '(min-width: 900px)'
const subscribeWide = (notify: () => void): (() => void) => {
  const query = window.matchMedia(WIDE)
  query.addEventListener('change', notify)
  return () => query.removeEventListener('change', notify)
}
const isWide = (): boolean => window.matchMedia(WIDE).matches

const EXAMPLES = ['hyphenation', 'typography', 'conditional', 'algorithm', 'dictionary', 'structured']

/** The estimator each method was made from, so a view of an old run is read with the settings that made it. */
const estimators = new WeakMap<object, CrfEstimator<TokenTable>>()

/** What one baked step holds: the model, its held-out scores and what is left of each template. */
interface Frame {
  readonly fitted: CrfFitted<TokenTable>
  readonly scores: Scores
  readonly survival: readonly TemplateSurvival[]
}

let texScores: Scores | undefined
const tex = (): Scores => {
  const { test } = wordSplit()
  texScores ??= scoreLabels(
    test.labels,
    test.words.map((w) => patternLabels(w)),
  )
  return texScores
}

export default function CrfHyphenationPage(): ReactNode {
  const [controls, setControls] = useState<Controls>(START)
  const wide = useSyncExternalStore(subscribeWide, isWide, () => true)
  const online = controls.algorithm === 'ap' || controls.algorithm === 'pa'
  const steps = online ? controls.epochs : controls.iterations
  const window: Window = useMemo(
    () => ({ radius: controls.radius, classes: controls.classes }),
    [controls.radius, controls.classes],
  )
  const extractor = useMemo(() => extractorOf(window), [window])
  const lines = useMemo(() => templateLines(window), [window])
  const data = useMemo(() => {
    const part = firstWords(wordSplit().train, controls.train)
    return crfData(part.tables, part.labels)
  }, [controls.train])
  const method = useMemo(() => {
    const estimator = crf<TokenTable>({
      features: extractor,
      algorithm: controls.algorithm,
      c1: controls.c1,
      c2: controls.c2,
      maxIterations: steps,
    })
    const made = estimator.method(data)
    estimators.set(made, estimator)
    return made
  }, [extractor, data, controls.algorithm, controls.c1, controls.c2, steps])
  const config = useMemo(() => defaults(method.config) as Config, [method])
  const solver = controls.algorithm === 'lbfgs' && controls.c1 > 0 ? 'OWL-QN' : controls.algorithm
  return (
    <div className="flex flex-col gap-4">
      <ControlPanel title="the CRF: data, templates, algorithm and penalties" columns={3}>
        <SpaceControls
          space={SPACE}
          value={controls}
          exclude={EXCLUDED[controls.algorithm]}
          onChange={(v) => setControls(v as unknown as Controls)}
        />
      </ControlPanel>
      <BakedSession<Frame>
        method={method as unknown as Method<unknown, Config>}
        config={config}
        steps={steps}
        frames={FRAMES}
        logs="defaults"
        bake={(session, t) => {
          const estimator = estimators.get(session.method as object) as CrfEstimator<TokenTable>
          const params = (session as unknown as Session<unknown, Config>).probe(t, 'model/params') as unknown
          const fitted = estimator.revive(params as CrfParams as never)
          const { test } = wordSplit()
          return {
            fitted,
            scores: scoreLabels(test.labels, fitted.predict(test.tables)),
            survival: templateSurvival(fitted, templateLines(window)),
          }
        }}
        onReset={() => setControls(START)}
      >
        {({ live, t, frame, scrub }) => (
          <div className="flex flex-col gap-4">
            <SessionPanels
              id="crf-hyphenation"
              live={live}
              at={t >= live.head ? null : t}
              onScrub={scrub}
              figure={CURVES}
              chrome={false}
              layout={[
                {
                  kind: 'multiples',
                  panel: { kind: 'curves', paths: PATHS },
                  over: { by: 'paths' },
                  columns: wide ? 4 : 2,
                  shared: false,
                },
              ]}
            />
            {frame === undefined ? null : (
              <>
                <div className="grid grid-cols-1 items-start gap-4 lg:grid-cols-2">
                  <Result frame={frame} t={t} solver={solver} />
                  <Survival frame={frame} />
                </div>
                <WordBox frame={frame} extractor={extractor} lines={lines} />
              </>
            )}
          </div>
        )}
      </BakedSession>
      <p className="max-w-3xl text-xs text-muted">
        Each step is one iteration of the {solver} solver on a linear-chain CRF that labels every letter of a word with
        a hyphen after it (B) or not (O). The held-out words are 400 of the Moby Hyphenator list's 3984 (the
        dictionary's syllable breaks, not TeX's typesetting ones); the TeX patterns were built from another dictionary
        whose exceptions overlap this list, and may break a word anywhere here, so they are a strong baseline, not a
        held-out one. Scrub the curves to see the model at an earlier step.
      </p>
    </div>
  )
}

const pct = (v: number): string => `${(100 * v).toFixed(1)}%`

function Result(props: { readonly frame: Frame; readonly t: number; readonly solver: string }): ReactNode {
  const { frame, t, solver } = props
  const mode = useColourMode()
  const baseline = tex()
  const option = useMemo(
    () =>
      barChart(
        ['word accuracy', 'hyphen precision', 'hyphen recall', 'hyphen F1'],
        [
          { name: 'CRF', values: [frame.scores.words, frame.scores.precision, frame.scores.recall, frame.scores.f1] },
          { name: 'TeX patterns', values: [baseline.words, baseline.precision, baseline.recall, baseline.f1] },
        ],
        { mode, figure: RESULT, y: { limits: [0, 1] }, labels: true },
      ),
    [frame, baseline, mode],
  )
  return (
    <section className="flex flex-col gap-2">
      <h2 className="text-sm font-medium text-fg">Held-out words: the CRF against the TeX patterns</h2>
      <Chart option={option} figure={RESULT} />
      <dl className="grid grid-cols-2 gap-x-6 gap-y-2 text-sm sm:grid-cols-4">
        <Fact name="step" value={String(t)} />
        <Fact name="solver" value={solver} />
        <Fact name="nonzero weights" value={String(frame.fitted.params.nonzeros)} />
        <Fact name="words right: CRF, TeX" value={`${pct(frame.scores.words)}, ${pct(tex().words)}`} />
      </dl>
    </section>
  )
}

function Fact(props: { readonly name: string; readonly value: string }): ReactNode {
  return (
    <div>
      <dt className="text-xs text-muted">{props.name}</dt>
      <dd className="font-medium tabular-nums">{props.value}</dd>
    </div>
  )
}

function Survival(props: { readonly frame: Frame }): ReactNode {
  const { survival } = props.frame
  const total = survival.reduce((n, s) => n + s.features, 0)
  const alive = survival.reduce((n, s) => n + s.nonzero, 0)
  return (
    <section className="flex flex-col gap-2">
      <h2 className="text-sm font-medium text-fg">
        Which templates carry weight ({alive} of {total} features nonzero)
      </h2>
      <ul className="flex flex-col gap-1 text-xs">
        {survival.map((s) => (
          <li key={s.template} className="grid grid-cols-[minmax(0,12rem)_1fr_5rem] items-center gap-2">
            <code className={`truncate ${s.nonzero === 0 ? 'text-muted line-through' : 'text-fg'}`} title={s.template}>
              {s.template}
            </code>
            <div className="h-2 overflow-hidden rounded-full bg-bg">
              <div
                className="h-full rounded-full bg-(--wb-accent)"
                style={{ width: `${s.features === 0 ? 0 : (100 * s.nonzero) / s.features}%` }}
              />
            </div>
            <span className="text-right tabular-nums text-muted">
              {s.nonzero} / {s.features}
            </span>
          </li>
        ))}
      </ul>
      <p className="text-xs text-muted">
        Of each template's features, those with a nonzero weight. Raise c1 and OWL-QN zeroes most of them, the
        long-window templates first.
      </p>
    </section>
  )
}

const CELL = 28
const LEFT = 84
const BAR = 64
const RULES = [
  { name: 'CRF', dot: 'var(--wb-accent)' },
  { name: 'TeX', dot: 'var(--wb-fg)' },
  { name: 'dictionary', dot: 'var(--wb-muted)' },
] as const

function WordBox(props: {
  readonly frame: Frame
  readonly extractor: ReturnType<typeof extractorOf>
  readonly lines: readonly string[]
}): ReactNode {
  const { frame, extractor } = props
  const [text, setText] = useState('hyphenation')
  const [chosen, setChosen] = useState<number | null>(null)
  const word = text
    .toLowerCase()
    .replace(/[^a-z]/g, '')
    .slice(0, 24)
  const reading = useMemo(() => (word.length < 2 ? undefined : readWord(frame.fitted, word)), [frame.fitted, word])
  if (reading === undefined)
    return (
      <section className="flex flex-col gap-2">
        <WordInput
          text={text}
          onText={(v) => {
            setText(v)
            setChosen(null)
          }}
        />
        <p className="text-sm text-muted">Type a word of at least two letters.</p>
      </section>
    )
  const n = reading.letters.length
  const likeliest = reading.probability.reduce((best, p, i) => (p > (reading.probability[best] ?? 0) ? i : best), 0)
  const gap = chosen !== null && chosen < n - 1 ? chosen : likeliest
  const marks: Record<string, readonly boolean[]> = {
    CRF: hyphensOf(reading.crf),
    TeX: hyphensOf(reading.tex),
    dictionary: reading.dictionary === undefined ? [] : hyphensOf(reading.dictionary),
  }
  return (
    <section className="flex flex-col gap-3">
      <WordInput
        text={text}
        onText={(v) => {
          setText(v)
          setChosen(null)
        }}
      />
      <div className="overflow-x-auto">
        <div className="relative" style={{ width: LEFT + n * CELL + 8 }}>
          {reading.probability.map((p, i) => (
            <button
              // biome-ignore lint/suspicious/noArrayIndexKey: gaps are positional
              key={i}
              type="button"
              aria-label={`gap ${i + 1}: ${Math.round(100 * p)}% hyphen`}
              onClick={() => setChosen(i)}
              className="absolute top-0 z-10 cursor-pointer bg-transparent"
              style={{ left: LEFT + (i + 1) * CELL - CELL / 2, width: CELL, height: BAR + 18 }}
            />
          ))}
          <svg
            width={LEFT + n * CELL + 8}
            height={BAR + 40 + RULES.length * 18}
            role="img"
            aria-label={`Hyphen probability in each gap of ${word}`}
            className="max-w-none"
          >
            {reading.probability.map((p, i) => {
              const x = LEFT + (i + 1) * CELL
              return (
                // biome-ignore lint/suspicious/noArrayIndexKey: gaps are positional
                <g key={i}>
                  <rect
                    x={x - 6}
                    y={BAR - p * BAR}
                    width={12}
                    height={Math.max(1, p * BAR)}
                    rx={2}
                    fill="var(--wb-accent)"
                    opacity={i === gap ? 1 : 0.55}
                  />
                  <text x={x} y={BAR + 12} textAnchor="middle" fontSize={9} fill="var(--wb-muted)">
                    {Math.round(100 * p)}
                  </text>
                  {i === gap ? <rect x={x - 8} y={BAR - 2} width={16} height={2} fill="var(--wb-fg)" /> : null}
                </g>
              )
            })}
            {reading.letters.map((c, i) => (
              <text
                // biome-ignore lint/suspicious/noArrayIndexKey: letters are positional
                key={i}
                x={LEFT + i * CELL + CELL / 2}
                y={BAR + 32}
                textAnchor="middle"
                fontSize={18}
                fill="var(--wb-fg)"
              >
                {c}
              </text>
            ))}
            <text x={LEFT - 8} y={BAR - 2} textAnchor="end" fontSize={11} fill="var(--wb-muted)">
              P(hyphen), %
            </text>
            {RULES.map((r, row) => (
              <g key={r.name}>
                <text x={LEFT - 8} y={BAR + 52 + row * 18} textAnchor="end" fontSize={11} fill="var(--wb-muted)">
                  {r.name}
                </text>
                {(marks[r.name] ?? []).map((on, i) =>
                  on ? (
                    // biome-ignore lint/suspicious/noArrayIndexKey: gaps are positional
                    <circle key={i} cx={LEFT + (i + 1) * CELL} cy={BAR + 48 + row * 18} r={5} fill={r.dot} />
                  ) : null,
                )}
              </g>
            ))}
          </svg>
        </div>
      </div>
      <dl className="grid grid-cols-1 gap-x-6 gap-y-1 text-sm sm:grid-cols-3">
        <Fact name="CRF" value={reading.crf} />
        <Fact name="TeX patterns" value={reading.tex} />
        <Fact name="dictionary" value={reading.dictionary ?? 'not in the word list'} />
      </dl>
      <Evidence frame={frame} extractor={extractor} word={word} gap={gap} probability={reading.probability[gap] ?? 0} />
    </section>
  )
}

const hyphensOf = (hyphenated: string): boolean[] => {
  const out: boolean[] = []
  let letters = 0
  for (const ch of hyphenated) {
    if (ch === '-') out[letters - 1] = true
    else {
      letters++
      out[letters - 1] = false
    }
  }
  return out.slice(0, -1)
}

function WordInput(props: { readonly text: string; readonly onText: (v: string) => void }): ReactNode {
  return (
    <div className="flex flex-wrap items-center gap-2 text-sm">
      <label className="flex items-center gap-2">
        <span className="font-medium text-fg">Try a word</span>
        <input
          value={props.text}
          onChange={(e) => props.onText(e.target.value)}
          spellCheck={false}
          autoCapitalize="none"
          className="w-44 rounded-md border border-line bg-bg px-2 py-1 text-fg"
        />
      </label>
      {EXAMPLES.map((w) => (
        <button
          key={w}
          type="button"
          onClick={() => props.onText(w)}
          className="cursor-pointer rounded-md border border-line bg-bg px-2 py-1 text-xs text-fg hover:bg-surface"
        >
          {w}
        </button>
      ))}
    </div>
  )
}

function Evidence(props: {
  readonly frame: Frame
  readonly extractor: ReturnType<typeof extractorOf>
  readonly word: string
  readonly gap: number
  readonly probability: number
}): ReactNode {
  const { frame, extractor, word, gap, probability } = props
  const items = useMemo(() => {
    return gapEvidence(frame.fitted, extractor, hyphenationTables([word])[0] as TokenTable, gap, 8)
  }, [frame.fitted, extractor, word, gap])
  const biggest = Math.max(1e-9, ...items.map((e) => Math.abs(e.push)))
  const letters = [...word]
  return (
    <div className="flex flex-col gap-1 text-xs">
      <h3 className="text-sm font-medium text-fg">
        Why {Math.round(100 * probability)}% between “{letters.slice(0, gap + 1).join('')}” and “
        {letters.slice(gap + 1).join('')}”?
      </h3>
      <p className="text-muted">
        The features that fire at “{letters[gap]}”, by how far they move the log-odds towards a hyphen (right, accent)
        or away from it (left). Click another gap to ask again.
      </p>
      <ul className="flex flex-col gap-1">
        {items.map((e) => (
          <li key={e.feature} className="grid grid-cols-[minmax(0,13rem)_1fr_3.5rem] items-center gap-2">
            <code className="truncate text-fg" title={e.feature}>
              {e.feature}
            </code>
            <div className="relative h-2 rounded-full bg-bg">
              <div
                className={`absolute top-0 h-full rounded-full ${e.push > 0 ? 'bg-(--wb-accent)' : 'bg-(--wb-muted)'}`}
                style={{
                  width: `${(50 * Math.abs(e.push)) / biggest}%`,
                  ...(e.push > 0 ? { left: '50%' } : { right: '50%' }),
                }}
              />
            </div>
            <span className="text-right tabular-nums text-muted">{e.push.toFixed(2)}</span>
          </li>
        ))}
      </ul>
    </div>
  )
}
