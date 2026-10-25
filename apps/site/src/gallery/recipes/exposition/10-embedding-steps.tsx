import { iris } from 'mdml-engine/datasets'
import { standardScaler } from 'mdml-engine/learn/preprocessing'
import { tsneSession, umapSession } from 'mdml-engine/methods/manifold'
import { key } from 'mdml-engine/random'
import type { Algorithm } from 'mdml-engine/trace'
import { lines, type Panel, points } from 'mdml-render/charts'
import { figure, gridSpec } from 'mdml-render/figure'
import {
  Choice,
  ControlPanel,
  FigureFrame,
  FigureGrid,
  NumberField,
  Player,
  useColourMode,
  useLiveRun,
} from 'mdml-render/react'
import { type ReactNode, useMemo, useRef, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Embedding steps',
  question: 'How do I watch a nonlinear embedding being optimised, live?',
  explain:
    '`tsneSession(data, hyper, key)` and `umapSession(…)` set a method up without running it — the calibrated affinities (or the fuzzy graph) and the initial layout — and return the optimiser as an `Algorithm`, the same one `tsne` and `umap` run when they fit. `useLiveRun` steps it a few iterations per frame as the player plays, so the embedding learns in front of you: early exaggeration pulls the species apart, then they settle. Steps already taken are kept, so you can scrub back. The perplexity (or the number of neighbours) sets how local the picture is.',
}

// region
const raw = iris()
const data = { ...raw, x: standardScaler().fit(raw).transform(raw.x) }
const labels = Array.from(raw.y.data, (c) => raw.classes[c] ?? String(c))
const FIGURE = figure({ width: 760, height: 380 })
const GRID = gridSpec(1, 2, { widthRatios: [3, 2], wspace: 0.12 })

type Method = 't-SNE' | 'UMAP'
interface Live {
  readonly y: readonly (readonly number[])[]
  readonly objective: number
}

// An algorithm's states read as positions and an objective, keeping each inner state so scrubbing back is exact.
function reading<S>(a: Algorithm<S>, read: (s: S) => Live): Algorithm<Live & { readonly inner: S }> {
  return {
    name: a.name,
    config: a.config,
    init: (k) => {
      const s = a.init(k)
      return { inner: s, ...read(s) }
    },
    step: (state, ctx) => {
      const s = a.step(state.inner, ctx)
      return { inner: s, ...read(s) }
    },
  }
}

// A method's session as one live algorithm, so t-SNE and UMAP run the same way.
function liveAlgorithm(method: Method, locality: number): { algorithm: Algorithm<Live>; maxSteps: number } {
  if (method === 't-SNE') {
    const s = tsneSession(data, { perplexity: locality, maxIter: 500 }, key(0))
    return { maxSteps: s.maxSteps, algorithm: reading(s.algorithm, (st) => ({ y: st.y, objective: st.kl })) }
  }
  const s = umapSession(data, { nNeighbors: Math.round(locality), nEpochs: 200 }, key(0))
  return { maxSteps: s.maxSteps, algorithm: reading(s.algorithm, (st) => ({ y: st.y, objective: st.crossEntropy })) }
}

export default function EmbeddingStepsRecipe(): ReactNode {
  const mode = useColourMode()
  const [method, setMethod] = useState<Method>('t-SNE')
  const [locality, setLocality] = useState(20)
  const { algorithm, maxSteps } = useMemo(() => liveAlgorithm(method, locality), [method, locality])
  const run = useLiveRun(algorithm, { maxSteps, stepsPerFrame: method === 't-SNE' ? 5 : 2 })
  // The objective of every step seen so far, for the curve (reset with each new run).
  const history = useRef<{ for: unknown; values: Map<number, number> }>({ for: null, values: new Map() })
  if (history.current.for !== algorithm) history.current = { for: algorithm, values: new Map() }
  history.current.values.set(run.t, run.state.objective)
  const objectiveName = method === 't-SNE' ? 'KL divergence' : 'cross-entropy'

  const panels = useMemo<Record<string, Panel>>(() => {
    const ts = [...history.current.values.keys()].sort((a, b) => a - b)
    const vs = ts.map((t) => history.current.values.get(t) ?? Number.NaN)
    return {
      r0c0: {
        layers: [
          points([{ x: run.state.y.map((r) => r[0] ?? 0), y: run.state.y.map((r) => r[1] ?? 0), groups: labels }], {
            mode,
          }),
        ],
        options: { title: `${method} · step ${run.t}`, x: { label: 'dimension 1' }, y: { label: 'dimension 2' } },
      },
      r0c1: {
        layers: [
          lines([{ x: ts, y: vs, name: objectiveName }], { mode }),
          points([{ x: [run.t], y: [run.state.objective], name: 'now', size: 9 }], { mode, colourOffset: 3 }),
        ],
        options: {
          title: objectiveName,
          x: { label: 'step', limits: [0, maxSteps] },
          y: { label: objectiveName },
          legend: false,
        },
      },
    }
  }, [run.state, run.t, method, objectiveName, maxSteps, mode])

  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <Choice label="method" value={method} onChange={setMethod} options={['t-SNE', 'UMAP'] as Method[]} />
        <NumberField
          label={method === 't-SNE' ? 'perplexity' : 'neighbours'}
          value={locality}
          onChange={setLocality}
          min={2}
          max={50}
          integer
        />
      </ControlPanel>
      <FigureFrame figure={FIGURE} playback={run.controller} filename="embedding-steps">
        <FigureGrid figure={FIGURE} grid={GRID} panels={panels} />
      </FigureFrame>
      <Player
        controller={run.controller}
        stamp={() => `step ${run.t} / ${maxSteps}${run.status === undefined ? '' : ` (${run.status})`}`}
      />
    </div>
  )
}
// endregion
