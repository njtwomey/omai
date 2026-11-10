import { chainDiagnostics, type ModelRun, type ModelRunStep, modelRun, type NodeState } from 'omai-engine/exposition'
import { defineModel, toStructure } from 'omai-engine/probprog/model'
import * as d from 'omai-engine/probprog/model/dists'
import { chainEssCurves } from 'omai-engine/sampling'
import { ChainDiagnosticsView, Choice, ControlPanel, ModelRunPlayer } from 'omai-render/react'
import { modelGraph } from 'omai-render/structures'
import { type ReactNode, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Inference on a model graph',
  question: 'How do I show what a run believes of each variable of a model, on the picture of the model?',
  explain:
    'The plate diagram of a model (`toStructure`, drawn with its plates crossing where they share a dimension, sizes as `S = 40`, data as dots) takes a `ModelRun` record: at each step the belief of each node is written in it with a small interval glyph under it, the messages that passed are on the edges, and the change and the evidence are charted below with the improper-update count. The same record serves a finished tape and a run in progress. For samplers, `ChainDiagnosticsView` draws the trace plots, the rank and ECDF-difference plots (with the simultaneous band), the autocorrelation, the effective sample size by quantile and as the draws accumulate, the energy with the E-BFMI, the pair plot with the divergent transitions marked and the R-hat table.',
  shows: ['modelRun', 'chainDiagnostics'],
}

// Toy data throughout: the records are made here, not run.
const exams = defineModel('exams', (b) => {
  const student = b.range('student', { size: 40 })
  const exam = b.range('exam', { size: 4 })
  const mu = b.latent('mu', d.normal(0, 1))
  const skill = b.latent('skill', d.normal(mu, 1), { dims: [student] })
  const difficulty = b.latent('difficulty', d.normal(0, 1), { dims: [exam] })
  const score = b.data('y', { dims: [student, exam], role: 'response' })
  b.observe('y', d.normal(skill.sub(difficulty), 1), score)
})
// The four difficulties are drawn one by one, so each can carry its own belief.
const picture = modelGraph(toStructure(exams), { expand: { difficulty: 4 } })

const TRUE_DIFFICULTY = [-0.9, -0.2, 0.4, 1.0]

// A little generator, so the toy data is the same on every visit.
function noise(seed: number): () => number {
  let s = seed
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0
    const u = (s + 1) / 4294967297
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0
    const v = (s + 1) / 4294967297
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v)
  }
}

function toyRun(): ModelRun {
  const steps: ModelRunStep[] = []
  const change: number[] = []
  const evidence: number[] = []
  const improper: number[] = []
  const at: number[] = []
  for (let t = 1; t <= 12; t++) {
    const r = 0.62 ** t
    const nodes: Record<string, NodeState> = {
      mu: { mean: 0.35 * (1 - r), sd: 0.2 + 0.8 * r, text: '' },
      skill: { mean: 0.1 * (1 - r), sd: 0.3 + 0.6 * r, text: '' },
    }
    for (const [i, truth] of TRUE_DIFFICULTY.entries())
      nodes[`difficulty[${i}]`] = { mean: truth * (1 - r), sd: 0.15 + 0.85 * r, text: '' }
    const k = t % 4
    steps.push({
      step: t,
      nodes,
      messages: [
        {
          from: 'y',
          to: `difficulty[${k}]`,
          text: `N(μ = ${(TRUE_DIFFICULTY[k] ?? 0).toFixed(2)}, σ² = ${(0.5 + r).toFixed(2)})`,
        },
        { from: 'y', to: 'skill', text: `N(μ = ${(0.1 * (1 - r)).toFixed(2)}, σ² = ${(1 + 2 * r).toFixed(2)})` },
      ],
    })
    at.push(t)
    change.push(1.2 * r ** 1.4)
    evidence.push(-92 + 11 * (1 - r))
    improper.push(t > 3 && t < 7 ? 1 : t >= 7 ? 2 : 0)
  }
  return modelRun({
    structure: picture,
    status: 'converged',
    series: [
      { name: 'change', label: 'largest change of a belief', steps: at, values: change, direction: 'lower' },
      { name: 'evidence', label: 'log evidence estimate', steps: at, values: evidence, direction: 'higher' },
      { name: 'improper', label: 'improper updates', steps: at, values: improper },
    ],
    steps: steps.map((s) => ({
      ...s,
      nodes: Object.fromEntries(
        Object.entries(s.nodes).map(([id, n]) => [
          id,
          { ...n, text: `${(n.mean ?? 0).toFixed(2)} ± ${(n.sd ?? 0).toFixed(2)}` },
        ]),
      ),
    })),
  })
}

function toyChains(): ReturnType<typeof chainDiagnostics> {
  const draw = (seed: number, centre: number, spread: number, drift = 0): number[] => {
    const g = noise(seed)
    let x = centre
    return Array.from({ length: 300 }, (_, i) => {
      x = centre + drift * Math.sin(i / 40) + 0.8 * (x - centre) + spread * g()
      return x
    })
  }
  const quantity = (
    seed: number,
    centre: number,
    spread: number,
    offsets: readonly number[] = [0, 0, 0, 0],
  ): number[][] => offsets.map((o, c) => draw(seed + 17 * c, centre + o, spread, o === 0 ? 0 : 0.4))
  const draws = {
    mu: quantity(1, 0.35, 0.2),
    'difficulty[0]': quantity(5, -0.9, 0.15),
    'difficulty[2]': quantity(9, 0.4, 0.15, [0, 0, 0.55, 0.9]),
    'difficulty[3]': quantity(13, 1.0, 0.16),
  }
  const g = noise(77)
  // The energy of each chain: a random walk that mostly forgets itself between draws (the E-BFMI near 1) on three chains,
  // and one that remembers (a low E-BFMI) on the fourth, the chain with the trouble.
  const energy = [0.2, 0.2, 0.2, 0.93].map((memory, c) => {
    let e = 0
    return Array.from({ length: 300 }, () => {
      e = memory * e + Math.sqrt(1 - memory * memory) * g() + (c === 3 ? 0.02 : 0)
      return e
    })
  })
  return chainDiagnostics(draws, {
    bins: 12,
    summary: [
      { site: 'mu', name: 'mu', rhat: 1.002, ess: 910 },
      { site: 'difficulty', name: 'difficulty[0]', rhat: 1.004, ess: 740 },
      { site: 'difficulty', name: 'difficulty[2]', rhat: 1.31, ess: 38 },
      { site: 'difficulty', name: 'difficulty[3]', rhat: 1.006, ess: 690 },
    ],
    divergent: [[], [], Array.from({ length: 300 }, (_, i) => (i === 120 || i === 121 ? 1 : 0)), []],
    energy,
    essCurves: chainEssCurves(draws, { points: 8 }),
  })
}

const RUN = toyRun()
const CHAINS = toyChains()

export default function ModelInference(): ReactNode {
  const [what, setWhat] = useState<'beliefs' | 'chains'>('beliefs')
  return (
    <div className="flex w-fit max-w-full flex-col gap-3">
      <ControlPanel>
        <Choice label="show" value={what} options={['beliefs', 'chains'] as const} onChange={setWhat} />
      </ControlPanel>
      {what === 'beliefs' ? (
        <ModelRunPlayer record={RUN} filename="model-inference" />
      ) : (
        <ChainDiagnosticsView record={CHAINS} />
      )}
    </div>
  )
}
