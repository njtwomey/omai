import { valueAndGrad } from '../../../lib/autodiff/index.ts'
import { WhiteboxError } from '../../../lib/base/index.ts'
import {
  coefficientTable,
  diagramRecord,
  type Explained,
  type Exposition,
  learningCurve,
} from '../../../lib/exposition/index.ts'
import type { Params } from '../../../lib/nn/index.ts'
import { type Key, key as makeKey } from '../../../lib/random/index.ts'
import {
  type Structure,
  type StructureEdge,
  type StructureNode,
  type StructureOverlay,
  structure,
} from '../../../lib/scene/index.ts'
import {
  mul,
  relu,
  sigmoid,
  square,
  sub,
  sum,
  type Tensor,
  tanh,
  tensor,
  toFlat,
  type Value,
} from '../../../lib/tensor/index.ts'
import { mlpNetwork } from './mlp.ts'

/** The activations a hand-worked network may use. */
export type BackpropActivation = 'logistic' | 'tanh' | 'relu'

/** Options of {@link backpropByHand}. */
export interface BackpropOptions {
  /** Layer widths, input first (default [2, 2, 1]). */
  readonly sizes?: readonly number[]
  /** The hidden and output activation (default 'logistic', as in the textbook derivations). */
  readonly activation?: BackpropActivation
  /** The training rows (default the four rows of XOR). */
  readonly x?: readonly (readonly number[])[]
  /** One target vector per row (default XOR's). */
  readonly y?: readonly (readonly number[])[]
  /** Gradient-descent steps, one row per step in turn (default 8). */
  readonly steps?: number
  readonly learningRate?: number
  /** The initialisation key (scikit-learn's uniform initialisation, as {@link mlpNetwork}). */
  readonly key?: Key
}

/** One unit's forward and backward quantities at one step. */
export interface UnitValues {
  readonly id: string
  readonly layer: number
  readonly unit: number
  /** The pre-activation z = Σ w·a + b (undefined for inputs). */
  readonly z?: number
  /** The activation a = σ(z) (the input value for inputs). */
  readonly a: number
  /** δ = ∂L/∂z, by the chain rule (undefined for inputs). */
  readonly delta?: number
}

/** One parameter's gradient at one step, by hand and by reverse-mode autodiff. */
export interface ParameterGradient {
  /** E.g. `W1[0,1]` (layer 1's weight from input unit 1 to unit 0) or `b2[0]`. */
  readonly name: string
  readonly value: number
  readonly byHand: number
  readonly autodiff: number
}

/** One step of the worked example. */
export interface BackpropStep {
  readonly step: number
  /** The training row used. */
  readonly row: number
  readonly target: readonly number[]
  /** L = ½ Σ (a - y)². */
  readonly loss: number
  readonly units: readonly UnitValues[]
  readonly gradients: readonly ParameterGradient[]
  /** The largest |by hand - autodiff| over the parameters. */
  readonly maxError: number
}

/** What {@link backpropByHand} returns. */
export interface BackpropResult {
  readonly sizes: readonly number[]
  readonly activation: BackpropActivation
  readonly steps: readonly BackpropStep[]
  /** The network as a diagram: a node per unit and the loss, an edge per weight. */
  readonly structure: Structure
  /**
   * Overlays to play over the structure: for each step, the forward pass layer by layer (z and a on each unit), the
   * loss, then the backward pass from the output (δ on each unit).
   */
  readonly frames: readonly StructureOverlay[]
  /** A caption per frame. */
  readonly captions: readonly string[]
  /** The largest disagreement between the hand-worked and the autodiff gradients over every step. */
  readonly maxError: number
}

const XOR_X = [
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1],
]
const XOR_Y = [[0], [1], [1], [0]]

function act(name: BackpropActivation, z: number): number {
  if (name === 'logistic') return 1 / (1 + Math.exp(-z))
  if (name === 'tanh') return Math.tanh(z)
  return Math.max(0, z)
}

// The derivative of the activation, from its output (and input, for the ReLU).
function actPrime(name: BackpropActivation, z: number, a: number): number {
  if (name === 'logistic') return a * (1 - a)
  if (name === 'tanh') return 1 - a * a
  return z > 0 ? 1 : 0
}

const unitId = (layer: number, unit: number, layers: number): string =>
  layer === 0 ? `x${unit}` : layer === layers - 1 ? `y${unit}` : `h${layer}_${unit}`

function networkStructure(sizes: readonly number[]): Structure {
  const nodes: StructureNode[] = []
  const edges: StructureEdge[] = []
  const L = sizes.length
  for (const [l, width] of sizes.entries())
    for (let j = 0; j < width; j++)
      nodes.push({
        id: unitId(l, j, L),
        label: l === 0 ? `x${j + 1}` : l === L - 1 ? `ŷ${j + 1}` : `h${l},${j + 1}`,
        role: l === 0 ? 'input' : l === L - 1 ? 'output' : 'operation',
        group: `layer${l}`,
      })
  nodes.push({ id: 'loss', label: 'L = ½ Σ (ŷ − y)²', role: 'output' })
  for (let l = 1; l < L; l++)
    for (let j = 0; j < (sizes[l] ?? 0); j++)
      for (let k = 0; k < (sizes[l - 1] ?? 0); k++)
        edges.push({ from: unitId(l - 1, k, L), to: unitId(l, j, L), label: `w${l}[${j},${k}]`, directed: true })
  for (let j = 0; j < (sizes[L - 1] ?? 0); j++) edges.push({ from: unitId(L - 1, j, L), to: 'loss', directed: true })
  return structure('computation-graph', nodes, edges, {
    groups: sizes.map((_, l) => ({
      id: `layer${l}`,
      label: l === 0 ? 'input' : l === L - 1 ? 'output' : `hidden ${l}`,
      kind: 'cluster' as const,
    })),
    title: 'A small network, worked by hand',
    direction: 'right',
  })
}

const fmt = (v: number): string => (Math.abs(v) < 1e-3 && v !== 0 ? v.toExponential(2) : v.toFixed(3))

/**
 * Backpropagation worked by hand on a small network, step by step beside reverse-mode autodiff: each step takes one
 * training row through the forward pass (every unit's pre-activation z and activation a), computes the loss
 * L = ½ Σ (ŷ − y)², runs the backward pass by the chain rule (δ = (a − y)·σ'(z) at the output, δ = σ'(z)·Σ w·δ below,
 * ∂L/∂w = δ·a of the unit below, ∂L/∂b = δ), checks every gradient against `valueAndGrad` of the same network, and
 * takes a gradient-descent step. The result carries the network's structure with overlays to play (forward values,
 * then deltas) and every quantity per step; the exposition is the gradient table of the first step (by hand, with the
 * autodiff disagreement as its error column), the loss per step against the largest disagreement (on its own axis),
 * and the network as a `diagram` with every frame's overlay and caption, to play over it.
 *
 * @param options - Widths, activation, rows, targets, steps, learning rate and key.
 * @returns The steps, the structure and its frames, and the exposition records.
 */
export function backpropByHand(options: BackpropOptions = {}): Explained<BackpropResult> {
  const sizes = options.sizes ?? [2, 2, 1]
  const activation = options.activation ?? 'logistic'
  const xs = options.x ?? XOR_X
  const ys = options.y ?? XOR_Y
  if (sizes.length < 2) throw new WhiteboxError('value', 'backpropByHand: needs at least an input and an output layer')
  if (xs.length !== ys.length || xs.length === 0)
    throw new WhiteboxError('value', 'backpropByHand: needs as many targets as rows, and at least one row')
  const L = sizes.length
  const lr = options.learningRate ?? 0.5
  // The same initial weights as the library network: W[l][j][k], b[l][j] for l = 1..L-1.
  const net = mlpNetwork(sizes, activation, options.key ?? makeKey(0))
  const names = net
    .children()
    .filter(([, m]) => m.kind === 'Linear')
    .map(([n]) => n)
  const W: number[][][] = [[]]
  const b: number[][] = [[]]
  for (const [i, name] of names.entries()) {
    const p = net.params[name] as { readonly weight: Tensor; readonly bias: Tensor }
    const inn = sizes[i] ?? 0
    const w = toFlat(p.weight)
    W.push(Array.from({ length: sizes[i + 1] ?? 0 }, (_, j) => w.slice(j * inn, (j + 1) * inn)))
    b.push(toFlat(p.bias))
  }
  // The autodiff side: the same computation on tensors, the output layer squashed by the activation too.
  const autodiffLoss = (params: Params, x: readonly number[], y: readonly number[]): Value => {
    let a: Value = tensor([[...x]])
    for (const name of names) {
      const layer = net.children().find(([n]) => n === name)?.[1] as { apply(p: Params, v: Value): Value }
      a = ACTIVATION[activation](layer.apply(params[name] as Params, a))
    }
    return mul(sum(square(sub(a, tensor([[...y]])))), 0.5)
  }
  const steps: BackpropStep[] = []
  const frames: StructureOverlay[] = []
  const captions: string[] = []
  const nSteps = options.steps ?? 8
  for (let t = 0; t < nSteps; t++) {
    const row = t % xs.length
    const x = xs[row] as readonly number[]
    const y = ys[row] as readonly number[]
    // Forward.
    const z: number[][] = [[]]
    const a: number[][] = [[...x]]
    for (let l = 1; l < L; l++) {
      const zl = (W[l] as number[][]).map((wj, j) =>
        wj.reduce((s, w, k) => s + w * (a[l - 1]?.[k] ?? 0), b[l]?.[j] ?? 0),
      )
      z.push(zl)
      a.push(zl.map((v) => act(activation, v)))
    }
    const out = a[L - 1] as number[]
    const loss = 0.5 * out.reduce((s, v, j) => s + (v - (y[j] ?? 0)) ** 2, 0)
    // Backward.
    const delta: number[][] = Array.from({ length: L }, () => [])
    delta[L - 1] = out.map((v, j) => (v - (y[j] ?? 0)) * actPrime(activation, z[L - 1]?.[j] ?? 0, v))
    for (let l = L - 2; l >= 1; l--)
      delta[l] = (a[l] as number[]).map(
        (v, k) =>
          actPrime(activation, z[l]?.[k] ?? 0, v) *
          (W[l + 1] as number[][]).reduce((s, wj, j) => s + (wj[k] ?? 0) * (delta[l + 1]?.[j] ?? 0), 0),
      )
    // Autodiff, on the same parameters.
    const params: Params = Object.fromEntries(
      names.map((name, i) => [
        name,
        { weight: tensor((W[i + 1] as number[][]).map((r) => [...r])), bias: tensor([...(b[i + 1] as number[])]) },
      ]),
    )
    const vg = valueAndGrad((p: Params) => autodiffLoss(p, x, y))(params)
    const gradients: ParameterGradient[] = []
    for (let l = 1; l < L; l++) {
      const g = vg.grad[names[l - 1] as string] as { readonly weight: Tensor; readonly bias: Tensor }
      const gw = toFlat(g.weight)
      const gb = toFlat(g.bias)
      const inn = sizes[l - 1] ?? 0
      for (let j = 0; j < (sizes[l] ?? 0); j++) {
        for (let k = 0; k < inn; k++)
          gradients.push({
            name: `W${l}[${j},${k}]`,
            value: W[l]?.[j]?.[k] ?? 0,
            byHand: (delta[l]?.[j] ?? 0) * (a[l - 1]?.[k] ?? 0),
            autodiff: gw[j * inn + k] ?? 0,
          })
        gradients.push({ name: `b${l}[${j}]`, value: b[l]?.[j] ?? 0, byHand: delta[l]?.[j] ?? 0, autodiff: gb[j] ?? 0 })
      }
    }
    const maxError = gradients.reduce((m, g) => Math.max(m, Math.abs(g.byHand - g.autodiff)), 0)
    const units: UnitValues[] = []
    for (let l = 0; l < L; l++)
      for (let j = 0; j < (sizes[l] ?? 0); j++)
        units.push({
          id: unitId(l, j, L),
          layer: l,
          unit: j,
          a: a[l]?.[j] ?? 0,
          ...(l === 0 ? {} : { z: z[l]?.[j] ?? 0, delta: delta[l]?.[j] ?? 0 }),
        })
    steps.push({ step: t, row, target: [...y], loss, units, gradients, maxError })
    // Frames: forward layer by layer, the loss, backward layer by layer.
    const forwardBadges: Record<string, Record<string, string | number>> = {}
    for (let l = 0; l < L; l++) {
      const ids: string[] = []
      for (let j = 0; j < (sizes[l] ?? 0); j++) {
        const id = unitId(l, j, L)
        ids.push(id)
        forwardBadges[id] = l === 0 ? { a: fmt(a[0]?.[j] ?? 0) } : { z: fmt(z[l]?.[j] ?? 0), a: fmt(a[l]?.[j] ?? 0) }
      }
      frames.push({ active: ids, badges: { ...forwardBadges } })
      captions.push(l === 0 ? `step ${t + 1}: row ${row + 1} enters` : `step ${t + 1}: forward through layer ${l}`)
    }
    forwardBadges['loss'] = { L: fmt(loss) }
    frames.push({ active: ['loss'], badges: { ...forwardBadges } })
    captions.push(`step ${t + 1}: the loss`)
    const backwardBadges: Record<string, Record<string, string | number>> = { ...forwardBadges }
    for (let l = L - 1; l >= 1; l--) {
      const ids: string[] = []
      for (let j = 0; j < (sizes[l] ?? 0); j++) {
        const id = unitId(l, j, L)
        ids.push(id)
        backwardBadges[id] = { ...(forwardBadges[id] ?? {}), δ: fmt(delta[l]?.[j] ?? 0) }
      }
      frames.push({ active: ids, badges: { ...backwardBadges } })
      captions.push(`step ${t + 1}: δ back through layer ${l}`)
    }
    // The gradient-descent step.
    for (let l = 1; l < L; l++)
      for (let j = 0; j < (sizes[l] ?? 0); j++) {
        const dj = delta[l]?.[j] ?? 0
        const wj = W[l]?.[j] as number[]
        for (let k = 0; k < wj.length; k++) wj[k] = (wj[k] ?? 0) - lr * dj * (a[l - 1]?.[k] ?? 0)
        ;(b[l] as number[])[j] = (b[l]?.[j] ?? 0) - lr * dj
      }
  }
  const first = steps[0] as BackpropStep
  const maxError = steps.reduce((m, s) => Math.max(m, s.maxError), 0)
  const exposition: Exposition[] = [
    coefficientTable({
      names: first.gradients.map((g) => `∂L/∂${g.name}`),
      estimates: first.gradients.map((g) => g.byHand),
      standardErrors: first.gradients.map((g) => Math.abs(g.byHand - g.autodiff)),
    }),
    learningCurve({
      sizes: steps.map((s) => s.step + 1),
      train: steps.map((s) => s.loss),
      validation: steps.map((s) => s.maxError),
      resource: 'step',
      labels: { train: 'loss on the step’s row', validation: 'largest |by hand − autodiff|' },
      secondaryAxis: true,
    }),
  ]
  const structure = networkStructure(sizes)
  exposition.push(
    diagramRecord({
      structure,
      steps: frames.map((overlay, i) => ({ overlay, ...(captions[i] === undefined ? {} : { caption: captions[i] }) })),
      stepName: 'pass',
    }),
  )
  return { sizes, activation, steps, structure, frames, captions, maxError, exposition }
}

const ACTIVATION: Readonly<Record<BackpropActivation, (v: Value) => Value>> = { logistic: sigmoid, tanh, relu }
