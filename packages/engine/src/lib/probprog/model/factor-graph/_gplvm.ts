/**
 * The detection of a latent variable model with random Fourier features in the spec of a model (the form `gplvmFourierModel` writes), made
 * once before any node is mapped: an observed Gaussian `y[point, output]` (the two in either order) whose mean is `sum over feature of cos(x . omega + phase) * W`,
 * with `x[point, coord]` a latent normal, `omega[coord, feature]` and `phase[feature]` known, and `W[output, feature]` a latent normal with a
 * known prior that nothing else reads. The weights of each output then become one Gaussian vector (`latentGplvmWeights`) and each point
 * one `gplvm-row` factor over its coordinates and those vectors (`_mappings/observed-gplvm.ts`). It is pure: it reads the spec and the
 * known values and makes no graph structure; the expression has to be written inline (a derived node in it is not followed, because the walk maps a derived node on its own). A model of this form that is outside the factor's limits (two or more coordinates per point, more
 * than `GPLVM_MOST_WEIGHTS` weights, a missing observation, a noise that differs between entries, a weight matrix that something else
 * reads) is not detected, and compiles by the scalar operators (`apply`, `product`, `sum`) as before.
 */

import { GPLVM_MOST_WEIGHTS } from '../../../message-passing/index.ts'
import { toFlat } from '../../../tensor/index.ts'
import type { BoundModel } from '../bind.ts'
import { auxId } from '../ids.ts'
import type { OperatorTable } from '../operators.ts'
import { type ExprSpec, type ModelNode, nodeRefs } from '../spec.ts'
import { evaluateKnown, isKnownExpr, type Known } from './_known.ts'

/**
 * The most coordinates per point the compiler gives to the `gplvm-row` factor. The factor itself takes two (its grid over two coordinates
 * agrees with brute force), but measured on the model with two coordinates it is not better than the scalar route and is four times slower
 * (24 points, 6 outputs, 10 features: both coordinates found in 3 of 6 seeds against 2 of 6, 63 s a run against 14 s; 40 points, 8 outputs,
 * 12 features: 170 s), so a model with two coordinates compiles by the scalar operators as before.
 */
export const GPLVM_COMPILED_COORDS = 1

/** What the detection found: the nodes of the form, their dimensions and the name of the weight vector. */
export interface GplvmPlan {
  /** The observed node, the coordinates' node, the weights' node, the frequencies' node and the phases' node (absent: no phase). */
  readonly observed: string
  readonly x: string
  readonly w: string
  readonly omega: string
  readonly phase: string | undefined
  /** The Gaussian vector variable of each output's weights (`features` coordinates each), made by the latent mapping of the weights' node. */
  readonly vectors: readonly string[]
  /** The dimensions: points, outputs, features and coordinates (the names, in the declared order of each node). */
  readonly dims: { readonly point: string; readonly output: string; readonly feature: string; readonly coord: string }
  /** The sizes of those dimensions. */
  readonly points: number
  readonly outputs: number
  readonly features: number
  readonly coords: number
}

/** The node of the given kind an expression is a reference to, or undefined (a derived node is not followed: the walk would map it on its own). */
function refTo<K extends ModelNode['kind']>(
  nodes: ReadonlyMap<string, ModelNode>,
  e: ExprSpec,
  kind: K,
): (ModelNode & { readonly kind: K }) | undefined {
  if (e.kind !== 'ref') return undefined
  const n = nodes.get(e.name)
  return n?.kind === kind ? (n as ModelNode & { readonly kind: K }) : undefined
}

/** The two arguments of a binary op, or undefined when the expression is not that op with two arguments. */
function binary(e: ExprSpec, op: string): readonly [ExprSpec, ExprSpec] | undefined {
  return e.kind === 'op' && e.op === op && e.args.length === 2
    ? [e.args[0] as ExprSpec, e.args[1] as ExprSpec]
    : undefined
}

/**
 * The latent variable model with random Fourier features of a spec, if it has that form and is within the limits of the factor.
 *
 * @param bound - The bound model.
 * @param nodes - The nodes an expression refers to, by name (every node but the observed ones).
 * @param known - The constants of the model.
 * @param modelOps - The model operator registry (it evaluates the known values the detection reads).
 * @returns The plan, or undefined.
 */
export function findGplvm(
  bound: BoundModel,
  nodes: ReadonlyMap<string, ModelNode>,
  known: Known,
  modelOps: OperatorTable,
): GplvmPlan | undefined {
  const isKnown = (e: ExprSpec): boolean => isKnownExpr(e, known, bound.customs)
  const knownNumbers = (e: ExprSpec): readonly number[] | undefined => {
    try {
      return toFlat(
        evaluateKnown(e, {
          modelOps,
          customs: bound.customs,
          ref: (name) => {
            const v = known.get(name)
            if (v === undefined) throw new Error(`'${name}' is not known`)
            return v
          },
        }).value,
      )
    } catch {
      // not a known value: the form is not this one
      return undefined
    }
  }
  const readers = (name: string): number => bound.spec.nodes.filter((n) => nodeRefs(n).includes(name)).length
  for (const node of bound.spec.nodes) {
    if (node.kind !== 'observed' || node.dist.family !== 'normal' || node.dims.length !== 2) continue
    const loc = node.dist.params['loc']
    const scale = node.dist.params['scale']
    if (loc === undefined || scale === undefined || !isKnown(scale) || !isKnown(node.value)) continue
    const sum = loc
    if (sum.kind !== 'op' || sum.op !== 'sum' || sum.args.length !== 1) continue
    const feature = sum.params?.['dim']
    const product = binary(sum.args[0] as ExprSpec, 'mul')
    if (typeof feature !== 'string' || product === undefined) continue
    // The weights and the cosine, in either order.
    const weightsFirst = refTo(nodes, product[0], 'latent') !== undefined
    const w = refTo(nodes, weightsFirst ? product[0] : product[1], 'latent')
    const cosine = weightsFirst ? product[1] : product[0]
    if (w === undefined || cosine.kind !== 'op' || cosine.op !== 'cos' || cosine.args.length !== 1) continue
    // The angle: x . omega + phase (the sum in either order), or x . omega alone.
    const angle = cosine.args[0] as ExprSpec
    let product2: readonly [ExprSpec, ExprSpec] | undefined
    let phase: (ModelNode & { readonly kind: 'data' }) | undefined
    const added = binary(angle, 'add')
    if (added !== undefined) {
      const first = binary(added[0], 'matmul')
      product2 = first ?? binary(added[1], 'matmul')
      phase = refTo(nodes, first === undefined ? added[0] : added[1], 'data')
      if (phase === undefined) continue
    } else product2 = binary(angle, 'matmul')
    if (product2 === undefined) continue
    const x = refTo(nodes, product2[0], 'latent')
    const omega = refTo(nodes, product2[1], 'data')
    if (x === undefined || omega === undefined) continue
    // The dimensions: x[point, coord], omega[coord, feature], phase[feature], W[output, feature], y[point, output].
    const [point, coord] = x.dims
    const [coordOf, featureOf] = omega.dims
    const [output, weightFeature] = w.dims
    if (x.dims.length !== 2 || omega.dims.length !== 2 || w.dims.length !== 2) continue
    if (coordOf !== coord || featureOf !== feature || weightFeature !== feature) continue
    // the observed node's dimensions follow its mean, so either order of the point and the output
    if (!(node.dims.includes(point as string) && node.dims.includes(output as string))) continue
    if (phase !== undefined && phase.dims.join() !== feature) continue
    if (point === undefined || coord === undefined || output === undefined) continue
    // What the factor needs of the latents: normals, the weights with a known prior that only this observation reads.
    if (x.dist.family !== 'normal' || (x.constraint !== undefined && x.constraint.kind !== 'real')) continue
    if (w.dist.family !== 'normal' || (w.constraint !== undefined && w.constraint.kind !== 'real')) continue
    const wLoc = w.dist.params['loc']
    const wScale = w.dist.params['scale']
    if (wLoc === undefined || wScale === undefined || !isKnown(wLoc) || !isKnown(wScale)) continue
    if (readers(w.name) !== 1) continue
    const noise = knownNumbers(scale)
    if (noise === undefined || noise.length === 0 || !noise.every((v) => v === noise[0] && v > 0)) continue
    if (bound.missing[node.name] !== undefined && bound.missing[node.name]?.count !== 0) continue
    if (bound.batch !== undefined && node.dims.includes(bound.batch.dim)) continue
    const sizes = bound.sizes
    const points = sizes[point] ?? 0
    const outputs = sizes[output] ?? 0
    const features = sizes[feature] ?? 0
    const coords = sizes[coord] ?? 0
    if (points < 1 || outputs < 1 || features < 1 || coords < 1) continue
    if (coords > GPLVM_COMPILED_COORDS || outputs * features > GPLVM_MOST_WEIGHTS) continue
    return {
      observed: node.name,
      x: x.name,
      w: w.name,
      omega: omega.name,
      phase: phase?.name,
      vectors: Array.from({ length: outputs }, (_, d) => auxId(w.name, 8, d)),
      dims: { point, output, feature, coord },
      points,
      outputs,
      features,
      coords,
    }
  }
  return undefined
}
