import { betaEta, gaussianMvPrior, wishartEta } from '../../../../message-passing/index.ts'
import { inv } from '../../../../numerics/linalg/index.ts'
import { toFlat } from '../../../../tensor/index.ts'
import type { NamedValue } from '../../operators.ts'
import type { DistSpec, ExprSpec, ModelNode } from '../../spec.ts'
import { type Atom, type Cell, prod, scaledOf, unflatten } from '../_cells.ts'
import type { Context } from '../_context.ts'
import { precisionExpr, unwrap } from '../_idioms.ts'
import { evalNamedValue, eventVector } from '../_params.ts'
import { Refusal } from '../_refusal.ts'
import { evalSym } from '../_symbolic/_affine.ts'
import { addPlated } from './plates.ts'

/**
 * A multivariate normal latent with a known mean and covariance (a Gaussian process prior over the inputs of the data: the
 * covariance a constant Gram matrix, or given by its Cholesky factor `scaleTril`): one `gaussian-mv` variable with a
 * dense `gaussian-mv` prior, whose components the likelihood reads (`probit(f)`, `logits`, a Gaussian row), so the
 * correlations between the elements are kept and the engines see a dense Gaussian vector.
 */
export function latentMultivariateNormal(cx: Context, node: ModelNode & { readonly kind: 'latent' }): void {
  const shape = cx.shapeOf(node.dims)
  const n = prod(shape)
  if (node.dims.length !== 1)
    throw new Refusal('structure', `'${node.name}' is a multivariateNormal: one vector over one dimension is mapped`)
  const params = node.dist.params
  // A random mean that is a linear map of another Gaussian vector (`matvec(F, previous)`): one dense Gaussian step between the two.
  if (params['loc'] !== undefined && !cx.isConst(params['loc'])) {
    latentTransition(cx, node, params['loc'])
    return
  }
  const known = (e: ExprSpec | undefined, what: string): number[] => {
    if (e === undefined || !cx.isConst(e))
      throw new Refusal('operator', `the ${what} of '${node.name}' depends on a random value: a known matrix is mapped`)
    return Array.from(toFlat(cx.evalNamed(e).value))
  }
  const loc = known(params['loc'], 'mean')
  const mean = loc.length === 1 ? new Array<number>(n).fill(loc[0] ?? 0) : loc
  let cov: number[]
  if (params['covariance'] !== undefined) cov = known(params['covariance'], 'covariance')
  else if (params['scaleTril'] !== undefined) {
    const L = known(params['scaleTril'], 'scale factor')
    cov = new Array<number>(n * n).fill(0)
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++) {
        let v = 0
        for (let k = 0; k <= Math.min(i, j); k++) v += (L[i * n + k] ?? 0) * (L[j * n + k] ?? 0)
        cov[i * n + j] = v
      }
  } else
    throw new Refusal(
      'structure',
      `'${node.name}' is a multivariateNormal with a precision: give a covariance or scaleTril`,
    )
  if (cov.length !== n * n) throw new Refusal('structure', `the covariance of '${node.name}' is not ${n} by ${n}`)
  cx.g.addVariable({ name: node.name, family: { kind: 'gaussian-mv', size: n } })
  cx.g.addFactor(gaussianMvPrior(node.name, mean, cov), node.name)
  cx.syms.set(node.name, {
    dims: node.dims,
    cells: Array.from({ length: n }, (_, j) => cx.g.compAtom(node.name, j, n)),
  })
  cx.layout(node, 'latent', 'gaussian-mv', 'packed', [node.name], true)
}

/** The Wishart latent that the precision (or covariance) of a multivariate normal node names, or undefined for any other form. */
export function wishartOf(
  cx: Context,
  node: ModelNode & { readonly dist: DistSpec },
): { readonly name: string; readonly variable: string; readonly d: number; readonly covariance: boolean } | undefined {
  for (const key of ['precision', 'covariance'] as const) {
    const e = node.dist.params[key]
    if (e === undefined) continue
    const u = unwrap(cx.nodes, e)
    if (u.kind !== 'ref') continue
    const w = cx.wisharts.get(u.name)
    if (w === undefined) continue
    if ((key === 'covariance') !== w.covariance)
      throw new Refusal(
        'structure',
        `'${node.name}' reads '${u.name}' as its ${key}, but '${u.name}' is ${w.covariance ? 'an inverse Wishart (a covariance): write { covariance: ... }' : 'a Wishart (a precision matrix): write { precision: ... }'}`,
      )
    return { name: u.name, ...w }
  }
  return undefined
}

/**
 * A Wishart latent (the precision matrix of a Gaussian) or an inverse-Wishart one (its covariance): one variable of the Wishart family
 * with the prior `W(df, scale)` (for the inverse Wishart `W(df, scale^-1)` on the precision, the site is its inverse). The matrix is
 * read only as the precision or covariance of a multivariate normal (`mvn-wishart`).
 */
export function latentWishart(cx: Context, node: ModelNode & { readonly kind: 'latent' }): void {
  const covariance = node.dist.family === 'inverseWishart'
  if (node.dims.length > 0)
    throw new Refusal('structure', `'${node.name}' is a ${node.dist.family} over [${node.dims}]: one matrix is mapped`)
  const dfExpr = node.dist.params['df']
  const scaleExpr = node.dist.params['scale']
  if (dfExpr === undefined || scaleExpr === undefined || !cx.isConst(dfExpr) || !cx.isConst(scaleExpr))
    throw new Refusal('operator', `the degrees of freedom and the scale of '${node.name}' must be known values`)
  const df = toFlat(cx.evalNamed(dfExpr).value)[0] as number
  const scale = cx.evalNamed(scaleExpr).value
  const d = scale.shape[0] as number
  if (scale.shape.length !== 2 || scale.shape[1] !== d)
    throw new Refusal('structure', `the scale of '${node.name}' must be a square matrix`)
  if (!(df > d - 1))
    throw new Refusal('data', `'${node.name}' has ${df} degrees of freedom: more than ${d - 1} are needed`)
  // W(df, S) on the precision; the inverse Wishart Psi on the covariance is W(df, Psi^-1) on the precision
  const matrix = covariance ? Array.from(toFlat(inv(scale))) : Array.from(toFlat(scale))
  const eta = wishartEta(df, matrix, d)
  const name = cx.g.addVariable({ name: node.name, family: { kind: 'wishart', size: d } })
  cx.g.addFactor({ operator: 'prior', args: [name], params: { eta } }, name)
  cx.wisharts.set(node.name, { variable: name, d, covariance })
  cx.layout(node, 'latent', 'wishart', 'scalar', [name], true, covariance ? { reciprocal: true } : {})
}

/**
 * The mean of a multivariate normal as the argument of `mvn-wishart`, per row: a known vector (or one per row), or a Gaussian vector
 * variable (a latent multivariate normal over a named dimension), shared by every row.
 */
export function vectorMean(
  cx: Context,
  node: ModelNode & { readonly dist: DistSpec },
  d: number,
  plateDims: readonly string[],
  vectorDim: string | undefined,
): { readonly at: (row: number) => string | { readonly const: readonly number[] }; readonly shared: boolean } {
  const e = node.dist.params['loc']
  if (e === undefined) throw new Refusal('structure', `'${node.name}': the mean is missing`)
  if (cx.isConst(e)) {
    const nv = cx.evalNamed(e)
    const kept = nv.dims.filter((x) => x !== vectorDim)
    const shape = cx.shapeOf(plateDims)
    if (nv.value.shape.length - kept.length !== 1 || nv.value.shape[nv.value.shape.length - 1] !== d)
      throw new Refusal('structure', `the mean of '${node.name}' must be a vector of ${d} (per row, when it varies)`)
    return {
      shared: kept.length === 0,
      at: (row) => ({
        const: eventVector(cx, { value: nv.value, dims: kept }, plateDims, unflatten(row, shape)),
      }),
    }
  }
  const sym = evalSym(cx, e)
  let vector: string | undefined
  const cells = sym.cells
  if (cells.length !== d)
    throw new Refusal('operator', `the mean of '${node.name}' must be one Gaussian vector of ${d} entries (or known)`)
  for (const [k, c] of cells.entries()) {
    const terms = c.kind === 'aff' ? [...c.aff.terms].filter(([, v]) => v !== 0) : []
    const atom = terms.length === 1 && terms[0]?.[1] === 1 ? cx.g.atoms.get(terms[0][0]) : undefined
    if (
      atom?.kind !== 'comp' ||
      atom.index !== k ||
      atom.size !== d ||
      (vector !== undefined && atom.vector !== vector)
    )
      throw new Refusal(
        'operator',
        `the mean of '${node.name}' must be a known vector or one whole Gaussian vector variable`,
      )
    vector = atom.vector
  }
  return { shared: true, at: () => vector as string }
}

/**
 * Gaussian vectors `x ~ N(loc, L^-1)` with `L` a Wishart latent (or `N(loc, Sigma)` with `Sigma` an inverse-Wishart one), one vector
 * per index of the plate dimensions before the last, which names the vector's axis (`dims: [group, coef]`; `dims: [coef]` is one
 * vector): a Gaussian vector variable and an `mvn-wishart` factor per vector. The mean is known or one Gaussian vector variable
 * (a population mean), so the vectors are the correlated random effects of a hierarchical model, and a likelihood reads their
 * coordinates as `theta.at(group)` summed against a design over the axis.
 */
export function latentMvnWishart(cx: Context, node: ModelNode & { readonly kind: 'latent' }): void {
  const w = wishartOf(cx, node) as NonNullable<ReturnType<typeof wishartOf>>
  if (node.dims.length === 0)
    throw new Refusal(
      'structure',
      `'${node.name}' is a multivariate normal with a Wishart precision: give the vector's axis a named dimension (dims: [..., axis])`,
    )
  const vectorDim = node.dims[node.dims.length - 1] as string
  const plateDims = node.dims.slice(0, -1)
  const d = cx.size(vectorDim)
  if (d !== w.d)
    throw new Refusal(
      'structure',
      `'${node.name}' has ${d} entries per vector, the matrix '${w.name}' is ${w.d} by ${w.d}`,
    )
  const shape = cx.shapeOf(plateDims)
  const rows = prod(shape)
  const mean = vectorMean(cx, node, d, plateDims, vectorDim)
  const cells: Cell[] = []
  const vars: string[] = []
  for (let r = 0; r < rows; r++) {
    const name = cx.g.addVariable({
      name: cx.elementName(node.name, plateDims, r),
      family: { kind: 'gaussian-mv', size: d },
    })
    cx.g.addFactor({ operator: 'mvn-wishart', args: [name, mean.at(r), w.variable] }, name)
    for (let k = 0; k < d; k++) cells.push(cx.g.compAtom(name, k, d))
    vars.push(name)
  }
  cx.syms.set(node.name, { dims: node.dims, cells })
  cx.layout(node, 'latent', 'gaussian-mv', plateDims.length === 0 ? 'packed' : 'elements', vars, true)
}

/**
 * Observed vectors `y_i ~ N(loc, L^-1)` with `L` a Wishart latent (or `N(loc, Sigma)`, `Sigma` an inverse Wishart): a factor
 * `mvn-wishart` per row on the observed vector, the exact conjugate update of the matrix when the mean is known.
 */
export function observedMvnWishart(cx: Context, node: ModelNode & { readonly kind: 'observed' }): void {
  const w = wishartOf(cx, node) as NonNullable<ReturnType<typeof wishartOf>>
  const d = w.d
  const nv = evalNamedValue(cx, node)
  if (nv.value.shape[nv.value.shape.length - 1] !== d || nv.value.shape.length !== node.dims.length + 1)
    throw new Refusal('structure', `'${node.name}' is observed as vectors of ${d} entries over [${node.dims}]`)
  const shape = cx.shapeOf(node.dims)
  const mean = vectorMean(cx, node, d, node.dims, undefined)
  if (plateVectors(cx, node, nv, d, mean, 'mvn-wishart', [w.variable])) return
  const vars: (string | null)[] = []
  for (let i = 0; i < prod(shape); i++) {
    const y = eventVector(cx, nv, node.dims, unflatten(i, shape))
    if (y.some((v) => Number.isNaN(v))) {
      vars.push(null)
      continue
    }
    const name = cx.g.addVariable({
      name: cx.elementName(node.name, node.dims, i),
      family: { kind: 'gaussian-mv', size: d },
      observed: y,
    })
    cx.g.addFactor({ operator: 'mvn-wishart', args: [name, mean.at(i), w.variable] }, name)
    vars.push(name)
  }
  cx.layout(node, 'observed', undefined, node.dims.length === 0 ? 'scalar' : 'observed', vars, false)
}

/**
 * The rows of an observed vector node with a shared mean as ONE factor of `operator` (`mvn-wishart`, `mvn-sd-corr`): the sample mean of the
 * rows, their count and their scatter about it are all the likelihood depends on, so the graph does not grow with the data. False
 * when the rows are not plated (`plates: 'expand'`, a mean that varies by row, no plate dimension).
 */
export function plateVectors(
  cx: Context,
  node: ModelNode & { readonly kind: 'observed' },
  nv: NamedValue,
  d: number,
  mean: { readonly at: (row: number) => string | { readonly const: readonly number[] }; readonly shared: boolean },
  operator: string,
  tail: readonly (string | { readonly const: number })[],
): boolean {
  if (!cx.vectorise || node.dims.length === 0 || !mean.shared) return false
  const shape = cx.shapeOf(node.dims)
  const rows: number[][] = []
  for (let i = 0; i < prod(shape); i++) {
    const y = eventVector(cx, nv, node.dims, unflatten(i, shape))
    if (!y.some((v) => Number.isNaN(v))) rows.push(y)
  }
  if (rows.length === 0) return false
  const n = rows.length
  const ybar = Array.from({ length: d }, (_, k) => rows.reduce((acc, y) => acc + (y[k] as number), 0) / n)
  const scatter = new Array<number>(d * d).fill(0)
  for (const y of rows)
    for (let a = 0; a < d; a++)
      for (let b = 0; b < d; b++)
        scatter[a * d + b] =
          (scatter[a * d + b] as number) +
          ((y[a] as number) - (ybar[a] as number)) * ((y[b] as number) - (ybar[b] as number))
  addPlated(cx, node, { operator, args: [{ const: ybar }, mean.at(0), ...tail], params: { count: n, scatter } })
  return true
}

/** A correlation latent: `rho = 2 u - 1` with `u ~ Beta(eta, eta)` (the LKJ prior of a 2 by 2 matrix), one beta variable with that prior. */
export function latentCorrelation(cx: Context, node: ModelNode & { readonly kind: 'latent' }): void {
  if (node.dims.length > 0)
    throw new Refusal('structure', `'${node.name}' is a correlation over [${node.dims}]: one correlation is mapped`)
  const e = node.dist.params['eta']
  const eta = e === undefined ? 1 : cx.isConst(e) ? (toFlat(cx.evalNamed(e).value)[0] as number) : Number.NaN
  if (!(eta > 0)) throw new Refusal('operator', `the concentration of '${node.name}' must be a known positive value`)
  const name = cx.g.addVariable({ name: node.name, family: { kind: 'beta' } })
  cx.g.addFactor({ operator: 'prior', args: [name], params: { eta: betaEta(eta, eta) } }, name)
  cx.correlations.set(node.name, name)
  cx.layout(node, 'latent', 'beta', 'scalar', [name], true, { correlation: true })
}

/** A standard deviation of a bivariate normal as the precision port of `mvn-sd-corr`: a known value, or the idiom `1 / sqrt(tau)` of a gamma latent. */
export function precisionPort(
  cx: Context,
  node: ModelNode & { readonly dist: DistSpec },
  param: string,
): string | { readonly const: number } {
  const e = node.dist.params[param]
  if (e === undefined) throw new Refusal('structure', `'${node.name}': ${param} is missing`)
  if (cx.isConst(e)) {
    const v = toFlat(cx.evalNamed(e).value)[0] as number
    if (!(v > 0)) throw new Refusal('data', `${param} of '${node.name}' must be positive`)
    return { const: 1 / (v * v) }
  }
  const tau = precisionExpr(cx.idiom, e)
  if (tau === undefined)
    throw new Refusal(
      'operator',
      `the ${param} of '${node.name}' depends on a random value that is not the precision idiom: write it as 1 / sqrt(tau) with tau a gamma latent`,
    )
  const cell = evalSym(cx, tau).cells[0] as Cell
  const scaled = scaledOf(cell)
  if (
    scaled === undefined ||
    scaled.scale !== 1 ||
    scaled.offset !== 0 ||
    cell.kind !== 'rv' ||
    cell.family !== 'gamma'
  )
    throw new Refusal(
      'operator',
      `the ${param} of '${node.name}' must be 1 / sqrt(tau) with tau a gamma latent (a known multiple or offset of it is not mapped)`,
    )
  return scaled.variable
}

/** The correlation of a bivariate normal as the beta port of `mvn-sd-corr`: a known value or a `correlation` latent. */
export function correlationPort(
  cx: Context,
  node: ModelNode & { readonly dist: DistSpec },
): string | { readonly const: number } {
  const e = node.dist.params['rho']
  if (e === undefined) throw new Refusal('structure', `'${node.name}': rho is missing`)
  if (cx.isConst(e)) {
    const v = toFlat(cx.evalNamed(e).value)[0] as number
    if (!(v > -1 && v < 1)) throw new Refusal('data', `rho of '${node.name}' must be in (-1, 1)`)
    return { const: v }
  }
  const u = unwrap(cx.nodes, e)
  const variable = u.kind === 'ref' ? cx.correlations.get(u.name) : undefined
  if (variable === undefined)
    throw new Refusal('operator', `the rho of '${node.name}' must be known or a latent with the correlation prior`)
  return variable
}

/**
 * Observed vectors `y_i ~ bivariateNormal(loc, scale1, scale2, rho)` with the two standard deviations known or `1 / sqrt(tau)` of gamma latents and
 * the correlation known or a `correlation` latent: one `mvn-sd-corr` factor for the rows of a shared mean (a plate), else one per row.
 */
export function observedBivariateNormal(cx: Context, node: ModelNode & { readonly kind: 'observed' }): void {
  const nv = evalNamedValue(cx, node)
  if (nv.value.shape[nv.value.shape.length - 1] !== 2 || nv.value.shape.length !== node.dims.length + 1)
    throw new Refusal('structure', `'${node.name}' is observed as vectors of 2 entries over [${node.dims}]`)
  const vectorDim = node.dist.eventDims?.[0]
  const mean = vectorMean(cx, node, 2, node.dims, vectorDim)
  const tail = [precisionPort(cx, node, 'scale1'), precisionPort(cx, node, 'scale2'), correlationPort(cx, node)]
  if (plateVectors(cx, node, nv, 2, mean, 'mvn-sd-corr', tail)) return
  const shape = cx.shapeOf(node.dims)
  const vars: (string | null)[] = []
  for (let i = 0; i < prod(shape); i++) {
    const y = eventVector(cx, nv, node.dims, unflatten(i, shape))
    if (y.some((v) => Number.isNaN(v))) {
      vars.push(null)
      continue
    }
    const name = cx.g.addVariable({
      name: cx.elementName(node.name, node.dims, i),
      family: { kind: 'gaussian-mv', size: 2 },
      observed: y,
    })
    cx.g.addFactor({ operator: 'mvn-sd-corr', args: [name, mean.at(i), ...tail] }, name)
    vars.push(name)
  }
  cx.layout(node, 'observed', undefined, node.dims.length === 0 ? 'scalar' : 'observed', vars, false)
}

/**
 * A multivariate normal latent whose mean is `F w + c` for a known matrix and another Gaussian vector `w` (the step of a linear
 * dynamical system, `x_t ~ N(F x_(t-1), Q)`), with a known covariance: one `gaussian-vector` variable and a `gaussian-transition`
 * factor between it and `w`, so a chain of steps with linear Gaussian readings is the Kalman smoother.
 */
export function latentTransition(cx: Context, node: ModelNode & { readonly kind: 'latent' }, locExpr: ExprSpec): void {
  if (node.dims.length !== 1)
    throw new Refusal('structure', `'${node.name}' is a multivariateNormal: one vector over one dimension is mapped`)
  const n = cx.size(node.dims[0] as string)
  const params = node.dist.params
  const covarianceExpr = params['covariance']
  if (covarianceExpr === undefined || !cx.isConst(covarianceExpr))
    throw new Refusal('operator', `the covariance of '${node.name}' must be a known matrix when its mean is random`)
  const covariance = Array.from(toFlat(cx.evalNamed(covarianceExpr).value))
  if (covariance.length !== n * n)
    throw new Refusal('structure', `the covariance of '${node.name}' is not ${n} by ${n}`)
  const loc = cx.over(evalSym(cx, locExpr), node.dims)
  let vector: string | undefined
  let columns = 0
  const forms = loc.map((c) => {
    if (c.kind !== 'aff')
      throw new Refusal('operator', `the mean of '${node.name}' must be a linear map of another Gaussian vector`)
    const terms = [...c.aff.terms].filter(([, k]) => k !== 0)
    for (const [key] of terms) {
      const atom = cx.g.atoms.get(key)
      if (atom?.kind !== 'comp' || (vector !== undefined && atom.vector !== vector))
        throw new Refusal('operator', `the mean of '${node.name}' must be a linear map of one Gaussian vector`)
      vector = atom.vector
      columns = atom.size
    }
    return { aff: c.aff, terms }
  })
  if (vector === undefined)
    throw new Refusal('operator', `the mean of '${node.name}' does not depend on a Gaussian vector`)
  const matrix = new Array<number>(n * columns).fill(0)
  const offset = forms.map(({ aff, terms }, i) => {
    for (const [key, k] of terms) matrix[i * columns + (cx.g.atoms.get(key) as Atom & { kind: 'comp' }).index] = k
    return aff.offset
  })
  cx.g.addVariable({ name: node.name, family: { kind: 'gaussian-mv', size: n } })
  cx.g.addFactor(
    {
      operator: 'gaussian-transition',
      args: [node.name, vector],
      params: { matrix, covariance, ...(offset.some((v) => v !== 0) ? { offset } : {}) },
    },
    node.name,
  )
  cx.syms.set(node.name, {
    dims: node.dims,
    cells: Array.from({ length: n }, (_, j) => cx.g.compAtom(node.name, j, n)),
  })
  cx.layout(node, 'latent', 'gaussian-mv', 'packed', [node.name], true)
}
