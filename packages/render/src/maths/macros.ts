/**
 * The shared TeX macro registry, following aifn-engine's maths macros: bold vectors and matrices, calligraphic sets,
 * number systems, probability and expectation, operators, delimiters and distributions, so every page and package
 * writes the same symbols, with `extendMacros` for adding more without touching the shared set.
 */

const LOWER = 'abcdefghijklmnopqrstuvwxyz'.split('')
const UPPER = LOWER.map((c) => c.toUpperCase())
const GREEK_LOWER = [
  'alpha',
  'beta',
  'gamma',
  'delta',
  'epsilon',
  'varepsilon',
  'zeta',
  'eta',
  'theta',
  'vartheta',
  'iota',
  'kappa',
  'lambda',
  'mu',
  'nu',
  'xi',
  'pi',
  'rho',
  'sigma',
  'tau',
  'upsilon',
  'phi',
  'varphi',
  'chi',
  'psi',
  'omega',
]
const GREEK_UPPER = ['Gamma', 'Delta', 'Theta', 'Lambda', 'Xi', 'Pi', 'Sigma', 'Upsilon', 'Phi', 'Psi', 'Omega']

function family(
  names: readonly string[],
  name: (s: string) => string,
  body: (s: string) => string,
): Record<string, string> {
  return Object.fromEntries(names.map((s) => [`\\${name(s)}`, body(s)]))
}

/**
 * The shared TeX macro registry: bold vectors (`\xvec`, `\thetavec`), matrices (`\Xmat`, `\Sigmamat`), calligraphic
 * sets (`\Dcal`), number systems and probability (`\reals`, `\expect`, `\iid`), operators (`\argmin`, `\KL`,
 * `\trace`), delimiters (`\norm{x}`, `\inner{x}{y}`) and distributions (`\Gauss`, `\Bern`, `\Dir`, `\GP`). Every
 * figure, label and live equation gets the same symbols; extend it with {@link extendMacros}.
 *
 * @example
 * ```ts
 * import { macros } from 'whitebox-render/maths'
 * macros['\\xvec'] // => '\\mathbf{x}'
 * ```
 */
export const macros: Readonly<Record<string, string>> = {
  // Vectors (bold lowercase & bold Greek)
  ...family(
    LOWER,
    (c) => `${c}vec`,
    (c) => `\\mathbf{${c}}`,
  ),
  ...family(
    GREEK_LOWER,
    (g) => `${g}vec`,
    (g) => `\\boldsymbol{\\${g}}`,
  ),
  ...family(
    GREEK_LOWER,
    (g) => `${g}b`,
    (g) => `\\boldsymbol{\\${g}}`,
  ),
  '\\ones': '\\mathbf{1}',
  '\\zeros': '\\mathbf{0}',

  // Matrices (bold uppercase & bold Greek)
  ...family(
    UPPER,
    (c) => `${c}mat`,
    (c) => `\\mathbf{${c}}`,
  ),
  ...family(
    GREEK_UPPER,
    (g) => `${g}mat`,
    (g) => `\\boldsymbol{\\${g}}`,
  ),
  ...family(
    GREEK_UPPER,
    (g) => `${g}b`,
    (g) => `\\boldsymbol{\\${g}}`,
  ),
  '\\eye': '\\mathbf{I}',
  '\\I': '\\mathbf{I}',
  '\\transpose': '^{\\mathsf{\\top}}',
  '\\tr': '\\top',

  // Sets and families (calligraphic capitals)
  ...family(
    UPPER,
    (c) => `${c}cal`,
    (c) => `\\mathcal{${c}}`,
  ),

  // Number systems and probability
  '\\reals': '\\mathbb{R}',
  '\\complex': '\\mathbb{C}',
  '\\rationals': '\\mathbb{Q}',
  '\\integers': '\\mathbb{Z}',
  '\\naturals': '\\mathbb{N}',
  '\\expect': '\\mathbb{E}',
  '\\prob': '\\mathbb{P}',
  '\\pr': '\\operatorname{Pr}',
  '\\indicator': '\\mathbb{I}',
  '\\iid': '\\mathrel{\\overset{\\text{iid}}{\\sim}}',
  '\\eqdef': '\\stackrel{\\mathrm{def}}{=}',

  // Operators
  '\\argmin': '\\operatorname*{arg\\,min}',
  '\\argmax': '\\operatorname*{arg\\,max}',
  '\\prox': '\\operatorname{prox}',
  '\\trace': '\\operatorname{tr}',
  '\\diag': '\\operatorname{diag}',
  '\\rank': '\\operatorname{rank}',
  '\\Span': '\\operatorname{span}',
  '\\sgn': '\\operatorname{sgn}',
  '\\var': '\\operatorname{var}',
  '\\cov': '\\operatorname{cov}',
  '\\corr': '\\operatorname{corr}',
  '\\KL': '\\operatorname{KL}',
  '\\entropy': '\\operatorname{H}',
  '\\Null': '\\operatorname{null}',
  '\\row': '\\operatorname{row}',
  '\\col': '\\operatorname{col}',
  '\\proj': '\\operatorname{proj}',
  '\\AIC': '\\operatorname{AIC}',
  '\\BIC': '\\operatorname{BIC}',

  // Delimiters
  '\\abs': '\\left\\lvert #1 \\right\\rvert',
  '\\norm': '\\left\\lVert #1 \\right\\rVert',
  '\\inner': '\\left\\langle #1,\\, #2 \\right\\rangle',
  '\\set': '\\left\\{ #1 \\right\\}',
  '\\paren': '\\left( #1 \\right)',
  '\\brack': '\\left[ #1 \\right]',
  '\\LP': '\\left(',
  '\\RP': '\\right)',
  '\\LB': '\\left[',
  '\\RB': '\\right]',
  '\\LC': '\\left\\{',
  '\\RC': '\\right\\}',
  '\\LA': '\\left\\langle',
  '\\RA': '\\right\\rangle',

  // Distributions
  '\\Gauss': '\\mathcal{N}',
  '\\Bern': '\\operatorname{Bern}',
  '\\Binom': '\\operatorname{Binom}',
  '\\Cat': '\\operatorname{Cat}',
  '\\Mult': '\\operatorname{Mult}',
  '\\Poisson': '\\operatorname{Poisson}',
  '\\Geom': '\\operatorname{Geom}',
  '\\Unif': '\\operatorname{Unif}',
  '\\Exp': '\\operatorname{Exp}',
  '\\GammaD': '\\operatorname{Gamma}',
  '\\Beta': '\\operatorname{Beta}',
  '\\Dir': '\\operatorname{Dir}',
  '\\NegBin': '\\operatorname{NegBin}',
  '\\InvGamma': '\\operatorname{InvGamma}',
  '\\LogNorm': '\\operatorname{LogNormal}',
  '\\MvGauss': '\\mathcal{N}',
  '\\Cauchy': '\\operatorname{Cauchy}',
  '\\Weibull': '\\operatorname{Weibull}',
  '\\Gumbel': '\\operatorname{Gumbel}',
  '\\StudentT': '\\operatorname{t}',
  '\\Laplace': '\\operatorname{Laplace}',
  '\\Wishart': '\\operatorname{Wishart}',
  '\\ChiSq': '\\chi^2',
  '\\Hypergeom': '\\operatorname{Hypergeom}',
  '\\FDist': '\\operatorname{F}',
  '\\GP': '\\mathcal{GP}',
}

/**
 * The registry extended with more macros (later ones override), as a new record: the shared registry is never
 * modified.
 *
 * @param more - Macros to add, keyed with their backslash (`'\R': '\mathbb{R}'`).
 * @param base - The registry to extend (default {@link macros}).
 * @returns The extended registry.
 * @example
 * ```ts
 * import { extendMacros } from 'whitebox-render/maths'
 * extendMacros({ '\\loss': '\\mathcal{L}' })['\\loss'] // => '\\mathcal{L}'
 * ```
 */
export function extendMacros(
  more: Readonly<Record<string, string>>,
  base: Readonly<Record<string, string>> = macros,
): Readonly<Record<string, string>> {
  return { ...base, ...more }
}
