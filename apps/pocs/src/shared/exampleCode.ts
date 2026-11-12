// The code shown at the foot of every Bayesian example page: the model the knobs build and the lines that run the chosen
// inference method, cut from the library's own source files by their `// region` markers (the files are imported as text,
// so the code on screen is the code that runs). A model that varies by a dropdown is written in the source as pieces (a
// head, one piece per choice, a tail), and `regions` picks the pieces of the current configuration.
import type { Config } from 'omai-engine/trace'
import { snippetOf } from '../../../site/src/shell/snippet.ts'

/** A piece of the code: a region of the source files, shown `depth` levels (two spaces each) in. */
export type Piece = readonly [name: string, depth: number]

/** What a page shows: its source files (by name, as text), and the pieces its configuration picks. */
export interface ExampleCode {
  /** The source files of the model, by file name (loaded on demand: the sources are large). */
  readonly files: () => Promise<Readonly<Record<string, string>>>
  /** The pieces of the model for a configuration, in the order they are shown (the inference lines follow). */
  readonly regions: (config: Config) => readonly Piece[]
  /** Configurations (over the defaults) that between them show every piece. */
  readonly variants: readonly Config[]
  /** The regions that choose, bind and run the inference method, in order (default: the scaffold's). */
  readonly inference?: readonly string[]
}

/** The scaffold's source: the lines that choose, bind and run the inference method. */
export const scaffold = (): Promise<string> => import('../examples/example-method.ts?raw').then((m) => m.default)

/** The source files a page's code is cut from, with the scaffold's (by name). */
export async function sourcesOf(code: ExampleCode): Promise<Readonly<Record<string, string>>> {
  return { ...(await code.files()), 'example-method.ts': await scaffold() }
}

const text = (config: Config, key: string, fallback = ''): string => {
  const v = config[key]
  return typeof v === 'string' ? v : fallback
}

/** The head and the tail of a model that is defined in one function of one file, around the pieces in between. */
const around = (name: string, ...middle: readonly Piece[]): readonly Piece[] => [
  [name, 0],
  ...middle,
  [`${name}:end`, 0],
]

const bayesian = (file: string): (() => Promise<Record<string, string>>) => {
  const loaders: Record<string, () => Promise<{ default: string }>> = {
    'classifier-method.ts': () => import('../examples/classifier-method.ts?raw'),
    'linear-regression-method.ts': () => import('../examples/linear-regression-method.ts?raw'),
    'robust-regression-method.ts': () => import('../examples/robust-regression-method.ts?raw'),
    'gp-classification-method.ts': () => import('../examples/gp-classification-method.ts?raw'),
    'conjugate.ts': () => import('../../../../packages/engine/src/lib/probprog/models/conjugate.ts?raw'),
    'schools.ts': () => import('../../../../packages/engine/src/lib/probprog/models/schools.ts?raw'),
    'radon-counties.ts': () => import('../../../../packages/engine/src/lib/probprog/models/radon-counties.ts?raw'),
    'mixture.ts': () => import('../../../../packages/engine/src/lib/probprog/models/mixture.ts?raw'),
    'irt-hierarchical.ts': () => import('../../../../packages/engine/src/lib/probprog/models/irt-hierarchical.ts?raw'),
    'bayesian-gam.ts': () => import('../../../../packages/engine/src/lib/probprog/models/bayesian-gam.ts?raw'),
  }
  return async () => ({ [file]: (await (loaders[file] as () => Promise<{ default: string }>)()).default })
}

/** The GAM page's files: its model, and the page itself (it runs the session on its own). */
const gamFiles = async (): Promise<Record<string, string>> => ({
  ...(await bayesian('bayesian-gam.ts')()),
  'Page.tsx': (await import('../pocs/bayesian-examples/bayesian-gam/Page.tsx?raw')).default,
})

/** The regions of the scaffold that choose, bind and run the inference method, in the order they are shown. */
const INFERENCE = ['inference:offer', 'inference:bind', 'inference:steps', 'inference:run'] as const

/** The code of each example page, by slug. */
export const exampleCode: Readonly<Record<string, ExampleCode>> = {
  classify: {
    files: bayesian('classifier-method.ts'),
    regions: (c) => around('classify', [`classify:${text(c, 'link', 'probit')}`, 1]),
    variants: [{ link: 'probit' }, { link: 'logistic' }],
  },
  'coin-flipping': {
    files: bayesian('conjugate.ts'),
    regions: (c) => [[text(c, 'kind', 'coin') === 'counts' ? 'coin:gamma-poisson' : 'coin:beta-bernoulli', 0]],
    variants: [{ kind: 'coin' }, { kind: 'counts' }],
  },
  'linear-regression': {
    files: bayesian('linear-regression-method.ts'),
    regions: (c) => {
      const noise = text(c, 'noiseModel', 'known')
      const prior = text(c, 'prior', 'gaussian')
      const sparsity = typeof c['sparsity'] === 'number' ? c['sparsity'] : 1
      const mine =
        prior === 'spikeSlab'
          ? 'spike-slab'
          : prior === 'ard' && sparsity !== 1
            ? 'ard-mixed'
            : ['ard', 'horseshoe'].includes(prior)
              ? prior
              : 'gaussian'
      return [
        ['linear', 0],
        [`linear:${noise === 'precision' ? 'precision' : noise === 'learned' ? 'learned' : 'known'}`, 2],
        ['linear:observe-end', 1],
        [`linear:${mine}`, 1],
        ['linear:end', 0],
      ]
    },
    variants: [
      { prior: 'gaussian', noiseModel: 'known' },
      { prior: 'spikeSlab', noiseModel: 'learned' },
      { prior: 'ard', noiseModel: 'precision', sparsity: 1 },
      { prior: 'ard', noiseModel: 'known', sparsity: 0.5 },
      { prior: 'horseshoe', noiseModel: 'learned' },
    ],
  },
  'robust-regression': {
    files: bayesian('robust-regression-method.ts'),
    regions: (c) => {
      const likelihood = text(c, 'likelihood', 'studentT')
      const scale = text(c, 'noiseModel', 'learned') === 'learned' ? 'learned' : 'known'
      const mine =
        likelihood === 'studentT'
          ? `student-${scale}-df-${text(c, 'dfModel', 'learned') === 'fixed' ? 'fixed' : 'learned'}`
          : `${likelihood}-${scale}`
      return around('robust', [`robust:${mine}`, 1])
    },
    variants: [
      { likelihood: 'gaussian', noiseModel: 'known' },
      { likelihood: 'laplace', noiseModel: 'learned' },
      { likelihood: 'mixture', noiseModel: 'known' },
      { likelihood: 'mixture', noiseModel: 'learned' },
      { likelihood: 'studentT', dfModel: 'learned', noiseModel: 'learned' },
      { likelihood: 'studentT', dfModel: 'learned', noiseModel: 'known' },
      { likelihood: 'studentT', dfModel: 'fixed', noiseModel: 'learned' },
      { likelihood: 'studentT', dfModel: 'fixed', noiseModel: 'known' },
      { likelihood: 'gaussian', noiseModel: 'learned' },
      { likelihood: 'laplace', noiseModel: 'known' },
    ],
  },
  'eight-schools': {
    files: bayesian('schools.ts'),
    regions: (c) => {
      const form = text(c, 'model', 'centred')
      if (form === 'no-pooling') return around('schools', ['schools:no-pooling', 1])
      if (form === 'complete-pooling') return around('schools', ['schools:mu', 1], ['schools:complete-pooling', 1])
      return around(
        'schools',
        ['schools:mu', 1],
        ['schools:tau', 1],
        [form === 'non-centred' ? 'schools:non-centred' : 'schools:centred', 1],
      )
    },
    variants: [{ model: 'centred' }, { model: 'non-centred' }, { model: 'complete-pooling' }, { model: 'no-pooling' }],
  },
  radon: {
    files: bayesian('radon-counties.ts'),
    regions: (c) => {
      const pooling = text(c, 'pooling', 'partial')
      if (pooling === 'pooled' || pooling === 'unpooled') return around('radon', [`radon:${pooling}`, 1])
      if (pooling === 'slopes') return around('radon', ['radon:hierarchy', 1], ['radon:slopes', 1])
      return around(
        'radon',
        ['radon:hierarchy', 1],
        ['radon:partial', 1],
        [text(c, 'parameterisation', 'non-centred') === 'centred' ? 'radon:centred' : 'radon:non-centred', 1],
      )
    },
    variants: [
      { pooling: 'pooled' },
      { pooling: 'unpooled' },
      { pooling: 'partial', parameterisation: 'centred' },
      { pooling: 'partial', parameterisation: 'non-centred' },
      { pooling: 'slopes' },
    ],
  },
  'gaussian-mixture': {
    files: bayesian('mixture.ts'),
    regions: (c) =>
      around('mixture', [text(c, 'spread', 'shared') === 'separate' ? 'mixture:spreads' : 'mixture:shared', 1]),
    variants: [{ spread: 'shared' }, { spread: 'separate' }],
  },
  'item-response': {
    files: bayesian('irt-hierarchical.ts'),
    regions: (c) => {
      if (text(c, 'model', 'rasch') !== '2pl') return around('irt', ['irt:rasch', 1])
      return around(
        'irt',
        ['irt:2pl', 1],
        [text(c, 'discriminationPrior', 'log-normal') === 'normal' ? 'irt:2pl-normal' : 'irt:2pl-log-normal', 1],
      )
    },
    variants: [
      { model: 'rasch' },
      { model: '2pl', discriminationPrior: 'log-normal' },
      { model: '2pl', discriminationPrior: 'normal' },
    ],
  },
  'gp-classification': {
    files: bayesian('gp-classification-method.ts'),
    regions: (c) => {
      const link = text(c, 'link', 'logistic') === 'probit' ? 'probit' : 'logistic'
      return text(c, 'hyper', 'fixed') === 'inferred'
        ? [['gp:function', 0], ...around('gp:inferred', [`gp:inferred:${link}`, 1])]
        : around('gp:fixed', [`gp:fixed:${link}`, 1])
    },
    variants: [
      { hyper: 'fixed', link: 'probit' },
      { hyper: 'fixed', link: 'logistic' },
      { hyper: 'inferred', link: 'probit' },
      { hyper: 'inferred', link: 'logistic' },
    ],
  },
  'bayesian-gam': {
    files: gamFiles,
    inference: ['gam:offer', 'gam:bind', 'gam:run'],
    regions: (c) => {
      const prior = text(c, 'prior', 'penalty')
      if (prior === 'ard') return around('gam:local', ['gam:local:ard', 1])
      if (prior === 'spikeSlab') return around('gam:local', ['gam:local:spike-slab', 1])
      const hyper = text(c, 'hyper', 'gammaSmooth')
      if (hyper === 'gammaSmooth') return [['gam:gamma-smooth', 0]]
      if (hyper === 'logNormal') return [['gam:log-normal', 0]]
      return around('gam:known-scale', [hyper === 'fixed' ? 'gam:known-scale:fixed' : 'gam:known-scale:learned', 1])
    },
    variants: [
      { prior: 'penalty', hyper: 'gammaSmooth' },
      { prior: 'penalty', hyper: 'logNormal' },
      { prior: 'penalty', hyper: 'gammaNoise' },
      { prior: 'penalty', hyper: 'fixed' },
      { prior: 'ard' },
      { prior: 'spikeSlab' },
    ],
  },
}

/**
 * The code of a page for a configuration: the pieces of its model (indented to where they sit), then the regions that choose,
 * bind and run the inference method.
 *
 * @param code - The page's entry in {@link exampleCode}.
 * @param config - The configuration.
 * @param files - The source files ({@link sourcesOf}).
 * @returns The code, the file the model is in, and the names of regions that were not found in the sources (none when it is whole).
 */
export function codeOf(
  code: ExampleCode,
  config: Config,
  files: Readonly<Record<string, string>>,
): { readonly code: string; readonly file: string; readonly missing: readonly string[] } {
  const missing: string[] = []
  const find = (name: string): { found: string; file: string } => {
    for (const [file, source] of Object.entries(files)) {
      const found = snippetOf(source, { region: name })
      if (found.trim() !== '') return { found, file }
    }
    missing.push(name)
    return { found: '', file: '' }
  }
  let file = ''
  const model = code
    .regions(config)
    .map(([name, depth]) => {
      const piece = find(name)
      file ||= piece.file
      const pad = '  '.repeat(depth)
      return piece.found
        .split('\n')
        .map((l) => (l === '' ? l : pad + l))
        .join('\n')
    })
    .join('')
  const inference = (code.inference ?? INFERENCE).map((name) => find(name).found).join('\n')
  return {
    code: `${model}\n// the inference: the method chosen from the library's table, run on the model\n${inference}`,
    file,
    missing,
  }
}
