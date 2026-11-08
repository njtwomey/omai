/**
 * Write the family files of the completeness survey from `docs/completeness/data/match-<lib>.json`.
 *
 *   node scripts/completeness-survey/report.ts
 *
 * Output: `docs/completeness/<family>.md` (sklearn, scipy-statsmodels, probabilistic, graphical-models, rl, torch,
 * graph-neural-networks, other) and `docs/completeness/data/counts.json`. `README.md` and `repertoire.md` are written
 * by hand. Run `match.ts` first. The prose of each family (what it covers, where omai is strong) is in this file.
 */
import { readdirSync, readFileSync, writeFileSync } from 'node:fs'
import { join, resolve } from 'node:path'

const ROOT = resolve(import.meta.dirname, '..', '..')
const DIR = join(ROOT, 'docs', 'completeness')
const DATA = join(DIR, 'data')

interface Row {
  lib: string
  area: string
  name: string
  status: 'full' | 'partial' | 'absent' | 'skip'
  how: string
  omai?: string
  note?: string
  home?: string
  size?: string
  priority?: string
  group?: string
}

interface Family {
  file: string
  title: string
  libs: string[]
  intro: string
  strong: string[]
}

const FAMILIES: Family[] = [
  {
    file: 'sklearn',
    title: 'scikit-learn and imbalanced-learn',
    libs: ['sklearn', 'imblearn'],
    intro:
      'scikit-learn is the reference for the estimator API, preprocessing, model selection, metrics, clustering, ' +
      'decomposition and the classical learners. omai does not copy the class-per-model layout: a family is one ' +
      'configurable estimator (`linearModel` covers LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression, ' +
      'Poisson, Gamma, Tweedie and SGD). Matching is therefore done by concept, with the nearest omai name written ' +
      'down in `scripts/completeness-survey/verdicts/sklearn.txt`.',
    strong: [
      'Linear and generalised linear models, path algorithms (LARS, coordinate descent), robust and quantile regression, with the solver as a slot.',
      'Trees, forests, boosting, stacking, voting; kernels (including approximations); SVMs; Gaussian processes with exact marginal likelihood.',
      'Model selection: nearly every splitter of scikit-learn plus nested CV, learning and validation curves, permutation tests; calibration and conformal prediction well beyond scikit-learn.',
      'Clustering (k-means variants, DBSCAN, OPTICS, HDBSCAN, spectral, affinity propagation, BIRCH), manifold learning (t-SNE, UMAP, PaCMAP, TriMap), covariance estimation.',
      'Every iterative routine is stepped and traced; scikit-learn offers `partial_fit` at best.',
    ],
  },
  {
    file: 'scipy-statsmodels',
    title: 'scipy, statsmodels and time-series libraries',
    libs: ['scipy', 'statsmodels', 'sktime', 'tslearn', 'prophet'],
    intro:
      'scipy supplies the numerical and statistical base (stats, optimize, linalg, signal, interpolate, integrate, ' +
      'spatial, special, sparse, fft, cluster); statsmodels the regression, GLM, test and time-series classics; ' +
      'sktime, tslearn and prophet the forecasting and time-series-learning layer. scipy and statsmodels were ' +
      'introspected; sktime, tslearn and prophet are from memory.',
    strong: [
      'Statistics: about 170 test exports, power analysis, effect sizes, resampling, copulas, extreme-value fits, smoothing; the distribution families of `dists` carry log-densities, sampling and bijectors under one contract.',
      'Optimisation: all the scipy minimisers plus optax-style optimisers, trust regions, LP/QP/MILP, global optimisers, and every one runs as a traced Algorithm.',
      'Time series: ARMA/ARIMA, exponential smoothing and ETS, STL and MSTL, VAR and VECM, Johansen, GARCH family, Kalman/EKF/UKF/particle filters, matrix profile, change-point detection, shapelets.',
      'Numerics: dense and sparse linear algebra, Krylov solvers, quadrature, interpolation, FFT/NUFFT, polynomials, inverse problems, ODE/PDE solvers with dense output and events.',
      'Signal processing and LTI systems cover scipy.signal at its full width, plus audio features.',
    ],
  },
  {
    file: 'probabilistic',
    title: 'Probabilistic programming and Bayesian workflow',
    libs: ['numpyro', 'arviz', 'pymc', 'tensorflow_probability'],
    intro:
      'NumPyro, PyMC, TensorFlow Probability and ArviZ cover probabilistic programs, inference engines, ' +
      'distributions and bijectors, and the Bayesian diagnostic workflow. torch.distributions is in torch.md. ' +
      'NumPyro and ArviZ were introspected; PyMC and TFP are from memory.',
    strong: [
      'Programs with `sample`, `observe`, plates and effect handlers; MCMC (HMC, NUTS, Gibbs, slice, MALA, SGLD, SMC, tempered SMC), variational inference with auto-guides, Laplace, ABC, exact enumeration and conjugate engines behind one `infer`.',
      'Diagnostics as numbers: R-hat, ESS, MCSE, HDI, divergences, BFMI, PSIS-LOO, WAIC, model comparison and stacking weights.',
      'Distributions: dozens of families, compositions (mixture, truncated, censored, transformed), copulas, GAMLSS families; about 50 bijector exports including coupling and autoregressive flows.',
      'Every sampler is a stepped algorithm with the chain on the tape, so a posterior can be watched filling in.',
    ],
  },
  {
    file: 'graphical-models',
    title: 'Graphical models',
    libs: ['pgmpy', 'hmmlearn'],
    intro:
      "pgmpy is the closest library to omai's `pgm` domain: factors, Bayesian and Markov networks, structure and " +
      'parameter learning, exact and approximate inference, causal inference. hmmlearn covers HMMs. Both were ' +
      'introspected. pyAgrum and libpgm were not installed and are not listed separately; their scope is the same.',
    strong: [
      'Exact inference: variable elimination with elimination-order heuristics, junction trees, belief propagation, max-product and MAP, MPLP, treewidth; approximate: Gibbs, likelihood weighting, rejection, mean field, variational message passing, EP, tree-reweighted BP.',
      'Models beyond pgmpy: CRFs (linear-chain and general), Gaussian MRFs, influence diagrams, plate models, factor graphs, Ising and Potts samplers (Swendsen-Wang, Wolff), HMM variants (HSMM, IO-HMM, factorial, GMM-HMM), linear dynamical systems.',
      'Structure learning (PC, GES, hill climbing, Chow-Liu, NOTEARS, DirectLiNGAM, FCI), identification (backdoor, front-door, instruments), counterfactuals and twin networks.',
      'Inference algorithms run stepwise on the tape (messages, eliminations, sweeps).',
    ],
  },
  {
    file: 'rl',
    title: 'Reinforcement learning',
    libs: ['gymnasium', 'stable_baselines3', 'cleanrl', 'pettingzoo'],
    intro:
      'gymnasium for environments, wrappers, spaces and vector environments; Stable-Baselines3 and CleanRL for the ' +
      'algorithms and their supporting pieces (policies, buffers, noise, callbacks); PettingZoo for multi-agent ' +
      'environments. gymnasium was introspected; the others are from memory.',
    strong: [
      'Tabular and function-approximation methods of Sutton and Barto (DP, MC, TD, n-step, eligibility traces, tile coding, Dyna, prioritised sweeping) plus bandits with regret bounds and best-arm identification.',
      'Deep RL: DQN family (double, dueling, C51, QR-DQN, IQN, Rainbow), PPO, TRPO, A2C, DDPG, TD3, SAC, HER, model-based (PETS, MBPO, Dreamer, MuZero), offline (BCQ, CQL, IQL, TD3+BC, Decision Transformer), imitation (DAgger, GAIL, MaxEnt IRL), multi-agent (VDN, QMIX, MADDPG), safe RL (shields, Lagrangian PPO), curiosity.',
      'Infrastructure: wrappers, vector environments, replay (uniform, prioritised, n-step), returns and advantage estimators including V-trace and Retrace, evaluation statistics with bootstrap intervals, AEC and parallel multi-agent APIs.',
      'Environments: classic control, toy text, bandits, MinAtar, grid worlds, several continuous-control tasks with analytic models, board games (tic-tac-toe, Connect Four, Go, Nim), Kuhn and Leduc poker, MPE-style particle worlds.',
    ],
  },
  {
    file: 'torch',
    title: 'PyTorch',
    libs: ['torch', 'torchvision', 'torchmetrics'],
    intro:
      'torch.nn (layers and containers), torch.nn.functional, torch.nn.init, torch.optim and its schedulers, ' +
      'torch.distributions, torch.autograd and torch.func, torch.utils.data, plus torchvision transforms and datasets ' +
      'and torchmetrics. All introspected from the installed versions.',
    strong: [
      'Layers: nearly all the convolution, pooling, normalisation, recurrent and attention layers of torch.nn, plus Mamba, RetNet, KAN, neural ODEs/CDEs/SDEs, Fourier neural operators, EGNN, mixture of experts, diffusion U-Net pieces, Hopfield layers.',
      'Optimisers: SGD with momentum, Adam family, Lion, LAMB, Adafactor, Shampoo, L-BFGS and 15 schedules as one slot; gradient clipping; EMA, SWAG.',
      'Autodiff: JAX-style `grad`, `vjp`, `jvp`, `vmap`, Hessians, Taylor mode, checkpointing, `checkPrimitive` for new primitives; differentiation is opt-in so there is no `no_grad`.',
      'Losses: about 190 exports, including contrastive, ranking, preference (DPO, KTO, ORPO), score matching and diffusion objectives.',
      'Interpretability and compression built in: LoRA, pruning, quantisation (GPTQ, AWQ), adversarial attacks and certification.',
    ],
  },
  {
    file: 'graph-neural-networks',
    title: 'Graph neural networks and graph libraries',
    libs: ['torch_geometric', 'networkx'],
    intro:
      'PyTorch Geometric (and DGL, which has the same scope) for GNN layers, pooling, batching, samplers, transforms ' +
      "and datasets; NetworkX for classical graph algorithms, since omai's `graph` domain is NetworkX-shaped. " +
      'NetworkX was introspected; PyG is from memory (it is not installed).',
    strong: [
      'Layers: GCN, GraphSAGE, GAT, GIN, relational GCN, EGNN and a graph transformer, on a general message-passing function; spectral filters (Chebyshev, heat kernel, wavelets).',
      'Sampling: neighbour sampling with variance estimates, layer-wise importance sampling, random-walk subgraphs, node2vec walks; positional encodings (Laplacian, random walk); Weisfeiler-Leman.',
      'Classical algorithms: centralities, community detection (Louvain, Leiden, Girvan-Newman, label propagation), flows and cuts, matching, shortest paths, spanning trees, isomorphism (VF2), percolation, epidemics on networks.',
      'Everything above that is iterative runs as a traced algorithm.',
    ],
  },
  {
    file: 'other',
    title: 'Other domains: causal, fairness, explainability, survival, anomaly, vision, text and tuning',
    libs: [
      'dowhy',
      'econml',
      'fairlearn',
      'shap',
      'captum',
      'lifelines',
      'scikit_survival',
      'pyod',
      'skimage',
      'nltk',
      'gensim',
      'optuna',
    ],
    intro:
      'The nearest libraries for the domains of omai that the main families do not cover: DoWhy and EconML (causal), ' +
      'fairlearn (fairness), SHAP and Captum (explainability), lifelines and scikit-survival (survival), PyOD ' +
      '(anomaly detection), scikit-image (vision), NLTK and gensim (text), Optuna (tuning). Introspected: fairlearn, ' +
      'shap, captum, scikit-image, nltk, gensim, optuna. From memory: DoWhy, EconML, lifelines, scikit-survival, ' +
      'PyOD. Kornia, spaCy, Hugging Face and similar were skipped: omai has no pretrained-model domain by design.',
    strong: [
      'Explainability: exact, kernel, sampling and tree SHAP, LIME, anchors, Integrated Gradients and the Captum attribution set, counterfactuals and recourse, TCAV, influence functions, data Shapley, mechanistic-interpretability tools (activation patching, logit lens, sparse autoencoders).',
      'Fairness: group, individual and intersectional measures, pre-, in- and post-processing methods, fairness under feedback, with `metricFrame`.',
      'Causal: regression adjustment, IPW, matching, AIPW, TMLE, DML, meta-learners, causal forests, instruments, RDD, synthetic control, difference-in-differences, sensitivity analysis.',
      'Survival: Kaplan-Meier, Nelson-Aalen, Cox (Breslow/Efron), AFT, log-rank family, Brier and dynamic AUC.',
      'Vision: OpenCV-shaped classical pipeline from filters through features, geometry, stereo and bundle adjustment.',
      'Text: tokenisers, stemmers, subword models, n-gram language models with smoothing, TF-IDF/BM25, Earley parsing and PCFG training, word2vec, fastText.',
    ],
  },
]

const PRI: Record<string, number> = { high: 3, medium: 2, low: 1 }
const SIZE: Record<string, number> = { L: 3, M: 2, S: 1 }

function cell(s: string | undefined): string {
  return (s ?? '').replace(/\|/g, '/').replace(/\s+/g, ' ').trim()
}

function shortArea(lib: string, area: string): string {
  return area.startsWith(`${lib}.`) ? area.slice(lib.length + 1) : area
}

function names(rows: Row[], max = 12): string {
  const n = rows.map((r) => r.name)
  const shown = n.slice(0, max).map((x) => `\`${x}\``)
  return n.length > max ? `${shown.join(', ')} and ${n.length - max} more` : shown.join(', ')
}

function main(): void {
  const rows: Row[] = readdirSync(DATA)
    .filter((f) => f.startsWith('match-') && f.endsWith('.json'))
    .sort()
    .flatMap((f) => JSON.parse(readFileSync(join(DATA, f), 'utf8')) as Row[])
  const versions: Record<string, { version: string; source: string }> = {}
  for (const r of rows) versions[r.lib] ??= { version: '', source: '' }
  const refs = (lib: string): { version: string; source: string } => {
    try {
      const d = JSON.parse(readFileSync(join(DATA, `ref-${lib}.json`), 'utf8')) as { version: string }
      return { version: d.version, source: 'introspection' }
    } catch {
      return { version: '', source: 'memory' }
    }
  }
  const counts: Record<string, unknown> = {}
  for (const fam of FAMILIES) {
    const out: string[] = []
    const famRows = rows.filter((r) => fam.libs.includes(r.lib))
    const tot = { full: 0, partial: 0, absent: 0, skip: 0, notLow: 0 }
    for (const r of famRows) {
      tot[r.status]++
      if ((r.status === 'absent' || r.status === 'partial') && (r.priority === 'high' || r.priority === 'medium'))
        tot.notLow++
    }
    counts[fam.file] = { ...tot, libs: {} as Record<string, unknown> }
    out.push(`# ${fam.title}`, '')
    out.push(`Part of the [completeness survey](README.md). ${fam.intro}`, '')
    out.push(
      'Counts are of names: "covered" means an omai export or a documented configuration does what the reference name does, ' +
        '"partial" means a different shape or fewer options, "absent" means nothing found by name or by concept, and ' +
        '"out of scope" is plumbing, GPU or production features, downloads and the like that omai should not copy. ' +
        'Semantics were not compared.',
      '',
    )
    out.push('## Summary', '')
    out.push('| library | source | area | covered | partial | absent | out of scope |')
    out.push('| --- | --- | --- | ---: | ---: | ---: | ---: |')
    for (const lib of fam.libs) {
      const lr = famRows.filter((r) => r.lib === lib)
      if (lr.length === 0) continue
      const v = refs(lib)
      const areas = [...new Set(lr.map((r) => r.area))].sort()
      const lt = { full: 0, partial: 0, absent: 0, skip: 0, notLow: 0 }
      for (const a of areas) {
        const c = { full: 0, partial: 0, absent: 0, skip: 0 }
        for (const r of lr.filter((x) => x.area === a)) {
          c[r.status]++
          if ((r.status === 'absent' || r.status === 'partial') && (r.priority === 'high' || r.priority === 'medium'))
            lt.notLow++
        }
        for (const k of ['full', 'partial', 'absent', 'skip'] as const) lt[k] += c[k]
        out.push(`|  |  | ${shortArea(lib, a)} | ${c.full} | ${c.partial} | ${c.absent} | ${c.skip} |`)
      }
      out.push(
        `| **${lib}** | ${v.source}${v.version ? ` ${v.version}` : ''} | **all** | **${lt.full}** | **${lt.partial}** | **${lt.absent}** | **${lt.skip}** |`,
      )
      ;(counts[fam.file] as { libs: Record<string, unknown> }).libs[lib] = { ...lt, ...v }
    }
    out.push('')
    out.push('## Where omai is already strong', '')
    for (const s of fam.strong) out.push(`- ${s}`)
    out.push('')
    out.push('## Absent and partial items', '')
    out.push(
      'Priority is high, medium or low by how many learners would expect the thing, how many pages it unlocks and how well ' +
        'it shows off a stepped, traced run; not by recency. Sizes: S a day or less, M a few days, L a week or more. ' +
        'Homes follow `docs/structure.md`. "Nearest in omai" is what a user would reach for today. Items are grouped; ' +
        'the full list of names per group is in `data/match-<lib>.json`.',
      '',
    )
    for (const lib of fam.libs) {
      const lr = famRows.filter((r) => r.lib === lib && (r.status === 'absent' || r.status === 'partial'))
      if (lr.length === 0) continue
      out.push(`### ${lib}`, '')
      const areas = [...new Set(lr.map((r) => r.area))].sort()
      // group rows: by explicit group, else by (area, status, note)
      const groups = new Map<string, Row[]>()
      for (const r of lr) {
        const key = r.group ? `g|${r.group}|${r.status}` : `a|${r.area}|${r.status}|${r.note ?? ''}|${r.omai ?? ''}`
        groups.set(key, [...(groups.get(key) ?? []), r])
      }
      const list = [...groups.values()].sort((a, b) => {
        const pa = Math.max(...a.map((r) => PRI[r.priority ?? 'low'] ?? 1))
        const pb = Math.max(...b.map((r) => PRI[r.priority ?? 'low'] ?? 1))
        return pb - pa || b.length - a.length
      })
      void areas
      out.push(
        '| reference (area) | what it is and why it matters | nearest in omai | proposed home | size | priority | status |',
      )
      out.push('| --- | --- | --- | --- | --- | --- | --- |')
      for (const g of list) {
        const first = g[0]
        if (!first) continue
        const withNote = g.find((r) => r.note) ?? first
        const pri = g.reduce(
          (m, r) => ((PRI[r.priority ?? 'low'] ?? 1) > (PRI[m] ?? 1) ? (r.priority ?? 'low') : m),
          'low',
        )
        const size = g.reduce((m, r) => ((SIZE[r.size ?? ''] ?? 0) > (SIZE[m] ?? 0) ? (r.size ?? '') : m), '')
        const home = g.find((r) => r.home)?.home
        const areaSet = [...new Set(g.map((r) => shortArea(lib, r.area)))]
        const label = first.group ? `**${first.group}**: ${names(g)}` : `${names(g)}`
        const what =
          cell(withNote.note) ||
          (first.omai ? 'only the nearest thing in the next column' : 'no counterpart found by name or concept')
        const status = [...new Set(g.map((r) => r.status))].join('/')
        out.push(
          `| ${cell(label)} (${cell(areaSet.slice(0, 2).join(', '))}${areaSet.length > 2 ? ', ...' : ''}) | ${what} | ${cell(first.omai) || 'nothing close'} | ${cell(home) || 'to be placed'} | ${size || '-'} | ${pri} | ${status} |`,
        )
      }
      out.push('')
    }
    writeFileSync(join(DIR, `${fam.file}.md`), `${out.join('\n')}\n`)
  }
  writeFileSync(join(DATA, 'counts.json'), `${JSON.stringify(counts, null, 1)}\n`)
  console.log(JSON.stringify(counts, null, 0))
}

main()
