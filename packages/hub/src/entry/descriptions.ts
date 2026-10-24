/**
 * One line per hub collection (and per folder of collections) for its index page: what the methods in it have in
 * common and what their demos show. Keyed by collection path, as entries' `collection`.
 */
export const collectionDescriptions: Readonly<Record<string, string>> = {
  sklearn:
    'Methods grouped as scikit-learn groups them, each live on small datasets with the records that explain its fit.',
  'sklearn/cluster':
    'Clustering: centroid, medoid, density, hierarchical and graph methods, each started on data that shows where it shines.',
  'sklearn/covariance': 'Covariance estimation, robust and sparse, and the outliers it exposes.',
  'sklearn/decomposition':
    'Linear and kernel factorisations: components, explained variance, loadings and the data in the new coordinates.',
  'sklearn/discriminant-analysis':
    'Gaussian class models: shared covariances give linear boundaries, own ones quadratic.',
  'sklearn/ensemble':
    'Many models made one: bagging and forests average, boosting adds learners stage by stage; watch the stages.',
  'sklearn/gaussian-process':
    'Distributions over functions: posterior bands, and the marginal likelihood the kernel is fitted on.',
  'sklearn/isotonic': 'Monotone regression by pooling adjacent violators.',
  'sklearn/kernel-ridge': 'Ridge regression in a kernel’s feature space, its penalty chosen by cross-validation.',
  'sklearn/linear-model':
    'Linear and generalised linear models: regularisation paths, cross-validation curves, intervals and robust fits.',
  'sklearn/manifold':
    'Non-linear embeddings in the plane: some unroll a manifold (watch the colours stay in order), some pull clusters apart.',
  'sklearn/mixture': 'Gaussian mixtures fitted by EM, step by step; the Bayesian one switches spare components off.',
  'sklearn/multiclass': 'Multiclass problems reduced to binary ones: one-vs-rest, one-vs-one and output codes.',
  'sklearn/naive-bayes':
    'Classes as products of independent per-feature likelihoods, and how much each feature adds to a prediction.',
  'sklearn/neighbors':
    'Methods built on distances to training points: neighbour votes, centroids, kernel densities and outlier factors.',
  'sklearn/random-projection': 'Random low-dimensional projections that roughly keep distances.',
  'sklearn/semi-supervised':
    'Learning from a few labels and many unlabelled points: labels flow along a graph, or a classifier labels its own data.',
  'sklearn/svm': 'Support vector machines: margins, the support vectors that set them, and the SMO solver’s progress.',
  'sklearn/tree': 'Decision trees grown split by split, every candidate split scored, then pruned by cost complexity.',
  generative:
    'Generative models learning toy data live: flows, autoencoders, GANs and diffusion, their samples and densities step by step.',
  'generative/flows': 'Normalising flows: invertible maps from a Gaussian to the data, trained by exact likelihood.',
  pyod: 'Anomaly detectors outside scikit-learn, after PyOD: scores, thresholds and the evidence behind each flag.',
  probabilistic: 'Probabilistic models and density estimators.',
  'probabilistic/density': 'Density estimation without a model: histograms and their kernel-smoothed cousins.',
  rl: 'A reinforcement-learning playground: agents learning live in their environments, episode by episode.',
  'rl/bandits': 'Bandits: exploration against exploitation, one pull at a time.',
  'rl/classic-control': 'Classic control: carts, poles and pendulums balanced by learned policies.',
  'rl/gridworlds': 'Grid worlds: value functions and policies spreading out from the goal.',
  tuning: 'Hyperparameter searches over a method and a dataset: every trial, and what the search learned.',
  'complex-systems':
    'Complex systems: simple local rules (cells, agents) and the large patterns they grow, played step by step.',
  evolutionary:
    'Evolutionary computation: populations selected, recombined and mutated, played generation by generation.',
  patterns:
    'Pattern mining: frequent itemsets found level by level or from a prefix tree, and the association rules they imply.',
  'classical-ai': 'Classical AI: searches, game players, constraint solvers and planners, each played step by step.',
  'classical-ai/adversarial':
    'Game-tree search: values backed up from the leaves, and the branches alpha-beta never needs to look at.',
  'classical-ai/csp':
    'Constraint satisfaction: queens, Sudoku and maps solved value by value, with the values each choice rules out.',
  'classical-ai/planning':
    'Classical planning: a search through world states guided by a relaxed plan, and the plan it finds carried out.',
  'classical-ai/search':
    'Path search on grid worlds and mazes: watch the frontier and the explored set grow, expansion by expansion.',
}

/**
 * The description of a hub collection, or of a folder of collections.
 *
 * @param path - A collection path (`sklearn/cluster`) or a folder of collections (`sklearn`).
 * @returns Its one-line description, if it has one.
 * @example
 * ```ts
 * import { collectionDescription } from 'whitebox-hub'
 * collectionDescription('sklearn/svm') === undefined // => false
 * collectionDescription('nowhere') // => undefined
 * ```
 */
export function collectionDescription(path: string): string | undefined {
  return collectionDescriptions[path]
}
