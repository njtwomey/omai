/**
 * The contract every structured model offers to a learner. A structured model scores a structured output y for an input
 * x with a linear function of a parameter vector, score(w; x, y) = w . phi(x, y), and a learner needs only four things of
 * it: the feature counts phi(x, y) of a labelling, the expected feature counts and log-partition function under the
 * model (for maximum likelihood), a decoder for the best labelling, optionally against a loss (for the perceptron,
 * passive-aggressive updates and structured SVMs), and the task loss. Linear chains, trees, grids, skip chains, general
 * graphs, semi-Markov and higher-order CRFs, MEMMs and latent-variable models each implement the oracle once and every
 * solver of `structuredSolvers` then trains them; none of them has to know about the others.
 */

/**
 * What a learner may ask of a structured model whose score is linear in its parameters. `X` is an input (the item and
 * edge attributes of a sequence, the node and edge attributes of a graph) and `Y` an output (a labelling).
 *
 * Implementations are pure: they never keep a weight vector, so a solver can evaluate any point. The parameter layout
 * (which index belongs to which feature) is the model's own and is the same in `features`, `expectations` and `decode`.
 *
 * @index structure The contract a structured model offers a learner: features, expectations, decoding, loss.
 */
export interface StructuredOracle<X, Y> {
  /** The number of parameters. */
  readonly size: number
  /**
   * The feature counts phi(x, y) of a labelling: the vector whose inner product with the weights is its score (counted
   * with attribute values, so a feature that fires twice counts twice).
   */
  features(x: X, y: Y): Float64Array
  /**
   * The expected feature counts E[phi(x, Y)] and the log-partition function log Z(x) under p(y | x) proportional to
   * exp(w . phi(x, y)): exact (forward-backward, a junction tree) or approximate (loopy belief propagation, the Bethe
   * approximation), as the model says in its own documentation.
   */
  expectations(w: ArrayLike<number>, x: X): { readonly counts: Float64Array; readonly logPartition: number }
  /**
   * The expected feature counts and log-partition function with the output clamped to what is known of `y`, for models
   * whose training labels are partial or whose labelling has latent parts (partial-label and hidden CRFs): the gradient
   * of the negative log-likelihood is then the free minus the clamped expectations. Omitted for a model whose labels are
   * complete, where the clamped counts are `features(x, y)` and the clamped log-partition function is `w . phi(x, y)`.
   */
  clamped?(w: ArrayLike<number>, x: X, y: Y): { readonly counts: Float64Array; readonly logPartition: number }
  /**
   * The highest-scoring labelling under the weights and its score. With `against` (a gold labelling) the score of every
   * labelling is raised by its task loss against that gold one (cost-augmented, or loss-augmented, decoding: what the
   * passive-aggressive update and the structured SVM's separation oracle need); the returned score includes the cost.
   */
  decode(w: ArrayLike<number>, x: X, options?: { readonly against?: Y }): { readonly y: Y; readonly score: number }
  /** The task loss between a gold and a predicted labelling (the Hamming distance for a sequence labeller). */
  loss(gold: Y, predicted: Y): number
}

/** A labelled example of a structured model. */
export interface StructuredExample<X, Y> {
  readonly x: X
  readonly y: Y
}
