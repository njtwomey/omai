/**
 * The registry entries of the datasets vendored from the UCI Machine Learning Repository: `uciEntries` makes one
 * vendored entry for each loader of `classification-uci.ts`, `regression-uci.ts` and `tables-uci.ts`, with the title,
 * description, size, licence and source of the provenance table, so the catalogue lists them beside the other real
 * datasets and the dataset selectors find them by kind, text and real flag.
 */

import { OmaiError } from '../base/index.ts'
import type { Dataset } from '../data/index.ts'
import { UCI_INFO, UCI_LICENCE, uciSource } from './_uci.ts'
import {
  balanceScale,
  banknoteAuthentication,
  ecoli,
  glassIdentification,
  haberman,
  ionosphere,
  lenses,
  seeds,
  sonar,
  spambase,
  vehicleSilhouettes,
  wholesaleCustomers,
  yeast,
} from './classification-uci.ts'
import { type DatasetEntry, vendoredEntry } from './registry.ts'
import {
  airfoilSelfNoise,
  concreteStrength,
  energyEfficiency,
  wineQualityRed,
  wineQualityWhite,
  yachtHydrodynamics,
} from './regression-uci.ts'
import {
  abalone,
  adultIncome,
  autoMpg,
  bikeSharingDaily,
  carEvaluation,
  congressionalVoting,
  dermatology,
  forestFires,
  heartDisease,
  hepatitis,
  mushroom,
  servo,
  studentMath,
  ticTacToe,
  zoo,
} from './tables-uci.ts'

/** The loaders of the UCI datasets by name, in the order of the provenance table. */
const LOADERS: readonly (readonly [name: string, load: () => Dataset])[] = [
  ['banknoteAuthentication', banknoteAuthentication],
  ['ionosphere', ionosphere],
  ['sonar', sonar],
  ['glassIdentification', glassIdentification],
  ['seeds', seeds],
  ['ecoli', ecoli],
  ['yeast', yeast],
  ['haberman', haberman],
  ['balanceScale', balanceScale],
  ['vehicleSilhouettes', vehicleSilhouettes],
  ['wholesaleCustomers', wholesaleCustomers],
  ['lenses', lenses],
  ['spambase', spambase],
  ['concreteStrength', concreteStrength],
  ['energyEfficiency', energyEfficiency],
  ['yachtHydrodynamics', yachtHydrodynamics],
  ['airfoilSelfNoise', airfoilSelfNoise],
  ['wineQualityRed', wineQualityRed],
  ['wineQualityWhite', wineQualityWhite],
  ['autoMpg', autoMpg],
  ['abalone', abalone],
  ['heartDisease', heartDisease],
  ['hepatitis', hepatitis],
  ['zoo', zoo],
  ['ticTacToe', ticTacToe],
  ['carEvaluation', carEvaluation],
  ['congressionalVoting', congressionalVoting],
  ['mushroom', mushroom],
  ['adultIncome', adultIncome],
  ['dermatology', dermatology],
  ['forestFires', forestFires],
  ['studentMath', studentMath],
  ['servo', servo],
  ['bikeSharingDaily', bikeSharingDaily],
]

/**
 * The registry entries of the UCI datasets: real tabular data, each a classification, regression or table dataset
 * (the table ones with categorical features and explicit missing values), tagged with its kind, size, licence (CC BY
 * 4.0) and source file.
 *
 * @returns One vendored entry per dataset.
 * @concept datasets
 * @example
 * ```ts
 * import { uciEntries } from 'omai-engine/datasets'
 * uciEntries().length // => 34
 * uciEntries().filter((e) => e.kind === 'table').length // => 15
 * ```
 */
export function uciEntries(): DatasetEntry[] {
  return LOADERS.map(([name, load]) => {
    const info = UCI_INFO[name]
    if (info === undefined) throw new OmaiError('value', `uciEntries: no provenance for '${name}'`)
    return vendoredEntry(name, load, {
      title: info.title,
      description: info.description,
      kind: info.kind,
      modality: 'tabular',
      size: info.size,
      licence: UCI_LICENCE,
      source: uciSource(name),
    })
  })
}
