/**
 * Real numeric regression datasets from the UCI Machine Learning Repository (all CC BY 4.0): concrete compressive
 * strength, energy efficiency (two targets), yacht hydrodynamics, airfoil self-noise and the red and white wine
 * quality. Each is a vendored payload generated from the published file by `oracle/datasets`.
 */

import type { RegressionData } from '../data/index.ts'
import {
  AIRFOIL_FEATURES,
  AIRFOIL_SHA256,
  AIRFOIL_SHAPE,
  AIRFOIL_TARGETS,
  AIRFOIL_X,
  AIRFOIL_Y,
} from './_data/airfoil.data.ts'
import {
  CONCRETE_FEATURES,
  CONCRETE_SHA256,
  CONCRETE_SHAPE,
  CONCRETE_TARGETS,
  CONCRETE_X,
  CONCRETE_Y,
} from './_data/concrete.data.ts'
import {
  ENERGY_FEATURES,
  ENERGY_SHA256,
  ENERGY_SHAPE,
  ENERGY_TARGETS,
  ENERGY_X,
  ENERGY_Y,
} from './_data/energy.data.ts'
import {
  WINE_QUALITY_RED_FEATURES,
  WINE_QUALITY_RED_SHA256,
  WINE_QUALITY_RED_SHAPE,
  WINE_QUALITY_RED_TARGETS,
  WINE_QUALITY_RED_X,
  WINE_QUALITY_RED_Y,
} from './_data/wine-quality-red.data.ts'
import {
  WINE_QUALITY_WHITE_FEATURES,
  WINE_QUALITY_WHITE_SHA256,
  WINE_QUALITY_WHITE_SHAPE,
  WINE_QUALITY_WHITE_TARGETS,
  WINE_QUALITY_WHITE_X,
  WINE_QUALITY_WHITE_Y,
} from './_data/wine-quality-white.data.ts'
import { YACHT_FEATURES, YACHT_SHA256, YACHT_SHAPE, YACHT_TARGETS, YACHT_X, YACHT_Y } from './_data/yacht.data.ts'
import { uciRegression } from './_uci.ts'

/**
 * The UCI concrete compressive strength data: cement, blast furnace slag, fly ash, water, superplasticizer, coarse and
 * fine aggregate (kg per cubic metre) and age in days of 1030 concrete mixes, with the compressive strength in MPa.
 * Read from the CSV rendering of the repository's spreadsheet. Embedded from the repository's `data.csv` (CC BY 4.0).
 *
 * @returns The dataset: features `x`, the target or targets `y`, and their names.
 * @concept datasets
 * @example
 * ```ts
 * import { concreteStrength } from 'omai-engine/datasets'
 * concreteStrength().x.shape // => [1030, 8]
 * ```
 */
export function concreteStrength(): RegressionData {
  return uciRegression('concreteStrength', {
    shape: CONCRETE_SHAPE,
    x: CONCRETE_X,
    y: CONCRETE_Y,
    features: CONCRETE_FEATURES,
    targets: CONCRETE_TARGETS,
    sha256: CONCRETE_SHA256,
  })
}

/**
 * The UCI energy efficiency data: eight attributes (relative compactness, surface, wall and roof area, height,
 * orientation, glazing area and its distribution) of 768 simulated buildings, with two targets, the heating and the
 * cooling load. Read from the CSV rendering of the repository's spreadsheet. Embedded from the repository's `data.csv`
 * (CC BY 4.0).
 *
 * @returns The dataset: features `x`, the target or targets `y`, and their names.
 * @concept datasets
 * @example
 * ```ts
 * import { energyEfficiency } from 'omai-engine/datasets'
 * energyEfficiency().y.shape // => [768, 2]
 * ```
 */
export function energyEfficiency(): RegressionData {
  return uciRegression('energyEfficiency', {
    shape: ENERGY_SHAPE,
    x: ENERGY_X,
    y: ENERGY_Y,
    features: ENERGY_FEATURES,
    targets: ENERGY_TARGETS,
    sha256: ENERGY_SHA256,
  })
}

/**
 * The UCI yacht hydrodynamics data: six hull coefficients and the Froude number of 308 full-scale experiments, with
 * the residuary resistance per unit weight of displacement. Embedded from the repository's `yacht_hydrodynamics.data`
 * (CC BY 4.0).
 *
 * @returns The dataset: features `x`, the target or targets `y`, and their names.
 * @concept datasets
 * @example
 * ```ts
 * import { yachtHydrodynamics } from 'omai-engine/datasets'
 * yachtHydrodynamics().x.shape // => [308, 6]
 * ```
 */
export function yachtHydrodynamics(): RegressionData {
  return uciRegression('yachtHydrodynamics', {
    shape: YACHT_SHAPE,
    x: YACHT_X,
    y: YACHT_Y,
    features: YACHT_FEATURES,
    targets: YACHT_TARGETS,
    sha256: YACHT_SHA256,
  })
}

/**
 * The UCI airfoil self-noise data: frequency, angle of attack, chord length, free-stream velocity and suction side
 * displacement thickness of 1503 wind-tunnel tests of NACA 0012 airfoils, with the scaled sound pressure level in
 * decibels. Embedded from the repository's `airfoil_self_noise.dat` (CC BY 4.0).
 *
 * @returns The dataset: features `x`, the target or targets `y`, and their names.
 * @concept datasets
 * @example
 * ```ts
 * import { airfoilSelfNoise } from 'omai-engine/datasets'
 * airfoilSelfNoise().x.shape // => [1503, 5]
 * ```
 */
export function airfoilSelfNoise(): RegressionData {
  return uciRegression('airfoilSelfNoise', {
    shape: AIRFOIL_SHAPE,
    x: AIRFOIL_X,
    y: AIRFOIL_Y,
    features: AIRFOIL_FEATURES,
    targets: AIRFOIL_TARGETS,
    sha256: AIRFOIL_SHA256,
  })
}

/**
 * The UCI wine quality data, the red wines: eleven physicochemical measurements of 1599 red Portuguese vinho verde
 * wines and their quality, the median of at least three tasters' scores (an integer). Often treated as classification
 * as well; the target is kept as the published score. Embedded from the repository's `winequality-red.csv` (CC BY
 * 4.0).
 *
 * @returns The dataset: features `x`, the target or targets `y`, and their names.
 * @concept datasets
 * @example
 * ```ts
 * import { wineQualityRed } from 'omai-engine/datasets'
 * wineQualityRed().x.shape // => [1599, 11]
 * ```
 */
export function wineQualityRed(): RegressionData {
  return uciRegression('wineQualityRed', {
    shape: WINE_QUALITY_RED_SHAPE,
    x: WINE_QUALITY_RED_X,
    y: WINE_QUALITY_RED_Y,
    features: WINE_QUALITY_RED_FEATURES,
    targets: WINE_QUALITY_RED_TARGETS,
    sha256: WINE_QUALITY_RED_SHA256,
  })
}

/**
 * The UCI wine quality data, the white wines: eleven physicochemical measurements of 4898 white Portuguese vinho verde
 * wines and their quality, the median of at least three tasters' scores (an integer). Often treated as classification
 * as well; the target is kept as the published score. Embedded from the repository's `winequality-white.csv` (CC BY
 * 4.0).
 *
 * @returns The dataset: features `x`, the target or targets `y`, and their names.
 * @concept datasets
 * @example
 * ```ts
 * import { wineQualityWhite } from 'omai-engine/datasets'
 * wineQualityWhite().x.shape // => [4898, 11]
 * ```
 */
export function wineQualityWhite(): RegressionData {
  return uciRegression('wineQualityWhite', {
    shape: WINE_QUALITY_WHITE_SHAPE,
    x: WINE_QUALITY_WHITE_X,
    y: WINE_QUALITY_WHITE_Y,
    features: WINE_QUALITY_WHITE_FEATURES,
    targets: WINE_QUALITY_WHITE_TARGETS,
    sha256: WINE_QUALITY_WHITE_SHA256,
  })
}
