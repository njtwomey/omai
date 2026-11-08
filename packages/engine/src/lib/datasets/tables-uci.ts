/**
 * Real mixed-type tables from the UCI Machine Learning Repository (all CC BY 4.0), with categorical and text columns
 * and explicit missing values (NaN, never imputed): Auto MPG, abalone, heart disease, hepatitis, zoo, tic-tac-toe, car
 * evaluation, congressional voting, mushroom and adult (stratified subsamples), dermatology, forest fires, student
 * performance, servo and daily bike sharing. `tableToXy` turns one into features and a target.
 */

import type { TableData } from '../data/index.ts'
import { ABALONE_COLUMNS, ABALONE_SHA256 } from './_data/abalone.data.ts'
import { ADULT_COLUMNS, ADULT_SHA256 } from './_data/adult.data.ts'
import { AUTO_MPG_COLUMNS, AUTO_MPG_SHA256 } from './_data/auto-mpg.data.ts'
import { BIKE_SHARING_COLUMNS, BIKE_SHARING_SHA256 } from './_data/bike-sharing.data.ts'
import { CAR_EVALUATION_COLUMNS, CAR_EVALUATION_SHA256 } from './_data/car-evaluation.data.ts'
import { CONGRESSIONAL_VOTING_COLUMNS, CONGRESSIONAL_VOTING_SHA256 } from './_data/congressional-voting.data.ts'
import { DERMATOLOGY_COLUMNS, DERMATOLOGY_SHA256 } from './_data/dermatology.data.ts'
import { FOREST_FIRES_COLUMNS, FOREST_FIRES_SHA256 } from './_data/forest-fires.data.ts'
import { HEART_DISEASE_COLUMNS, HEART_DISEASE_SHA256 } from './_data/heart-disease.data.ts'
import { HEPATITIS_COLUMNS, HEPATITIS_SHA256 } from './_data/hepatitis.data.ts'
import { MUSHROOM_COLUMNS, MUSHROOM_SHA256 } from './_data/mushroom.data.ts'
import { SERVO_COLUMNS, SERVO_SHA256 } from './_data/servo.data.ts'
import { STUDENT_MATH_COLUMNS, STUDENT_MATH_SHA256 } from './_data/student-math.data.ts'
import { TIC_TAC_TOE_COLUMNS, TIC_TAC_TOE_SHA256 } from './_data/tic-tac-toe.data.ts'
import { ZOO_COLUMNS, ZOO_SHA256 } from './_data/zoo.data.ts'
import { uciTable } from './_uci.ts'

/**
 * The UCI Auto MPG data as a table: miles per gallon (the target), cylinders, displacement, horsepower, weight,
 * acceleration, model year, origin (1 American, 2 European, 3 Japanese, as published) and the car's name (a text
 * column) of 398 cars. Six horsepower values are missing in the file and stay missing (NaN). Embedded from the
 * repository's `auto-mpg.data` (CC BY 4.0).
 *
 * @returns The table, with `mpg` as its target.
 * @concept datasets
 * @example
 * ```ts
 * import { rowCount } from 'omai-engine/data'
 * import { autoMpg } from 'omai-engine/datasets'
 * rowCount(autoMpg()) // => 398
 * autoMpg().columns['horsepower']?.kind // => 'numeric'
 * ```
 */
export function autoMpg(): TableData {
  return uciTable(
    'autoMpg',
    { columns: AUTO_MPG_COLUMNS, sha256: AUTO_MPG_SHA256 },
    { target: 'mpg', text: ['carName'] },
  )
}

/**
 * The UCI abalone data as a table: the sex (female, infant, male), length, diameter, height and four weights of 4177
 * abalone, and the number of rings (the target), which with 1.5 gives the age in years. Embedded from the repository's
 * `abalone.data` (CC BY 4.0).
 *
 * @returns The table, with `rings` as its target.
 * @concept datasets
 * @example
 * ```ts
 * import { rowCount } from 'omai-engine/data'
 * import { abalone } from 'omai-engine/datasets'
 * rowCount(abalone()) // => 4177
 * abalone().columns['sex']?.kind // => 'categorical'
 * ```
 */
export function abalone(): TableData {
  return uciTable('abalone', { columns: ABALONE_COLUMNS, sha256: ABALONE_SHA256 }, { target: 'rings' })
}

/**
 * The UCI heart disease data, the processed Cleveland file, as a table: 13 attributes (age, sex, chest pain type,
 * blood pressure, cholesterol, fasting blood sugar, resting ECG, maximum heart rate, exercise-induced angina, ST
 * depression, slope, major vessels coloured, thalassemia) of 303 patients, and the target `num`, 0 for no disease to 4
 * for its degree. The categorical attributes keep the published numeric codes. Four `ca` and two `thal` values are
 * missing and stay missing (NaN). Embedded from the repository's `processed.cleveland.data` (CC BY 4.0).
 *
 * @returns The table, with `num` as its target.
 * @concept datasets
 * @example
 * ```ts
 * import { rowCount } from 'omai-engine/data'
 * import { heartDisease } from 'omai-engine/datasets'
 * rowCount(heartDisease()) // => 303
 * heartDisease().columns['num']?.kind // => 'categorical'
 * ```
 */
export function heartDisease(): TableData {
  return uciTable('heartDisease', { columns: HEART_DISEASE_COLUMNS, sha256: HEART_DISEASE_SHA256 }, { target: 'num' })
}

/**
 * The UCI hepatitis data as a table: age, sex, 12 yes/no symptoms and treatments (coded 1 no, 2 yes as published; sex
 * 1 male, 2 female), five laboratory values and histology of 155 patients, with the outcome (DIE or LIVE) as target.
 * The file marks 167 values missing and they stay missing (NaN). Embedded from the repository's `hepatitis.data` (CC
 * BY 4.0).
 *
 * @returns The table, with `class` as its target.
 * @concept datasets
 * @example
 * ```ts
 * import { rowCount } from 'omai-engine/data'
 * import { hepatitis } from 'omai-engine/datasets'
 * rowCount(hepatitis()) // => 155
 * hepatitis().columns['class']?.kind // => 'categorical'
 * ```
 */
export function hepatitis(): TableData {
  return uciTable('hepatitis', { columns: HEPATITIS_COLUMNS, sha256: HEPATITIS_SHA256 }, { target: 'class' })
}

/**
 * The UCI zoo data as a table: 15 yes/no attributes (0 or 1) and a leg count of 101 animals, whose name is a text
 * column, and the target `type` (mammal, bird, reptile, fish, amphibian, insect, invertebrate), named as in the file's
 * documentation. Embedded from the repository's `zoo.data` (CC BY 4.0).
 *
 * @returns The table, with `type` as its target.
 * @concept datasets
 * @example
 * ```ts
 * import { rowCount } from 'omai-engine/data'
 * import { zoo } from 'omai-engine/datasets'
 * rowCount(zoo()) // => 101
 * zoo().columns['type']?.kind // => 'categorical'
 * ```
 */
export function zoo(): TableData {
  return uciTable(
    'zoo',
    { columns: ZOO_COLUMNS, sha256: ZOO_SHA256 },
    {
      target: 'type',
      text: ['animal'],
      levels: { type: ['mammal', 'bird', 'reptile', 'fish', 'amphibian', 'insect', 'invertebrate'] },
    },
  )
}

/**
 * The UCI tic-tac-toe endgame data as a table: the nine squares (x, o or b for blank) of all 958 possible final
 * positions when x plays first, and the target `class`, positive when x has three in a row. Embedded from the
 * repository's `tic-tac-toe.data` (CC BY 4.0).
 *
 * @returns The table, with `class` as its target.
 * @concept datasets
 * @example
 * ```ts
 * import { rowCount } from 'omai-engine/data'
 * import { ticTacToe } from 'omai-engine/datasets'
 * rowCount(ticTacToe()) // => 958
 * ticTacToe().columns['class']?.kind // => 'categorical'
 * ```
 */
export function ticTacToe(): TableData {
  return uciTable('ticTacToe', { columns: TIC_TAC_TOE_COLUMNS, sha256: TIC_TAC_TOE_SHA256 }, { target: 'class' })
}

/**
 * The UCI car evaluation data as a table: buying price, maintenance price, doors, persons, boot size and safety of
 * every one of 1728 cars (each attribute with its levels in their natural order), and the target `class`,
 * unacceptable, acceptable, good or very good. The classes are very unbalanced (1210 unacceptable). Embedded from the
 * repository's `car.data` (CC BY 4.0).
 *
 * @returns The table, with `class` as its target.
 * @concept datasets
 * @example
 * ```ts
 * import { rowCount } from 'omai-engine/data'
 * import { carEvaluation } from 'omai-engine/datasets'
 * rowCount(carEvaluation()) // => 1728
 * carEvaluation().columns['class']?.kind // => 'categorical'
 * ```
 */
export function carEvaluation(): TableData {
  return uciTable(
    'carEvaluation',
    { columns: CAR_EVALUATION_COLUMNS, sha256: CAR_EVALUATION_SHA256 },
    {
      target: 'class',
      levels: {
        class: ['unacc', 'acc', 'good', 'vgood'],
        buying: ['low', 'med', 'high', 'vhigh'],
        maintenance: ['low', 'med', 'high', 'vhigh'],
        doors: ['2', '3', '4', '5more'],
        persons: ['2', '4', 'more'],
        luggageBoot: ['small', 'med', 'big'],
        safety: ['low', 'med', 'high'],
      },
    },
  )
}

/**
 * The UCI congressional voting records as a table: how each of 435 members of the US House voted in 1984 on 16 issues
 * (y or n) and their party (democrat or republican). The file's `?` is not an unknown vote but a vote other than yea
 * or nay (absent, paired, abstained); it is kept as a missing value (392 of them). Embedded from the repository's
 * `house-votes-84.data` (CC BY 4.0).
 *
 * @returns The table, with `party` as its target.
 * @concept datasets
 * @example
 * ```ts
 * import { rowCount } from 'omai-engine/data'
 * import { congressionalVoting } from 'omai-engine/datasets'
 * rowCount(congressionalVoting()) // => 435
 * congressionalVoting().columns['party']?.kind // => 'categorical'
 * ```
 */
export function congressionalVoting(): TableData {
  return uciTable(
    'congressionalVoting',
    { columns: CONGRESSIONAL_VOTING_COLUMNS, sha256: CONGRESSIONAL_VOTING_SHA256 },
    { target: 'party' },
  )
}

/**
 * The UCI mushroom data as a table, a stratified subsample of 2000 of its 8124 mushrooms (the full file is too large
 * to vendor): 22 categorical attributes in the published one-letter codes and the target `class` (e edible, p
 * poisonous). The sample keeps the class shares (1036 edible, 964 poisonous, as 4208 and 3916 of 8124) and the
 * original row order; the rows are chosen by the SHA-256 of their index, so the sample is reproducible. The `?` of
 * `stalk-root` stays missing (NaN). Embedded from the repository's `agaricus-lepiota.data` (CC BY 4.0).
 *
 * @returns The table, with `class` as its target.
 * @concept datasets
 * @example
 * ```ts
 * import { rowCount } from 'omai-engine/data'
 * import { mushroom } from 'omai-engine/datasets'
 * rowCount(mushroom()) // => 2000
 * mushroom().columns['class']?.kind // => 'categorical'
 * ```
 */
export function mushroom(): TableData {
  return uciTable('mushroom', { columns: MUSHROOM_COLUMNS, sha256: MUSHROOM_SHA256 }, { target: 'class' })
}

/**
 * The UCI adult (census income) data as a table, a stratified subsample of 2500 of the 32561 rows of its training file
 * (which is too large to vendor): age, work class, final weight, education, marital status, occupation, relationship,
 * race, sex, capital gain and loss, hours per week and native country, and the target `income` (at most 50K or above
 * 50K). The sample keeps the income shares (1898 and 602, as 24720 and 7841 of 32561) and the original row order; the
 * rows are chosen by the SHA-256 of their index, so the sample is reproducible. The `?` of work class, occupation and
 * native country stay missing (NaN). Embedded from the repository's `adult.data` (CC BY 4.0).
 *
 * @returns The table, with `income` as its target.
 * @concept datasets
 * @example
 * ```ts
 * import { rowCount } from 'omai-engine/data'
 * import { adultIncome } from 'omai-engine/datasets'
 * rowCount(adultIncome()) // => 2500
 * adultIncome().columns['income']?.kind // => 'categorical'
 * ```
 */
export function adultIncome(): TableData {
  return uciTable('adultIncome', { columns: ADULT_COLUMNS, sha256: ADULT_SHA256 }, { target: 'income' })
}

/**
 * The UCI dermatology data as a table: 11 clinical and 22 histopathological features (degrees 0 to 3, family history 0
 * or 1) and the age of 366 patients, with the diagnosed disease as target. Eight ages are missing and stay missing
 * (NaN). Embedded from the repository's `dermatology.data` (CC BY 4.0).
 *
 * @returns The table, with `disease` as its target.
 * @concept datasets
 * @example
 * ```ts
 * import { rowCount } from 'omai-engine/data'
 * import { dermatology } from 'omai-engine/datasets'
 * rowCount(dermatology()) // => 366
 * dermatology().columns['age']?.kind // => 'numeric'
 * ```
 */
export function dermatology(): TableData {
  return uciTable(
    'dermatology',
    { columns: DERMATOLOGY_COLUMNS, sha256: DERMATOLOGY_SHA256 },
    {
      target: 'disease',
      levels: {
        disease: [
          'psoriasis',
          'seboreic dermatitis',
          'lichen planus',
          'pityriasis rosea',
          'chronic dermatitis',
          'pityriasis rubra pilaris',
        ],
      },
    },
  )
}

/**
 * The UCI forest fires data as a table: map coordinates, month, day of the week, four Canadian fire weather indices
 * (FFMC, DMC, DC, ISI), temperature, humidity, wind and rain of 517 fires, and the burnt area in hectares (the target;
 * 247 are zero, and the distribution is very skewed). Embedded from the repository's `forestfires.csv` (CC BY 4.0).
 *
 * @returns The table, with `area` as its target.
 * @concept datasets
 * @example
 * ```ts
 * import { rowCount } from 'omai-engine/data'
 * import { forestFires } from 'omai-engine/datasets'
 * rowCount(forestFires()) // => 517
 * forestFires().columns['month']?.kind // => 'categorical'
 * ```
 */
export function forestFires(): TableData {
  return uciTable(
    'forestFires',
    { columns: FOREST_FIRES_COLUMNS, sha256: FOREST_FIRES_SHA256 },
    {
      target: 'area',
      levels: {
        month: ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
        day: ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun'],
      },
    },
  )
}

/**
 * The UCI student performance data, the mathematics file, as a table: school, sex, age, family background, study
 * habits, alcohol use, absences and the grades `G1` and `G2` of 395 secondary-school students, with the final grade
 * `G3` (0 to 20) as target. The repository describes the data as anonymised. Embedded from the repository's
 * `student.zip/student-mat.csv` (CC BY 4.0).
 *
 * @returns The table, with `G3` as its target.
 * @concept datasets
 * @example
 * ```ts
 * import { rowCount } from 'omai-engine/data'
 * import { studentMath } from 'omai-engine/datasets'
 * rowCount(studentMath()) // => 395
 * studentMath().columns['school']?.kind // => 'categorical'
 * ```
 */
export function studentMath(): TableData {
  return uciTable('studentMath', { columns: STUDENT_MATH_COLUMNS, sha256: STUDENT_MATH_SHA256 }, { target: 'G3' })
}

/**
 * The UCI servo data as a table: the motor and screw type (letters A to E, categorical) and the position and velocity
 * gain settings of 167 servomechanisms, with the rise time (the target), a very non-linear function of them. Embedded
 * from the repository's `servo.data` (CC BY 4.0).
 *
 * @returns The table, with `riseTime` as its target.
 * @concept datasets
 * @example
 * ```ts
 * import { rowCount } from 'omai-engine/data'
 * import { servo } from 'omai-engine/datasets'
 * rowCount(servo()) // => 167
 * servo().columns['motor']?.kind // => 'categorical'
 * ```
 */
export function servo(): TableData {
  return uciTable('servo', { columns: SERVO_COLUMNS, sha256: SERVO_SHA256 }, { target: 'riseTime' })
}

/**
 * The UCI bike sharing data, the daily file, as a table: the date (a text column), season, year, month, holiday,
 * weekday, working day, weather situation, normalised temperature, feeling temperature, humidity and wind speed of 731
 * days of 2011 and 2012, and the rentals: casual and registered users, and their sum `cnt` (the target). The running
 * index of the file is dropped. Embedded from the repository's `day.csv` (CC BY 4.0).
 *
 * @returns The table, with `cnt` as its target.
 * @concept datasets
 * @example
 * ```ts
 * import { rowCount } from 'omai-engine/data'
 * import { bikeSharingDaily } from 'omai-engine/datasets'
 * rowCount(bikeSharingDaily()) // => 731
 * bikeSharingDaily().columns['cnt']?.kind // => 'numeric'
 * ```
 */
export function bikeSharingDaily(): TableData {
  return uciTable(
    'bikeSharingDaily',
    { columns: BIKE_SHARING_COLUMNS, sha256: BIKE_SHARING_SHA256 },
    { target: 'cnt', text: ['dteday'] },
  )
}
