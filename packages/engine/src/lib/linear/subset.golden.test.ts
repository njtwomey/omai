import { learnGolden, str } from '../../testing/learn.ts'
import type { Tensor } from '../tensor/index.ts'
import { type SubsetCriterion, type SubsetMethod, subsetRegression } from './subset.ts'

learnGolden('subset selection against statsmodels OLS information criteria', 'linear/subset', {
  subset: (inputs, p) => {
    const [x, y] = inputs as [Tensor, Tensor]
    const f = subsetRegression({ method: str(p, 'method') as SubsetMethod, criterion: str(p, 'criterion') as SubsetCriterion }).fit({ x, y })
    return { selected: f.params.selected, criterionValue: f.params.criterionValue, path: f.params.path.map((s) => s.criterion) }
  },
})
