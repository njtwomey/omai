# omai pocs

Proofs of concept: single live pages of what the library can do, kept out of the site. `make pocs` serves them at
http://localhost:5310 (the site is `make dev`, port 5300). The shell, colours and controls are the site's own (the app
imports `apps/site/src/shell`), so a page looks the same as it would on the site.

Inclusion only: a page exists because someone asked for it, not because the library has the feature. To add one, make
`src/pocs/<slug>/poc.ts` (title, summary, tags, date: this is what the index searches) and `src/pocs/<slug>/Page.tsx`
(the default export is the page); nothing else needs editing. Use the library's own estimators and slots (an
estimator's `.method(data)` run with `useSession`, its config as `SpaceControls`, panels from `SessionPanels`) so a
page stays a composition; anything with substance belongs in `packages/engine/src/lib`. A page that proves its worth can
be promoted to the site's gallery by moving its recipe there.

Groups: pocs that belong together are listed together. Declare a group in `src/groups.ts` (slug, title, summary) and join
it with `group: '<slug>'` in the poc's `poc.ts`; the index shows one section per group (a group's own page is
`/group/<slug>`), a search lists matches flat with the group on each card, and a poc with no group is listed under "More".
A group with no pocs is not shown.

A new page's styles arrive with a restart: Tailwind scans for classes when `make pocs` starts, so stop it and start it
again after adding a page whose layout looks unstyled.
