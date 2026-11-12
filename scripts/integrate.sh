#!/usr/bin/env bash
# Merge an agent's branch into main the same way every time, so a merge costs one command and one look at a summary:
#
#   scripts/integrate.sh <branch> ["merge message"] [extra vitest filters...]
#
# 1. merges the branch (no fast-forward); the generated files that conflict (llms.txt, the agent index, the API surface)
#    take main's side and are regenerated; any other conflict stops the script and is listed (exit 2);
# 2. regenerates the barrels, the agent index and the API surface, commits the merge;
# 3. runs the checks (barrels, agent index, API surface, layers) and tsc for the packages the merge touched;
# 4. clears the Vite and Vitest caches (a stale glob hid new pages more than once) and nudges the dev server's file lists;
# 5. runs the EP suite (`make ep-check`) with a long timeout plus a vitest filter per changed engine domain, and prints one
#    line per step.
# Light by default (the fan matters): the merge, the generated files, the static checks and tsc, and only the tests of the changed
# engine domains' own files; FULL=1 adds the whole EP suite and the broad filters. A branch is merged after it has merged main itself (agents are told to), so conflicts here are rare and small.
set -u
branch="${1:?usage: scripts/integrate.sh <branch> [message] [vitest filters...]}"
message="${2:-Merge $branch}"
shift $(( $# >= 2 ? 2 : 1 ))
cd "$(dirname "$0")/.."
summary=()
step() { summary+=("$1: $2"); }

if [ -n "$(git status --porcelain --untracked-files=no)" ]; then echo "working tree not clean: commit or stash first"; exit 1; fi
before="$(git rev-parse HEAD)"
git merge --no-ff "$branch" -m "$message" >/dev/null 2>&1
generated='^(llms\.txt|docs/api-surface\.txt|packages/[^/]+/llms\.txt|packages/[^/]+/index/.*)$'
conflicts="$(git diff --name-only --diff-filter=U)"
if [ -n "$conflicts" ]; then
  other="$(echo "$conflicts" | grep -Ev "$generated" || true)"
  if [ -n "$other" ]; then echo "CONFLICTS to resolve by hand:"; echo "$other"; echo "(merge left in progress; generated files not yet touched)"; exit 2; fi
  echo "$conflicts" | xargs git checkout --ours -- 2>/dev/null
fi
make barrels agent-index api-surface >/dev/null 2>&1
git add -A
if ! git diff --cached --quiet; then git commit -q --no-edit >/dev/null 2>&1 || true; fi
after="$(git rev-parse HEAD)"
changed="$(git diff --name-only "$before" "$after")"
step merge "$(git log --oneline -1)"

run() { local name="$1"; shift; if "$@" >/tmp/integrate-$$.log 2>&1; then step "$name" ok; else step "$name" "FAILED (see below)"; tail -15 /tmp/integrate-$$.log >&2; fi; }
run checks make barrels-check agent-index-check api-surface-check layers
if echo "$changed" | grep -q '^packages/engine/'; then run tsc-engine bash -c 'cd packages/engine && mise exec -- pnpm exec tsc -p . --noEmit'; fi
if echo "$changed" | grep -q '^packages/render/'; then run tsc-render bash -c 'cd packages/render && mise exec -- pnpm exec tsc -p . --noEmit'; fi
if echo "$changed" | grep -Eq '^(apps/pocs|apps/site|packages/)'; then run tsc-pocs bash -c 'cd apps/pocs && mise exec -- pnpm exec tsc -p . --noEmit'; fi
rm -rf node_modules/.vitest-cache node_modules/.vite apps/pocs/node_modules/.vite 2>/dev/null
touch apps/pocs/src/shared/ep/atomSource.ts apps/pocs/src/registry.ts 2>/dev/null
filters="$(echo "$changed" | sed -n 's#^packages/engine/src/lib/\([^/]*\(/[^/]*\)\?\)/.*#\1#p' | sort -u | tr '\n' ' ')"
export OMAI_TEST_WORKERS="${OMAI_TEST_WORKERS:-3}" OMAI_TEST_TIMEOUT=60000
# shellcheck disable=SC2086
if [ "${FULL:-0}" = "1" ]; then
  run tests mise exec -- pnpm exec vitest run ep- message-passing probprog/model probprog/inference apps/pocs/src/registry $filters "$@"
else
  changed_tests="$(git diff --name-only --diff-filter=AM "$before" "$after" | grep -E '\.test\.tsx?$' | tr '\n' ' ')"
  if [ -n "$changed_tests" ]; then
    export OMAI_TEST_WORKERS=1
    # shellcheck disable=SC2086
    run tests nice -n 15 mise exec -- pnpm exec vitest run $changed_tests "$@"
  else step tests "none changed (light mode: FULL=1 for the EP suite)"; fi
fi
echo; printf '%s\n' "${summary[@]}"
