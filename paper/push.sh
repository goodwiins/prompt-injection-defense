#!/usr/bin/env bash
# Commit + push the audit/fix pass to GitHub.
# Run from the repo root on your Mac:
#   bash paper/push.sh
#
# This script:
#   1. Removes a stale .git/index.lock left over from the sandbox session.
#   2. Stages every change made during the audit pass.
#   3. Creates one signed-off commit on the paper-fix-april-2026 branch.
#   4. Pushes to origin and sets upstream tracking.

set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
echo "Repo root: $REPO_ROOT"

# 1. Clear stale lock if present.
if [[ -f .git/index.lock ]]; then
    echo "Removing stale .git/index.lock"
    rm -f .git/index.lock
fi

# 2. Confirm we're on the audit branch (or switch to it).
CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [[ "$CURRENT_BRANCH" != "paper-fix-april-2026" ]]; then
    echo "Switching to paper-fix-april-2026 (was $CURRENT_BRANCH)"
    git checkout paper-fix-april-2026 2>/dev/null || \
        git checkout -b paper-fix-april-2026
fi

# 3. Stage everything.
git add -A

# Show what we're about to commit.
echo
echo "=== Files to commit ==="
git diff --cached --stat | tail -40
echo

# 4. Commit.
COMMIT_MSG="paper: audit pass — rebuilt bibliography, re-scoped to detection-only

This commit captures the first wave of the FIX_PLAN.md remediation:

* CRITIQUE.md, FIX_PLAN.md, HANDOFF.md, paper_annotated.tex — audit
  artefacts documenting findings and next steps.
* references.bib — full rebuild. Removed 8 placeholder arXiv IDs and
  Anonymous-author entries; replaced with verified citations
  (Lee & Tiwari 2410.07283, Chen et al. StruQ 2402.06363, Chen et al.
  SecAlign 2410.05451, Si et al. PromptArmor 2507.15219, Li et al.
  InjecGuard 2410.22770, Fan & Li PeerGuard 2505.11642, Chao et al.
  JailbreakBench 2404.01318, Liu et al. 2306.05499, Perplexity
  BrowseSafe 2511.20597). Citation keys in paper.tex updated.
* paper.tex — Detection-only re-scope:
    - New title.
    - Abstract rewritten: dropped 'three-layer' language, removed
      TensorTrust, reconciled 97.2/97.3, 38.1/43.3, BrowseSafe.
    - Introduction restructured around the actual contribution.
    - Threat model: explicit out-of-scope items, no undefined jargon.
    - System Design: §4.2/§4.3 collapsed into a single 'future work'
      paragraph; §4 now describes only the detector that's evaluated.
    - Methodology: training-data section rewritten to match
      models/injection_aware_mpnet_metadata.json (2667/570/413, not
      2800/2800/1400). Spearman 0.904 properly attributed.
    - Conclusion: corrected, internally consistent numbers; broken
      cumulative-bypass paragraph removed; code-release URL added.
* results/legacy/ — moved 9 stale result files here (with README) so
  reviewers can't reproduce contradictory numbers from earlier models.
* evaluation/evaluate_mpnet_on_tensortrust.py — new ready-to-run eval
  script for TensorTrust.

Open work tracked in paper/HANDOFF.md (TensorTrust eval run, IDR script,
Wilson CIs, threshold sweep figure, GCG/AutoDAN restore, P2 polish)."

git commit -m "$COMMIT_MSG"

# 5. Push, setting upstream if needed.
echo
echo "=== Pushing to origin ==="
git push -u origin paper-fix-april-2026

echo
echo "Done. Open a PR at:"
echo "  https://github.com/goodwiins/prompt-injection-defense/compare/main...paper-fix-april-2026"
