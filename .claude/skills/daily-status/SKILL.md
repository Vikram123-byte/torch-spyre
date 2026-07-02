---
name: daily-status
description: "Generate a daily status update for any squad's Slack workflow. Pulls recent git activity, open PRs, and blockers to produce a formatted update matching the team's standup template."
---

# Daily Status Update Generator

Generate a status update for a squad Slack standup thread. Adapts to
whatever format the team uses (P/B/R, Yesterday/Today/Blockers, custom).

## Arguments

The user may pass arguments to customize the output:

- `/daily-status` — default P/B/R format, last 24 hours, current repo
- `/daily-status since Friday` — custom time window
- `/daily-status --repos torch-spyre,spyre-inference` — multi-repo
- `/daily-status --format ytb` — Yesterday/Today/Blockers format
- `/daily-status --label vLLM` — filter to issues with a specific label

## Steps

1. Identify the user by running `git config user.name`,
   `git config user.email`, and `gh api user --jq .login`.

2. The repo scope defaults to the current repo from
   `gh repo view --json nameWithOwner --jq .nameWithOwner`. If the user
   specifies `--repos`, iterate over each. If the user specifies an org,
   query all repos under that org.

3. Use P/B/R format unless the user specifies otherwise:

   | Format flag | Sections |
   |---|---|
   | `pbr` (default) | Progress / Blockers / Awaiting Re-Review / WIP |
   | `ytb` | Yesterday / Today / Blockers |
   | `custom` | Ask the user for section names |

4. The time window defaults to "last 24 hours". Parse natural language
   like "since Friday", "this week", "last 3 days". Convert to an ISO
   timestamp (`<ISO_TIMESTAMP>`) for `gh` queries and a git `--since`
   value (`<WINDOW>`) for `git log`.

5. Gather open and merged PRs in one query (used by steps 6–8):

   ```bash
   gh pr list --author="@me" --state all --limit 30 \
     --json number,title,state,mergedAt,updatedAt,isDraft,reviewDecision,statusCheckRollup,headRefName
   ```

   Fetch commit SHAs for each merged PR in the window (for dedup in
   step 6):

   ```bash
   gh pr view <NUMBER> --json commits --jq '.commits[].oid'
   ```

6. Gather progress activity, scoped to the time window and repo list:

   ```bash
   # Commits authored in the time window (try name, then email if empty)
   git log --all --author="$(git config user.name)" --since="<WINDOW>" \
     --format="%h %s" --no-merges
   git log --all --author="$(git config user.email)" --since="<WINDOW>" \
     --format="%h %s" --no-merges

   # Issues assigned to the user (optionally filtered by label)
   gh issue list --assignee="@me" --state open --limit 20 \
     --json number,title,labels,updatedAt \
     --jq '.[] | "#\(.number): \(.title) [\(.labels | map(.name) | join(", "))]"'
   ```

   If `--label` is specified, add `--label "<LABEL>"` to the issue query.

   **Build the Progress section from:**

   - Merged PRs where `mergedAt > <ISO_TIMESTAMP>`: one bullet each,
     e.g. `#1234: <title> (merged)`. Do **not** also list their commits.
   - Commits from `git log` that are **not** SHAs belonging to a merged
     PR in the window (dedup against step 5). Use commit-message bullets
     only; do **not** prefix with a PR number (step 7 owns open PRs).

   Open PRs are classified in step 7 only — Blockers, Awaiting Re-Review,
   or WIP. Do **not** list them in Progress. Draft PRs belong in **WIP**
   only.

7. Classify open PRs (from step 5). Each PR belongs in **at most one**
   section. Apply rules in this order:

   | Condition | Section |
   |---|---|
   | `isDraft == true` | **WIP** — not a blocker |
   | `reviewDecision == "CHANGES_REQUESTED"` and `updatedAt > <ISO_TIMESTAMP>` | **Awaiting Re-Review** |
   | `reviewDecision == "CHANGES_REQUESTED"` but not updated in window | Skip (stale; omit unless user asks) |
   | CI failure (`statusCheckRollup` contains `FAILURE`) | **Blockers** |
   | `reviewDecision` is `REVIEW_REQUIRED`, `null`, or pending (non-draft) | **Blockers** — awaiting first review |
   | `reviewDecision == "APPROVED"` and CI passing | Not a blocker |

   **Consolidate to one bullet per PR.** Combine signals on a single line,
   e.g. `#1234: <title> — CI failing, awaiting review`.

   If nothing is blocking, write "None" under **Blockers**.

8. **Awaiting Re-Review** — PRs where you pushed after review feedback
   and reviewers need to re-check. This is **not** the same as a blocker
   waiting on first-pass review.

   ```bash
   # From step 5 data: isDraft == false, reviewDecision == "CHANGES_REQUESTED",
   # updatedAt > <ISO_TIMESTAMP>
   ```

   If nothing awaits re-review, write "None".

9. **WIP** — draft PRs (`isDraft == true`). List separately so work-in-
   progress does not look like a blocker:

   ```
   - #567: <title> (draft)
   ```

   Omit the **WIP** section entirely if there are no drafts.

10. Format the output as a message matching the chosen template, ready
    to paste into Slack. Start with a scope line:

    ```
    *Status for <git user.name>, <time window> (<repo or repo list>)*
    ```

    ### P/B/R format (default)

    ```
    **P:**
    - <progress bullets>

    **B:**
    - <blocker bullets or "None">

    **R:** *(Awaiting Re-Review)*
    - <re-review bullets or "None">

    **WIP:**
    - <draft PR bullets>
    ```

    Omit **WIP** when empty.

    ### Yesterday/Today/Blockers format

    ```
    **Yesterday:**
    - <same content as Progress above>

    **Today:**
    - <open assigned issues not in Blockers/WIP>
    - <open non-draft PRs you are actively working on>

    **Blockers:**
    - <blocker bullets or "None">
    ```

    For **Today**, infer from open assigned issues and open non-draft PRs.
    Exclude items already listed under Blockers or WIP.

    ### Formatting rules

    - One line per bullet. Use PR/issue numbers (e.g., `#1234`).
    - For multi-repo output, prefix with `repo-name#1234` only when bullets
      span more than one repo; otherwise skip the repo prefix.
    - No emojis unless the user requests them.
    - Keep it concise — the reader skims dozens of these in a thread.
    - **Dedup:** never show the same PR in both Progress and another
      section. Merged PRs appear once under Progress, not as commits.

11. Show the formatted update and ask if the user wants to adjust
    anything before posting.

## Notes

- If `gh` is not authenticated, instruct the user to run `gh auth login`.
- If `git log --author` returns nothing with `user.name`, fall back to
  `user.email`. If both fail, warn the user their git identity may not
  match their GitHub commits.
- For multi-repo updates, group bullets by repo only when items come from
  more than one repo. Otherwise skip the repo header.
- The skill works in any GitHub-hosted project, not just torch-spyre.
  It reads the repo context from the current working directory.
- The output is scoped to whoever is running it via their git/gh
  identity. No squad configuration needed.
