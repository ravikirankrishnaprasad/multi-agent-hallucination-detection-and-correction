# Terminology Reframe (journal_v2)

## Why

The rejected v1 manuscript and the repository described the system as a
"multi-agent" / "agent-inspired" framework. The editor correctly noted that the
implementation is a set of **deterministic modular processing stages**, not
genuinely independent reasoning agents: the components perform fixed
retrieval/similarity/threshold arithmetic with no autonomy, no independent
reasoning, and no inter-agent communication. Retaining "agent" language
overclaims the contribution.

## Decision

journal_v2 describes the system as a **lightweight modular retrieval-grounded
hallucination detection and conservative correction pipeline**.

Replace terminology as follows, where technically accurate:

| v1 term | journal_v2 term |
|---|---|
| multi-agent framework | modular retrieval-grounded pipeline |
| agent (generic) | module / stage / component |
| generation agent | (input) response source / generation stage |
| verification agent | verification module / verifier |
| correction agent | correction module / corrector |
| agent orchestration | pipeline orchestration |

## Rules

1. Do **not** describe the system as multi-agent anywhere in journal_v2 code,
   docs, or the rewritten manuscript.
2. Do **not** create fake "agent" wrappers around deterministic functions just to
   justify the old term.
3. Use "agent" only if a genuinely independent reasoning component is later
   implemented (e.g. an LLM-based verifier and an LLM-based corrector that reason
   independently and reconcile). Any such work must be clearly labelled and is
   currently **future work**, out of Phase 1 scope.
4. Do **not** rename the GitHub repository in Phase 1. The repository name is left
   unchanged to preserve existing links, the dissertation tag, and the citation;
   the reframing is documented here and noted in the README instead.

## Scope in Phase 1

Phase 1 does not rename or modify the v1 scripts (they are preserved as the
historical, rejected baseline). New journal_v2 code uses neutral module/stage
naming from the outset. A README note records the reframing for readers arriving
from the dissertation or preprint.
