# Linear Ticket + Branch Naming Convention Design

**Date:** 2026-03-31
**Status:** Implemented

## Problem

The repo had no issue-tracking integration. Features were developed without linkage to a project management system, making it impossible to trace code changes back to requirements or planned work. Branch names followed no convention, and PRs had no structured metadata.

## Solution

Enforce a mandatory Linear ticket ↔ branch ↔ commit linkage for every feature. Use the Linear MCP server for agent-level ticket discovery and creation. Use GitHub Actions for CI-level branch name validation on PRs.

---

## Components

### 1. Linear MCP Server

- **Server:** `https://mcp.linear.app/mcp` (Linear's hosted MCP)
- **Auth:** OAuth 2.1 with dynamic client registration (browser flow, no API keys in files)
- **Setup:** `claude mcp add --transport http linear-server https://mcp.linear.app/mcp`
- **Scope:** User-level Claude Code config — no repo files modified
- **Project:** "Trading Databricks" in the "Solo" team

### 2. Branch Naming Convention

```
type/TRD-NNN-short-description
```

| Component | Values | Example |
|-----------|--------|---------|
| `type` | `feat`, `fix`, `chore`, `docs`, `test`, `refactor` | `feat` |
| `TRD-NNN` | Linear ticket identifier | `TRD-42` |
| `short-description` | lowercase, hyphen-separated, ≤ 5 words | `add-news-silver-layer` |

**Examples:**
```
feat/TRD-42-add-news-silver-layer
fix/TRD-43-tick-timestamp-cast
chore/TRD-44-update-cluster-config
test/TRD-45-bar-construction-unit-tests
```

### 3. Commit Message Convention

```
TRD-NNN type: short imperative description
```

**Examples:**
```
TRD-42 feat: add silver news dedup transform
TRD-42 test: add silver news pytest assertions
TRD-43 fix: correct tick timestamp timezone cast
```

### 4. GitHub Actions Branch Validation

Job `validate-branch` in `.github/workflows/deploy.yml`:
- Runs on `pull_request` events only
- Regex: `^(feat|fix|chore|docs|test|refactor)/TRD-[0-9]+-[a-z0-9-]+$`
- Fails the PR check if the branch name does not match
- Runs in parallel with `validate` and `test` jobs — no blocking dependency

### 5. CLAUDE.md Rules

Two insertion points:

**Non-Negotiable Rules** (agent-level mandate):
> Every feature branch must reference a Linear ticket. Before invoking `using-git-worktrees`, look up or create a ticket in the "Trading Databricks" Linear project using the `linear-server` MCP.

**Development Workflow section** (procedure):
Full branch + commit conventions with examples, plus explicit superpowers chain placement.

---

## Superpowers Workflow Integration

The ticket lookup step is inserted between `writing-plans` and `using-git-worktrees`:

```
brainstorming
    ↓
writing-plans
    ↓
[look up or create Linear ticket via linear-server MCP]  ← NEW
    ↓
using-git-worktrees  (branch named type/TRD-NNN-desc)
    ↓
subagent-driven-development  (commits prefixed TRD-NNN)
    ↓
finishing-a-development-branch
```

CLAUDE.md rules override all superpowers skills, so no skill files need modification. The insertion is purely declarative in CLAUDE.md.

---

## Constraints

- One ticket per branch — never mix multiple ticket IDs on one branch
- Branch names use only lowercase letters, digits, and hyphens in the description segment
- The `linear-server` MCP must be connected before starting any feature work
- If MCP is unavailable, ask the user for the ticket ID before proceeding
