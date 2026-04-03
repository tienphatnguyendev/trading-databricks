# Fix CI Tests Implementation Plan

**Status:** ✅ FINISHED

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide the necessary Databricks environment variables to the pytest step in CI to allow Databricks Connect to authenticate.

**Architecture:** Inject `DATABRICKS_HOST` and `DATABRICKS_TOKEN` as environment variables into the test step.

**Tech Stack:** GitHub Actions.

---

### Task 1: Add Databricks Secrets to Test Job

**Files:**
- Modify: `.github/workflows/deploy.yml`

- [ ] **Step 1: Add the environment variables**

Update `.github/workflows/deploy.yml` so the test job includes the necessary credentials.

```yaml
  test:
    name: Unit tests
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
      - name: Install dependencies
        run: uv sync --extra dev
      - name: Run tests
        env:
          DATABRICKS_HOST: ${{ secrets.DATABRICKS_HOST }}
          DATABRICKS_TOKEN: ${{ secrets.DATABRICKS_TOKEN }}
        run: uv run pytest -v; code=$?; [ $code -eq 5 ] && echo "No tests collected (stubs only)" || exit $code
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/deploy.yml
git commit -m "ci: provide databricks credentials to test job"
```
