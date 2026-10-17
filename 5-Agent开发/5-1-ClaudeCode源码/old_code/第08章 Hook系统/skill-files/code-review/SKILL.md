---
name: code-review
description: Review code for bugs, performance issues, and style problems
---

When reviewing code, check the following:

1. **Correctness** — logical errors, off-by-one, missing edge cases, null/None handling
2. **Security** — command injection, path traversal, XSS, SQL injection, hardcoded secrets
3. **Performance** — unnecessary allocations, N+1 queries, blocking I/O, missing caches
4. **Style** — naming consistency, dead code, overly complex expressions, missing imports
5. **Robustness** — error handling, timeout guards, resource cleanup, input validation

Output your review as a list of findings. Each finding should be a single line prefixed with one of:
- `[BUG]` for correctness issues
- `[SEC]` for security issues
- `[PERF]` for performance issues
- `[STYLE]` for style issues
- `[OK]` if nothing found in a category
