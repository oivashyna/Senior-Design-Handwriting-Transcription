"""
HTS prompt template renderer.

Supports:
  - {{VARIABLE}}                    -> simple substitution
  - {{VAR.attr}}                    -> attribute/key access (used inside FOREACH)
  - {{#IF VAR}} ... {{/IF}}         -> block kept iff VAR is truthy
  - {{#IF VAR == value}} ... {{/IF}} -> block kept iff VAR equals value
  - {{#IF VAR != value}} ... {{/IF}} -> block kept iff VAR not equal value
  - {{#IF A or B}} / {{#IF A and B}} -> compound conditions (each side any of
                                        the above forms)
  - {{#UNLESS VAR}} ... {{/UNLESS}} -> block kept iff VAR is falsy
  - {{#FOREACH item IN LIST}} ... {{/FOREACH}} -> repeat body per element of
                                        LIST (a list of dicts); inside the body,
                                        {{item}} and {{item.key}} resolve per row.

Resolution order: FOREACH loops expand first, then conditionals (so variables
inside a dropped block never render), then remaining {{VAR}} tokens substitute.
Nested conditionals resolve innermost-first.
"""

import re

# Innermost {{#IF ...}}...{{/IF}} or {{#UNLESS ...}}...{{/UNLESS}} block
# (no nested opener inside the body -> matches innermost first).
_BLOCK_RE = re.compile(
    r"\{\{#(IF|UNLESS)\s+((?:(?!\}\}).)*?)\}\}"  # opener + condition expr (no }} inside)
    r"((?:(?!\{\{#(?:IF|UNLESS)\b|\{\{/(?:IF|UNLESS)\}\}).)*?)"  # body: no nested opener or closer
    r"\{\{/\1\}\}",  # matching closer
    re.DOTALL,
)

_VAR_RE = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_.]*)\s*\}\}")

# Innermost FOREACH (no nested FOREACH opener in the body).
_FOREACH_RE = re.compile(
    r"\{\{#FOREACH\s+([A-Za-z_]\w*)\s+IN\s+([A-Za-z_]\w*)\s*\}\}"
    r"((?:(?!\{\{#FOREACH\b).)*?)"
    r"\{\{/FOREACH\}\}",
    re.DOTALL,
)

# A single comparison clause: VAR, VAR == val, VAR != val.
_CLAUSE_RE = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_.]*)\s*"  # variable
    r"(?:(==|!=)\s*(.*?)\s*)?$"  # optional comparison
)


def _truthy(value):
    """Falsy: None, False, '', 0, 'false'/'no'/'0'/'off'/'none' (case-insensitive)."""
    if value is None or value is False:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    s = str(value).strip()
    if s == "":
        return False
    return s.lower() not in {"false", "no", "0", "off", "none"}


def _lookup(name, variables):
    """Resolve a possibly-dotted name against variables. Returns (present, value)."""
    parts = name.split(".")
    if parts[0] not in variables:
        return False, None
    cur = variables[parts[0]]
    for p in parts[1:]:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        elif hasattr(cur, p):
            cur = getattr(cur, p)
        else:
            return False, None
    return True, cur


def _eval_clause(expr, variables):
    m = _CLAUSE_RE.match(expr)
    if not m:
        raise ValueError(f"Malformed condition: {expr!r}")
    name, op, rhs = m.group(1), m.group(2), m.group(3)
    present, val = _lookup(name, variables)

    if op is None:
        return present and _truthy(val)

    if rhs is not None and len(rhs) >= 2 and rhs[0] in "'\"" and rhs[-1] == rhs[0]:
        rhs = rhs[1:-1]
    lhs = "" if val is None else str(val)
    return lhs == rhs if op == "==" else lhs != rhs


def _eval_condition(expr, variables):
    """Support a single top-level 'or' / 'and' between two clauses."""
    for kw, combine in ((" or ", any), (" and ", all)):
        if kw in expr:
            parts = expr.split(kw)
            return combine(_eval_clause(p, variables) for p in parts)
    return _eval_clause(expr, variables)


def _resolve_blocks(template, variables):
    while True:
        new = _BLOCK_RE.sub(lambda m: _apply_block(m, variables), template)
        if new == template:
            return new
        template = new


def _apply_block(match, variables):
    kind, expr, body = match.group(1), match.group(2).strip(), match.group(3)
    cond = _eval_condition(expr, variables)
    keep = cond if kind == "IF" else (not cond)
    return body if keep else ""


def _expand_foreach(template, variables):
    def expand(match):
        item_name, list_name, body = match.group(1), match.group(2), match.group(3)
        rows = variables.get(list_name) or []
        out = []
        for row in rows:
            scoped = dict(variables)
            scoped[item_name] = row
            # Resolve this row's body fully (vars + any inner conditionals).
            out.append(render(body, scoped, strict=False))
        return "".join(out)

    while True:
        new = _FOREACH_RE.sub(expand, template)
        if new == template:
            return new
        template = new


def render(template, variables, *, strict=False):
    """
    Render `template` against `variables`.

    strict=True raises KeyError on any leftover {{VAR}} with no value;
    strict=False (default) replaces unknown vars with the empty string.
    """
    template = _expand_foreach(template, variables)
    resolved = _resolve_blocks(template, variables)

    def sub_var(m):
        name = m.group(1)
        present, v = _lookup(name, variables)
        if present:
            return "" if v is None else str(v)
        if strict:
            raise KeyError(f"Missing variable: {name}")
        return ""

    return _VAR_RE.sub(sub_var, resolved)


def render_clean(template, variables, *, strict=False):
    """render() plus whitespace tidy: collapse 3+ blank lines, strip edges."""
    out = render(template, variables, strict=strict)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip() + "\n"
