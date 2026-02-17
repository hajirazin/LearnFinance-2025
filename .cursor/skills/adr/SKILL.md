---
name: adr
description: Create or refine Architecture Decision Records (ADRs) for complex technical changes. Use when the user runs /adr, asks to create an ADR, document an architecture decision, or needs to record technical design choices. Strictly no code edits -- ADR content only.
---

# Architecture Decision Record (ADR) Mode

**Strict constraint:** Do NOT edit, generate, or suggest any code changes, Jira tasks, or implementations. Focus solely on creating or refining ADRs.

## Where ADRs live

ADRs are **technical decisions for developers** and belong in MKDocs:

- **Location:** `docs/pages/adr/<title>.md`
- **Format:** MKDocs/HTML-friendly (double line breaks, Mermaid diagrams, relative links)

### Documentation split

| Aspect | Wiki (Confluence) | Jira | MKDocs (ADRs) |
|--------|-------------------|------|---------------|
| **Purpose** | Product context | Execution tracking | Technical decisions |
| **Audience** | Product, Sales, Executives | Dev team, Scrum master | Developers, Architects |
| **Language** | Business, benefits | User stories, ACs | Technical, architecture |
| **Content** | WHY build it | HOW to execute | HOW to architect |

## ADR creation workflow

### 1. Reference the plan

Build strictly on the existing plan from plan mode. If the plan is unclear or incomplete, suggest switching back to plan mode to refine first.

### 2. Assess need

Confirm the ADR is warranted:

- Big changes impacting DDD domains or cross-aggregate boundaries
- Infrastructure or platform decisions
- New integration points with external services

If not needed, suggest skipping to implementation.

### 3. Ask detailed questions (5-10)

Prioritize questioning over assuming:

- "How should this align with DDD aggregates?"
- "What are the performance/security implications?"
- "Any integration points with external services?"
- "Root causes of current limitations?"
- "What alternatives were considered and rejected?"

### 4. Build the ADR

Follow this mandatory structure:

```markdown
# ADR-NNN: <Title>

## Status
Proposed | Accepted | Deprecated | Superseded by ADR-XXX

## Date
YYYY-MM-DD

## Decision Makers
<names/roles>

## Context
<Business context -- why this decision is needed>

## Decision
<The architecture decision made>

## Alternatives Considered
### Option A: ...
### Option B: ...
### Option C: ...

## Consequences
### Positive
### Negative
### Risks

## DDD Alignment
<How this respects domain boundaries, aggregates, services>

## Implementation Notes
<High-level steps and risks -- NO code>
```

### 5. DDD emphasis

Every ADR must respect and document:

- Domain boundaries
- Aggregate roots affected
- Service layer implications
- Bounded context relationships

### 6. Iterate

Support multiple refinement rounds based on answers.

### 7. Exit check

End by asking: "Is the ADR complete and validated? Ready to move to implementation, or back to plan mode?"

## When to add a business summary in Wiki

If the decision has business implications for non-technical stakeholders:

1. Create full ADR in MKDocs (technical details)
2. Create business summary in Wiki (why it matters for product/business)
3. Link between them

## MKDocs content scope

What belongs in MKDocs alongside ADRs:

- Detailed system architecture (diagrams, service descriptions, data flows)
- API specifications and contracts
- Database schemas and migrations
- Local development setup
- Deployment guides
- CI/CD pipeline documentation
- Security architecture
- Code patterns and examples
