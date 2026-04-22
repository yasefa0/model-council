# Lens Templates

Lenses are structured thinking frameworks that focus the Model Council's analysis through a specific decision-making lens. Instead of a generic prompt, a lens gives the council a shared vocabulary and set of dimensions to evaluate.

## How It Works

When you pass `--lens <name>` to the council, it loads the corresponding `.md` file from this directory and prepends it to the prompt as a system-level framing. All three panel models and the chairman receive the same lens context.

```bash
python council.py --lens career-bet "Should I take this offer?"
python council.py --lens leap "I'm thinking about moving to a new city for a job"
```

## Included Lenses

| Lens | File | Best For |
|------|------|----------|
| `career-bet` | `career-bet.md` | Evaluating job offers, opportunities, or pivots |
| `leap` | `leap.md` | Major life or career decisions with high stakes |

## Creating a Custom Lens

1. Create a new `.md` file in this directory, e.g. `lenses/my-lens.md`
2. Add a YAML front matter block with `name` and `description`
3. Write your framework as a set of named dimensions with guiding questions

### Template

```markdown
---
name: my-lens
description: One-line description of what this lens is for
---

# My Lens Framework

- **Dimension 1**: Guiding question or prompt for this dimension
- **Dimension 2**: Guiding question or prompt for this dimension
- **Dimension 3**: Guiding question or prompt for this dimension
- **Dimension 4**: Guiding question or prompt for this dimension
```

### Tips

- **Keep dimensions to 4–6.** More than that dilutes focus.
- **Name dimensions clearly.** The council models use the dimension names as anchors.
- **Write questions, not instructions.** "What evidence exists vs. what's inferred?" works better than "Analyze the evidence."
- **Match the lens to a decision type**, not a topic. A good lens works across many specific scenarios of the same type.

## Example: Career Bet Lens in Action

```
$ python council.py --lens career-bet "I have two job offers: one at a startup, one at a big tech company"

[Stage 1] Collecting panel responses...
[Stage 2] Peer ranking...
[Stage 3] Chairman synthesis...

## Chairman Synthesis

**Signal vs. Noise**: The startup offer shows real traction (Series B, 40% YoY growth)...
**Optionality**: Big tech opens more doors short-term but the startup equity...
...
```
