# Output summary — $source_title

*This file is the citable scientific record of the run. Claims extracted during gate 2 will cite specific sections of this document. Fill each `<FILL>` marker with honest, specific content. Where you did not measure something, say so — unknown is not the same as unverified.*

## Purpose

<FILL — one short paragraph: why this run was performed. What question was it meant to answer?>

## Physical configuration

<FILL — the physical setup: geometry, materials, initial conditions, boundary conditions, driving sources or forces. Enough detail that a reader could reproduce.>

## Numerical configuration

- **Solver:** $solver_name, built from commit `$solver_commit`
- **Inputs:** $input_files
- **Hardware:** $hardware
- **Run date:** $run_date

<FILL — anything else about the numerics that isn't captured in `input.ini`: why specific numerical choices were made, any deviations from default configuration, build-time flags.>

## Runs performed

<FILL — if this source captures multiple logical runs (baseline + restart test, parameter sweep, etc.) per AGENTS.md v0.2 §3.1 one-envelope rule, describe each in a separate subsection and tag claims with the `run:` frontmatter key. Delete this section if this source is a single run.>

## Results

<FILL — what you observed. Separate what was measured (artifact-backed) from what was inferred (reasoning). Quantify where possible. If you inspected outputs and then deleted them, say so explicitly — that becomes a provenance meta-claim in gate 2.>

## Limitations and unverified assertions

<FILL — what this run does NOT license. No grid-convergence study, single-configuration test, evidence not re-verifiable from rossum alone, etc. Honest scope-bounding here prevents downstream over-citation.>

## Artifacts

### In rossum (committed)
- `input.ini` — the input file(s) used for this run
- `output_summary.md` — this document

### Outside rossum
<FILL — paths to raw output data if not committed to rossum. State why (size, binary format, etc.). This is the `raw_data_location` in source.yaml.>

## External references

<FILL — any theoretical source, prior experiment, or analytical result that this run was compared against. These become `external_refs:` entries on extracted claims per AGENTS.md v0.2 §2.5.>
