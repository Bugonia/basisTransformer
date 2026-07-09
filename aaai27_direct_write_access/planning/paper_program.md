# Paper Program

## Paper 0: Residual Stream as a Write-Economy

Claim:

> Transformer residual streams are best understood as shared write ledgers where
> different modules own different basis families and coefficient generators.

Use:

- conceptual framing;
- existing enwik8 topology and basis-carry experiments;
- open-model basis inventory.

## Paper 1: Basis-Level Attribution for Transformer Predictions

Claim:

> Token logits can be decomposed into layer/module/basis write contributions,
> giving a more direct explanation than attention maps alone.

Minimum evidence:

- one open model;
- prompt-level attribution reports;
- causal removal of top contributions.

## Paper 2: Coefficient Trajectories Reveal Hidden Computation Paths

Claim:

> Final hidden states hide path information; coefficient trajectories expose
> behaviorally meaningful computation routes.

Minimum evidence:

- prompt-pair trajectory distance;
- contrast with final hidden-state distance;
- case studies for correct vs wrong or grounded vs unsupported answers.

## Paper 3: Direct Write Access Is an Architectural Resource

Claim:

> Attention and FFN both need direct residual write access; coefficient-only
> modulation is weaker than direct heterogeneous basis writes.

Minimum evidence:

- existing standard/parallel/block/carry results;
- additional robustness across optimizer, depth, and scale;
- pretrained-model activation interventions.

## Paper 4: Grounding and Hallucination as Basis-Support Mismatch

Claim:

> Hallucination can be detected when answer logits are driven by high-level
> language-prior writes without stable evidence-mediated Attention support.

Minimum evidence:

- RAG-style prompt set;
- grounding score;
- comparison with entropy, attention mass, and self-consistency.

## Paper 5: Architectural-Basis-Aware Sparse Autoencoders

Claim:

> SAE dictionaries should be connected to native Transformer write dictionaries
> rather than treated as completely post-hoc bases.

Minimum evidence:

- ordinary SAE baseline;
- architectural-basis-constrained SAE;
- reconstruction, sparsity, interpretability, and causal ablation comparison.

