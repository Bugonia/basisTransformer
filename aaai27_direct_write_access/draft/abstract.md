# Draft Abstract

Transformer residual streams are often described as shared representation
spaces through which Attention heads, feed-forward networks (FFNs), embeddings,
and unembeddings communicate. This view, while useful, does not distinguish two
architecturally different roles: directly writing learned directions into the
residual stream and merely modulating the coefficients of another module's
writes. We use a basis/coefficient view of Transformer blocks in which
Attention output projections and FFN down projections define distinct learned
write-basis families, while their coefficients are generated dynamically from
the mixed residual history. Under this view, the standard Transformer block is
not only a composition of Attention and FFN computations; it is a dual-write
system in which both modules retain direct residual-stream write access.

We test this interpretation through controlled decoder-only language-modeling
experiments on enwik8. In a five-seed, parameter-matched topology sweep,
standard Attention-then-FFN blocks obtain the lowest test loss. Reversing order
while preserving both direct write families causes only a small paired
degradation, whereas block-composed and carry variants that remove one direct
write outlet remain substantially weaker. These results do not by themselves
isolate direct write access from all normalization and optimization-path
effects, but they support a write-economy hypothesis: residual streams are
shared spaces where model quality depends on which heterogeneous modules can
write directly and how their dynamic coefficients are coupled.

## Shorter Abstract Target

For an 8-page top-conference paper, shorten to 180-220 words after the open-model
experiments are complete.
