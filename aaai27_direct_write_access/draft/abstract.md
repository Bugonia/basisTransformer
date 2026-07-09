# Draft Abstract

Transformer residual streams are often described as shared representation
spaces through which Attention heads, feed-forward networks (FFNs), embeddings,
and unembeddings communicate. This view, while useful, does not distinguish two
architecturally different roles: directly writing learned directions into the
residual stream and merely modulating the coefficients of another module's
writes. We introduce a basis/coefficient view of Transformer blocks in which
Attention output projections and FFN down projections define distinct learned
write-basis families, while their coefficients are generated dynamically from
the mixed residual history. Under this view, the standard Transformer block is
not only a composition of Attention and FFN computations; it is a dual-write
system in which both modules retain direct residual-stream write access.

We test this interpretation through controlled decoder-only language-modeling
experiments on enwik8. Standard Attention-then-FFN blocks outperform parallel,
order-reversed, block-composed, and carry variants. Crucially, carry variants
give the removed module additional influence over coefficient generation but
still deny it direct write access, and they remain substantially weaker than
standard dual-write blocks. A complementary output-projection absorption control
shows that an Attention output projection can be functionally redundant when it
is not a direct residual write outlet, sharpening the distinction between
parameterization and write access. We further outline pretrained open-model
diagnostics that map the same write-basis structure onto GPT-2, Pythia, and
Qwen-style models through basis inventory, logit attribution, and
counterfactual write patching.

These results support the view that direct write access is an architectural
resource in Transformers. The framework reframes residual streams from passive
representation spaces into write economies, where model quality depends on
which heterogeneous basis families can directly write and how their dynamic
coefficients are coupled.

## Shorter Abstract Target

For an 8-page top-conference paper, shorten to 180-220 words after the open-model
experiments are complete.
