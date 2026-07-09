# Introduction Skeleton

## Paragraph 1: Residual Streams Are Central but Underspecified

Message:

Transformer residual streams are the shared communication substrate of modern
language models, but current analyses usually ask what information is present in
the stream rather than how different modules obtain write access to it.

Draft:

> The residual stream is the central communication channel of a Transformer
> language model. Token embeddings, Attention heads, feed-forward networks
> (FFNs), normalization layers, and the unembedding all read from or write to
> this shared state. Most analyses therefore treat the residual stream as a
> representation space: a sequence of hidden vectors whose geometry, features,
> or linear probes reveal what the model knows at a given layer. This
> perspective is powerful, but it leaves a basic architectural question
> under-specified: which modules are allowed to directly write their own learned
> directions into the residual stream?

## Paragraph 2: The Missing Distinction

Message:

Direct writing and coefficient modulation are not the same. A module can shape
another module's coefficients without owning final write directions.

Draft:

> We argue that this distinction matters. A sublayer may influence the residual
> stream in two different ways. It may directly write an update through its own
> output basis, or it may only shape the coefficients used by a later sublayer.
> These roles are conflated in the standard description of Transformer blocks,
> where Attention and FFN are often treated as two nonlinear transformations
> connected by residual paths. In a standard pre-normalization block, however,
> both sublayers are direct authors of the residual stream: Attention writes
> through its output projection, and the FFN writes through its down projection.
> In contrast, block-composed variants such as `h + FFN(Attention(h))` allow
> Attention to affect FFN coefficients but remove Attention's direct write
> outlet.

## Paragraph 3: Basis/Coefficient Framework

Message:

Define learned write bases and dynamic coefficients.

Draft:

> We formalize this distinction through a basis/coefficient view of residual
> updates. The columns of an Attention output projection form an Attention
> write-basis family, while the columns of an FFN down projection form an FFN
> write-basis family. The corresponding coefficients are not fixed parameters:
> Attention coefficients combine context-dependent routing weights with value
> coordinates, and FFN coefficients are nonlinear activations generated from the
> current residual state. A standard block therefore produces updates of the
> form
> \[
> \Delta H_l = B_l^A c_l^A(H_l) + B_l^F c_l^F(H_l, B_l^A c_l^A(H_l)),
> \]
> where both basis families write directly, and both coefficient functions are
> coupled through the mixed residual history.

## Paragraph 4: Experimental Question

Message:

If this view is right, removing direct write access should hurt more than merely
changing order or coefficient coupling.

Draft:

> This view makes a concrete prediction: architectures that preserve both
> direct write families should remain closer to the standard Transformer than
> architectures that allow one module only to modulate another module's
> coefficients. We test this prediction by comparing standard, parallel,
> order-reversed, block-composed, and carry variants under matched
> language-modeling settings. The carry variants are especially diagnostic:
> they pass additional intermediate signals across layers, giving the removed
> module a stronger role in coefficient generation, but they still do not
> restore its direct write basis.

## Paragraph 5: Contributions

Message:

List the paper's concrete contributions.

Draft:

> Our contributions are fourfold. First, we introduce a basis/coefficient
> formalism that separates direct residual write access from coefficient
> modulation in Transformer blocks. Second, we derive the Attention and FFN
> write-basis families for standard decoder-only Transformers and show how their
> coefficients depend on residual history. Third, we provide controlled
> language-modeling experiments showing that dual direct write access explains a
> substantial part of the standard block advantage, beyond ordering and
> optimization effects. Fourth, we extend the framework toward pretrained open
> models through basis inventory, logit attribution, and counterfactual
> write-patching diagnostics.

## Paragraph 6: Implication and Boundary

Message:

State the design principle while avoiding overclaiming.

Draft:

> These results suggest that residual connections should not be viewed only as
> optimization aids. They allocate write rights in a shared computational
> workspace. For the model scales studied here, Attention and FFN appear to
> benefit from retaining separate, heterogeneous direct write outlets. The
> framework does not imply that every learned direction is semantically
> monosemantic, nor does it replace existing accounts of residual-stream
> geometry. Instead, it adds a structural layer: who can write, through which
> basis, with coefficients generated by which residual history.
