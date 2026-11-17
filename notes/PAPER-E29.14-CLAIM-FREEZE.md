# E29.14 paper claim freeze

**Frozen:** 2026-11-16, before any new replication response.

## Central claim

On the registered GPT-2-medium lexical-induction panels, a semantically defined
changed-lexical-position intervention is quantitatively equivalent to its
whole-sequence counterpart under fixed large component compositions.  The
equivalence was prospectively confirmed from 24 through 47 simultaneously
patched transformer sublayers.

The microscopic anchor is the layer-0 MLP residual update: by itself, replacing
that update at changed lexical positions transfers almost the complete donor
response in both directions, while shared/static positions and the adjacent
layer-0 attention update do not.  The response changes continuously under
dose interpolation.

## Supporting claims

1. Direct add-one and drop-one interventions identify a broad attention-class
   backup, with layer-18 through layer-21 attention updates the strongest
   registered late band in GPT-2 medium.
2. The all-minus-layer-0-MLP composition nearly reaches the donor endpoint, so
   the early MLP is a sufficient trigger but not globally necessary.
3. Earlier predictor and sparse-graph failures motivate measuring the interface
   by exact paired interventions rather than interpreting a fitted interaction
   graph as a direct causal circuit.

## Claims explicitly excluded

- universality across models, tasks, or prompt distributions;
- a Boolean gate, a universal low-rank direction, or algebraic commutation;
- an independently routed early-MLP and late-attention parallel circuit;
- recovery of a portable direct sparse interaction graph;
- superiority of tensor predictors over parameter-matched nonlinear baselines;
- novelty of activation patching, position-aware circuits, interchange
  interventions, sparse recovery, or backup-mechanism discovery by themselves.

## Existing confirmatory evidence

- E29.14a: bidirectional lexical localization, matched attention control, and
  graded dose on twelve response-blind fresh contexts.
- E29.14b: paired all-position versus changed-lexical-position interventions at
  24 and 32 patched components, including nonsaturated hub-off masks.
- E29.14c: the same paired comparison at 36, 40, 44, and all 47 non-hub
  components; c36 remains nonsaturated.
- E29.14d: prospective direct add-one and drop-one tests of the nominated late
  attention backup.

## Completion gates for the paper

The manuscript may describe the result as a model-specific case study using the
existing evidence.  A broader main-track claim additionally requires:

1. a preregistered replication on a distinct checkpoint/model and a
   meaningfully different behavior;
2. bidirectional transfer, semantic-scope localization, a neighboring-component
   control, and at least one nonsingleton composition condition;
3. random/static-position and energy-matched intervention controls;
4. a full-distribution metric alongside the registered scalar response when
   technically feasible;
5. a portable clean-environment inference-to-score replay and public commands;
6. a current related-work comparison and an explicit discovery/confirmation
   chronology.

If the independent replication fails, the failure is retained and the paper is
submitted only with the narrow GPT-2-medium case-study claim.  No threshold,
prompt, model, or intervention may be changed after response inspection and
then presented as confirmatory.
