# Candidate precision review

Candidate precision is measured over the candidates eligible to enter the investigation fan-out for each distinct drug
and historical cutoff. Each prediction is reviewed once for its drug, cutoff, and disease.

## Assign labels

A candidate is **valid** when it is outside the drug's approved scope at the cutoff and independent evidence supports a
therapeutic hypothesis for that drug-disease pair. Evidence may include drug-specific clinical studies, consistent human
observational evidence, or a pharmacological mechanism supported by disease biology and evidence from the same drug class.

A candidate is **invalid** when it is already approved at the cutoff, represents only a broader parent or unrelated sibling
of the supported disease, describes an adverse effect or procedure rather than a therapeutic indication, conflicts with the
drug's pharmacological direction, or lacks an independent therapeutic rationale.

A candidate is **uncertain** when the available evidence cannot distinguish a defensible hypothesis from a false candidate.
Uncertain cases are not silently removed from the measurement.

Every decision records a reason category, a factual rationale, and the evidence used. Missing evidence produces an uncertain
decision, not a guessed label.

## Calculate precision

Reviewed precision divides valid decisions by valid plus invalid decisions. The report also gives a lower bound that counts
uncertain decisions as invalid and an upper bound that counts them as valid. Micro precision weights every prediction
equally. Macro precision gives each drug equal weight. A 95 percent Wilson interval accompanies the reviewed micro estimate.

The metric is precision at the configured investigation limit. It does not describe the full candidate universe beyond that
limit and it is not FDA-approval precision.
