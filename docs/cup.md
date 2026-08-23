# Cumulative Utility Parity audit and implementation mapping

## Pre-change audit

| Paper variable | Previous repository variable/path | Previous update | Match? |
|---|---|---|---|
| `A_k(t)` | logical `BaselineClient.availability` | full trace read in `main_server.py` | Yes: independent telemetry existed. |
| `pi_hat_k(t)` | `AvailabilityEstimator` / `availability_seen` | from full telemetry | Partly: estimator was correct, but the production scheduler did not preserve the full CUP accounting path. |
| `S_k(t)` | `selected` | `FairnessSchedulerController.select()` | **No:** the scheduler filtered to currently available clients. |
| `P_k(t)` | effectively `selected` | selected clients were sent directly to training | **No:** selection and participation collapsed. |
| `Delta u_k(t)` | one global test accuracy | credited to every selected client | **No:** not a per-client marginal utility. |
| `u_k(t)` | `utility_sums` and `utility_log` | increased on selection | **No:** unavailable selected clients could not be represented and global accuracy was substituted for client utility. |
| `d_k(t)` | scheduler conditional deficit / participation debt variants | selection-dependent | **No:** it was not literally `sum_(s<t)(1-P_k(s))`. |
| surrogate | generic accounting/training mode primitives | not integrated into the production logical loop | **No:** empirical utility-only semantics were not explicit. |
| fairness metrics | utilities derived from attributed global accuracy; Gini from selections | round end | **No:** neither normalized cumulative utility nor realized participation was used consistently. |

The corrected production round is now:

`full trace A -> telemetry-only pi_hat -> available-candidate CUP score -> binary S -> P=A*S -> train P -> importance-corrected aggregate -> shared per-client evaluation -> Delta u -> realized u -> debt -> utility-only surrogate -> metrics/log`.

The audited path requires logical scheduling. The legacy physical-only path lacks
a complete logical-population telemetry vector and is rejected rather than
silently producing incomparable baseline accounting.

## Paper-to-code mapping

| Variable | File/class | Persistent variable | Update timing |
|---|---|---|---|
| `A_k(t)` | `main_server.py`; `CumulativeUtilityParity.observe_availability` | per-round telemetry plus `availability_count` | Before selection for every client. |
| `S_k(t)` | `CumulativeUtilityParity.select_clients` | `selection_count` | Decision is made over the currently available candidate set; counted at round end. |
| `P_k(t)=A_k(t)S_k(t)` | `realize_participation` | `participation_count` | Execution gate immediately after selection. |
| `pi_hat_k(t)` | `observe_availability` | `availability_estimate` | `availability_count/(t+1)` from telemetry only. |
| `Delta u_k(t)` | `_utility_increment` | row `utility_increment` | After common per-client evaluation; accuracy gain by default. |
| `u_k(t)` | `end_round` | `utility` | Incremented only if `P=1`. |
| `u_tilde_k(t)` | `end_round` | `normalized_utility` | `utility/pi_hat`; NaN is explicitly logged for zero observed availability. |
| `d_k(t)` | `end_round` | `participation_debt` | Adds `1-P_k(t)` after utility accounting; never reset. |
| reactive `p_k(t)` | `reactive_scores` | `last_scores` | Before sampling: `alpha/(pi_hat+epsilon)*(1+lambda*d)`. |
| normalized selection marginal | `fixed_size_inclusion_probabilities` | `last_selection_probabilities` | Water-filled to sum exactly `m`. |
| `r_k*` and `tau*` | `oracle_maxmin_rates` | transient oracle rates | Optional `CUP_SCHEDULER=oracle_maxmin`. |
| surrogate staleness | `end_round` | `last_real_participation_round` | `t-tau_k` only for a nonparticipant with prior real participation. |
| surrogate reliability | `end_round` | logged `surrogate_weight` | `eta0 exp(-CUP_SURROGATE_DECAY*staleness)`. |

## Scheduler semantics

- **Population scored:** every currently available logical client. Scores remain
  logged for all clients, but unavailable clients receive zero inclusion
  probability and cannot consume the round budget.
- **Current availability filter:** the complete `A_t` vector updates the estimator,
  then only `A_k(t)=1` clients enter fixed-size sampling.
- **Reactive score:** `alpha_k/(pi_hat_k+epsilon) * (1+CUP_DEBT_LAMBDA*d_k)`.
- **Cold start:** denominator epsilon plus `CUP_INVERSE_AVAILABILITY_CLIP`; clipping
  is visible through the logged score and run configuration.
- **Normalization:** priorities are water-filled into inclusion marginals in
  `[0,1]` summing to `m`.
- **Sampling:** pivotal dependent rounding, without replacement, with cardinality
  `min(m, |A_t|)` and the computed marginals.
- **Oracle:** separately computes Lemma 2 `tau*` and `r*_k`; it is not conflated
  with the reactive scheduler.

## Aggregation semantics

CUP uses a Horvitz-Thompson client-delta correction for the uniform-client target
objective. A participating client's delta multiplier is
`(1/N)/(selection_inclusion_probability * pi_hat)`, with an explicit numerical
clip (`CUP_AGGREGATION_CLIP`). Scheduler scores themselves are never model
weights. Surrogates never enter aggregation in empirical `utility_only` mode.

**MANUSCRIPT AMBIGUITY — aggregate correction.** The supplied specification
requires objective-preserving importance correction but does not give an exact
finite-round estimator for fixed-size dependent sampling plus unknown true
availability. The implementation uses exact scheduling marginals and empirical
telemetry `pi_hat` in an unnormalized Horvitz-Thompson delta. An alternative is a
self-normalized Hájek estimator, which has lower variance but finite-sample bias.
The explicit correction clip introduces the bounded-variance bias anticipated by
the convergence discussion.

## Utility and surrogates

- Default `CUP_UTILITY_METRIC=accuracy_gain` uses consecutive per-client global
  model accuracy differences and retains negative gains.
- `loss_reduction` is a distinct opt-in nonnegative mode using returned local
  pre/post empirical loss.
- Only `P=1` adds the increment to real utility.
- `CUP_SURROGATE=false` is the default. When enabled, only
  `CUP_SURROGATE_MODE=utility_only` is accepted. It maintains separate surrogate
  utility and never changes `P`, counts, returned updates, or the global model.

**MANUSCRIPT AMBIGUITY — utility clipping.** Accuracy gain is retained with its
sign because the supplied experimental definition is a consecutive accuracy
difference. The alternative nonnegative convention materially changes Jain/CV
and is available only as the separately named loss-reduction mode.

**MANUSCRIPT AMBIGUITY — SELECTFAIR.** The specification does not uniquely state
the fixed-size weighted sampler. Pivotal dependent rounding was chosen because it
preserves a strict binary budget and known first-order inclusion marginals needed
by the correction. Sequential PPS without replacement is an alternative but has
less convenient inclusion probabilities.

**MANUSCRIPT AMBIGUITY — alpha.** `alpha_k=1` is the neutral default because no
client-specific alpha policy was supplied. A different policy changes selection
priorities and must be introduced under an explicit configuration.

**Warm start:** CUP uses the same full global initialization and ordinary local
training as the baselines (`alpha=1` semantics). No CUP-only personalized warm
start is silently enabled.

## Method distinction

| Method | Fairness target | Selection | Server objective/update | Temporal availability state |
|---|---|---|---|---|
| CUP | availability-normalized cumulative utility | inverse availability + participation debt / max-min oracle | HT-corrected target-objective delta | explicit |
| q-FFL | q-weighted client loss | baseline participation | q-FedAvg | no CUP state |
| AFL | worst-case mixture loss | baseline participation | minimax lambda/model update | no CUP state |
| FairFedCS | equitable selection + reputation/performance | Lyapunov/reputation CSI | FedAvg after selection | selection fairness state |
| PHP-FL | participation/model heterogeneity | PHP-FL baseline selection | DEAL/ISPU auxiliary update | PHP-FL-specific |
| FedAvg-random | none | random population intent | FedAvg | none |
| Uniform-available | none | uniform among available | FedAvg | current availability only |
| FedProx | optimization heterogeneity | baseline random | proximal local objective + FedAvg | none |
