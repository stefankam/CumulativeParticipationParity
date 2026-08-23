# Stateful fair-FL baselines

This document traces the logical round implemented by the repository:

1. `server/availability.py` produces one shared availability realization.
2. `server/main_server.py` builds eligible `BaselineClient` records and invokes
   the method's selector.
3. `client/topology_client.py` evaluates `loss_at_global = F_k(w_t)` and then
   runs the configured local epochs.
4. The client returns an explicit record containing client ID, model state,
   sample count, pre-update loss, and final training loss.
5. `server/fl_methods.py` applies the method-specific server update and retains
   persistent state. Ordinary FedAvg remains the fallback only for methods that
   specify ordinary averaging.

`utility_log` remains compatibility bookkeeping for cumulative experiment
utility. It is not q-FFL loss, AFL lambda, FairFedCS reputation/queue state, or
PHP-FL personalization state.

## q-FFL / q-FedAvg

- **Source:** Li et al., *Fair Resource Allocation in Federated Learning*,
  q-FedAvg Algorithm 1; reference implementation
  <https://github.com/litian96/fair_flearn>.
- **Objective:** `sum_k p_k F_k(w)^(q+1)/(q+1)`.
- **Selection:** the common uniform-available selection realization.
- **Local update:** configured ordinary local training, starting at `w_t`.
  Before training, the client evaluates the mean example loss over its complete
  local loader; this is `F_k(w_t)`.
- **Server update:** `Delta_k=F_k(w_t)^q(w_t-w_k)` and
  `h_k=q F_k(w_t)^(q-1)||w_t-w_k||^2 + L F_k(w_t)^q`, followed by
  `w_{t+1}=w_t-sum(Delta_k)/sum(h_k)`.
- **State:** no cross-round optimizer state is required.
- **Availability:** only returned records from the shared eligible set enter the
  solver.
- **Configuration:** `Q_FFL_Q`, `Q_FFL_L`, and `Q_FFL_EPS`.
- **L mapping:** `Q_FFL_L` is the explicit Lipschitz-related constant in the
  solver. It is deliberately not inferred as the reciprocal client SGD rate.

## AFL

- **Source:** Mohri, Sivek, and Suresh, *Agnostic Federated Learning*.
- **Objective:** `min_w max_lambda sum_k lambda_k F_k(w)` over the probability
  simplex.
- **Selection/local update:** common availability and ordinary configured local
  training.
- **Server state:** one persistent lambda entry for every logical client,
  initialized uniformly.
- **Model update:** selected client deltas are weighted by their pre-round AFL
  lambda mass, renormalized over the sampled participants.
- **Lambda update:** exponentiated-gradient ascent on observed client losses,
  followed by simplex normalization. A nonparticipant gets no current loss
  gradient; its existing unnormalized mass is retained. This is the documented
  stochastic partial-participation adaptation.
- **Configuration:** `AFL_LAMBDA_INIT` (`uniform` or a full comma-separated
  simplex vector), `AFL_LAMBDA_LR`, and `AFL_MODEL_LR`. These are distinct from
  client SGD learning rate.
- **Logging:** lambda min/max/sum/entropy, leader, observed clients, and sampled
  weighted objective.
- **FIDELITY DEVIATION:** the original convex AFL development samples source
  distributions under assumptions different from intermittent cross-device FL.
  Conditional renormalization of persistent lambda over the realized participant
  set is this simulator's explicit stochastic model-step adaptation.

## FairFedCS

- **Source:** Shi et al., *Fairness-Aware Client Selection for Federated
  Learning*.
- **Selection:** persistent suitability `Psi_i(t)=sigma*r_i(t)+Q_i(t)` is ranked
  only over the shared available set.
- **Queue state:** `Q_i(t+1)=[Q_i(t)+K/N-x_i(t)]_+` persists across rounds.
- **Reputation:** an exponentially smoothed nonnegative local contribution,
  where the repository maps contribution to the measured local empirical-loss
  decrease `max(0,F_k(w_t)-F_k(w_k))`.
- **Aggregation:** ordinary sample-weighted FedAvg after selection; no
  `utility_log`-derived weight is used.
- **Configuration:** `FAIRFEDCS_SIGMA` and `FAIRFEDCS_REPUTATION_DECAY`.
- **FIDELITY DEVIATION:** the repository does not expose the paper's full
  device-quality/reputation telemetry. Empirical local-loss decrease is used as
  the explicit contribution observation; it is not claimed to be identical to
  every reputation signal evaluated by the paper.

## PHP-FL

- **Source:** Wu et al., *A Fair Federated Learning Method for Handling Client
  Participation Probability Inconsistencies in Heterogeneous Environments*;
  official repository <https://github.com/Siyuan01/PHP-FL-main>.
- **Architecture:** each participating logical client retains a personalized
  local model plus an auxiliary model initialized from the shared auxiliary
  model. Only auxiliary updates go to the server.
- **DEAL:** local and auxiliary logits form an equal ensemble for supervised
  loss. Bidirectional temperature-scaled KL alignment trains both ends.
- **ISPU:** a persistent EMA of absolute local parameter changes represents
  importance. The least-important progress-dependent fraction receives shared
  parameters before local training; the mask and importance history persist.
- **Inactive clients:** no function is invoked for them, so their local model,
  importance, and mask are unchanged.
- **Server update:** a dedicated auxiliary-record path sample-averages only
  successful auxiliary submissions. Personalized models are never averaged.
- **Configuration:** `PHP_DEAL_ALIGN_WEIGHT`, `PHP_DEAL_TEMPERATURE`,
  `PHP_ISPU_INITIAL_RATIO`, `PHP_ISPU_RATIO_GROWTH`, and
  `PHP_ISPU_IMPORTANCE_DECAY`.
- **FIDELITY DEVIATION:** this simulator exposes one homogeneous ResNet family,
  while PHP-FL targets heterogeneous architectures. Matching local/auxiliary
  shapes are used as the homogeneous special case. The external official source
  was not vendored into this repository; consequently the equal ensemble,
  parameter-change importance measure, and linear update-ratio schedule are an
  explicit semantic adapter and **must not be described as a byte-for-byte or
  fully faithful port of the official implementation**.
