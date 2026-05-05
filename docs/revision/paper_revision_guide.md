# Paper Revision Guide

This guide proposes targeted manuscript edits for the LeWM audit revision. Each item gives the intended location, reviewer weakness addressed, current text where applicable, and insertion-ready LaTeX prose.

## G1. Abstract Scope Weakening

**Location:** `paper/main.tex`, abstract, final sentence.

**Addresses:** W1.

**Current text:**

```latex
We recommend reporting pool-level and selection-level metrics alongside endpoint metrics for latent world models that use CEM-style planning.
```

**Revised text:**

```latex
For LeWM-style latent planners that select actions by terminal latent costs under iterative sampling-based optimization, we recommend reporting pool-level and selection-level metrics alongside endpoint metrics.
```

**If the submitted abstract contains the broader phrase from the review copy, replace it directly:**

Current:

```latex
any latent world model that plans with terminal latent costs can be audited this way
```

Revised:

```latex
latent planners with terminal-cost objectives can be audited this way when their optimizer-produced candidate pools and evaluation-time physical costs are available
```

## G2. Introduction Scope Weakening

**Location:** `paper/sections/introduction.tex`, paragraph 1.

**Addresses:** W1.

**Current text:**

```latex
Self-supervised world models that learn dynamics in latent space have become a practical route to goal-conditioned control from pixels. Recent JEPA-style models train without reward supervision, yet achieve high planning success on manipulation tasks.
```

**Revised text:**

```latex
Self-supervised world models that learn dynamics in latent space are an increasingly practical route to goal-conditioned control from pixels. Recent JEPA-style models train without reward supervision and can achieve high planning success on selected manipulation benchmarks.
```

**Location:** `paper/sections/introduction.tex`, paragraph 4.

**Addresses:** W1.

**Current text:**

```latex
Our results apply specifically to planning with Euclidean terminal latent costs under CEM-style optimization, but the evaluation protocol we propose is more general. Endpoint metrics are not wrong; they measure whether a representation carries task information, not whether CEM can use that information during search. Planning evaluation should therefore include the distributions produced by the optimizer itself. We propose a minimal reporting protocol with three views. Report endpoint metrics on a fixed benchmark set. Report pool metrics on the candidates produced by the planner. Report selection metrics for the action that is actually executed. This protocol extends beyond LeWM; any latent world model that plans with terminal latent costs can be audited this way. The goal is a planning-compatible test of geometry, not another endpoint-only score.
```

**Revised text:**

```latex
Our empirical claims apply to LeWM with Euclidean terminal latent costs under CEM-style optimization, evaluated on PushT and OGBench-Cube. The evaluation principle is broader, but it should be applied with the same instrumentation: endpoint metrics are not wrong; they measure whether a representation carries task information, not whether an iterative planner can use that information during search. Planning evaluation should therefore include the distributions produced by the optimizer itself. We propose a minimal reporting protocol with three views. Report endpoint metrics on a fixed benchmark set. Report pool metrics on the candidates produced by the planner. Report selection metrics for the action that is actually executed. For latent planners with terminal-cost objectives, this audit is directly applicable when final candidate pools can be saved and an evaluation-time physical cost is available. The goal is a planning-compatible test of geometry, not another endpoint-only score.
```

**Optional caption weakening for Figure 1:**

Current:

```latex
\caption{Endpoint-planning decoupling in latent world models. ...
```

Revised:

```latex
\caption{Endpoint-planning decoupling in terminal-cost latent planning. ...
```

## G3. New Mechanism Attribution Subsection

**Location:** Insert in `paper/sections/decoupling.tex` after `\subsection{Five-case taxonomy}` and its concluding paragraph, before `\subsection{Per-subset structure in PushT}`.

**Addresses:** W2.

**Current text:** NEW PARAGRAPH.

**New text:**

```latex
\subsection{Mechanism attribution: convergence compression versus local representation failure}

The gap $\DeltaCEM$ could arise from several mechanisms. CEM might compress the final candidate pool into a physically narrow neighborhood where any cost function has little signal; the learned encoder might preserve endpoint order globally but lose local order within the optimizer's final pool; or predictor bias might distort the imagined terminal states. To separate these explanations, we recomputed final-pool ranking with a privileged V1 hinge oracle cost, denoted $C_\mathrm{V1}$, on the same default CEM final pools. This analysis changes only the ranking function used for diagnosis: the candidate pools, model predictions, and executed rank-1 actions are unchanged.

Across the 100 PushT pairs, the learned Euclidean cost gives $\Rpool(\Cmodel)=0.084$, while the V1 oracle gives $\Rpool(C_\mathrm{V1})=0.666$. The same pattern holds in the hard subsets. In the 16 invisible-quadrant pairs, where the final pools contain little success mass ($M_\mathrm{pool\_success}=1.4\%$), $\Rpool(C_\mathrm{V1})=0.775$ while $\Rpool(\Cmodel)=0.022$. In the 21 sign-reversal pairs, $\Rpool(C_\mathrm{V1})=0.925$ while $\Rpool(\Cmodel)=-0.040$. Even in the latent-favorable subset, where the learned cost performs comparatively better, V1 remains stronger: $\Rpool(C_\mathrm{V1})=0.417$ versus $\Rpool(\Cmodel)=0.197$.

These numbers rule out pure CEM convergence compression as the dominant explanation and clearly exceed the pre-registered ``mixed'' zone. The mixed rule was designed for cases where the privileged ranker itself had only weak final-pool signal, e.g., $\Rpool(C_\mathrm{V1}) \in [0.05,0.15]$. Here $\Rpool(C_\mathrm{V1})=0.666 \gg 0.15$ while $\Rpool(\Cmodel)=0.084 \approx 0$, so the overall 100-pair result fits the local-representation-failure pattern: V1 retains substantial ranking power inside the CEM final pool, whereas the learned Euclidean cost does not. Per subset, the invisible-quadrant and sign-reversal groups show the strongest representation failure; ordinary and latent-favorable pairs retain partial learned-cost signal alongside stronger V1 signal.
```

## G4. New Discussion Paragraph on MPPI

**Location:** Insert in `paper/sections/discussion.tex` after `\subsection{Implications for latent world model evaluation}` or as a new subsection before `\subsection{Intervention paths motivated by the framework}`.

**Addresses:** W4.

**Current text:** NEW PARAGRAPH.

**New text:**

```latex
\subsection{Optimizer dependence: CEM versus MPPI}

We also tested whether endpoint-planning decoupling is specific to CEM's hard elite truncation. On 30 PushT pairs with three seeds per pair, we replaced the CEM update with an MPPI-style soft weighting update over all 300 candidates, using the same model, action parameterization, horizon, and terminal latent cost. MPPI achieved lower rank-1 planning success than CEM in this fixed-budget comparison (38.9\% versus 53.3\%), consistent with slower convergence under 30 iterations. However, the pool geometry gives the relevant diagnostic. MPPI preserved substantially more physical diversity in its final pools: the mean standard deviation of $C_\mathrm{real\_state}$ was 29.478 under MPPI versus 7.947 under CEM, a 3.7$\times$ increase. Correspondingly, $\Rpool(\Cmodel)$ rose from 0.115 under CEM to 0.194 under MPPI, while $\Rpool(C_\mathrm{V1})$ rose from 0.537 to 0.707.

The comparison supports a decomposed attribution. CEM hard truncation exacerbates the gap by compressing the candidate pool, but soft weighting does not eliminate it. Using $\Rendpoint=0.495$ at $m=192$ as the reference, MPPI retains only about 39\% of the endpoint ranking signal ($0.194/0.495$). The CEM-to-MPPI improvement in $\Rpool(\Cmodel)$ accounts for roughly 21\% of the endpoint-pool gap, $(0.194-0.115)/(0.495-0.115)$. Thus endpoint-planning decoupling is partially optimizer-specific and partially optimizer-general: CEM's hard truncation worsens the local pool geometry, but most of the lost ranking signal remains representation-driven.
```

## G5. New Discussion Paragraph on Privilege-Free Monitoring

**Location:** Insert in `paper/sections/discussion.tex` after the MPPI paragraph, or in the limitations section immediately after the paragraph on privileged physical costs.

**Addresses:** W5.

**Current text:** NEW PARAGRAPH.

**New text:**

```latex
\subsection{Privilege-free monitoring signals}

The three-level reporting protocol uses $C_\mathrm{real\_state}$ to compute $\Rpool$ and selection regret, so the full diagnostic is an evaluation-time audit rather than a deployable monitor. We therefore tested whether quantities available from the learned cost alone predict high-regret pairs. The strongest global proxy was the standard deviation of the learned cost among the top-30 final-pool candidates, which correlated with selection regret at Spearman $\rho=0.314$ ($p=0.001$). Within the invisible-quadrant subset, the full-pool learned-cost spread was stronger, with $\rho=0.588$ ($p=0.017$). As a cross-check, learned-cost spread tracked privileged physical spread: $\rho(\mathrm{std}(\Cmodel), \mathrm{std}(C_\mathrm{real\_state}))=0.482$ ($p<0.001$).

These proxies are not replacements for the full audit, but they show that partial monitoring is possible without simulator access. In particular, low learned-cost diversity in the final pool can flag candidate-pool compression and elevated selection risk. The practical conclusion is therefore intermediate: privileged physical costs remain necessary for definitive $\DeltaCEM$ measurement, while cost-spread proxies provide statistically significant warning signals that can be logged by a deployed latent planner.
```

## G6. Limitations Update

**Location:** Replace or revise `paper/sections/limitations.tex`, paragraphs 1, 4, and 5.

**Addresses:** W1, W3, W4.

**Current text:**

```latex
The discussion proposes a broader reporting protocol, but the evidence has bounded scope: all experiments use one latent world model, LeWM, on two single-object manipulation tasks with low-dimensional simulator states. The framework is model-agnostic, but cross-model generalization to PLDM, DINO-WM, TD-MPC, or other latent planners is not tested. PushT and OGBench-Cube do not cover locomotion, navigation, deformable objects, multi-object interaction, semantic planning, broad exploration data, mixed-quality behavior, or online collection.
```

**Revised text:**

```latex
The evidence has bounded scope. All experiments use a single latent world model, LeWM, and two single-object manipulation environments, PushT and OGBench-Cube. The framework is intended to be model-agnostic, but cross-model generalization to PLDM, DINO-WM, TD-MPC, or other latent planners is not tested. The environments also do not cover locomotion, navigation, deformable objects, multi-object interaction, semantic planning, broad exploration data, mixed-quality behavior, or online collection.
```

**Current text:**

```latex
Cube-specific evidence has narrower support: full projected CEM uses a 25-pair subset rather than the full 100-pair benchmark, the inverted-U pattern received mixed support after multi-seed verification, and the orthogonal sanity check verified data integrity at the file level only.
```

**Revised text if Phase E completes:**

```latex
Cube-specific evidence remains narrower than PushT but is strengthened by the extended full projected-CEM run: the full projected-CEM cell uses the first 50 Cube pairs with three projection seeds, while the Cube re-rank-only cell uses all 100 pairs with three seeds. This supports the cross-environment $\DeltaCEM$ claim, but Cube still has fewer full-planning pairs than PushT and should not be interpreted as a complete environment-general validation.
```

**Revised text if Phase E is incomplete:**

```latex
Cube-specific evidence has narrower support: full projected CEM uses a 25-pair subset rather than the full 100-pair benchmark, the inverted-U pattern received mixed support after multi-seed verification, and the orthogonal sanity check verified data integrity at the file level only. We therefore treat Cube as cross-environment support for the endpoint-pool diagnostic, not as a fully powered second benchmark.
```

**Current text:**

```latex
The physical task costs used for pool metrics are privileged evaluation signals: appropriate for diagnosis, but unavailable to a deployed planner. The proposed protocol is an evaluation protocol, not a new planning objective.
```

**Revised text:**

```latex
The physical task costs used for pool metrics are privileged evaluation signals: appropriate for diagnosis, but unavailable to a deployed planner. The proposed protocol is an evaluation protocol, not a new planning objective. Our privilege-free proxy analysis suggests that learned-cost spread can partially monitor high-regret cases without simulator access, but this proxy has only been validated on PushT and does not yet replace $\Rpool$ or selection-regret measurement across environments.
```

**Current text:**

```latex
The finding is specific to CEM-style optimization. MPPI, gradient-based planners, learned proposals, or value-guided search may interact differently with the same latent geometry, requiring redefined optimizer-local distributions and selection metrics.
```

**Revised text:**

```latex
CEM remains the primary optimizer studied in this paper. The MPPI comparison on a 30-pair PushT subset (90 scored pools across three seeds) provides partial cross-optimizer evidence: soft weighting preserves more physical pool diversity and improves $\Rpool(\Cmodel)$, but it does not close the endpoint-pool gap. Gradient-based planners, learned proposals, value-guided search, and closed-loop replanning may interact differently with the same latent geometry and would require their own optimizer-local pool definitions.
```

## G7. Table 2 Note and Cube Rows

**Location:** `paper/sections/decoupling.tex`, Table `\ref{tab:cem-gap-protocols}` and its surrounding text.

**Addresses:** W3.

### If Phase E Completes

**Current Cube full projected-CEM rows:**

```latex
Cube & Full projected CEM & 8 & .477 & .030 & .447 & 64.0 \\
Cube & Full projected CEM & 64 & .575 & .006 & .569 & 72.0 \\
Cube & Full projected CEM & 192 & .603 & .008 & .595 & 60.0 \\
```

**Replacement rows after `results/revision/cube_full_proj_cem_extended.json` completes:**

```latex
Cube & Full projected CEM & 8 & <Rendpoint_m8> & <Rpool_m8> & <DeltaCEM_m8> & <Success_m8> \\
Cube & Full projected CEM & 64 & <Rendpoint_m64> & <Rpool_m64> & <DeltaCEM_m64> & <Success_m64> \\
Cube & Full projected CEM & 192 & <Rendpoint_m192> & <Rpool_m192> & <DeltaCEM_m192> & <Success_m192> \\
```

Use the 50-pair, three-seed aggregate means from:

```text
results/revision/cube_full_proj_cem_extended.json
```

Recommended caption suffix:

```latex
Cube full projected-CEM rows use the 50-pair, three-seed extended run; Cube re-rank-only rows use the 100-pair, three-seed fixed-pool evaluation.
```

Recommended surrounding sentence:

```latex
The extended Cube full projected-CEM run preserves the qualitative conclusion of the original 25-pair smoke: endpoint ranking remains substantially larger than final-pool ranking under matched projected-CEM planning, so the Cube evidence supports the endpoint-pool diagnostic while remaining smaller than the PushT evaluation.
```

### If Phase E Is Incomplete

**Footnote for the existing Cube full projected-CEM rows:**

```latex
\footnotetext{Cube full projected-CEM rows are from the original 25-pair, single-seed run. A larger 50-pair, three-seed extension was launched for the revision but was not complete at submission time. We therefore use Cube as supporting cross-environment evidence and avoid claims that depend on the single-seed inverted-U pattern.}
```

**Replacement sentence after Table 2 discussion:**

```latex
Because the Cube full projected-CEM cell is smaller than the PushT full projected-CEM cell, we interpret Cube conservatively: it tests whether the endpoint-pool gap appears outside PushT, but we do not treat the Cube inverted-U success pattern as a primary claim unless the extended run confirms it.
```

## Suggested Reviewer-Response Framing

```latex
We revised the manuscript to narrow the scope of the claims, added a privileged-ranker mechanism attribution experiment, added a privilege-free proxy analysis, and added a cross-optimizer MPPI comparison. The new results distinguish three components of the phenomenon. First, V1 oracle ranking remains strong inside the same CEM pools where the learned Euclidean latent cost collapses, indicating local representation failure rather than mere CEM convergence compression. Second, MPPI soft weighting preserves more physical pool diversity and improves $\Rpool(\Cmodel)$, showing that CEM hard truncation exacerbates but does not create the entire endpoint-pool gap. Third, learned-cost spread provides a statistically significant but partial monitoring signal without simulator access. We also revised the abstract, introduction, and limitations to state explicitly that the empirical evidence is for LeWM on PushT and OGBench-Cube, with CEM as the primary optimizer.
```
