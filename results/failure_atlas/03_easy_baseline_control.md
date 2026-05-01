# Easy Baseline Control

## 1. Failure Type Name

Easy baseline control.

This is the control regime where initial and goal block poses are nearly identical or require only very small corrections. These pairs confirm that the evaluation pipeline, simulator execution, CEM candidate generation, and success metric behave as expected when the representation is not stressed by large geometric change.

## 2. Representative Trajectories

| Pair | Category | Successes | Block displacement | Rotation required | Encoder corr |
|---:|---|---:|---:|---:|---:|
| 25 | Easy | 37/40 | 0.0 | 0.000 | 0.104 |
| 26 | Easy | 37/40 | 0.5 | 0.192 | 0.151 |
| 28 | Easy | 33/40 | 0.0 | 0.000 | 0.393 |
| 29 | Easy | 31/40 | 0.0 | 0.000 | 0.694 |

The low encoder correlations on some easy pairs are not evidence of failure by themselves because many actions have tied or near-tied real-state costs around zero. In this regime, success is high and the model objective reliably produces acceptable behavior.

## 3. Blow-Up Step With Physical Event Annotation

| Pair | Blow-up step t* | Mean predictor error | Max predictor error | Annotation |
|---:|---:|---:|---:|---|
| 25 | 4 | 3.446 | 3.786 | Small prediction error under a near-stationary goal; CEM_late succeeds. |
| 26 | 8 | 9.188 | 11.111 | Ratio is inflated because real latent motion is near zero; despite the numerical ratio, the selected CEM_late action succeeds physically. |

## 4. Three-Cost Ranking

For pair `25`, the planner-best `CEM_late` action has `C_model = 1.991`, `C_real_z = 10.302`, and `C_real_state = 0.328`, and it succeeds. Many dataset actions also have `C_real_state = 0.0`, so rank agreement is artificially low due to ties.

For pair `29`, the planner-best `CEM_late` action has `C_model = 1.613`, `C_real_z = 7.424`, and `C_real_state = 1.136`, and it succeeds. The real-state-best actions are tied at zero physical cost, but all are successful.

The control confirms that the pipeline recognizes easy goals and that high success is achievable when physical displacement and rotation are small.

## 5. Decision-Tree Assignment

**Control, not a failure case.** These pairs anchor the Case B/E interpretation by showing that LeWM works when encoder geometry is not stressed by large displacement or rotation.
