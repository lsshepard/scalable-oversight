<!-- Liam's summary of his earlier ideation with Codex, as pasted into the planning conversation on
2026-11-16. Transcribed by Claude; Liam's own copy is authoritative if anything differs. Liam's framing:
"dont take any of this for granted since its just discussion". -->

# Context for continuing Liam Shepard's CS2823 project

This summarizes the development of the project, the current mathematical setup, results suggested or derived during the conversation, unresolved questions, and important corrections. Treat proposed results as conditional arguments to verify, rather than established theorems.

## 1. Project and current objective

Liam Shepard is developing a project on mechanism design for scalable AI oversight for Harvard CS2823R. His supervisor is **Lily Lavitas**.

The original goal was to design a reward mechanism in which **honest supervision is the unique Nash equilibrium**, then test whether training supervisors with that reward encourages honest reporting and disrupts collusion.

The central idea is to give different supervisors secretly modified versions of the same complex output. The mechanism knows the modifications. It rewards agreement after adjusting for those changes, together with accurate reporting of planted behavior.

The conversation has moved from writing the proposal to developing the mathematical model. **The immediate task is to concretize and investigate the private-mixture mechanism described below**, preserving the spirit of scalable oversight.

Liam's stated priorities for choosing assumptions and models are:

1. Realism.
2. Tractability.
3. Meaningful results.

He wants genuine deliberation, alternative approaches, and counterexamples. He does not want unnecessary formal detail in the proposal, but is now interested in substantive mathematical analysis.

## 2. Motivation and maintained assumptions

The motivating problem is that a designer cannot fully understand or verify a complicated model output or reasoning trace. Capable supervisors can interpret it, but might report dishonestly or coordinate on the same false interpretation.

Agreement alone does not establish honesty: supervisors could share an incorrect or dishonest transformation.

The initial assumptions are:

- Supervisors are sufficiently capable to interpret the relevant behavior.
- They seek to maximize the mechanism's reward.
- They cannot identify the planted perturbations.
- They initially behave as reporting functions of their observed inputs.

Liam explicitly pushed back when the assistant repeatedly raised audit recognition as an objection: **undetectability is an assumption of the baseline model**. It should be accepted when analyzing that baseline. Later, Liam requested variations of assumptions to improve realism, so weaker hiding assumptions can be explored as clearly labeled alternatives.

Correlated accidental errors are **not a main object of study**. Liam wants to focus on incentives and deliberate misreporting among capable models.

## 3. The actual private-mixture setup

Let:

- \(U\) be a space of complex representations, such as embeddings of logs or reasoning traces.
- \(V\) be a simpler, interpretable semantic/report space.
- \(u\in U\) be an original complex log whose truthful report the designer cannot verify.
- \(T:U\to V\) be the ideal truthful translation.
- \(F_i:U\to V\) be supervisor \(i\)'s actual reporting strategy.
- \(o_i\in U\) be a reference/perturbation chosen by the mechanism.
- \(w_i=T(o_i)\in V\) be its independently known, trusted translation.
- \(a_i\in(0,1)\) be a mixing weight.

Supervisor \(i\) receives only
\[
z_i=a_i u+(1-a_i)o_i.
\]

**Crucial clarification from Liam:** the supervisor is not given \(o_i\) or \(w_i\) separately. Those are known to the mechanism. The supervisor sees the mixture.

The mechanism computes
\[
\widehat v_i=
\frac{F_i(z_i)-(1-a_i)w_i}{a_i}
\]
and rewards agreement between recovered estimates from different supervisors and different references.

Liam originally wrote a correction involving \(F(w)\). That was a notation/type error if \(F:U\to V\), because \(w\in V\). The intended correction uses the known \(w\), **not the supervisor's own \(F_i(o_i)\)**.

That distinction is fundamental.

### Mixing-weight privacy

The conversation proposed independently varying \(a_i\), while keeping it private. Liam explicitly clarified privacy of \(o_i,w_i\), but **privacy of \(a_i\) was an assistant-proposed design variation, not an independently confirmed original requirement**.

A supervisor's ability to observe or infer \(a_i\) matters. A reporting strategy that can condition on \(a_i\) can potentially adapt its distortions to the demixing operation.

## 4. Occasional calibration checks

Liam proposed sometimes directly checking the known translations.

The correctly typed check is
\[
F_i(o)=w,
\]
not \(F_i(w)=o\).

One implementation is to sometimes set \(a=0\). On those rounds:

- The supervisor receives the reference input \(o\).
- It is not told that this is a calibration round.
- The mechanism scores its report directly against \(w\).
- The demixing formula is not used, because it would divide by zero.

This changes the earlier "never sees \(o\) by itself" description: ordinary comparison rounds contain mixtures; occasional hidden calibration rounds contain a reference alone.

Calibration addresses shared offsets and helps identify the intended translation, but its implications depend on reference coverage and the strategy class.

## 5. Why approximate linearity was proposed

Liam's reasoning:

- The complex and simple representations are both semantic spaces.
- They may encode similar semantic factors or dimensions.
- Therefore a linear transformation between them might be a reasonable approximation.
- The approximation could have an error term.
- One could bound how that error propagates through the reward mechanism.
- Vector-space modeling might also allow direct empirical experiments in embedding spaces.

The assistant agreed that this is a plausible empirical hypothesis, but distinguished it from a logical consequence:

- Sharing semantics does not by itself imply linearly aligned coordinates.
- If both spaces encode common semantic factors approximately linearly, approximate linear translation is plausible.
- Arbitrary nonlinear encodings of the same semantics need not be linearly related.
- Arbitrary embedding interpolation need not correspond to a meaningful intermediate log.

The assumption most directly relevant to the proposed reward is **mixture preservation**, rather than a global linear approximation:
\[
T(au+(1-a)o)
=
aT(u)+(1-a)T(o)+d(u,o,a).
\]

Exact preservation gives an affine translation on an appropriate domain under standard regularity conditions. Approximate preservation is the current relaxation of interest.

Relevant literature discussed:

- Park, Choe, and Veitch, *The Linear Representation Hypothesis and the Geometry of Large Language Models*:
  https://proceedings.mlr.press/v235/park24c.html
- *mixup: Beyond Empirical Risk Minimization*:
  https://arxiv.org/abs/1710.09412
- *Manifold Mixup*:
  https://proceedings.mlr.press/v97/verma19a.html

These support investigating linear semantic structure/interpolation, not assuming the entire proposed translation law is already established.

## 6. Important counterexamples and corrections

### A. Subtracting the supervisor's own reference report would fail

For the alternative mechanism
\[
\frac{F_i(au+(1-a)o)-(1-a)F_i(o)}a,
\]
every affine reporting function recovers \(F_i(u)\), however dishonest.

Supervisors using the same false affine transformation would agree perfectly.

**This objection does not apply directly to Liam's intended mechanism**, because the mechanism subtracts independently known \(w=T(o)\).

### B. Fixed mixing weights permit shared constant bias

If \(T\) is affine and all supervisors use
\[
F_i(x)=T(x)+c,
\]
then, for fixed common \(a\),
\[
\widehat v_i=T(u)+c/a.
\]
They agree perfectly on a biased answer.

Calibration checks expose this bias:
\[
F_i(o)-w=c.
\]

Independent private weights also make the same offset appear as \(c/a_i\), creating disagreement. That is a candidate additional constraint, not a general proof by itself.

### C. References covering only a subspace leave other distortions unconstrained

Example:
\[
T(x,y)=x+y,
\qquad o=(t,0),\quad w=t.
\]
The dishonest transformation
\[
F(x,y)=x-y
\]
matches every reference and yields
\[
\widehat v=u_1-u_2
\]
for every reference and mixing weight.

For \(u=(2,3)\), supervisors consistently recover \(-1\), while truth is \(5\).

This illustrates a coverage problem: references must constrain the relevant semantic directions.

### D. Even spanning reference points need not suffice for nonlinear strategies

A sharper example:
\[
T(x)=x,\quad o\in\{0,1\},\quad a=\tfrac12,
\]
and
\[
F(x)=x+c\sin(4\pi x).
\]

Then \(F(0)=0\) and \(F(1)=1\), so both reference checks pass. Either reference produces
\[
\widehat v=u+2c\sin(2\pi u).
\]

Thus calibration and perfect agreement can coexist with nonlinear misreporting.

**Scope qualification:** this is a limited-reference-coverage variation. Calibration inputs concentrated at endpoints and ordinary mixed inputs are not statistically indistinguishable. Do not present this counterexample as satisfying a strong full-distribution hiding assumption.

### E. Correlation and numerical agreement are different

Liam initially said to reward "correlated" recovered vectors.

The assistant recommended beginning with squared numerical disagreement because literal Pearson correlation ignores offsets and positive rescaling and does not produce the convex quadratic argument below.

**Liam has not explicitly confirmed that squared distance should replace correlation.** Results using squared loss must be labeled as results for that variation.

### F. Identifying truth is different from implementing it as an equilibrium

A binary example showed that a uniquely correct assignment can coexist with a dishonest Nash equilibrium when agreement outweighs the gain from correcting a discrete report.

However, the assistant later corrected an overgeneralization: in continuous vector reporting with quadratic losses, a small step toward truth may improve calibration to first order while incurring disagreement only to second order. Thus arbitrarily weak positive calibration can sometimes eliminate a shared bias.

Discrete-game thresholds should not be imported uncritically into the continuous quadratic model.

## 7. Conditional unique-Nash argument

A proposed squared-loss game:
\[
L_i=
\lambda\,\mathbb E_{\mathrm{checks}}
\|F_i(o)-T(o)\|^2
+
\sum_{j\ne i}\beta_{ij}
\mathbb E_{\mathrm{comparisons}}
\|\widehat v_i-\widehat v_j\|^2,
\]
with symmetric nonnegative comparison weights.

A common potential is
\[
\Phi=
\lambda\sum_i\mathbb E\|F_i(o)-T(o)\|^2
+
\sum_{i<j}\beta_{ij}
\mathbb E\|\widehat v_i-\widehat v_j\|^2.
\]

A unilateral reporting change affects the player's loss exactly as it affects this potential.

Under an appropriate convex strategy class with continuous vector reports, this is a differentiable convex quadratic in reporting functions. If:

- truthful \(T\) belongs to the strategy class;
- \(T\) preserves the mixtures exactly;
- reference/comparison coverage makes truthful reporting the only zero-loss profile;
- the relevant expectations/function-space optimization are well defined;

then truthful reporting is the unique equilibrium outcome, up to irrelevant zero-probability inputs.

**Important limitations:**

- Coverage/identification is a substantive assumption to prove, not a conclusion of convexity.
- A finite collection of spanning references identifies an affine function, but not arbitrary nonlinear functions.
- Convexity in reporting functions does not imply convexity of neural-network training.
- A function-space Nash theorem does not prove RL convergence.
- Repeated interaction, communication, and history-dependent strategies have not been fully analyzed.
- "Unique Nash equilibrium" alone is not automatically a complete coalition-proofness guarantee.

## 8. Approximation error and why the equilibrium can be biased

Assume
\[
T(au+(1-a)o)
=
aT(u)+(1-a)T(o)+d(u,o,a),
\qquad \|d\|\leq\varepsilon.
\]

A truthful supervisor produces
\[
\widehat v=T(u)+d/a.
\]

Different references can create different residuals, so honest supervisors can receive a disagreement penalty. A slightly distorted reporting function might reduce that penalty while sacrificing a little calibration accuracy.

A schematic illustration:
\[
L(b)=\lambda b^2+(\varepsilon-b)^2.
\]
The optimum is
\[
b^*=\frac{\varepsilon}{1+\lambda},
\]
which is unique but not exactly truthful when \(\varepsilon\ne0\).

This example illustrates the tradeoff; it is not the full mechanism.

### Conditional quantitative bound already derived

For two supervisors, let:

- \(\lambda>0\) be effective calibration strength, including check frequency;
- \(\beta\) be effective agreement strength;
- \(a_i\ge a_{\min}>0\);
- calibration translations be correct;
- calibration coverage control errors on comparison inputs;
- the feasible reporting class be convex and contain \(T\).

Then the proposed bound is
\[
\left(
\sum_{i=1}^2
\mathbb E\|F_i^*-T\|^2
\right)^{1/2}
\leq
\frac{\varepsilon}{a_{\min}}
\sqrt{\frac{\beta}{\lambda}}.
\]

The norm initially concerns the calibration distribution. Transfer to comparison or original inputs requires the appropriate coverage/distribution relation.

Derivation sketch: write \(e_i=F_i^*-T\), let \(r\) be truthful demixing disagreement, and let \(Be\) be the disagreement induced by strategic distortions. Then
\[
\Phi=\lambda\mathcal E^2+\beta\|r+Be\|_{L^2}^2,
\qquad
\|r\|_{L^2}\le 2\varepsilon/a_{\min}.
\]
First-order optimality along a shrinkage toward truth gives
\[
\lambda\mathcal E^2
\leq
\beta(\|r\|y-y^2)
\leq
\beta\|r\|^2/4.
\]

A conservative bound for each final recovered estimate is
\[
\left(\mathbb E\|\widehat v_i^*-T(u)\|^2\right)^{1/2}
\leq
\frac{\varepsilon}{a_{\min}}
+
\frac{\varepsilon}{a_{\min}^2}
\sqrt{\frac{\beta}{\lambda}}.
\]

These are population RMS bounds, not uniform pointwise guarantees. Finite-sample versions and unrestricted nonlinear strategy coverage remain unresolved.

## 9. More realistic approximation through local curvature

The last exploration improved the uniform-error interpretation.

If \(T\)'s Jacobian is \(M\)-Lipschitz along the relevant interpolation paths, then
\[
\|T(au+(1-a)o)-aT(u)-(1-a)T(o)\|
\leq
\frac M2a(1-a)\|u-o\|^2.
\]

After demixing:
\[
\boxed{
\|\widehat v-T(u)\|
\leq
\frac M2(1-a)\|u-o\|^2.
}
\]

Thus the earlier \(\varepsilon/a\) bound is conservative when the error specifically comes from curvature: the defect itself scales with \(a\).

This suggests:

- use local rather than globally linear approximations;
- investigate semantically nearby trusted references;
- treat curvature error, reporting noise, and reference-label error separately.

Reporting noise still gets amplified by \(1/a\). Reference-label error contributes a factor involving \((1-a)/a\).

The difficult practical question is whether trusted, understandable references exist sufficiently close to genuinely difficult logs.

## 10. Reward variations proposed for numerical stability

Instead of dividing inside the reward, define
\[
q_i=F_i(z_i)-(1-a_i)w_i.
\]

Under exact truthful reporting, \(q_i=a_iT(u)\). Penalize
\[
\|a_jq_i-a_iq_j\|^2.
\]

This checks the same equality as normalized agreement for positive weights, but changes effective comparison weighting and avoids division-driven gradient amplification.

With multiple supervisors, another candidate is
\[
\min_v\sum_i\|q_i-a_i v\|^2,
\]
giving
\[
\widehat v=\frac{\sum_i a_iq_i}{\sum_i a_i^2}.
\]

These are suggestions to analyze, not finalized mechanism choices.

## 11. Quantitative identification as a better research target

The assistant suggested replacing the vague phrase "sufficient perturbations" with a condition such as
\[
\mathcal D(e)\geq
\kappa\,
\mathbb E_{\mathrm{relevant\ inputs}}
\sum_i\|e_i\|^2,
\]
where \(\mathcal D(e)\) is the ideal penalty induced by calibration and mixture comparisons.

Interpretation:

- \(\kappa=0\): some false reporting directions are invisible.
- Small \(\kappa>0\): uniqueness may hold but robustness and learning can be weak.
- A useful design should keep \(\kappa\) appreciable as the reporting class becomes richer.

In a finite linear parameterization, this is a smallest-eigenvalue/singular-value question.

A suggested general bound, conditional on this inequality and the convex quadratic formulation, is
\[
\text{reporting error}
\leq
\frac{R+\sqrt{\zeta}}{\sqrt{\kappa}},
\]
where:

- \(R\) is the weighted residual produced by truthful reporting due to imperfect mixtures/reference labels;
- \(\zeta\) is the excess expected potential loss above its optimum.

The hard research task is establishing \(\kappa\) from realistic reference geometry and private mixing. The inequality must not simply be assumed without investigating its content.

## 12. Finite computational experiments performed

Small examples were calculated locally. They are preliminary mathematical experiments, not LLM experiments or general proofs.

The most relevant experiment:

- \(U=[0,1]\).
- Truth \(T(x)=x\).
- Two independent reporting functions, each piecewise linear on a grid.
- Known references \(o=0,1\), with \(w=o\).
- Original inputs sampled uniformly over a finer finite grid.
- Independent reference/weight choices for both supervisors.
- Quadratic potential with calibration coefficient \(0.1\), comparison coefficient \(0.9\).
- All combinations in the finite design were included.
- The matrix was analyzed relative to a normalized report-error norm.

Results for 81 reporting values per supervisor:

| Weight choices | Zero-loss error directions | Smallest normalized eigenvalue |
|---|---:|---:|
| Fixed \(a=0.5\) | 39 | 0 |
| \(a\in\{0.3,0.7\}\) | 0 | approximately 0.1648 |
| \(a\in\{0.8,0.9\}\) | 0 | approximately 0.0080 |

Across grid sizes 21, 41, 81, 161:

- Fixed \(0.5\): nullities 9, 19, 39, 79.
- Weights \(0.3,0.7\): no null directions; smallest eigenvalues approximately 0.1984, 0.1756, 0.1648, 0.1608.
- Weights \(0.8,0.9\): no null directions; smallest eigenvalues approximately 0.0162, 0.0114, 0.0080, 0.0060.

Interpretation: randomized weights appear to eliminate shared errors in this finite class, while near-one weights can give weaker quantitative control.

These tests use limited, recognizable reference coverage. They do not instantiate perfect statistical indistinguishability. They also do not prove behavior for arbitrary continuous functions, high-dimensional semantic spaces, or neural strategies.

Other numerical checks verified the periodic counterexample, the curvature formula for \(T(x)=x^2\), and a quadratic perturbation bound over random finite-dimensional instances.

## 13. Distributional hiding variations

Perfect hiding remains the maintained baseline assumption. For realistic variants, distinguish:

1. inability to identify the hidden reference component;
2. inability to distinguish a calibration round from a comparison round;
3. inability to infer the private mixing weight.

These are different requirements.

A derived observation:
If \(U,O\) are independent and identically distributed with finite covariance, convex mixing changes covariance:
\[
\operatorname{Cov}(aU+(1-a)O)
=
[a^2+(1-a)^2]\operatorname{Cov}(U).
\]

A suggested controlled alternative:
\[
z=au+bo,\qquad a^2+b^2=1.
\]
For independent centered Gaussian inputs this preserves the marginal distribution exactly. A linear truthful map allows
\[
T(u)=[T(z)-bT(o)]/a.
\]

This is a synthetic benchmark. It is not a demonstrated model of natural semantic embeddings. Affine offsets require appropriate centering/correction.

Relevant source:
https://arxiv.org/abs/1710.11381

## 14. Broader approaches explored earlier

Before the mixture mechanism was clarified, the assistant considered:

- hidden audits of scoped behavioral claims;
- known transformation relations with trusted anchors;
- one-sided logical interventions;
- distributional transfer guarantees;
- affine embedding models as benchmarks.

Useful retained observations:

- Planting a known violation can establish a positive label without linearity.
- Positive-only audits permit "always flag a violation."
- Removing one violation does not establish that a whole log is clean.
- Scoped questions make known positive and negative interventions more credible.
- Agreement can sometimes stabilize dishonesty rather than improve incentives.
- An audits-only baseline is essential.

For a binary auditing model, a proposed sufficient honesty condition was
\[
\alpha(v)\lambda[1-2\varepsilon(v)]
>
[1-\alpha(v)]\mu,
\]
where \(\alpha(v)\) is posterior audit probability given the full pre-report view.

**This was a different, discrete model. Its reward thresholds are not the main result for the current continuous mixture mechanism.**

A transfer bound was also discussed:
\[
\operatorname{Err}_P
\leq
\operatorname{Err}_{Q,\text{audit labels}}
+\varepsilon+\operatorname{TV}(P,Q),
\]
for a fixed reporting policy and a shared semantic labeling rule. It is conditional on genuine distributional control; failure of one detector does not establish small total variation.

## 15. Related literature worth investigating

The closest newly identified connection is to **program self-testing/self-correction and random self-reducibility**:

- Blum, Luby, and Rubinfeld, *Self-Testing/Correcting with Applications to Numerical Problems*:
  https://people.csail.mit.edu/ronitt/papers/selftestcorrect.pdf
- Gemmell, Lipton, Rubinfeld, Sudan, and Wigderson, *Self-Testing/Correcting for Polynomials and for Approximate Functions*:
  https://www.math.ias.edu/~avi/PUBLICATIONS/MYPAPERS/GLRSW91/madhu.pdf
- Hyers, *On the Stability of the Linear Functional Equation*:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC1078310/

These may supply tools for proving approximate consistency implies proximity to a structured function. They do not automatically establish correctness relative to the intended semantic translation, or strategic equilibrium guarantees.

Other sources discussed:

- Gao, Wright, and Leyton-Brown, *Incentivizing Evaluation via Limited Access to Ground Truth: Peer-Prediction Makes Things Worse*:
  https://arxiv.org/abs/1606.07042
- Ben-David et al., *A Theory of Learning from Different Domains*:
  https://escholarship.org/uc/item/2nv1j9sc
- *MixUp as Locally Linear Out-Of-Manifold Regularization*:
  https://arxiv.org/abs/1809.02499

Do not claim novelty for the algebraic verification ideas without comparing this literature.

## 16. Empirical plans and uncertainties

The empirical design remains contingent on the theory.

Candidate material:

- chain-of-thought traces from varied monitoring tasks;
- reported traces/examples from the Hugging Face incident;
- existing monitoring-collusion and peer-verification experiments.

Crucial distinction agreed during the conversation: **a trace containing misconduct does not itself motivate a supervisor to conceal it.** The experiment must establish an informative dishonest starting policy.

Possible elicitation methods:

- collusive instructions;
- prefilled beginnings of supervisors' own collusive reasoning;
- existing prompt-based collusion setups;
- additional fine-tuning if prompting is insufficient.

Prompting should produce actual false reports, not merely collusive-sounding reasoning. Supervisors must remain free to change strategies in response to rewards. A permanent instruction to lie can introduce a competing objective.

Keep prompting conditions fixed across reward comparisons. Success against prompted collusion would establish a controlled result, not automatically generalize to independently acquired deceptive incentives.

Proposed comparisons:

- calibration/planted-behavior reward alone;
- agreement alone;
- combined reward;
- fixed versus privately randomized mixing weights.

Main empirical outcomes:

- faithful reporting of original behavior;
- recovery from dishonest initial policies;
- learning to identify planted content instead of becoming honest;
- sensitivity to mixture validity, reference coverage, and reward weights.

Use independent training seeds, held-out examples, and confidence intervals capturing variation between runs and evaluation examples.

Relevant resources:

- Järviniemi, *Subversion via Focal Points*:
  https://arxiv.org/abs/2507.03010
  https://github.com/ollijarviniemi/subversion_via_focal_points
- Shi, Zhang, and Yang, *Emergent Collusion in Long-Horizon LLM Agent Interaction*:
  https://arxiv.org/abs/2609.24967
  https://huggingface.co/datasets/SALT-NLP/agent-collusion
- *Model Organisms of Collusion*:
  https://arxiv.org/abs/2610.30582
- Monitor Red Teaming:
  https://huggingface.co/datasets/ScaleAI/mrt
- OpenAI monitorability evaluation release:
  https://alignment.openai.com/monitorability-evals

These were identified through prior browsing. Suitability and current artifact availability should be checked before committing to them.

## 17. Original proposal, supervisor feedback, and writing decisions

The original proposal cited:

1. Bergemann, Koh, and Morris, *Mechanism Design for Alignment and Control*:
   https://arxiv.org/abs/2609.01595
2. Irving, Christiano, and Amodei, *AI Safety via Debate*:
   https://arxiv.org/abs/1805.00899
3. Qiu, Carroll, and Allen, *Truthfulness Despite Weak Supervision*:
   https://arxiv.org/abs/2601.20299

Liam's criticism of Bergemann et al.:

- Stylized applications express capability and alignment in a shared, low-dimensional action space.
- The oversight example relies on a monitor caring about particular future outcomes it can influence.
- Those preferences and representations may be difficult to operationalize.
- Liam's mechanism directly rewards reports about already-produced outputs.

The criticism was explicitly scoped to the stylized applications; the paper's general framework is broader.

Qiu et al.'s theoretical guarantees were described as addressing unilateral deception and leaving collusion open. Their training experiment uses DPO. These distinctions were used to sharpen the proposal's conceptual comparison.

Lily's feedback emphasized:

- specify what is supervised;
- specify models, perturbations, and dataset size;
- stage experiments;
- use multiple seeds and confidence intervals;
- consider cross-validation.

Liam replied that he intended to work out the theoretical model first and use it to guide the empirical design, asking whether empirical details could remain open. **No affirmative answer from Lily was established in this chat.**

A meeting had been booked for November 6 at 12:30 p.m. Its outcome was not discussed.

Proposal assignment:

- roughly two pages;
- NeurIPS 2025 LaTeX template;
- four sections: background/motivation, related work, problem statement/evaluation, challenges/risks;
- final submission deadline was updated to October 27.

Writing preferences and corrections:

- Research motivation, not personal thesis motivation.
- Evaluation plans belong in the problem statement.
- Keep the proposal conceptual and concise.
- Code-based perturbations concern empirical implementation; theoretical modeling comes first.
- "Deterministic" meant predefined code transformations rather than an adversarial generator.
- A third adversarial generator might conceal artifacts better but adds another collusion opportunity.
- Avoid formulaic prose, invented compounds, excessive repetition, em dashes, and "not X, but Y" framing.
- Do not repeatedly reopen assumptions already deliberately adopted.

## 18. Files and document state

Project directory:
`/Users/liamshepard/.codex/.chatgpt-projects/g-p-6ac50cdaa80881919317a167be733d28`

Files under `sources/` are read-only synced reference material.

The proposal source was created at:
`final-project-proposal/main.tex`

It used the official `neurips_2025.sty`, with preprint formatting and six references.

An exported PDF was inspected:
`/Users/liamshepard/Downloads/CS2823_Project_proposal.pdf`

On October 27 it was verified to have:

- exactly two US Letter pages;
- all four required sections;
- six resolved references and embedded links;
- correct author, email, and supervisor;
- no clipped or overlapping text;
- embedded fonts;
- text matching the then-current draft.

**Important later state:** on November 13, an absolute-path inspection found that `main.tex` contained only the six-character string `double`. The assistant did not restore or edit it. Do not assume that the original proposal remains in that source file. The exported PDF may be the best preserved copy.

## 19. Best next work

Continue from the specific private-mixture mechanism, rather than replacing it with generic auditing without explaining the tradeoff.

The most promising next steps suggested were:

1. State precisely what supervisors observe, especially whether mixing weights are private and whether strategies can use history.
2. Fix the reward: literal correlation versus calibrated squared agreement.
3. Analyze exact identification with sparse trusted references and privately randomized weights while permitting nonlinear strategies.
4. Prove quantitative control, not just uniqueness.
5. Replace a global approximation term with local curvature where credible.
6. Separate semantic approximation, reference error, reporting noise, finite sampling, and optimization error.
7. Investigate classical self-testing/correction results before claiming novelty.
8. Empirically test the mixture-preservation assumption and reference geometry before RL.

The strongest unresolved research question is:

**Can a realistic collection of trusted semantic references and private mixtures constrain every materially dishonest reporting function strongly enough to yield robust, approximately truthful equilibria—and does mixing add useful supervision beyond calibration alone?**
