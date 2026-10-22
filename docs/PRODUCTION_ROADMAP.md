# ShopSense production engineering roadmap

## Purpose and current baseline

This roadmap describes an evidence-gated path from the current ShopSense analytical prototype to a reliable, tested, deployable machine-learning application. A phase is complete only when its stated acceptance criteria are verified; local tests alone do not establish production readiness.

Repository evidence at the time this roadmap was written:

- Transaction loading, quality auditing, cleaning, and artifact export are implemented in `src/preprocessing.py`.
- Customer features, repeat-purchase labels, split generation, and metadata export are implemented in `src/features.py`.
- EDA is documented in `notebooks/01_eda.ipynb`; synthetic-data tests are in `tests/`.
- The Python 3.12 test workflow is `.github/workflows/tests.yml`.
- There is no model-training/evaluation module, segmentation implementation, application/UI, deployment configuration, production telemetry, or software `LICENSE` file in the repository.
- The configured observation and outcome periods, safe feature schema, and split policy are documented in `README.md` and generated modeling metadata. Confirm these contracts against the current code and metadata before consuming newly generated artifacts.

**Status key:** `completed` means the phase's objective has objective repository or run evidence; `in progress` means implementation exists but its acceptance criteria are not all met; `blocked` means a prerequisite prevents meaningful progress; `planned` means implementation has not started. Statuses describe repository evidence, not team assignment or approval.

## Phase A — Repository and CI reliability

**Objective and rationale:** Ensure contributors and reviewers receive dependable, reproducible checks on the actual pull-request revision. A workflow file and a local test pass are not proof that GitHub accepted and ran the workflow.

**Required implementation work:**
- Validate GitHub Actions syntax, workflow triggers, permissions, and job configuration with an available workflow validator.
- Preserve pull-request and push triggers; install dependencies and run the full suite on the supported Python version.
- Confirm the workflow check appears and succeeds on a real pull request.
- Define and document formatting, linting, type-checking, and test policies appropriate to this repository. Add a tool only with an explicit dependency/configuration change and CI integration.
- Keep CI independent of the private/local raw workbook and generated artifacts.

**Relevant files:** `.github/workflows/tests.yml`, `requirements.txt`, `pyproject.toml`, `tests/`, `README.md`.

**Acceptance criteria:**
- A workflow validator reports no syntax/configuration errors.
- A pull request receives the expected test status check, and the check passes for the reviewed commit.
- The CI job installs the declared dependencies and runs the same documented test command.
- CI succeeds without access to local datasets, notebooks' cached outputs, or developer-specific paths.
- The chosen formatting/lint/type-check policy is documented and enforced, or an explicit decision records that a check is not yet required.

**Dependencies/prerequisites:** Repository access to GitHub Actions; supported Python version and dependency manifests.

**Risks/failure scenarios:** Invalid YAML or GitHub expression semantics can prevent job scheduling; workflow permissions may be excessive; dependency resolution can drift; tests may accidentally depend on ignored local data.

**Tests and verification commands:**

```powershell
.\.venv\Scripts\python.exe -m pytest -v
git diff --check
actionlint .github\workflows\tests.yml
```

Run `actionlint` only where installed; do not treat an unavailable validator as a pass. Verify the final PR check in GitHub Actions. The workflow currently defines Python 3.12 installation and pytest on push and pull request. A successful GitHub run for the current reviewed revision remains a release gate, not inferable from local execution.

**Definition of done:** Workflow syntax is validated, CI policy is explicit, and the relevant pull-request check is observed passing.

**Status: In progress.** Workflow and test job exist in the repository; local tests are available. A GitHub pull-request run and a repository-level formatting/lint/type-check policy still require verification/decision.

## Phase B — Data pipeline reliability and governance

**Objective and rationale:** Make every transformation auditable, reproducible, and safe to rerun, with explicit contracts for raw inputs and generated outputs.

**Required implementation work:**
- Formalize raw workbook, Phase 2 cleaned transactions, positive-purchase rows, customer feature/label tables, and split schemas. Record required columns, types, nullability, keys, uniqueness, date semantics, and compatibility/version policy.
- Document exact handling of duplicate rows, missing/unparseable dates, cancellation invoice markers, negative/zero quantities and prices, returns, missing `CustomerID`, and missing `Description`. Preserve the distinction between transaction-level and customer-level eligibility.
- Add configurable data-quality thresholds or gates for material schema/quality drift, with actionable reporting instead of silent changes.
- Add stable regression fixtures or data fingerprints that do not require committing the full workbook or exposing private data.
- Make each multi-file artifact generation atomic: build and validate a complete run in a temporary/versioned location, then publish it as one consistent artifact set. Define behavior for interruption and disk-full failures.
- Record source-data fingerprint, resolved source identifier, relevant runtime and dependency versions, effective configuration, code revision where available, processing timestamp, and output checksums in audit metadata.
- Preserve `.gitignore` protections for `data/raw/` and `data/processed/`; verify data artifacts are neither staged nor tracked.
- Document UCI provenance, attribution, current source terms, and known source limitations. Do not infer a redistribution license without verification.

**Relevant files/modules:** `src/preprocessing.py`; proposed `src/data_contracts.py` or focused contract helpers; `data/raw/`, `data/processed/`; `tests/test_preprocessing.py`; proposed pipeline integration tests; `.gitignore`; `README.md`.

**Acceptance criteria:**
- Contract checks reject missing/duplicate required columns and invalid field types with actionable errors.
- Audit metadata identifies the exact input and effective cleaning configuration and is valid JSON.
- Quality thresholds have documented units/denominators and tests for both threshold pass and fail.
- An injected export failure leaves the previously published artifact set intact and does not advertise a partial run as complete.
- Repeated processing of the same source/configuration has identical data outputs; only explicitly defined run metadata may vary.
- Tests demonstrate that raw data remains unchanged and raw/generated files remain ignored and untracked.
- Dataset provenance and applicable use/redistribution terms are recorded with source links.

**Dependencies/prerequisites:** Stable Phase 2/3 data contracts; source terms review; CI from Phase A.

**Risks/failure scenarios:** Upstream workbook revisions or Excel parser changes can alter outputs; incomplete date coverage can create false negatives; invoice-number coercion can corrupt identifiers; partial writes can mix run versions; hashes prove identity, not completeness or permission to redistribute.

**Tests and verification commands:**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_preprocessing.py -v
.\.venv\Scripts\python.exe -m pytest tests\test_features.py -v
git check-ignore -v -- "data/raw/Online Retail.xlsx" data/processed/train.csv
git ls-files data/raw data/processed
```

Add temporary-directory failure-injection tests for multi-artifact publication. Never regenerate or overwrite production/valuable artifacts during a check without explicit approval.

**Definition of done:** Versioned contracts, reproducible provenance, tested quality gates, and atomic artifact publication are in place and exercised in CI.

**Status: In progress.** Current cleaning, audit, and artifact generation exist and local data paths are ignored. Atomic publication, source fingerprints/runtime lineage, threshold gates, and regression governance are not implemented.

## Phase C — Feature engineering and target validity

**Objective and rationale:** Ensure customer features and labels represent the stated prediction problem without customer leakage, future-derived features, or incomplete outcome observation.

**Required implementation work:**
- Verify configured observation/outcome dates against actual source coverage and explicitly define inclusive calendar boundaries, reference date, timezone behavior, and eligible cohort.
- Maintain a strict feature allowlist. Features must use observation purchases only; the target must use qualifying outcome invoices only.
- Check customer-level train/validation/test disjointness and that the split union contains each eligible customer exactly once.
- Cover date-boundary timestamps, missing and blank IDs, cancellation/return/unusual invoices, invalid/missing dates, empty cohorts, absent target classes, numerical finiteness, and deterministic outputs.
- Version the feature schema, target/window configuration, and generated metadata so downstream model artifacts can declare compatible inputs.
- Clarify that date min/max containment establishes observed bounds only; it cannot independently certify that the source feed is complete.

**Relevant files/modules:** `src/features.py`, `src/preprocessing.py`, `tests/test_features.py`, `data/processed/modeling_metadata.json`, `README.md`; proposed feature-schema version or contract module.

**Acceptance criteria:**
- Features are derived only from qualifying observation-window rows; labels count distinct qualifying outcome invoices.
- A test changing outcome rows leaves every observation-derived feature unchanged.
- CustomerID and target are not in the feature allowlist; raw invoice IDs/timestamps and outcome-derived fields are absent from model matrices.
- Output contains one row per eligible customer, non-null binary labels, and finite documented numeric feature values.
- Every eligible customer appears exactly once across disjoint splits with the same schema.
- Incomplete outcome coverage fails before labels are emitted.
- Metadata records windows, source coverage, feature definitions/version, target convention, eligibility, class counts, split policy, and warnings.

**Dependencies/prerequisites:** Phase B stable input contract; Phase A automated CI.

**Risks/failure scenarios:** Incorrect date endpoints or timezones shift purchases between windows; invoice-format changes duplicate invoice counts; too few examples in a class make three-way stratification invalid; repeat-purchase behavior is horizon-specific.

**Tests and verification commands:**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_features.py -v
.\.venv\Scripts\python.exe -m pytest -v
```

Add contract tests for feature schema version compatibility and a test that deliberately corrupts customer membership to prove leakage/integrity checks reject it.

**Definition of done:** Feature/label contracts, eligibility, window semantics, leakage tests, and split integrity are versioned and enforced in CI.

**Status: In progress.** RFM-like observation features, outcome labels, coverage checks, deterministic stratified customer splits, and synthetic tests exist. Feature schema versioning and production-grade input lineage/coverage evidence remain open.

## Phase D — Modeling and credible evaluation

**Objective and rationale:** Establish whether the target is predictable and whether any model is useful for a defined retail decision, without contaminating evaluation data.

**Required implementation work:**
- Define the operational decision, prediction timing, unit of action, and costs of false positives and false negatives with stakeholders before selecting a threshold.
- Establish a simple, transparent baseline before comparing more complex supervised classifiers.
- Compare planned candidate classifiers using identical splits, explicit safe features, and consistent preprocessing/evaluation.
- Put imputation, scaling, encoding, feature selection, and any resampling inside a training-fitted pipeline. Fit learned transformations on train only; use validation for selection.
- Use the held-out test set once for final evaluation after choices are fixed; document any repeated test access.
- Report precision, recall, F1, ROC-AUC when class support permits, confusion matrices, class balance, threshold trade-offs, and calibration where probabilities drive decisions.
- Add an out-of-time evaluation on a later calendar period if the source supports enough complete observation/outcome windows. It is distinct from a random customer split and must preserve the prediction chronology.
- Investigate leakage, imbalance, overfitting, calibration, subgroup/segment performance differences, stability, and uncertainty. Do not set universal score targets without business justification.
- Version training configuration, data/feature schema, model artifact, and evaluation report. Do not claim performance until reproducible run evidence exists.

**Relevant files/modules:** proposed `src/models/`, `src/evaluation.py`, `src/training.py`; `src/features.py`; `data/processed/train.csv`, `validation.csv`, `test.csv`; proposed `reports/model_evaluation/`; `tests/`.

**Acceptance criteria:**
- A deterministic baseline and candidate comparison can be rerun from documented data and configuration.
- Automated checks prove preprocessing is fit only on training data and test rows are never passed to model selection.
- Evaluation reports include class counts, metrics with definitions, confusion matrix, threshold policy, model/config/data identifiers, and limitations.
- Test set access and out-of-time evaluation are explicitly separated from model selection.
- Business owners document the relevant false-positive/false-negative costs and accept the decision thresholding approach.
- Performance is reported without unsupported claims; uncertainty and cohort limitations are stated.

**Dependencies/prerequisites:** Phases A-C complete; business decision and error-cost input; a verified feature contract.

**Risks/failure scenarios:** Leakage can inflate scores; the fixed outcome horizon can produce temporal drift; class/segment imbalance can obscure harm; repeated test tuning invalidates the test estimate; source dates may not support a robust temporal holdout.

**Tests and verification commands:** Proposed tests should be runnable with:

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

Document exact training/evaluation commands only after the training interface exists. Review metrics and artifacts from a real run; no scores are currently available.

**Definition of done:** Baseline, candidate comparison, leakage controls, final holdout protocol, and business-reviewed evaluation evidence are reproducible and accepted.

**Status: Planned.** No model-training or model-evaluation implementation or results are present.

## Phase E — Customer segmentation

**Objective and rationale:** Explore customer grouping as an independent descriptive/operational task, not as a substitute for repeat-purchase prediction.

**Required implementation work:**
- Define a concrete business purpose and action that segments are intended to support.
- Choose customer features appropriate to the segmentation period and question; document missingness and feature scaling.
- Evaluate candidate clustering approaches and cluster counts using appropriate internal diagnostics plus stability and interpretability checks.
- Profile segments in understandable business terms and assess whether profiles are reproducible and useful to retail decisions.
- Keep segmentation feature and artifact contracts separate from supervised targets unless a documented integration requires otherwise.
- Record fit data, preprocessing, algorithm configuration, random seed, schema version, and segment profiles.

**Relevant files/modules:** proposed `src/segmentation/`, `tests/test_segmentation.py`, generated non-versioned segment artifacts, `README.md`, and later application integration.

**Acceptance criteria:**
- Segment creation is deterministic for the same inputs/configuration.
- Stability is quantified across resamples or time slices; segment profiles and population coverage are reported.
- Feature scaling is fit only on the segmentation reference data and applied consistently.
- Profiles are reviewed against a stated retail use case; no segment is called commercially meaningful solely because a clustering algorithm returned it.
- Missing IDs and outliers have explicit treatment and audit counts.

**Dependencies/prerequisites:** Phase B data contracts; a documented business use case; Phase C customer entity contract.

**Risks/failure scenarios:** Clusters may be unstable, dominated by scale/outliers, or not actionable; segmentation can be confused with prediction; drift can change segment assignments over time.

**Tests and verification commands:** Proposed segmentation tests run through `.\.venv\Scripts\python.exe -m pytest -v`; add stability, schema, determinism, and profiling checks before publishing segment outputs.

**Definition of done:** Reproducible segments have stability evidence, interpretable profiles, documented limits, and stakeholder-confirmed use.

**Status: Planned.** No clustering implementation or segment evaluation is present.

## Phase F — Production application and integration

**Objective and rationale:** Connect verified data and model contracts through an application boundary while keeping processing, inference, segmentation, and presentation independently testable.

**Required implementation work:**
- Decide the application architecture and deployment target only after reviewing actual use cases, dependencies, data sensitivity, and operating constraints.
- Define module/service boundaries for preprocessing, feature generation, model training, inference, segmentation, and UI.
- Integrate only versioned, verified model artifacts and compatible feature/data contracts.
- Add configuration management, input validation, clear user-facing errors, and safe behavior for missing/invalid data or unavailable artifacts.
- Add unit, integration, and end-to-end smoke tests for valid and invalid user flows.
- Avoid embedding private paths, credentials, model artifacts, or data in code or UI configuration.

**Relevant files/modules:** proposed `app/` or `src/application/`, `src/inference.py`, `src/segmentation/`, contract modules, application tests, dependency manifest, and deployment configuration.

**Acceptance criteria:**
- Application start-up validates configuration and artifact compatibility and fails with actionable errors.
- Valid requests produce outputs tied to explicit model/data/schema versions.
- Invalid inputs, missing artifacts, and downstream failures return bounded, useful errors without exposing sensitive implementation details.
- Integration/smoke tests cover the declared deployment interface and do not require production credentials.
- Application boundaries and intended users/permissions are documented.

**Dependencies/prerequisites:** Phases B-D; Phase E if segmentation is exposed; approved user requirements and deployment constraints.

**Risks/failure scenarios:** UI and model schema drift; unsafe input rendering or file handling; silent fallback to stale artifacts; uncontrolled data exposure; architecture choice that cannot meet operating constraints.

**Tests and verification commands:** Add application-specific tests to the project suite; run `.\.venv\Scripts\python.exe -m pytest -v` and a documented local smoke test after the application exists. No application command is specified now because there is no application implementation.

**Definition of done:** A documented application boundary consumes only compatible, validated artifacts and passes integration/smoke tests in a non-production environment.

**Status: Planned.** No UI, inference service, or application integration exists.

## Phase G — Security and operational readiness

**Objective and rationale:** Protect data and credentials and make failures, performance, and releases observable and recoverable.

**Required implementation work:**
- Keep secrets out of source code, notebooks, logs, and version control. Document approved secret storage, rotation, and least-privilege access for any deployment environment.
- Add dependency vulnerability and supply-chain review, including transitive dependencies and update policy.
- Define privacy, retention, access, and deletion requirements for any real customer data; use least data necessary.
- Add structured logs with correlation/run identifiers and useful operational errors while excluding sensitive identifiers and data values.
- Add health/readiness checks appropriate to the chosen runtime.
- Define monitoring for input-data drift, prediction drift, model performance when labels become available, latency, resource use, and failure rates.
- Version models and data; define reproducible builds, rollback, backup, and recovery procedures. Test recovery rather than only documenting it.
- Document incident ownership and operational response appropriate to the selected deployment target.

**Relevant files/modules:** proposed security/dependency policy, workflow additions, configuration examples without secrets, application health/logging modules, monitoring definitions, model/data manifests, operational documentation under `docs/`.

**Acceptance criteria:**
- Automated secret scanning and dependency vulnerability checks run on the intended code paths; findings have a documented triage process.
- Access roles and retention rules are documented and reviewed for the target environment.
- Logs are structured, useful for diagnosis, and contain no raw customer-level data or secrets in test cases.
- Health/readiness checks distinguish unavailable service, missing model, and unready dependencies.
- Drift/failure alerts have defined signals, thresholds grounded in baseline evidence, and response instructions.
- A rollback and recovery exercise restores a known compatible application/model/data version.

**Dependencies/prerequisites:** Selected application/deployment architecture; Phases B, D, and F; privacy/security review for actual data use.

**Risks/failure scenarios:** Credential leakage; vulnerable dependency; personal data over-logging; alert fatigue; unnoticed drift; incompatible rollback; untested backup.

**Tests and verification commands:** Add automated scanning and health/rollback tests to CI and deployment checks. Run the selected scanners only after tool and policy selection; record versions and results. Local `pytest` is necessary but not a substitute for these controls.

**Definition of done:** Security controls, monitoring, operational response, and tested rollback/recovery are active for the chosen environment.

**Status: Planned.** No production security, monitoring, health, or recovery controls are implemented.

## Phase H — CI/CD, deployment, and release

**Objective and rationale:** Promote only tested, compatible, traceable application and model versions through controlled environments with verified rollback.

**Required implementation work:**
- Automate unit/integration tests, workflow validation, static checks, dependency/security checks, and application build checks as appropriate.
- Validate configuration and deployment in a non-production environment before production release.
- Define compatibility among application release, model artifact, feature schema, and input data contracts.
- Add release gates, artifact provenance, smoke tests, observability checks, and explicit approval boundaries.
- Document deployment, rollback, and recovery instructions for the actual selected target.
- Define measurable release acceptance criteria from system and business requirements; do not invent arbitrary universal thresholds.
- Record whether deployment occurred, the deployed version, environment, verification results, and rollback outcome. Never infer success from a build alone.

**Relevant files/modules:** `.github/workflows/`, proposed deployment manifests/scripts, model/data manifests, release documentation under `docs/`, CI/CD settings external to the repository.

**Acceptance criteria:**
- A tested build produces identifiable versioned artifacts and dependency/configuration manifests.
- Non-production deployment passes smoke tests and readiness checks using representative, non-sensitive test data.
- Release gates verify model/data/application compatibility and required approval.
- Rollback to the previous known-good release is successfully exercised and documented.
- Production deployment is reported only after the target environment confirms health and the release is verified.

**Dependencies/prerequisites:** Phases A, B, D, F, and G; an approved deployment target; release and rollback ownership.

**Risks/failure scenarios:** CI passes but deployment configuration fails; artifacts are mismatched; secrets or data enter build artifacts; failed rollout leaves a partial release; rollback is unavailable.

**Tests and verification commands:** Use CI checks and a target-specific non-production deployment/smoke procedure once those files and target exist. Preserve deployment logs and verification evidence; no deployment command can be specified from the current repository because deployment tooling is absent.

**Definition of done:** Release promotion and rollback work in the chosen environment and have recorded verification evidence.

**Status: Planned.** No deployment configuration or release pipeline exists.

## Phase I — Documentation and final acceptance

**Objective and rationale:** Give maintainers enough verified guidance to reproduce, operate, review, and recover the system, and reserve the production-ready claim for an evidence-based final gate.

**Required implementation work:**
- Maintain an architecture diagram and data-flow explanation that match deployed code.
- Document reproducible local setup, data retrieval/provenance, schemas, feature/target definitions, model and segmentation methodology, and environment configuration.
- Document known limitations, troubleshooting, privacy/security practices, release procedures, rollback, and recovery.
- Publish only verified performance, reliability, and operational evidence with dataset/model/environment versions.
- Provide maintainer and contributor guidance, ownership/escalation routes appropriate to the project, and public-safe coordination notes.
- Complete a final production-readiness checklist spanning tests/CI, data governance, model validity, application security, monitoring, release, and recovery.

**Relevant files/modules:** `README.md`, this roadmap, proposed `docs/architecture.*`, data/model cards, operational runbooks, contribution guidance, and release evidence.

**Acceptance criteria:**
- Documentation commands and repository paths are tested against the current branch.
- Architecture and data-flow documentation matches code and deployed interfaces.
- Data and model documentation identifies provenance, schema/version, intended use, limitations, and evaluation protocol.
- Performance and reliability claims point to reproducible reports and run identifiers.
- Final checklist has evidence for each required gate; unresolved risks are explicitly accepted by the appropriate project decision-makers.
- The term “production-ready” is used only after all required gates for the actual deployment context are met.

**Dependencies/prerequisites:** All applicable earlier phases and evidence from real CI, evaluation, non-production deployment, and operations exercises.

**Risks/failure scenarios:** Documentation drifts from implementation; public documentation exposes sensitive information; unsupported metrics/claims mislead users; acceptance gates are treated as paperwork without evidence.

**Tests and verification commands:**

```powershell
git diff --check
.\.venv\Scripts\python.exe -m pytest -v
```

Also verify links/commands and review CI, evaluation, deployment, and recovery evidence for the exact release candidate.

**Definition of done:** Maintainers can reproduce and operate the system, all required release evidence is linked, and outstanding risks have explicit disposition.

**Status: In progress.** A public README and this roadmap document exist; architecture diagrams, model/data cards, operational runbooks, and final acceptance evidence remain outstanding.

## Integration contracts and handoffs

These are responsibility-based workstreams, not individual assignments or deadlines. This roadmap is in the public repository, so confidential project coordination must remain in an approved private location.

| Workstream | Responsibility | Handoff contract and dependency |
|---|---|---|
| Data quality and preparation | EDA, preprocessing, feature engineering, labels, and dataset contracts. | Publish versioned schemas, qualifying-purchase rules, observation/outcome windows, `CustomerID`, `repeat_purchase`, safe feature allowlist, metadata, and reproducible train/validation/test artifacts. Communicate contract changes before downstream use. |
| Supervised learning | Baselines, candidate comparison, evaluation, and model selection. | Consume only the declared safe feature allowlist; fit learned transformations on training rows only; use validation for selection; reserve test for final evaluation; return model artifact requirements and evaluation evidence without changing input semantics silently. |
| Customer segmentation | Customer-level segmentation, cluster stability, and interpretation. | Define a segmentation-specific customer feature and preprocessing contract; keep its purpose and outputs distinct from the supervised target unless a reviewed interface explicitly joins them. |
| Application integration and release | User experience, artifact integration, deployment, and release engineering. | Consume versioned and validated data/model/segment interfaces; reject incompatible schema/model versions explicitly; preserve provenance, health, monitoring, and rollback evidence. |

**Current supervised-learning data handoff:** Consume `data/processed/train.csv`, `validation.csv`, and `test.csv`. The target is `repeat_purchase`; retain `CustomerID` only for auditing and split-integrity checks. The current safe feature allowlist is `recency_days`, `frequency`, `monetary`, `distinct_products`, `active_purchase_days`, `average_invoice_value`, `average_items_per_invoice`, and `customer_tenure_days` (also recorded in `modeling_metadata.json`). Fit learned transformations on training rows only, use validation for model selection, and reserve test for final evaluation. This contract is a current prototype interface; version it before compatibility is promised.

**Shared change rule:** Changes to transaction semantics, identifiers, date windows, feature names/types, target definition, or artifact schema require an explicit version/compatibility decision, updated tests and metadata, and communication to every consumer. No workstream should mutate another workstream's contract implicitly.

## Final production-readiness gate

Before describing ShopSense as production-ready, the release candidate must have evidence that:

- CI and pull-request checks passed on the exact revision; workflow syntax and dependency/security checks are valid.
- Data provenance, source terms, contracts, quality gates, atomic publication, and lineage are established.
- Features/labels and evaluation protocol are independently reviewed for leakage and temporal validity.
- Model performance and operational thresholds are supported by reproducible evaluation and business decision criteria.
- Segments, if exposed, have stability and usefulness evidence rather than only cluster assignments.
- The deployed application validates inputs, reports failures usefully, protects secrets/data, and exposes appropriate health signals.
- Monitoring, version compatibility, rollback, recovery, and privacy/access requirements are tested in the target environment.
- Documentation, runbooks, limitations, and release evidence match the actual system.

Passing the existing unit tests is necessary but not sufficient for this gate.
