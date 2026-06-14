# Data Card — LendingClub Loans (2007–2018)

## Source

[LendingClub statistics page](https://www.lendingclub.com/info/download-data.action) (since 2020 only on Kaggle).
Hosted mirror used in this project: `wordsforthewise/lending-club` on Kaggle.

## Schema

~150 columns shipped; 30 used here (application-time only).

Key columns we keep:

| Column | Type | Description |
|---|---|---|
| id | int | Unique loan ID |
| issue_d | date | Loan funding date |
| loan_amnt | float | Amount requested ($500–40k) |
| term | int | 36 or 60 months |
| int_rate | float | APR (%) |
| grade / sub_grade | str | LendingClub risk tier |
| emp_title / emp_length | str / float | Employer name / years |
| home_ownership | str | RENT / MORTGAGE / OWN / OTHER |
| annual_inc | float | Borrower-reported income |
| verification_status | str | Verified / Source / Not Verified |
| purpose / title | str | Loan purpose |
| zip_code / addr_state | str | First 3 digits + xx, US state |
| dti | float | Debt-to-income ratio |
| delinq_2yrs, inq_last_6mths, open_acc, etc. | int | Credit bureau aggregates |
| loan_status / last_pymnt_d | str / date | Outcome (used **only** for label construction) |

## Known biases / caveats

- **US consumer P2P loans, not Indian SME loans.** Domain transfer requires care: DTI, credit-bureau coverage, and income verification all differ.
- **Stated income** — `annual_inc` is borrower-reported. The `verification_status` flag indicates whether LendingClub verified it; "Not Verified" rows are an obvious fraud-risk slice.
- **Pre-2014 vs. post-2014** — LendingClub tightened underwriting in 2014; the loss distributions are different across periods. Time-split respects this.
- **No protected-class attributes** — no gender, race/caste, religion. Fairness analysis runs against legitimate proxies only.
- **Survivorship** — the dataset is *accepted* loans only; declined applicants aren't included. Fraud models trained only on accepted data will under-fit the upper tail of the score distribution.

## Versioning

We pin to the Kaggle mirror's `accepted_2007_to_2018Q4.csv` snapshot. If the source data is refreshed, freeze the snapshot via DVC or a content-addressed S3 path.
