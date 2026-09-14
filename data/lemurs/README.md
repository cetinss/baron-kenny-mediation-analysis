# README: LEMURS Differentially Private Survey Dataset

**Title:** Aim High, Stay Private: Differentially Private Synthetic Data Enables
Public Release of Behavioral Health Information with High Utility

**Authors:** Mohsen Ghasemizade, Juniper Lovato, Chris Danforth, Peter Sheridan
Dodds, Laura S.P. Bloomfield, Matthew Price, Joseph Near

**Institution:** University of Vermont, Burlington, VT, USA

**Corresponding Author:** Mohsen Ghasemizade (mghasemi@uvm.edu)

**Related Publication:** *Journal of the American Medical Informatics Association*, 2025

---

## Dataset Overview

This dataset is a **differentially private (DP) synthetic version** of the Survey
dataset collected during Phase 1 of the Lived Experiences Measured Using Rings
Study (LEMURS). LEMURS is a longitudinal study that recruited approximately 600
first-year college students at the University of Vermont beginning in Fall 2022.
Participants completed weekly surveys and wore Oura rings to track sleep,
physiological signals, stress, and general well-being.

The original Survey dataset contains 108 columns capturing behavioral,
psychological, and lifestyle data. Because the original data cannot be shared
publicly due to re-identification risks, this release provides a DP synthetic
version generated using the **AIM (Adaptive and Iterative Mechanism)** algorithm
at a privacy budget of **ε (epsilon) = 5**, which was identified as the optimal
balance between data utility and participant privacy.

**Differential Privacy (DP)** is a mathematical framework that protects individual
privacy by adding controlled random noise to the data. The privacy budget ε
controls the trade-off: smaller ε = stronger privacy but less fidelity to the
original data. A value of ε = 5 was selected because it maintained key
statistical and analytical properties of the original dataset while resisting
linkage and membership inference attacks.

---

## Files Included

| Filename | Description |
|---|---|
| `survey_syn_5.csv` | Differentially private synthetic Survey dataset generated with AIM at ε = 5 |
| `survey_codebook-2.xlsx` | Codebook describing all column names, survey questions, and response encodings |
| `README.md` | This documentation file |

---

## Dataset Structure

- **File:** `survey_syn_5.csv`
- **Format:** Comma-Separated Values (CSV)
- **Unit of observation:** One row = one weekly survey response for one participant
- **Participant identifier:** `record_id` — a pseudo-anonymized numerical ID.
  Each participant may contribute multiple rows (one per study week).
- **Note:** Because this is synthetic data, `record_id` values do not correspond
  to real individuals.
- **Missing values:** Represented as `N/A`. These reflect questions that were
  not applicable or not answered for a given week (e.g., follow-up questions
  shown only if a prior question was answered "Yes").
- **Qualitative (text-based) columns** from the original dataset have been
  removed, as the DP algorithm operates only on quantitative values.

---

## Variable Descriptions

All column names follow the format `F1_[Domain]_[Variable]`, where `F1`
refers to Form 1 (the weekly survey instrument). Full question text and
response codes for every column are provided in `survey_codebook-2.xlsx`.
A summary of the thematic domains is provided below.

### Oura Ring App Usage (`F1_App_*`)
Questions about how participants used the Oura ring app during the week.

| Column | Description | Values |
|---|---|---|
| `F1_App_numtakeoff` | Number of times the Oura ring was removed during the week | Continuous (count) |
| `F1_App_sleeprecovery` | Used Sleep Recovery feature | 1 = Yes, 0 = No |
| `F1_App_meditation` | Used Meditation feature | 1 = Yes, 0 = No |
| `F1_App_breathwork` | Used Breathwork feature | 1 = Yes, 0 = No |
| `F1_App_periodpredictor` | Used Period Predictor feature | 1 = Yes, 0 = No |
| `F1_App_otherring` | Used another wearable ring | 1 = Yes, 0 = No |
| `F1_App_encourgesleep` | Ring encouraged earlier sleep | 1 = Yes, 0 = No |
| `F1_App_encouragemove` | Ring encouraged more movement | 1 = Yes, 0 = No |
| `F1_App_encourageworkout` | Ring encouraged more workouts | 1 = Yes, 0 = No |
| `F1_App_encouragetakeeasy` | Ring encouraged taking it easier | 1 = Yes, 0 = No |
| `F1_App_encourageother` | Ring encouraged other behavior | 1 = Yes, 0 = No |

### Physical Activity (`F1_Activity_*`)
Based on the International Physical Activity Questionnaire (IPAQ).

| Column | Description | Values |
|---|---|---|
| `F1_Activity_numvigorous` | Days of vigorous activity in past 7 days | 0–7 (days/week) |
| `F1_Activity_nummoderate` | Days of moderate activity in past 7 days | 0–7 (days/week) |
| `F1_Activity_hoursminutes1` | Hours of vigorous activity per day | Continuous (hours) |
| `F1_Activity_hoursminutes4` | Minutes of vigorous activity per day | Continuous (minutes) |
| `F1_Activity_numwalking` | Days of walking ≥10 minutes in past 7 days | 0–7 (days/week) |
| `F1_Activity_hoursminutes2` | Hours of walking per day | Continuous (hours) |
| `F1_Activity_hoursminutes3` | Minutes of walking per day | Continuous (minutes) |
| `F1_Activity_hoursminutes5` | Minutes of sitting on a weekday | Continuous (minutes) |
| `F1_Activity_hoursminutes6` | Hours of sitting on a weekday | Continuous (hours) |

### Caffeine Consumption (`F1_Caffeine_*`)

| Column | Description | Values |
|---|---|---|
| `F1_Caffeine_caffeinethisweek` | Consumed caffeine this week | 1 = Yes, 0 = No |
| `F1_Caffeine_coffee` | Drank coffee | 1 = Yes, 0 = No |
| `F1_Caffeine_tea` | Drank tea | 1 = Yes, 0 = No |
| `F1_Caffeine_soda` | Drank soda | 1 = Yes, 0 = No |
| `F1_Caffeine_energydrink` | Drank energy drink | 1 = Yes, 0 = No |
| `F1_Caffeine_workoutdrink` | Drank workout/pre-workout drink | 1 = Yes, 0 = No |
| `F1_Caffeine_otherdrink` | Drank other caffeinated drink | 1 = Yes, 0 = No |
| `F1_Caffeine_whencaffeine*` | Time of day caffeine was consumed | 1 = Yes, 0 = No (see codebook for time windows) |
| `F1_Caffeine_whycaffeine*` | Reason for caffeine consumption | 1 = Yes, 0 = No (see codebook for reason categories) |

### Time Outdoors (`F1_TimeOut_*`)

| Column | Description | Values |
|---|---|---|
| `F1_TimeOut_Days` | Days spent in nature last week | 0–7 (days/week) |
| `F1_TimeOut_Hours` | Hours spent outdoors | Continuous (hours) |
| `F1_TimeOut_Minutes` | Minutes spent outdoors | Continuous (minutes) |
| `F1_TimeOut_When` | Time of day outdoors | 1=Early Morning, 2=Mid-Morning, 3=Early Afternoon, 4=Late Afternoon, 5=Early Evening, 6=Night |
| `F1_TimeOut_Cycling` through `F1_TimeOut_Other` | Type of outdoor activity | 1 = Yes, 0 = No (see codebook) |
| `F1_TimeOut_Location` | Type of outdoor environment | 1=Suburban, 2=Urban, 3=Rural |

### Social Engagement (`F1_Social_*`)

| Column | Description | Values |
|---|---|---|
| `F1_Social_numbersocial` | Number of social engagements this week | Continuous (count) |
| `F1_Social_howmanyfriends` | Number of friends spent time with | Continuous (count) |
| `F1_Social_howmmanyinperson` | Number of in-person interactions | Continuous (count) |
| `F1_Social_videocall` | Number of video call interactions | Continuous (count) |
| `F1_Social_phone` | Number of phone interactions | Continuous (count) |
| `F1_Social_talkfamily` | Times saw or talked to family | Continuous (count) |
| `F1_Social_moreorless` | Satisfaction with social frequency | 1=More than desired, 2=Less than desired, 3=Just right |

### Sleep (`F1_Sleep_*`)

| Column | Description | Values |
|---|---|---|
| `F1_Sleep_sleephours` | Average nightly sleep hours this week | Continuous (hours) |
| `F1_Sleep_sleepquality` | Self-rated sleep quality | 1=Very good, 2=Fairly good, 3=Fairly bad, 4=Very bad |
| `F1_Sleep_sleepamount` | Perceived adequacy of sleep | 1=Too little, 2=Adequate, 3=Ideal, 4=Too much |

### Stress Events (`F1_Stress_*`)

| Column | Description | Values |
|---|---|---|
| `F1_Stress_stressevent` | Experienced a stressful event this week | 1 = Yes, 0 = No |
| `F1_Stress_describeevent` | Nature of the stressful event | 1=Social, 2=Romantic, 3=Familial, 4=Financial, 5=Academic, 6=Physical, 7=Mental |
| `F1_Stress_stressacademic` | Had an academic test this week | 1 = Yes, 0 = No |
| `F1_Stress_stresspaper` | Had a paper/project due this week | 1 = Yes, 0 = No |

### Perceived Stress Scale (`F1_PSS_*`)
The PSS is a validated 10-item psychological scale measuring perceived stress
over the past week. Higher total scores indicate greater perceived stress.

| Column | Description | Values |
|---|---|---|
| `F1_PSS_stressupset` | Upset by unexpected events | 0=Never, 1=Almost Never, 2=Sometimes, 3=Fairly Often, 4=Very Often |
| `F1_PSS_stresscontrol` | Felt unable to control important things | 0–4 (same scale) |
| `F1_PSS_stressnervous` | Felt nervous and stressed | 0–4 |
| `F1_PSS_stressconfident` | Felt confident handling problems | 0–4 |
| `F1_PSS_stressthings` | Felt things were going your way | 0–4 |
| `F1_PSS_stresscope` | Could not cope with required tasks | 0–4 |
| `F1_PSS_stressirritations` | Able to control irritations | 0–4 |
| `F1_PSS_stressontop` | Felt on top of things | 0–4 |
| `F1_PSS_stressangered` | Angered by things outside control | 0–4 |
| `F1_PSS_stressdifficulties` | Difficulties piling up uncontrollably | 0–4 |

### Generalized Anxiety Disorder Scale (`F1_GAD_*`)
The GAD-7 is a validated 7-item scale measuring anxiety symptoms over the
past two weeks.

| Column | Description | Values |
|---|---|---|
| `F1_GAD_nervous` | Feeling nervous, anxious, or on edge | 0=Not at all, 1=Several days, 2=More than half the days, 3=Nearly every day |
| `F1_GAD_worry` | Unable to stop or control worrying | 0–3 (same scale) |
| `F1_GAD_worrydifferent` | Worrying too much about different things | 0–3 |
| `F1_GAD_troublerelaxing` | Trouble relaxing | 0–3 |
| `F1_GAD_restless` | Too restless to sit still | 0–3 |
| `F1_GAD_annoyed` | Easily annoyed or irritable | 0–3 |
| `F1_GAD_afraid` | Feeling afraid something awful might happen | 0–3 |
| `F1_GAD_totalscore` | GAD-7 total score (sum of 7 items) | 0–21; 0–4=Minimal, 5–9=Mild, 10–14=Moderate, 15–21=Severe anxiety |
| `F1_GAD_howdifficult` | Difficulty functioning due to anxiety symptoms | 0=Not at all, 1=Several days, 2=More than half days, 3=Nearly every day |

### Study Week

| Column | Description | Values |
|---|---|---|
| `week` | Study week number | Continuous (integer); week 1 = first week of the study semester |

---

## Abbreviations

| Abbreviation | Definition |
|---|---|
| DP | Differential Privacy |
| ε (epsilon) | Privacy budget — controls the privacy-utility trade-off in DP |
| AIM | Adaptive and Iterative Mechanism — the DP synthetic data generator used |
| LEMURS | Lived Experiences Measured Using Rings Study |
| PSS | Perceived Stress Scale |
| GAD | Generalized Anxiety Disorder (scale) |
| GAD-7 | 7-item Generalized Anxiety Disorder scale |
| TST | Total Sleep Time |
| HR | Heart Rate |
| HRV | Heart Rate Variability |
| UMAP | Uniform Manifold Approximation and Projection |
| IPAQ | International Physical Activity Questionnaire |
| UVM | University of Vermont |

---

## Data Generation Method

The synthetic data was generated using the **AIM (Adaptive and Iterative
Mechanism)** algorithm, a state-of-the-art workload-aware DP synthetic data
generator. AIM allocates noise strategically across the most informative
marginal distributions of the data. The privacy budget was set to **ε = 5**,
which was identified in the associated publication as the lowest ε at which:

- Marginal L1/L2 errors stabilized
- Spearman correlation structure was preserved
- Random forest regression R² scores plateaued (~0.64)
- UMAP cluster geometry closely matched the original
- Membership inference attacks (via TAPAS framework) showed no success

10 synthetic datasets were generated for each ε value tested (ε = 1, 2, 5,
10, 20, 50, 100), and the dataset with the lowest L1/L2 errors was selected
for this release.

---

## Privacy and Ethics

The original LEMURS dataset contains sensitive behavioral health and
physiological data from college students and cannot be publicly released due
to re-identification risks. This synthetic release was designed to enable
open science while protecting participant privacy. Participants provided
informed consent for data collection under IRB oversight at the University
of Vermont. The DP guarantee at ε = 5 ensures that no individual record
can be reliably re-identified, even when combined with auxiliary data.

---

## Software and Reproducibility

- **DP generator:** AIM (via `smartnoise-sdk` Python library)
- **Analysis environment:** Python 3.x
- **Key libraries:** `pandas`, `numpy`, `scikit-learn`, `umap-learn`,
  `statsmodels`, `matplotlib`, `seaborn`
- Code for reproducing the analyses is available in the associated publication.

---

## Citation

If you use this dataset, please cite the associated publication:

> Ghasemizade M, Lovato J, Danforth C, Dodds PS, Bloomfield LSP, Price M,
> Near J. Aim High, Stay Private: Differentially Private Synthetic Data
> Enables Public Release of Behavioral Health Information with High Utility.
> *Journal of the American Medical Informatics Association*, 2025.
