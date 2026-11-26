# Disease Mappings

These IDs were looked up on 2026-11-17 against the Open Targets Platform GraphQL API, MONDO (via EBI OLS4) and the NLM MeSH lookup service. The "MeSH linked in Open Targets" column lists the MeSH cross-reference stored on the Open Targets disease record. The "MeSH by exact name" column lists the MeSH descriptor found by searching MeSH directly for the disease name.

## Cardiovascular and cerebrovascular

| Term | Open Targets ID (name) | MONDO | MeSH linked in Open Targets | MeSH by exact name |
|---|---|---|---|---|
| Hypertensive disorder | MONDO_0005044 | MONDO:0005044 | D006973 | D006973 Hypertension |
| Myocardial infarction | MONDO_0005068 | MONDO:0005068 | D009203 | D009203 Myocardial Infarction |
| Heart attack | MONDO_0005068 (myocardial infarction) | MONDO:0005068 | D009203 | none; "Heart Attack" is an entry term under D009203 Myocardial Infarction |
| Myocardial ischemia | MONDO_0024644 | MONDO:0024644 | D017202 | D017202 Myocardial Ischemia |
| Ischemia | EFO_0000556 | none | D007511 | D007511 Ischemia |
| Stroke | MONDO_0005098 (stroke disorder) | MONDO:0005098 | D020521 | D020521 Stroke |
| Stroke disorder | MONDO_0005098 | MONDO:0005098 | D020521 | none; MeSH descriptor is "Stroke" (D020521) |
| Ischemic stroke | MONDO_1060198 | MONDO:1060198 | none | D000083242 Ischemic Stroke |

MONDO has only site-specific ischemia terms (brain, myocardial, limb, spinal cord), not a general one.

Stroke covers both ischemic and hemorrhagic stroke; ischemic stroke is its child term in MONDO.

## Metabolic

| Term | Open Targets ID (name) | MONDO | MeSH linked in Open Targets | MeSH by exact name |
|---|---|---|---|---|
| Prediabetes | MONDO_0006920 (prediabetes syndrome) | MONDO:0006920 | D011236 | D011236 Prediabetic State |

## Sleep

| Term | Open Targets ID (name) | MONDO | MeSH linked in Open Targets | MeSH by exact name |
|---|---|---|---|---|
| Narcolepsy | MONDO_0021107 | MONDO:0021107 | D009290 | D009290 Narcolepsy |
| Narcolepsy-cataplexy syndrome | MONDO_0016158 | MONDO:0016158 | none | none; "Narcolepsy-Cataplexy Syndrome" is an entry term under D009290 Narcolepsy |

MONDO and Open Targets treat narcolepsy-cataplexy syndrome as a child of narcolepsy, but MeSH folds both into the single Narcolepsy descriptor.

## Liver

| Term | Open Targets ID (name) | MONDO | MeSH linked in Open Targets | MeSH by exact name |
|---|---|---|---|---|
| Hepatic steatosis | MONDO_0004790 (fatty liver disease) | MONDO:0004790 | D005234 | D005234 Fatty Liver |
| Non-alcoholic fatty liver / steatotic liver disease | EFO_1001248 (non-alcoholic fatty liver) | MONDO:0013209 (MASLD), not used by Open Targets | none | D065626 Non-alcoholic Fatty Liver Disease |

Fatty liver disease is the broader parent of NAFLD/MASLD and includes alcohol-related fatty liver.

MONDO:0013209 cross-references MeSH D065626, but Open Targets does not use that term.

## Substance use

| Term | Open Targets ID (name) | MONDO | MeSH linked in Open Targets | MeSH by exact name |
|---|---|---|---|---|
| Nicotine dependence | MONDO_0008575 | MONDO:0008575 | D014029 | D014029 Tobacco Use Disorder (broader) |
| Smoking cessation | EFO_0004319 | none | D016540 | D016540 Smoking Cessation |
| Cocaine use disorder | EFO_0010445 | none (closest: MONDO:0005186 cocaine dependence, MONDO:0004456 cocaine abuse) | none | D019970 Cocaine-Related Disorders (broader) |
| Amphetamine-related disorders | none (MONDO_0003969 not in Open Targets) | MONDO:0003969 (amphetamine abuse) | none | D019969 Amphetamine-Related Disorders |
| Methamphetamine dependence | MONDO_0005419 | MONDO:0005419 | none | none; closest is D019969 Amphetamine-Related Disorders (broader) |

Open Targets files smoking cessation under "biological_process", not a disease therapeutic area.

Neither MONDO cocaine term cross-references MeSH.

MONDO:0003969 (amphetamine abuse) is the only MONDO term that cross-references MeSH D019969, and Open Targets does not include it.
