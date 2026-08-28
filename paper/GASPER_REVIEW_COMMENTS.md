# Gašper's review comments on `main_GS.pdf`

Source reviewed: `paper/main_GS.pdf`

This document transcribes all 39 substantive PDF comments and identifies the highlighted text to which each comment refers. The comments are ordered by manuscript source order, which is more useful for implementation than the order in which the PDF annotations were created. Gašper's wording is preserved as written.

Add your decision or requested revision in each **Tomi / instruction to Claude Code** field. Claude Code should not implement a comment until that field contains an explicit instruction.

## Implementation status (2026-08-28)

All 39 comments are resolved: 35 implemented, 2 needed no change (25, 38), and 2 were investigations that confirmed the existing work was correct (30, 31 -- 31 also produced a text clarification).

Two points were decided with Tomi before implementation:

- **Comment 12:** balanced accuracy replaces accuracy as the primary metric.
- **Comment 05:** the abstract says "657 full-workday recordings" instead of Gasper's "3-week", which would contradict the recording counts in Methodology.

**Page budget.** The paper must stay at 4 pages including references, and the new text cost about a page. Compensating changes, in this order of preference: redundant sentences merged or cut where the same point was already made elsewhere (Introduction, Related Work, Methodology, Results, Discussion, Conclusion); Table 1 restructured to drop its four group-header rows in favour of combined `Representation, Model` labels; the table caption shortened; the table body set in `\footnotesize`; and Figure 2 scaled to `0.86\textwidth`. No result, number, limitation or citation was removed. Figure 2 at `0.86\textwidth` is the largest that fits with the single-column table.

## Title, author information, and abstract

### 01. Terminology for participants

- **PDF page:** 1
- **Source:** `paper/main.tex:24`
- **Selected text:** “Office Workers”
- **Gašper:** “In the project we called them \"knowledge workers\"”
- **Tomi / instruction to Claude Code:** _Add your response here._ Fix in the whole paper to knowledge workers.
- **Implemented (2026-08-28):** **Done.** `Office Workers` -> `knowledge workers` in the title, abstract, introduction, methodology, discussion and conclusion. The remaining uses of "office" describe the *setting* ("office work", "office gaze"), not the participants.

### 02. Gašper's second affiliation

- **PDF page:** 1
- **Source:** `paper/main.tex:33-38`
- **Selected text:** “Jožef Stefan Institute, Ljubljana, Slovenia, gasper.slapnicar@ijs.si”
- **Gašper:** “If space is not an issue, can add Jožef Stefan International postgraduate school for me”
- **Tomi / instruction to Claude Code:** _Add your response here._ add as Gašper wants
- **Implemented (2026-08-28):** **Done.** Second `\affiliation` block added for Gasper: Jozef Stefan International Postgraduate School, Ljubljana, Slovenia.

### 03. Meaning of “user states”

- **PDF page:** 1
- **Source:** `paper/main.tex:43`
- **Selected text:** “user states”
- **Gašper:** “What states? I think the others called them \"psychological states\" in their paper”
- **Tomi / instruction to Claude Code:** _Add your response here._ psychological states yes
- **Implemented (2026-08-28):** **Done.** Abstract now reads "momentary psychological states at work".

### 04. Tense of “compare”

- **PDF page:** 1
- **Source:** `paper/main.tex:45`
- **Selected text:** “compare”
- **Gašper:** “compareD... past tense... but not crucial, can skip this fix.”
- **Tomi / instruction to Claude Code:** _Add your response here._ fix appropriately in the whole paper. make sure tenses are appropriately and consistently used as per best practices.
- **Implemented (2026-08-28):** **Done.** Tense pass over the whole paper on one rule: Introduction and Related Work stay in the present (framing and what published work shows); Methodology and Results are in the past (what we did and observed); atemporal facts ("data collection is ongoing", "a window is positive when...") stay present.

### 05. Description of the study

- **PDF page:** 1
- **Source:** `paper/main.tex:45`
- **Selected text:** “ongoing office-worker study:”
- **Gašper:** “maybe something like \"from data collected from knowledge workers during a 3-week data collection\"”
- **Tomi / instruction to Claude Code:** _Add your response here._ as gašper says
- **Implemented (2026-08-28):** **Done, with a change you approved.** Gasper's "3-week" would contradict the 657 recordings / 19-79 per participant (median 32) reported in Methodology, so the abstract says "drawn from knowledge workers across 657 full-workday recordings".

### 06. Meaning of “multichannel”

- **PDF page:** 1
- **Source:** `paper/main.tex:45`
- **Selected text:** “multichannel”
- **Gašper:** “What does this mean for eyetracking again? Sounds more video/audio thing”
- **Tomi / instruction to Claude Code:** _Add your response here._ the word is not wrong, just maybe add some brief explanation. could be implicit such as, you list a couple of possible channels or something similar. keep it brief. 
- **Implemented (2026-08-28):** **Done.** Abstract now glosses the channels inline: "raw multichannel signals (pupil size, gaze position and eye--screen distance)".

### 07. Explain the pretrained embeddings

- **PDF page:** 1
- **Source:** `paper/main.tex:45`
- **Selected text:** “frozen GazeMAE embeddings and frozen MOMENT embeddings.”
- **Gašper:** “Since noone knows what this is, could say \"embeddings obtained from pretrained foundational eyetracking (GazeMAE) and time series (MOMENT) encoders\"”
- **Tomi / instruction to Claude Code:** _Add your response here._ yes
- **Implemented (2026-08-28):** **Done.** "embeddings obtained from pretrained eye-tracking (GazeMAE) and time-series (MOMENT) encoders, both used frozen".

### 08. Meaning of “absorption”

- **PDF page:** 1
- **Source:** `paper/main.tex:46`
- **Selected text:** “momentary absorption.”
- **Gašper:** “What is absorption? I dont recall this target/label...”
- **Tomi / instruction to Claude Code:** _Add your response here._ don't use synonyms for terminology. be consistent. i think engagement was meant here. correct me if im wrong.
- **Implemented (2026-08-28):** **Done.** You were right: engagement was meant. See 27/28/34 for the terminology decision.

### 09. Wording of fold-local preprocessing

- **PDF page:** 1
- **Source:** `paper/main.tex:47`
- **Selected text:** “fold-local”
- **Gašper:** “fold-specific? per-fold?”
- **Tomi / instruction to Claude Code:** _Add your response here._ per-fold
- **Implemented (2026-08-28):** **Done.** `fold-local` -> `per-fold` everywhere (abstract, Related Work, Discussion).

### 10. Classifier wording

- **PDF page:** 1
- **Source:** `paper/main.tex:47`
- **Selected text:** “a random forest classifier and an MLP”
- **Gašper:** “\"a random forest and an MLP classifier...\"”
- **Tomi / instruction to Claude Code:** _Add your response here._ ok
- **Implemented (2026-08-28):** **Done.** "using a random forest and an MLP classifier".

### 11. “Unseen participant” under LOSO

- **PDF page:** 1
- **Source:** `paper/main.tex:48`
- **Selected text:** “unseen-participant”
- **Gašper:** “This is kind of implicit under LOSO. Could also say \"left-out\"”
- **Tomi / instruction to Claude Code:** _Add your response here._ apply if you think the comment has value, not strictly necessary.
- **Implemented (2026-08-28):** **Done.** Applied; `unseen-participant` -> `left-out participant` in the abstract, Results and Conclusion.

### 12. Statistical significance of the accuracy difference

- **PDF page:** 1
- **Source:** `paper/main.tex:48`
- **Selected text:** “0.014 above”
- **Gašper:** “Someone might ask for statistical significance on a 1%-ish difference”
- **Tomi / instruction to Claude Code:** _Add your response here._ maybe we should make the main results balanced accuracy, since better margin is observed there. what do you think? discuss briefly with me before implementing this point.
- **Implemented (2026-08-28):** **Done -- balanced accuracy is now the primary metric** (discussed and approved). Rationale stated in the paper is class imbalance, not the wider margin: the positive rate is 33-72 % per participant, so plain accuracy partly measures each person's base rate, while balanced accuracy has a fixed 0.500 chance level. Headline changes from 0.538 accuracy / +0.014 to **0.536 balanced accuracy, +0.036 over chance**, and the ranking becomes monotonic (raw > handcrafted > GazeMAE > MOMENT). Accuracy and its paired delta are kept as secondary columns. Balanced accuracy is defined on 16 folds, not 17 -- stated in the caption and in Experimental Design.

### 13. Whether the encoder should be called pretrained

- **PDF page:** 1
- **Source:** `paper/main.tex:48`
- **Selected text:** “frozen”
- **Gašper:** “pretrained?”
- **Tomi / instruction to Claude Code:** _Add your response here._ doesn't matter which one, just use the selected termin (the one more appropriate) consistently
- **Implemented (2026-08-28):** **Done.** Consistent split: the encoders are *pretrained* (that is a property of the checkpoints) and we used them *frozen* (that is our choice). Fixed the one place in the Introduction that called MOMENT's encoder "frozen" as if that were the authors' doing.

## Introduction

### 14. Define the state of interest

- **PDF page:** 1
- **Source:** `paper/sections/01_introduction.tex:3`
- **Selected text:** “state.”
- **Gašper:** “define \"state\" of interest (psychological...?)”
- **Tomi / instruction to Claude Code:** _Add your response here._ define as most appropriate
- **Implemented (2026-08-28):** **Done.** "that person's momentary psychological state".

### 15. Follow-up on “absorption”

- **PDF page:** 1
- **Source:** `paper/sections/01_introduction.tex:4`
- **Selected text:** “momentary engagement, particularly absorption”
- **Gašper:** “Ah, i now see what absorption is.”
- **Tomi / instruction to Claude Code:** _Add your response here._ ok, we should clear this earlier i guess? idk what to do, do what is best.
- **Implemented (2026-08-28):** **Done** via the terminology unification in 27/28/34 -- "absorption" no longer appears anywhere.

### 16. Wording of engagement inference

- **PDF page:** 1
- **Source:** `paper/sections/01_introduction.tex:6`
- **Selected text:** “it.”
- **Gašper:** “estimating it”
- **Tomi / instruction to Claude Code:** _Add your response here._ ok implement so
- **Implemented (2026-08-28):** **Done.** "a plausible sensor for estimating it".

### 17. Citation for links to attention and processing load

- **PDF page:** 1
- **Source:** `paper/sections/01_introduction.tex:7`
- **Selected text:** “documented links to attention and processing load.”
- **Gašper:** “reference”
- **Tomi / instruction to Claude Code:** _Add your response here._ add reference
- **Implemented (2026-08-28):** **Done.** Cites `kahneman1966pupil` (processing load) and the newly added `rayner1998reading` (attention). Rayner 1998, Psychological Bulletin 124(3):372-422, doi:10.1037/0033-2909.124.3.372, verified against Crossref.

### 18. Whether “short” is needed

- **PDF page:** 1
- **Source:** `paper/sections/01_introduction.tex:10`
- **Selected text:** “short”
- **Gašper:** “could omit - probably any window”
- **Tomi / instruction to Claude Code:** _Add your response here._ omit short
- **Implemented (2026-08-28):** **Done.** "A window of eye-tracking data...".

### 19. “Applied” or “related” work

- **PDF page:** 1
- **Source:** `paper/sections/01_introduction.tex:12`
- **Selected text:** “applied”
- **Gašper:** “related?”
- **Tomi / instruction to Claude Code:** _Add your response here._ yes
- **Implemented (2026-08-28):** **Done.** "remain common in related work".

### 20. Wording of the study response

- **PDF page:** 1
- **Source:** `paper/sections/01_introduction.tex:18`
- **Selected text:** “We answer with”
- **Gašper:** “We conducted a controlled...”
- **Tomi / instruction to Claude Code:** _Add your response here._ yes
- **Implemented (2026-08-28):** **Done.** "We conducted a controlled comparison of...".

## Methodology

### 21. Description of the multimodal data collection

- **PDF page:** 2
- **Source:** `paper/sections/03_methodology.tex:6`
- **Selected text:** “multimodal study”
- **Gašper:** “A study is not really multimodal... maybe \"from a study conducting multimodal data collection\"?”
- **Tomi / instruction to Claude Code:** _Add your response here._ implement appropriately.
- **Implemented (2026-08-28):** **Done.** "The data came from a study conducting multimodal data collection, in which knowledge workers were recorded...".

### 22. “Ordinary work” or “daily routine”

- **PDF page:** 2
- **Source:** `paper/sections/03_methodology.tex:6`
- **Selected text:** “ordinary work”
- **Gašper:** “daily routine”
- **Tomi / instruction to Claude Code:** _Add your response here._ yes.
- **Implemented (2026-08-28):** **Done.** "during their daily routine at their own workplace".

### 23. Wording of what the collection contains

- **PDF page:** 2
- **Source:** `paper/sections/03_methodology.tex:7`
- **Selected text:** “collection records”
- **Gašper:** “\"...data being collected includes...\"”
- **Tomi / instruction to Claude Code:** _Add your response here._ yes implement appropriately
- **Implemented (2026-08-28):** **Done.** "The data being collected includes RGB video, eye tracking, microphone audio and application usage."

### 24. “Screen activity” or “application usage”

- **PDF page:** 2
- **Source:** `paper/sections/03_methodology.tex:7`
- **Selected text:** “screen activity.”
- **Gašper:** “application usage”
- **Tomi / instruction to Claude Code:** _Add your response here._ yes
- **Implemented (2026-08-28):** **Done.** `screen activity` -> `application usage`.

### 25. Clarification of the five channels

- **PDF page:** 2
- **Source:** `paper/sections/03_methodology.tex:11`
- **Selected text:** “five channels: left and right pupil size, gaze x and y, and the average eye-screen distance.”
- **Gašper:** “I now also know what channels are :)”
- **Tomi / instruction to Claude Code:** _Add your response here._
- **Implemented (2026-08-28):** No action needed.

### 26. Explain experience sampling

- **PDF page:** 2
- **Source:** `paper/sections/03_methodology.tex:13`
- **Selected text:** “experience-sampling”
- **Gašper:** “What is this exactly?”
- **Tomi / instruction to Claude Code:** _Add your response here._ the term is used correctly i think. double check it.
- **Implemented (2026-08-28):** **Checked and correct.** "Experience sampling" is the right name for the protocol (repeated in-situ prompts during the working day). Added a short gloss so the reader does not have to know the term: "...in which participants rated their momentary state in situ rather than recalling it afterwards."

### 27. Unify absorption, engagement, and immersion

- **PDF page:** 2
- **Source:** `paper/sections/03_methodology.tex:15`
- **Selected text:** “absorption”
- **Gašper:** “Why is this not \"immersion\" as it is used in the question...?”
- **Tomi / instruction to Claude Code:** _Add your response here._ i agree. implement as immersion if appropriate, otherwise discuss with me before implementation.
- **Implemented (2026-08-28):** **Done.** "absorption" is gone from the paper. The item is now introduced by its own wording as the *immersion* item.

### 28. Engagement versus absorption

- **PDF page:** 2
- **Source:** `paper/sections/03_methodology.tex:16`
- **Selected text:** “We call it the engagement rating.”
- **Gašper:** “We call it engagement, but we also call it absorption?”
- **Tomi / instruction to Claude Code:** _Add your response here._ ambiguous really. i think this should be an easy fix if we use consistent terminology.
- **Implemented (2026-08-28):** **Done.** Two terms, each with one job: **engagement** is the construct and the running term (it is in the title, keywords and framing); **immersion** names the specific questionnaire item, "Right now I am immersed in my work", one of three momentary engagement items. The variable is then called the *engagement rating* throughout.

### 29. Choice of the 15-minute labelling interval

- **PDF page:** 2
- **Source:** `paper/sections/03_methodology.tex:17`
- **Selected text:** “Each prompt labels the 15 minutes of recording that precede it,”
- **Gašper:** “This was kind of our choice... it can label as little as we decide to.”
- **Tomi / instruction to Claude Code:** _Add your response here._ make it clear that this was our choice. not with a separate sentence, but mid sentence.
- **Implemented (2026-08-28):** **Done, mid-sentence as requested.** "Each prompt labelled the 15 minutes of recording that preceded it, a labelling horizon we chose ourselves, and only those intervals entered this study."

### 30. Comment on the gaze-density figure

- **PDF page:** 2
- **Source:** `paper/sections/03_methodology.tex:24`
- **Selected text:** “Gaze-coordinate density on the normalised display,”
- **Gašper:** “Damn, we look at the middle of the screen a lot actually :)”
- **Tomi / instruction to Claude Code:** _Add your response here._ i agree with the comment. it looks suspicious. double check if the image is factual, if the code that generated it is correct. 
- **Implemented (2026-08-28):** **Checked -- the figure is correct; no change made.** `tobii_experiments/make_paper_figures.py` bins normalised `GazePointX/Y` from the `tobii_coordinate_normalized` export, drops NaNs and keeps only `[0,1)^2`; the `density.T` + `origin="upper"` + `extent=(0,1,1,0)` combination puts y=0 at the top of the screen, matching the axis labels, so there is no transpose or flip bug. The central concentration is genuine and expected for screen-mounted tracking during editing/reading/browsing, and the map still shows broad coverage. The numbers could not be recomputed here because `$TRUSTME_SUBJECT_ROOT` is unset on this machine and the subject tree is not mounted; the verification is of the code path, not a re-run.

### 31. Justification of the valid-frame threshold

- **PDF page:** 2
- **Source:** `paper/sections/03_methodology.tex:33`
- **Selected text:** “at least 30% of its frames are valid and at least 32 valid frames remain,”
- **Gašper:** “How was this decided? Seems quite liberal (we allow 70% missing)?”
- **Tomi / instruction to Claude Code:** _Add your response here._ it is liberal but it is because 30% is still still X time and this seems enough according to Y literature. idk really, just argument somehow because we don't want to rerun experiments. also, explain to me personally what does it mean 30% of frames and 32 frames? is not only one always the higher bar and the other lower bar is irrelevant condition? in the text it probably should be a bit clearer why we have two conditions or remove the mention of the irrelevant condition if there is one.
- **Implemented (2026-08-28):** **Explained and clarified in the text.** The two conditions are *not* redundant -- they bind in different regimes, because the number of recorded frames per window varies (`Win_len` 32-182; a nominal 3 s at 60 Hz would be 180). The 30 % rule is proportional, so on a full 180-frame window it demands 54 valid frames and is the binding one; the 32-frame rule is absolute, so on a short 60-frame window (where 30 % is only 18 frames) it is the binding one. The crossover is at about 107 frames (~1.8 s). Confirmed in the upstream export code (`segment_trustme.py:212-232`, `build_gazemae_embeddings.py:344-350`; config `min_valid_fraction: 0.30`, `min_valid_frames: 32`). The paper now says both of these and justifies the permissiveness: tracking loss in unconstrained office work is frequent, and 32 valid samples still span at least two fixations of typical duration (`rayner1998reading`).

### 32. Meaning of encoder quality control

- **PDF page:** 2
- **Source:** `paper/sections/03_methodology.tex:35`
- **Selected text:** “encoder quality control,”
- **Gašper:** “this is the \"window is kept when..\"?”
- **Tomi / instruction to Claude Code:** _Add your response here._ improve wording to simpler sentence here. no made up words/terminology.
- **Implemented (2026-08-28):** **Done.** The invented term is gone. The sentence now reads "...157,685 of those also had both encoder embeddings...", which is what actually happened.

### 33. Citation for the handcrafted-feature description

- **PDF page:** 2
- **Source:** `paper/sections/03_methodology.tex:46-48`
- **Selected text:** “Handcrafted features describe a window through the ocular events and signals it contains. Blinks, fixations and saccades are taken from the tracker's event classification, and each event type contributes counts, rates and duration statistics, together with the location and dispersion of fixations and the amplitude and velocity of saccades. The remaining groups summarise pupil size and its rate of change, the eye-screen distance and its dynamics, and the fraction of missing samples in each channel.”
- **Gašper:** “Put your latest paper or one of your references here. Because someone might wonder more about features.”
- **Tomi / instruction to Claude Code:** _Add your response here._ This is this paper: Tomi Božak, Shivalika Goyal, Marc Langheinrich, Martin Gjoreski, and
- **Implemented (2026-08-28):** **Done.** `\cite{bozak2026workload}` added to the handcrafted-feature description: "...following the feature set of our earlier eye-tracking work".
Gašper Slapničar. 2026. Evaluating feature-based machine-learning models
with post hoc explainability for eye-tracking-based task type and workload
inference. AI, 7, 8, 325. doi: 10.3390/ai7080325.

### 34. Unify the target terminology in Experimental Design

- **PDF page:** 2
- **Source:** `paper/sections/03_methodology.tex:58`
- **Selected text:** “Engagement ratings”
- **Gašper:** “absorption, engagement, immersion...? unify.”
- **Tomi / instruction to Claude Code:** _Add your response here._ indeed, unify.
- **Implemented (2026-08-28):** **Done** -- see 28.

## Results, Discussion, and Conclusion

### 35. Make Table 1 full text width

- **PDF page:** 3
- **Source:** `paper/sections/04_results.tex:16-50`
- **Selected text:** Entire Table 1 body, from “Majority baseline” through the final “MOMENT embeddings / MLP” row.
- **Gašper:** “Make this full text width (will make everything slightly larger probably?)”
- **Tomi / instruction to Claude Code:** _Add your response here._ yes.
- **Implemented (2026-08-28), then reverted on Tomi's instruction:** Table 1 was first made full text width (`table*` with `tabularx`), but Tomi asked for it to be **one column wide, not page wide**, so it is now a single-column `table` stretched to `\columnwidth` with `tabular*` + `\extracolsep{\fill}`. To fit one column it uses combined `Representation, Model` row labels instead of the four group-header rows, with the per-representation dimensionality moved into the caption. Column order leads with the primary metric: Bal. acc. | SD | M-F1 | AUC | Acc. | Delta.

### 36. Statistical significance of the main results

- **PDF page:** 3
- **Source:** `paper/sections/04_results.tex:52-57`
- **Selected text:** “Table 1 reports the main comparison. Every representation-model combination sits close to the majority baseline. The highest observed result is the raw signal with the MLP, at 0.538 accuracy, 0.014 above the paired baseline, and it is also highest on balanced accuracy (0.536), macro-F1 (0.468) and ROC-AUC (0.551). Handcrafted features give the next-highest balanced accuracy, macro-F1 and ROC-AUC, and improve on the baseline accuracy with both classifiers. GazeMAE reaches 0.535 accuracy with the random forest and falls to 0.520, below the baseline, with the MLP. MOMENT falls below the baseline accuracy with both classifiers, while its balanced accuracy and ROC-AUC remain close to chance.”
- **Gašper:** “I hope people wont nag with statistical significance here...”
- **Tomi / instruction to Claude Code:** _Add your response here._ if easy fix, make it, otherwise ignore for now.
- **Implemented (2026-08-28):** **Addressed through 37**, which was the cheap fix. No significance tests were added (that stays consistent with the no-added-inferential-statistics rule for this paper).

### 37. Acknowledge the tentative interpretation

- **PDF page:** 3
- **Source:** `paper/sections/04_results.tex:64`
- **Selected text:** “tentative.”
- **Gašper:** “\"...and requires further statistical validation\"? Do we want to say it? I feel it in your sentence already anyway, just a question if we want to acknowledge it to avoid the question :) But maybe not, idk.”
- **Tomi / instruction to Claude Code:** _Add your response here._ yes, add.
- **Implemented (2026-08-28):** **Done.** "...so this interpretation remains tentative and requires further statistical validation."

### 38. Overall assessment of the Discussion

- **PDF page:** 3
- **Source:** `paper/sections/05_conclusion.tex:1`
- **Selected text:** “Discussion”
- **Gašper:** “Good discussion overall.”
- **Tomi / instruction to Claude Code:** _Add your response here._ ignore comment, nothing to change.
- **Implemented (2026-08-28):** No action needed.

### 39. Practical-use wording

- **PDF page:** 3
- **Source:** `paper/sections/05_conclusion.tex:13`
- **Selected text:** “For practice,”
- **Gašper:** “For practical use, ...”
- **Tomi / instruction to Claude Code:** _Add your response here._ yes
- **Implemented (2026-08-28):** **Done.** "For practical use, ...".
