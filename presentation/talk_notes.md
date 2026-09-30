# SKUI 2026 talk — what to say

Target: about 12 minutes (~1 min per slide) plus questions. Slide 10 is backup.

## 1. Title
- Thank the chair; I'm Tomi Božak from the Department of Intelligent Systems, JSI.
- Joint work with Zoja Anžur and Gašper Slapničar.
- Topic: which way of representing eye-tracking data works best for predicting engagement at work.

## 2. Introduction
- Engagement (immersion in work) changes during the day; application logs show little of it.
- A screen-mounted eye tracker records gaze and pupil size continuously, without the user doing anything.
- Before training any model we must choose how to represent a window of gaze data.
- Options: raw signal, handcrafted features, or embeddings from pretrained models.
- Published results use different data, labels and splits, so they can't be compared directly — that's our gap.

## 3. Four representations
- Every model sees the same 3-second windows: pupil size (L/R), gaze x/y, eye–screen distance.
- Raw: the signal itself, resampled and flattened — no domain knowledge.
- Handcrafted: blinks, fixations, saccades, pupil statistics — interpretable, from our earlier work.
- GazeMAE: pretrained on eye-tracking data; MOMENT: general time-series foundation model. Both used frozen, out of the box.
- On top: the same classifiers — random forest and an MLP.

## 4. Data
- Real-world data: knowledge workers recorded at their own desks during normal workdays.
- 17 participants, 657 full workdays; eye tracker at 60 Hz.
- Several times a day they answered a short questionnaire; our target: “Right now I am immersed in my work”, 0–6.
- Each answer labels the 15 minutes before it.
- Left plot: gaze mostly in the screen centre. Right: ratings; red ticks show each participant's mean — very different between people.

## 5. Windows and target
- Recordings cut into 3-second windows with 25 % overlap; windows with too much tracking loss removed → about 140 thousand windows.
- Ratings differ a lot between people (means 2.2–4.9), so an absolute threshold would mostly learn who the person is.
- So we centre ratings per participant and predict above vs. below their own mean.
- For the test participant we use the average of the training participants' means — no test labels leak in.

## 6. Methodology
- Three predictors on every representation: majority baseline, random forest and an MLP.
- Leave-one-subject-out: train on 16 people (green), test on the one left out (orange); repeat 17 times.
- Imputation, scaling and feature selection are fitted on training people only — no leakage.
- Main measure is balanced accuracy, because class balance differs per person; chance is always 0.5.

## 7. Results
- Bars start at 0.5 = chance; above the line is better than guessing.
- Raw signal is best with both classifiers: 0.536 with random forest, 0.534 with MLP (best AUC 0.553).
- Handcrafted next (0.521 with MLP); GazeMAE just above chance; MOMENT at chance.
- Honest part: everything is close to chance, and the spread across participants (SD 0.075) is bigger than the differences.
- Full table with all metrics on the backup slide.

## 8. t-SNE
- Same four representations projected to 2-D.
- Top row coloured by engagement rating: no region follows engagement in any representation.
- Bottom row coloured by participant: clear clusters in raw, weaker in handcrafted and GazeMAE, almost none in MOMENT.
- So the strongest signal in gaze is who the person is, not how engaged they are — this is why testing on unseen people matters.

## 9. Conclusion
- Under a fair, subject-disjoint comparison, the simplest input — raw signal — gave the best result; handcrafted close behind.
- Frozen pretrained encoders did not help: likely a mismatch — trained on lab eye-tracking or generic time series, not day-long office gaze.
- Predicting engagement in new people from 3-s windows in real office work is hard.
- Practical advice: start with raw or handcrafted features; don't expect a frozen encoder to help.
- Next: fine-tune the encoders, personalise to new users, try longer windows.
- Thank you — happy to take questions.

## 10. Backup: all metrics
- Only if asked. Paper Table 1.
- GazeMAE RF has good plain accuracy (.540), but plain accuracy partly reflects each person's class balance — that's why balanced accuracy is primary.
