# Vision as Teacher — Human Activity Recognition

Activity recognition from a waist-mounted phone, rebuilt around the question that
decides whether any of the numbers mean anything: **does the model work on a person
it has never seen?**

Under leave-one-subject-out over all 30 participants, a linear model on 288
hand-crafted features reaches **94.2 macro-F1**. The same model on a random window
split reports 97.8 — the 3.5-point difference is the cost of letting one
participant appear on both sides of the train/test line.

> **Status.** The sensor-only pipeline below is complete and reproducible. The
> vision arm and cross-modal distillation are planned, not built.

## Data

[UCI HAR](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones):
30 participants, six activities, a Samsung Galaxy S II worn at the waist. The
archive ships pre-segmented inertial signals — 2.56 s windows at 50 Hz (128
samples) across 9 channels: body acceleration, angular velocity and total
acceleration on x/y/z. **10,299 labelled windows**, each tagged with the
participant who produced it.

The dataset is committed, so a clone runs with no download step. It is stored
repacked rather than as the published text files, which cost 364 MB to say the
same thing:

| | Published | Committed | Contents |
|---|---:|---:|---|
| `data/uci_har_inertial.npz` | 190.0 MB | **24.3 MB** | the 9-channel windows every model here trains on |
| `data/uci_har_features561.npz` | 92.5 MB | **19.2 MB** | the authors' own 561 engineered features, for comparison |
| `data/combined.zip` | 20.5 MB | **8.0 MB** | per-subject traces, read by the archived notebooks |
| `data/self_collected/` | — | 3.5 MB | 24 phone recordings collected |
| `data/docs/` | — | 25 KB | the dataset's own README, feature list and label key |


### Citation

> Anguita, D., Ghio, A., Oneto, L., Parra, X., & Reyes-Ortiz, J. L. (2013).
> *A Public Domain Dataset for Human Activity Recognition Using Smartphones.*
> 21st European Symposium on Artificial Neural Networks (ESANN 2013).


## License

MIT — see [LICENSE](LICENSE).
