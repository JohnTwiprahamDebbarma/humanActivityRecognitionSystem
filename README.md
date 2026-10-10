# Human Activity Recognition System

Human Activity Recognition (HAR) is the task of identifying what a person is doing
from sensor data. This project recognises six activities from the accelerometer of a
smartphone, using Decision Trees:

`WALKING`, `WALKING_UPSTAIRS`, `WALKING_DOWNSTAIRS`, `SITTING`, `STANDING`, `LAYING`

The models are trained on the public [UCI-HAR dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
and then tested on data collected separately with a mobile phone. The repository also
contains a Decision Tree implemented from scratch.

## What has been done, and where

| Step | What is done | Where |
|---|---|---|
| 1 | The windows of the UCI-HAR dataset are joined into one continuous signal for every participant and activity | `CombineScript.py`, input in `UCI HAR Dataset/`, output in `Combined/` |
| 2 | A 10 second sample is taken from every signal, and the samples are split into train, test and validation sets | `MakeDataset.py` |
| 3 | The dataset is loaded and explored for the first time | `testing.ipynb` |
| 4 | The waveforms of the six activities are plotted and compared | `1_HAR_Analysis.ipynb`, Section 1 |
| 5 | Static and dynamic activities are compared using the linear acceleration | `1_HAR_Analysis.ipynb`, Section 2 |
| 6 | A Decision Tree is trained on the raw data, for one depth and then for depths 2 to 8 | `1_HAR_Analysis.ipynb`, Sections 3 and 4 |
| 7 | The raw data and the TSFEL features are each reduced to two features with PCA and plotted | `1_HAR_Analysis.ipynb`, Section 5 |
| 8 | A Decision Tree is trained on the TSFEL features and compared with the raw data, and the final model is saved | `1_HAR_Analysis.ipynb`, Section 6, saved in `Final Model/` |
| 9 | The activities on which the final model performs badly are found | `1_HAR_Analysis.ipynb`, Section 7 |
| 10 | The depth of the tree is chosen again using the validation set | `2_HAR_Depth_Selection.ipynb` |
| 11 | Data is collected with a mobile phone, preprocessed, and used to test the saved models, with the featurized data and with the raw data | `3_HAR_Deployment.ipynb`, `Collected Data/` |
| 12 | A Decision Tree is implemented from scratch | `Decision Tree/` |

## Results

| Model | Tested on | Accuracy |
|---|---|---|
| Decision Tree (depth 9) on the raw data | UCI-HAR test set, 36 samples | 52.8% |
| Decision Trees of depths 2 to 8 on the raw data | UCI-HAR test set | 44.4% to 52.8% |
| Decision Tree on the raw data, depth 4 chosen on the validation set | UCI-HAR test set | 61.1% |
| **Decision Tree (depth 4) on the TSFEL features after PCA (final model)** | UCI-HAR test set | **69.4%** |
| Final model, featurized data | Collected data, 24 recordings | 33.3% |
| Decision Tree on the raw data | Collected data, 24 recordings | 20.8% |

Guessing one of the six activities at random gives 16.7%.

- Static and dynamic activities can be told apart without any machine learning model.
- Featurizing the data with TSFEL works better than using the raw data: 69.4% against 52.8%.
- The final model confuses the dynamic activities with each other, and the static
  activities with each other, but never a static activity with a dynamic one. Walking
  upstairs is its worst activity.
- The models trained on the UCI-HAR dataset do not work on the collected data. The main
  reason is that the phone app records the acceleration without gravity, whereas the
  UCI-HAR data includes gravity.

## Repository structure

```
CombineScript.py              joins the UCI-HAR windows into one signal per participant and activity
MakeDataset.py                takes the 10 second samples and makes the train, test and validation sets
testing.ipynb                 first experiments with the dataset
1_HAR_Analysis.ipynb          analysis of the data, Decision Trees on the raw data and on the TSFEL features
2_HAR_Depth_Selection.ipynb   choosing the depth of the Decision Tree using the validation set
3_HAR_Deployment.ipynb        testing the saved models on the data collected with a mobile phone
UCI HAR Dataset/              the downloaded UCI-HAR dataset, read by CombineScript.py
  train/                      data of 21 participants, raw signals in Inertial Signals/
  test/                       data of 9 participants, raw signals in Inertial Signals/
Combined/                     output of CombineScript.py: 180 csv files (accx, accy, accz)
  Train/<ACTIVITY>/Subject_<n>.csv    21 participants
  Test/<ACTIVITY>/Subject_<n>.csv     9 participants
Collected Data/               24 recordings collected with a mobile phone, 4 for each activity
Final Model/                  models saved by 1_HAR_Analysis.ipynb and loaded by 3_HAR_Deployment.ipynb
  pca_features.sav            PCA fitted on the TSFEL features
  final_tree.sav              final Decision Tree (depth 4), trained on the two PCA features
  raw_tree.sav                Decision Tree trained on the raw data
Decision Tree/                Decision Tree implemented from scratch
  base.py                     the DecisionTree class
  utils.py                    entropy, gini index and information gain
  metrics.py                  accuracy, precision, recall, RMSE, MSE and MAE
  usage.py                    four test cases
UCI HAR Dataset.names         description of the dataset from its authors
requirements.txt              libraries needed to run the code
```

## Dataset

The [UCI-HAR dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
contains data from 30 participants. Each participant performed the six activities while
wearing a Samsung Galaxy S II smartphone on the waist. The accelerometer and the
gyroscope of the phone were recorded at 50 Hz, and the recordings are published as
windows of 2.56 seconds (128 samples) which overlap by 50%.

The downloaded dataset is included in the `UCI HAR Dataset` folder, as published by its
authors, so nothing needs to be downloaded to run the code. It has a `train` folder
(21 participants, 7352 windows) and a `test` folder (9 participants, 2947 windows):

| Files | Contents |
|---|---|
| `Inertial Signals/total_acc_*.txt` | total acceleration along the x, y and z axes, one window in each row |
| `Inertial Signals/body_acc_*.txt`, `Inertial Signals/body_gyro_*.txt` | body acceleration and angular velocity |
| `subject_*.txt`, `y_*.txt` | participant and activity label of each window |
| `X_*.txt` | 561 features calculated by the authors of the dataset |
| `README.txt`, `features.txt`, `features_info.txt`, `activity_labels.txt` | description of the dataset |

This project uses only the raw accelerometer data, that is, the total acceleration
along the x, y and z axes from the `Inertial Signals` folder of the dataset, together
with the participant and label files. It is measured in units of g, so it includes
gravity. The body acceleration, the gyroscope data and the 561 features of the authors
are not used.

## 1. Preprocessing

### `CombineScript.py`

**What:** For every participant and every activity, the script collects all the windows
and joins them into one continuous signal. Since consecutive windows overlap by 50%,
only the second half of each window (samples 64 to 128) is kept, so that no sample is
repeated. The signals are saved as csv files with the columns `accx`, `accy` and `accz`
in the `Combined` folder, with one folder for each activity.

**Why:** The published windows are only 2.56 seconds long. Joining them gives a long
signal for each participant and activity, from which a sample of any length can be cut.

The `Combined` folder is already included in this repository, so the script does not
need to be run. To run it again, see [How to run](#how-to-run).

### `MakeDataset.py`

**What:** The script reads the 180 csv files and keeps 500 samples from each, starting
from sample 100. At 50 Hz this is 10 seconds of activity. The result is an array of
shape `(180, 500, 3)`: 180 samples (30 participants and 6 activities), 500 time steps and 3 axes. The
activities are labelled 1 to 6:

| Label | Activity |
|---|---|
| 1 | WALKING |
| 2 | WALKING_UPSTAIRS |
| 3 | WALKING_DOWNSTAIRS |
| 4 | SITTING |
| 5 | STANDING |
| 6 | LAYING |

The 180 samples are then split into train, test and validation sets of 108, 36 and 36
samples (60%, 20% and 20%). The split is stratified, so every activity has the same
number of samples in each set.

**Why:** A fixed sample of 10 seconds for every participant and activity gives a small
and balanced dataset. A fixed random seed and a sorted list of files make the split the
same on every run and on every computer, so the results can be reproduced.

The notebooks import the sets directly:

```python
from MakeDataset import X_train, X_test, y_train, y_test, X_val, y_val
```

## 2. First experiments: `testing.ipynb`

**What:** The notebook in which the dataset was explored for the first time. It loads the
train, test and validation sets, checks their shapes, groups the samples of the train
set by activity using their labels, and makes the first 3D plots of the acceleration
along the three axes for each activity.

**Why:** It shows how the experiments started, before the analysis in
`1_HAR_Analysis.ipynb`.

**Result:** The train set has 108 samples, 18 for each activity, and the test and
validation sets have 36 samples each, which makes 180 in total. Every sample has 500
time steps and 3 axes. In the 3D plots, the points of Laying, Sitting and Standing stay
in small clusters, whereas the points of the three walking activities are spread out.
This difference between the static and the dynamic activities is looked at in detail in
`1_HAR_Analysis.ipynb`.

## 3. Analysis and models: `1_HAR_Analysis.ipynb`

### Section 1: Waveform of each activity

**What:** The acceleration of each activity is plotted in two ways: as points in 3D
(a_x, a_y, a_z), and against time for a_x, a_y, a_z and the linear acceleration.

**Why:** To see the differences and similarities between the activities before training
any model.

**Result:** The static activities (sitting, standing, laying) are almost constant, and
the dynamic activities (walking, walking upstairs, walking downstairs) are scattered.
The static activities also differ in their mean values: a_x is almost zero while
laying, and a_y is around 0.25 while sitting. The three dynamic activities look similar
to each other and are harder to tell apart.

### Section 2: Static v/s dynamic activities

**What:** The linear acceleration acc_x² + acc_y² + acc_z² is plotted for every sample
of each activity, along with its mean and standard deviation for each activity.

**Why:** To find out whether a machine learning model is needed to separate the static
activities from the dynamic ones.

**Result:** No model is needed. The linear acceleration of the static activities stays
close to 1 (only gravity), with a standard deviation of about 0.02. For the dynamic
activities the standard deviation is about 0.5 to 0.8. A threshold on the standard
deviation is enough to separate the two groups.

### Section 3: Decision Tree on the raw data

**What:** A Decision Tree (entropy criterion, depth 9) is trained on the linear
acceleration timeseries, where each of the 500 time steps is one feature. It is trained
on the train and validation sets together (144 samples) and tested on the test set (36
samples). The tree is saved as `Final Model/raw_tree.sav`.

**Why:** This is the baseline: how well can a Decision Tree do on the data as it is?
The tree is saved so that it can be tested on the collected data later.

**Result:** 52.8% accuracy. The confusion matrix shows that the errors are mostly within
the dynamic activities and within the static activities.

### Section 4: Decision Trees of varying depths

**What:** Decision Trees of depths 2 to 8 are trained on the same raw data. The accuracy
and the confusion matrix are reported for each depth, and the accuracies are plotted.

**Why:** To see whether the accuracy changes when the depth is increased.

**Result:** The accuracy is low for small depths (44.4% at depth 2), increases to 52.8%
at depths 4 and 5, and then stays at 50% for depths 6 to 8. A tree of depth 2 has at
most four leaves and cannot separate six activities. After that, a deeper tree does not
help, because the value of the signal at a single time step is not a useful feature:
the steps of different people do not happen at the same time.

### Section 5: PCA on the raw data and on the TSFEL features

**What:** PCA is used to compress the 500 time steps of the linear acceleration into two
features, which are shown in a scatter plot. Then [TSFEL](https://tsfel.readthedocs.io/en/latest/)
is used to extract features from each timeseries (statistical, temporal and spectral
features), PCA is applied to these features to obtain two features, and they are shown
in a scatter plot again. The fitted PCA is saved as `Final Model/pca_features.sav`.

**Why:** The scatter plots show how well the activities are separated in each case. The
PCA is saved because the same transformation has to be applied to the collected data.

**Result:** With PCA on the raw data, the activities are not separated. With PCA on the
TSFEL features, the static activities form one tight cluster, away from the dynamic
activities, and walking downstairs is separated from the other two dynamic activities.

### Section 6: Decision Tree on the TSFEL features

**What:** Decision Trees of depths 2 to 8 are trained on the two PCA features. They are
evaluated with 10-fold cross validation and on the test set, and the test accuracies
are plotted against those of the raw data from Section 4. The tree of depth 4 is taken
as the final model and saved as `Final Model/final_tree.sav`.

**Why:** To see whether featurizing works better than using the raw data.

**Result:** Featurizing works better. The accuracy on the test set is 69.4% at depths 3
and 4, against a best of 52.8% for the raw data. The mean cross validation accuracy is
between 61% and 72%. For depths above 4 the test accuracy is lower again, because a
deep tree on two features starts to fit single samples.

### Section 7: Activities where the model performance is bad

**What:** The confusion matrix and the classification report of the final model.

**Result:** The worst activity is walking upstairs (F1 score of 0.46), followed by
walking (0.62). The model confuses the dynamic activities with each other. Within the
static activities, some laying and standing samples are predicted as sitting. A static
activity is never confused with a dynamic one.

## 4. Choosing the depth using the validation set: `2_HAR_Depth_Selection.ipynb`

**What:** Decision Trees of depths 2 to 8 are trained on the train set alone and
compared on the validation set, using accuracy, precision, recall and F1 score. The
tree of the best depth is then trained on the train and validation sets together and
tested on the test set.

**Why:** In Section 4 of `1_HAR_Analysis.ipynb` the depths are compared on the test set. Choosing the depth
on the validation set keeps the test set for the final measurement only.

**Result:** The best depth on the validation set is 4. This tree has an accuracy of
61.1% on the test set.

## 5. Deployment: `3_HAR_Deployment.ipynb`

### Data collection

| | |
|---|---|
| Activities | Walking, Walking Up, Walking Down, Laying, Sitting, Standing |
| Device | OnePlus mobile phone |
| App | Physics Toolbox Suite |
| Sampling frequency | about 200 Hz |
| Recordings | 24: 4 for each activity, between 13.5 and 20 seconds long |

The recordings are in the `Collected Data` folder, named by activity and number, for
example `Sitting1.csv` and `WalkingUp3.csv`. Each file has the columns `time`,
`ax (m/s^2)`, `ay (m/s^2)`, `az (m/s^2)` and `aT (m/s^2)`, where `aT` is the magnitude
of the acceleration.

`Walking1.csv` contains a second, unrelated capture after a block of corrupt bytes,
where the time starts again. The notebook keeps only the first capture, which is the
walking recording.

### Preprocessing

**What:** Every recording goes through the same steps:

1. The magnitude `aT` is squared, to get the linear acceleration that the models were
   trained on.
2. A Butterworth band-pass filter (3 to 20 Hz) and a median filter are applied.
3. The data is resampled from about 200 Hz to 50 Hz, and missing values are filled by
   linear interpolation.
4. The part between 2 seconds and 12 seconds is kept, which gives 10 seconds of data.

**Why:** The models were trained on 10 second samples at 50 Hz, so the collected data
has to be brought to the same form. The filters reduce the noise. The first 2 seconds
and the end of each recording are removed because the phone is being handled then.

### Testing the models

**What:** The models saved by `1_HAR_Analysis.ipynb` are tested on the 24 recordings in
two ways:

- **Featurized data:** the TSFEL features are passed through the saved PCA and given to the final Decision Tree.
- **Raw data:** the timeseries itself is given to the Decision Tree trained on the raw data.

The accuracy, the confusion matrix and the classification report are reported for both.

**Result:**

| | Accuracy | What the model predicts |
|---|---|---|
| Featurized data | 33.3% (8 of 24) | all the static recordings as Laying, all the stair recordings as Walking Down, walking as Standing or Laying |
| Raw data | 20.8% (5 of 24) | 22 of the 24 recordings as Walking Down |

The models do not recognise the activities of the collected data. The featurized model
gives only three different answers, and it is correct only for the recordings whose
activity happens to be one of those answers. It does still separate the static
recordings from the stair recordings.

**Why it failed:**

- The app records the linear acceleration with gravity removed, whereas the total
  acceleration of the UCI-HAR dataset includes gravity. A phone at rest reads 0 in the
  collected data and 1 g in the UCI-HAR data.
- The units are different: m/s² in the collected data, and g in the UCI-HAR data.
- The filters applied to the collected data are not the same as the processing of the
  UCI-HAR data.
- The sensor configurations may be different (the sensitivity, the inbuilt filters and
  the range of operation).
- The phone was not positioned in the same way as the waist-mounted phone of the
  UCI-HAR dataset.
- The difference in walking style (a minor reason).

## 6. Decision Tree from scratch: `Decision Tree/`

**What:** A Decision Tree written without using any machine learning library. It works
for all four cases:

- discrete input, discrete output
- real input, real output
- real input, discrete output
- discrete input, real output

| File | Contents |
|---|---|
| `base.py` | The `Node` and `DecisionTree` classes: `fit()` builds the tree recursively, `predict()` runs it on test inputs and `plot()` prints it |
| `utils.py` | Check for real or discrete values, entropy, gini index, variance and information gain |
| `metrics.py` | Accuracy, precision, recall, RMSE, MSE and MAE |
| `usage.py` | Four test cases, one for each case above, with both criteria |

How it works:

- A column is treated as discrete if its dtype is `category`, and as real otherwise.
- For a discrete output the split is chosen by information gain (entropy) or by gini
  index. For a real output it is chosen by the reduction in mean squared error.
- A discrete feature is split into one branch for each of its values. A real feature is
  split into two branches at a threshold, and the thresholds tried are the midpoints
  between consecutive values of the feature.
- A leaf predicts the most common class for a discrete output, and the mean for a real
  output.
- The tree stops growing at `max_depth` (5 by default), or when all the samples of a
  node have the same output.

**Why:** To understand how a Decision Tree works by building one, instead of only using
a library.

The HAR notebooks above use the Decision Tree of scikit-learn. This implementation is a
separate part of the project and is not used in them.

To run the test cases:

```bash
cd "Decision Tree"
python usage.py
```

Part of the output, for discrete input and discrete output:

```
?(X1 in [3 0 4 1 2])
3: 	?(X4 in [4 2 0 3])
	4: 		?(X0 in [0 1 3])
		0: 			Class/Value 0
		1: 			Class/Value 2
		3: 			Class/Value 0
	2: 		Class/Value 2
	...
Criteria : information_gain
Accuracy:  0.9666666666666667
```

## Limitations

- The dataset is small: 180 samples, of which 36 are in the test set. One test sample is
  2.8% of accuracy, so small differences in accuracy are not significant.
- The split is over samples and not over participants, so the same participants are in
  the train and test sets, with different activities.
- Only the linear acceleration acc_x² + acc_y² + acc_z² is used. It does not contain the
  direction of gravity, which is what distinguishes sitting, standing and laying.
- PCA is applied to the TSFEL features without scaling them, so the two PCA features are
  dominated by the TSFEL features with the largest values.
- The data was collected with the linear accelerometer of the app, which removes
  gravity. To test the models properly, the total acceleration including gravity should
  be recorded, in the same units and with the phone in the same position as in the
  UCI-HAR dataset.

## How to run

Install the libraries:

```bash
pip install -r requirements.txt
```

Open the notebooks from this folder, so that `MakeDataset.py`, `Combined/`,
`Collected Data/` and `Final Model/` are found:

```bash
jupyter notebook
```

Run `1_HAR_Analysis.ipynb` first. It saves the models in `Final Model/`, which
`3_HAR_Deployment.ipynb` loads. The saved models are included, so the deployment
notebook can also be run directly. If scikit-learn reports a version mismatch when
loading them, run `1_HAR_Analysis.ipynb` once to save them again.

To make the `Combined` folder again from the dataset in `UCI HAR Dataset/`:

```bash
python CombineScript.py
```

The outputs in the notebooks were produced with Python 3.14, numpy 2.5, pandas 3.0,
scikit-learn 1.9, scipy 1.18, matplotlib 3.11, seaborn 0.13, plotly 7.1 and TSFEL 0.2.
The number of TSFEL features and the exact accuracies can change with other versions.
The interactive plotly plots are shown when the notebooks are run, but not in the
GitHub preview.

## Citation

Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. Reyes-Ortiz.
*A Public Domain Dataset for Human Activity Recognition Using Smartphones.* 21st
European Symposium on Artificial Neural Networks, Computational Intelligence and
Machine Learning (ESANN 2013), Bruges, Belgium, 24-26 April 2013.

## License

The code is under the MIT license. See [LICENSE](LICENSE).

The UCI-HAR dataset in `UCI HAR Dataset/`, and the `Combined/` folder made from it,
belong to the authors of the dataset. Their terms are given in
`UCI HAR Dataset/README.txt`: the publication above has to be cited, and commercial use
is not allowed.
