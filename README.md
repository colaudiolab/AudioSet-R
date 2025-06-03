# AudioSet-R
Official implementation: "AudioSet-R: A Refined AudioSet with Multi-Stage LLM Label Reannotation"

## :loudspeaker: News:
March, 2025: We submitted our paper on the AudioSet-R Dataset to the ACM MM2025 Dataset track.

## :information_desk_person: Overview:

![image](https://github.com/colaudiolab/AudioSet-R/blob/main/Illustration/Flowchart.png)

***Illustrates the proposed three-stage relabeling framework for AudioSet.***

## :musical_score: Datasets:

The dataset used in this study is AudioSet dataset: **Balanced training set (20550 .wav files)**, **Evaluate set (18885 .wav files)**.

***AudioSet official website***: [AudioSet](https://research.google.com/audioset//index.html), [Github](https://github.com/audioset/ontology)

We provide the json file of the audioset-R audio label and the json file of the original AudioSet audio label:

:+1:***AudioSet-R***: The json file of [balanced training set](https://github.com/colaudiolab/AudioSet-R/blob/main/AudioSet-R_train.json) and [evaluate set](https://github.com/colaudiolab/AudioSet-R/blob/main/AudioSet-R_eval.json)

***Original AudioSet***: The json file of [balanced training set](https://github.com/colaudiolab/AudioSet-R/blob/main/balanced_train.json) and [evaluate set](https://github.com/colaudiolab/AudioSet-R/blob/main/evaluate_set.json)

## Example:

![image](https://github.com/colaudiolab/AudioSet-R/blob/main/Illustration/case.png)

***The detailed analysis for three-round audio content extraction.***

## :tada: Statistic:
We provide a comparison of the training and fine-tuning performance of AudioSet-R on various supervised and self-supervised audio classification models:
![image](https://github.com/colaudiolab/AudioSet-R/blob/main/Illustration/Statistic.png)

## :runner: Run:
If you want to perform label prediction on other datasets, please follow the steps below:

`cd ./pipeline`

***Please read*** [`./pipeline/README.md`](https://github.com/colaudiolab/AudioSet-R/blob/main/pipeline/README.md)

## Cites:
If you found this repo is helpful, please consider citing our papers:
```

```
