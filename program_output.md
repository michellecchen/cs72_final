# Program outputs 

We have three sorts of output which might be relevant for the reader. The first two pertain to the accuracy of our machine learning models (BART zero shot classification and BERT sentiment analysis). For zero-shot classification, we will provide screenshots of both some of the classifiers for both positive and negative sentences for each of the companies, as well as a confusion matrix which evaluates the performance of the classifier in general. This data is also available as `.txt` in the directory `confusionData`. For BERT sentiment analysis, we will provide screenshots of classification reports (accuracy, precision, recall, f1-score, and support) of the fine-tuning results with various sets of parameters.

We also show the output that these two algorithms have been used to work toward; a series of timelines which aggregate sentiments over time for each company, per topic (diversity and inclusion, culture and values, work life balance, senior management, career opportunities, compensation and benefits), in relation to major lawsuits brought against the company. For each company, we will visualize the gold labeled (pros/cons sentences from _Glassdoor_ and _Indeed_) and BERT-predicted data (neutral review bodies from _Indeed_) separately; the comparison between timelines will, in turn, serve to validate our BERT model's sentiment predictions.

## Timelines

All of these images, as well as their accompanying textual descriptions, can be browsed in our [Figma](https://www.figma.com/file/7mWaWkFRK852axe3XGD6iB/Temporally-located-sentiment-analysis-on-gaming-company-employee-reviews-(CS72)%3A-Timelines?node-id=0%3A1). We highly recommend taking a look: all of these following images can be viewed at their highest possible resolution. The textual descriptions contains clickable in-line links to sources.

### Riot Games

**Gold labels**

![](https://i.imgur.com/YXied4M.png)

**Unlabeled**

![](https://i.imgur.com/BWifANR.png)

**Timeline of events**

![](https://i.imgur.com/Vu14Gyp.png)

### Blizzard

**Gold labels**

![](https://i.imgur.com/cy63na4.png)

**Unlabeled**

![](https://i.imgur.com/CUtf12p.png)

**Timeline of events**

![](https://i.imgur.com/BLEwFXY.png)

### Sony

**Gold labels**

![](https://i.imgur.com/GlEXbO6.png)

**Unlabeled**

![](https://i.imgur.com/SzqnBEE.png)

**Timeline of events**

![](https://i.imgur.com/4Tm9qTv.png)

### Ubisoft

**Gold labels**

![](https://i.imgur.com/YRp8hgQ.png)

**Unlabeled**

![](https://i.imgur.com/qOXvUXJ.png)

**Timeline of events**

![](https://i.imgur.com/tjYKiGK.png)

## Zero shot classification with BART
### Classifiers (50 for each company, 25 per valence)

Riot positive sentences:
![riot pos](https://github.com/michellecchen/cs72_final/blob/main/confusionData/screenshots/riotPosSS.png?raw=true)

Riot negative sentences:
![riot neg](https://github.com/michellecchen/cs72_final/blob/main/confusionData/screenshots/riotNegSS.png?raw=true)

Sony positive sentences:
![sony pos](https://github.com/michellecchen/cs72_final/blob/main/confusionData/screenshots/sonyPosSS.png?raw=true)

Sony negative sentences:
![sony neg](https://github.com/michellecchen/cs72_final/blob/main/confusionData/screenshots/sonyNegSS.png?raw=true)

Ubisoft positive sentences:
![ubisoft pos](https://github.com/michellecchen/cs72_final/blob/main/confusionData/screenshots/ubisoftPosSS.png?raw=true)

Ubisoft negative sentences:
![ubisoft neg](https://github.com/michellecchen/cs72_final/blob/main/confusionData/screenshots/ubisoftNegSS.png?raw=true)

Activision Blizzard positive sentences:
![activision pos](https://github.com/michellecchen/cs72_final/blob/main/confusionData/screenshots/actPosSS.png?raw=true)

Activision Blizzard negative sentences:
![activision neg](https://github.com/michellecchen/cs72_final/blob/main/confusionData/screenshots/actNegSS.png?raw=true)


### Confusion matrix 
We labeled 185 sentences by hand according to topic, then compared them with the predicted labels. As you can see, labels 2 and 6, culture and values and compensation and benefits, are the most contentious, with the highest amount of misclassifications. 

![zero shot confusion matrix](https://github.com/michellecchen/cs72_final/blob/main/confusionData/screenshots/confusionMatrix.png?raw=true)

Here we can clearly see the precision, recall, f1 score, and support per label. 

![riot pos](https://github.com/michellecchen/cs72_final/blob/main/confusionData/screenshots/classificationFromMatrix.png?raw=true)

Out of 61 sentences that were labeled by us as pertaining to culture and values, only 26 were classified by the model as such; this is most likely because we placed many statements which we found to be ambiguous in this category, whereas the model made a more definitive decision. For example, we might classify a sentence like "Great company to work for" as culture and values, whereas the model might classify it as career opportunities. 

Out of 46 sentences we labeled as pertaining to compensation and benefits, about half were labeled as that, whereas half were labeled as work life balance or culture and values. We used the compensation and benefits label as a sort of trash bin for sentences that we weren't sure how to label, which weren't relevant to any label ("No cons to speak of"), and sentences that weren't actually coherent sentences (one memorable "sentence" was simply "g"). Since the classifier must make a determination and doesn't "throw away" any sentences, this label was bound to have a lower recall.


## BERT sentiment analysis

### Running on CPU
Classification report for BERT fine-tuned with two epochs, 75:16:8 training:validation:testing data split, and 12300 total sentences:

(note: the split was originally 70:15:15, but we were unable to run predictions on the 15% testing dataset using CPU + the normal version of Colaboratory, so we halved the testing data)

![two epochs](https://github.com/michellecchen/cs72_final/blob/main/BERTscores/twoEpochFullTrainValHalfTest.png?raw=true)


Classification report for BERT fine-tuned with four epochs, 70:15:15 training:validation:testing data split, and 5171 total sentences:

![four epochs](https://github.com/michellecchen/cs72_final/blob/main/BERTscores/fourEpochsHalfEverything.png?raw=true)

### Running on GPU, various batch sizes

As you can see, we were only able to achieve 65% accuracy while running with up to four epochs. Each of those CPU training runs also took more than four hours. To be able to run with more epochs, we reduced our batch sizes and asked some friends with GPUs to run the GPU-pushed version of our notebook. On GPU, we were able to train the BERT with batch size 32 and 6 epochs, for example, in around fifteen minutes--a massive improvement over the hours it took to fine-tune BERT on CPU for even a few epochs. We also tried experimenting with DistilBERT--fine-tuning DistilBERT ran even faster, but to lesser accuracy. 

The following fine-tuned models were all tested on 1000 samples owing to GPU memory constraints. They were all trained on a total of 12300 sentences, split 70:15:15 training:validation:testing. 

#### DistilBERT

Report for DistilBERT fine-tuned with 4 epochs (batch size 32):

![four epochs](https://github.com/michellecchen/cs72_final/blob/main/BERTscores/distilBERT4epochs32batch1000s.png?raw=true)

Report for DistilBERT fine-tuned with 10 epochs (batch size 4:

![ten epochs](https://github.com/michellecchen/cs72_final/blob/main/BERTscores/distilBERT10epochs4batch1000s.png?raw=true)

Report for DistilBERT fine-tuned with 14 epochs (batch size 4:

![ten epochs](https://github.com/michellecchen/cs72_final/blob/main/BERTscores/distilBERT14epochs4batch1000s.png?raw=true)

#### BERT Base Uncased

Report for BERT fine-tuned with 12 epochs (batch size 128):

![twelve epochs](https://github.com/michellecchen/cs72_final/blob/main/BERTscores/BERT12epochs128batch1000s.png?raw=true)

Report for BERT fine-tuned with 16 epochs (batch size 64):

![sixteen epochs](https://github.com/michellecchen/cs72_final/blob/main/BERTscores/BERT16epochs64batch1000s.png?raw=true)

Report for BERT fine-tuned with 18 epochs (batch size 64):

![eighteen epochs](https://github.com/michellecchen/cs72_final/blob/main/BERTscores/BERT18E.png?raw=true)

We achieved maximum accuracy with the BERT fine-tuned for 16 epochs at 81%--the accuracy fell of very slightly again with 18 epochs to 79%, perhaps as a result of overfitting or perhaps randomly. In either case, since these two BERTs were of similar quality, we used the smaller and more accurate of the two to classify our Indeed neutral sentences.



