# Program outputs 

We have three sorts of output which might be relevant for the reader. The first two pertain to the accuracy of our machine learning models (BART zero shot classification and BERT sentiment analysis). For zero shot classification, we will provide screenshots of both some of the classifiers for both positive and negative sentences for each of the companies, as well as a confusion matrix which evaluates the performance of the classifier in general. This data is also available as `.txt` in the directory `confusionData`. For BERT sentiment analysis, we will provide screenshots of classification reports (accuracy, precision, recall, f1-score, and support) of the fine-tuning results with various sets of parameters.

Third and finally, we will show the output that these two algorithms have been used to work toward; a series of timelines which aggregate sentiments over time for each company, per topic (diversity and inclusion, culture and values, work life balance, senior management, career opportunities, compensation and benefits), in relation to major lawsuits brought against the company. For each company, we will visualize the gold labeled (pros/cons sentences from Glassdoor and Indeed) and BERT-predicted data (neutral review bodies from Indeed) separately; the comparison between timelines will, in turn, serve to validate our BERT model's sentiment predictions.

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

### Running on CPU, batch size 64
Classification report for BERT fine-tuned with two epochs, 75:16:8 training:validation:testing data split, and 12300 total sentences:

(note: the split was originally 70:15:15, but we were unable to run predictions on the 15% testing dataset using CPU + the normal version of Colaboratory, so we took halved the testing data)

![two epochs](https://github.com/michellecchen/cs72_final/blob/main/BERTscores/twoEpochFullTrainValHalfTest.png?raw=true)


Classification report for BERT fine-tuned with four epochs, 70:15:15 training:validation:testing data split, and 5171 total sentences:

![four epochs](https://github.com/michellecchen/cs72_final/blob/main/BERTscores/fourEpochsHalfEverything.png?raw=true)

### Running on GPU, various batch sizes

As you can see, we were only able to achieve 65% accuracy while running with up to four epochs. Each of those CPU training runs also took more than four hours. To be able to run with more epochs, we reduced our batch sizes and asked some friends with GPUs to run the GPU-pushed version of our notebook. On GPU, we were able to train the BERT with batch size 32 and 6 epochs, for example, in around fifteen minutes. 


## Timelines

