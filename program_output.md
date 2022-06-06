# Program outputs 

We have three sorts of output which might be relevant for the reader. The first two pertain to the accuracy of our machine learning models (BART zero shot classification and BERT sentiment analysis). For zero shot classification, we will provide screenshots of both some of the classifiers for both positive and negative sentences for each of the companies, as well as a confusion matrix which evaluates the performance of the classifier in general. This data is also available as `.txt` in the directory `confusionData`. For BERT sentiment analysis, we will provide screenshots of classification reports (accuracy, precision, recall, f1-score, and support) of the fine-tuning results with various sets of parameters.

Third and finally, we will show the output that these two algorithms have been used to work toward; a series of timelines which aggregate sentiments over time for each company, per topic (diversity and inclusion, culture and values, work life balance, senior management, career opportunities, compensation and benefits), in relation to major lawsuits brought against the company. For each company, we will visualize the gold labeled (pros/cons sentences from Glassdoor and Indeed) and BERT-predicted data (neutral review bodies from Indeed) separately; the comparison between timelines will, in turn, serve to validate our BERT model's sentiment predictions.

## Zero shot classification with BART
### Classifiers (50 for each company, 25 per valence)

Riot positive sentences:

Riot negative sentences:

Sony positive sentences:

Sony negative sentences:

Ubisoft positive sentences:

Ubisoft negative sentences:

Activision Blizzard positive sentences:

Activision Blizzard negative sentences:


### Confusion matrix 
We labeled 185 sentences by hand according to topic, then compared them with the predicted labels:



## BERT sentiment analysis
Classification report for BERT fine-tuned with two epochs, 75:16:8 training:validation:testing data split, and 12300 total sentences:

(note: the split was originally 70:15:15, but we were unable to run predictions on the 15% testing dataset using CPU + the normal version of Colaboratory, so we took halved the testing data)






Classification report for BERT fine-tuned with four epochs, 70:15:15 training:validation:testing data split, and 5171 total sentences:


## Timelines

