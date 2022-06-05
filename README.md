# Temporally located sentiment analysis on gaming company employee reviews using a fine-tuned BERT
## Written by Michelle Chen and Leah Ryu

GitHub usernames: michellecchen, nonsensicle

Emails: michelle.chen.22@dartmouth.edu, leah.ryu.22@dartmouth.edu

Welcome to our CS72 (Accelerated Computational Linguistics) final project for spring 2022! Please read through the following, which discusses motivation, algorithms, data, code organization, and setup instructions.

### Motivation and proposal

Many workplaces, when hiring, advertise themselves as inclusive or progressive to pique the interest of prospective employees. These publicity statements, however, do not necessarily translate into equitable treatment for actual employees behind the scenes. In the interest of preserving a company’s public image, incidents of harassment and discrimination often go undisclosed — especially when committed by authority/seniority figures against lower-ranking employees — and will only become known to outsiders when a great amount of harm has been committed, across a large stretch of time (i.e. exposés & employment discrimination lawsuits). Due to their containment as “insider knowledge” in the majority of cases, it is difficult for prospective employees to learn of these abuses before applying to work at the places they are committed.

There are avenues to “insider knowledge” that job seekers may access. One of them comes in the form of online reviews, written by current and former employees, on sites such as Glassdoor & Indeed. These textual reviews, however, are unstructured, and complaints about cafeteria food can be easily mixed in with complaints about wage disparities. Moreover, companies often have well over 300 reviews on Glassdoor alone. Therefore, users must choose between (a) investing a large sum of time into reading every review — an impracticality, when considering how many companies the average person must apply to when looking for work — or (b) only reading the most recent, which is insufficient for a complete and clear picture of what it’s like to work there.

Given the above problem, we propose a system that (a) parses all employee-written reviews on Glassdoor and Indeed for four given gaming software companies, respectively – Activision Blizzard, Riot Games, Sony, and Ubisoft –  which have faced lawsuits or allegations of discrimination/harrassment from former employees (b) classifies every sentence from each review by topic – essentially, we want to filter out reviews having to do with free food and merch, for example, (c) performs a sentiment analysis on the sentences classified as relating to interpersonal workplace dynamics (i.e. workplace culture, authority figures and their behavior, etc.), and (d) investigates how employee sentiment changes (if at all) leading up to, as well as in the aftermath of, significant events relating to harassment/discrimination in the company, as documented by exposés and lawsuits (i.e. targeted layoffs, relevant authority figures entering/leaving positions, etc.). Part D will be visualized using timelines.

### Algorithms / approach
We want to examine reviews relating to a few topics predefined by ourselves as the researchers: Culture and Values, Diversity and Inclusion, Work/Life Balance, Senior Management, Compensation and Benefits, and Career Opportunities. To do this, we apply zero-shot classification to our data. Zero-shot classification is an unsupervised learning algorithm which allows us to give a predefined set of topics and match an input string, such as a sentence, to one of those topics by probability. We can have those probabilities add up to 1 and simply choose the topic with the highest probability, or we can calculate the probability of each class independently; in our case, we will be using the zero-shot classification pipeline from Hugging Face (based on transformers), so we can reasonably assume that the former approach will yield accurate results. Here is the Python documentation from Hugging Face, as well as a use-case example from Towards Data Science: https://huggingface.co/facebook/bart-large-mnli, https://towardsdatascience.com/zero-shot-text-classification-with-hugging-face-7f533ba83cd6. We also manually classify around 200 sentences so that we can generate a confusion matrix to evaluate the zero shot classification.

For the sentiment analysis, we fine-tune a BERT classifier to label each review as positive or negative. To do this, we rely on a dataset whose contents have already been labeled as positive or negative. Fortunately, Glassdoor reviews have corresponding “pros” and “cons” sections — which we can apply as default, to review sentences as positive or negative. Each Indeed review also has an optional pros/cons structure, as well as a general commentary section. We plan on using the definitively labeled pros/cons sentences as training and validation data — then, using the BERT to label the “general” commentary. As output, we will receive labels for each sentence, which we can then aggregate into a score for each company per topic. 
Here is the BERT documentation from Hugging Face: https://huggingface.co/docs/transformers/model_doc/bert.  
Here are a couple of resources on fine-tuning a BERT (the first source is about positive/negative classification, explicitly, and we owe great thanks/ a good deal of our code to it): https://www.geeksforgeeks.org/fine-tuning-bert-model-for-sentiment-analysis/#:~:text=Google%20created%20a%20transformer%2Dbased,dataset%20would%20lead%20to%20overfitting, https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/BERT/Fine_tuning_BERT_(and_friends)_for_multi_label_text_classification.ipynb,  https://www.analyticsvidhya.com/blog/2021/12/fine-tune-bert-model-for-sentiment-analysis-in-google-colab/, https://huggingface.co/docs/transformers/training. 

We use Python to make timelines.

### Data 

We have scraped 5232 reviews from Glassdoor and 667 reviews from Indeed. We leveraged the online web service Page2API to do so, using [this tutorial on scraping reviews from Glassdoor.] (https://www.page2api.com/blog/how-to-scrape-glassdoor-reviews/) To scrape for reviews from Indeed is essentially the same process, but 

```
api_url = "https://www.page2api.com/api/v1/scrape/encoded/eyJhcGlfa2V5IjoiNzhhNjNkNWM3MTI2ZGM4ODIxNmNhODBmZWUzODIyZGRlMThkNmZiOCIsImJhdGNoIjp7InVybHMiOiJodHRwczovL3d3dy5pbmRlZWQuY29tL2NtcC9BY3RpdmlzaW9uL3Jldmlld3M_ZmNvdW50cnk9QUxMJnN0YXJ0PVswLCA4MCwgMjBdJmxhbmc9ZW4iLCJjb25jdXJyZW5jeSI6MSwibWVyZ2VfcmVzdWx0cyI6dHJ1ZX0sInJhdyI6eyJrZXkiOiJyZXZpZXdzIiwiZm9ybWF0IjoiY3N2In0sInBhcnNlIjp7InJldmlld3MiOlt7Il9wYXJlbnQiOiJbaXRlbXByb3A9cmV2aWV3XSIsInRpdGxlIjoiaDJbZGF0YS10ZXN0aWQ9dGl0bGVdID4-IHRleHQiLCJhdXRob3JfdHlwZSI6InNwYW5baXRlbXByb3A9YXV0aG9yXSA-PiB0ZXh0IiwiY29udGVudCI6InNwYW5baXRlbXByb3A9cmV2aWV3Qm9keV0gPj4gdGV4dCIsInJhdGluZyI6Im1ldGFbaXRlbXByb3A9cmF0aW5nVmFsdWVdID4-IGNvbnRlbnQiLCJwcm9zIjoiZGl2W2RhdGEtdG4tY29tcG9uZW50PSdyZXZpZXdEZXNjcmlwdGlvbiddICsgZGl2ID4gZGl2IGRpdjpudGgtb2YtdHlwZSgxKSA-IGRpdiA-IHNwYW4gPj4gdGV4dCIsImNvbnMiOiJkaXZbZGF0YS10bi1jb21wb25lbnQ9J3Jldmlld0Rlc2NyaXB0aW9uJ10gKyBkaXYgPiBkaXYgZGl2Om50aC1vZi10eXBlKDIpID4gZGl2ID4gc3BhbiA-PiB0ZXh0In1dfX0"
payload = {
  "api_key": "78a63d5c7126dc88216ca80fee3822dde18d6fb8",
  "batch": {
    "urls": "https://www.indeed.com/cmp/Riot-Games/reviews?fcountry=ALL&start=[0, 40, 20]&lang=en",
    "concurrency": 1,
    "merge_results": True
  },
  "raw": {
    "key": "reviews", "format": "csv"
  },
  "parse": {
    "reviews": [
      {
        "_parent": "[itemprop=review]",
        "title": "h2[data-testid=title] >> text",
        "author_type": "span[itemprop=author] >> text",
        "content": "span[itemprop=reviewBody] >> text",
        "rating": "meta[itemprop=ratingValue] >> content",
        "pros": "div[data-tn-component='reviewDescription'] + div > div div:nth-of-type(1) > div > span >> text",
        "cons": "div[data-tn-component='reviewDescription'] + div > div div:nth-of-type(2) > div > span >> text"
      }
    ]
  }
}

req = requests.get(api_url)
url_content = req.content
csv_file = open('allActivisionBlizzardIndeed.csv', 'ab')

csv_file.write(url_content)
csv_file.close()
```

### Code organization



### Conda instructions

Make a new anaconda environment called `cs72`. In the anaconda terminal type

`conda create --name cs72 python=3.7.11`

Activate the new environment with

`conda activate cs72`
