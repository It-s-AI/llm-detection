# Incentive mechanism

For validating we use two types of data, which is balanced in proportion 1:1.

### Human-written texts
To gather human-written validation data we use the Pile dataset.

The Pile is a 825 GiB diverse, open source language modelling data set that consists of 22 smaller, high-quality datasets combined together. It includes web-crawled data, financial, med, law, arxiv, github and also about 15 different topics.

### AI-generated texts
For AI-generated text collection, we need to obtain prompts and then generate texts based on these prompts. While for human texts we take samples from Pile dataset we have to generate ai-samples from the same data-source, so that the only difference between them was human/ai written.

So, as prompts we take a random sample and then use part of it as text begging and ask LLMs to generate a completion for it.

We use the Ollama GitHub repository to run Large Language Models and generate completions for these prompts. As LLMs we use 30+ SOTA models from the top of LLM-Arena.

We also randomly select generation parameters for LLM during validation to make the dataset more diverse.

### Data augmentation to prevent cheating
To prevent remembering Pile dataset and make it stablier to overfitting we add some augmentation to both ai-generated and human-written texts. First of all we select a random sequence of consecutive sentences from a given text. Then we add in a random place (or two) misspelling (about 10 different char-based augs) or remove a random adjective.

These augmentations don't allow miners to precalculate hashes on Pile dataset and then use them to determine whether this text is present in the human set of data or not.

## Reward counting
Based on [Detecting LLM-Generated Text in Computing Education](https://arxiv.org/pdf/2307.07411.pdf) 
article we decided to dived our reward on 3 parts:

#### F1 score
We decided to use it instead of classic accuracy, because
it better represent quality of model especially on binary-classification tasks.

#### False Positive score
FP_score = 1 - FP / len(samples).

It is usually more important not to mistakenly classify human-written text as AI-generated than the other way around.
It is preferable to tolerate a few more instances of student cheating or read some AI-generated emails than to wrongly penalize a real student or miss an important letter.

#### AP score
AP summarizes a precision-recall curve by calculating the weighted mean of precisions achieved at each threshold.
This allows us to evaluate the quality of the model's ranking.


The final reward is the average of these three values.
