
## Perplexity approach

We made a solid baseline solution based on counting [perplexity of fixed-length models](https://huggingface.co/docs/transformers/perplexity). 
For counting PPL we use a fresh phi-2 model from microsoft, which has been released at the end of 2023. 
We also trained a linear model on the phi-2 outputs, to make probabilities more representative. 

On our local validation with baseline model got overall accuracy about 89%, you can find accuracy per data source below:

| Data Source               | Accuracy |
|---------------------------|----------|
| LLM (gemma:7b)            | 0.939    |
| LLM (neural-chat)         | 0.856    |
| LLM (zephyr:7b-beta)      | 0.964    |
| LLM (vicuna)              | 0.981    |
| LLM (mistral)             | 0.963    |
| Human-data                | 0.841    |