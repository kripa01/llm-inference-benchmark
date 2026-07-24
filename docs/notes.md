

What is the difference between training and inference?
Why does an LLM need a tokenizer?
Why is the model file hundreds of megabytes even though it doesn't store books or web pages?
Why can't an LLM read raw English text directly?
Why doesn't the tokenizer just split on spaces?
Why do you think words like "healthcare" might become two tokens?

# Learning Notes

## LLM Inference

A language model takes tokenized input and predicts the probability of future tokens.

## Benchmark Metrics

- Load time
- Inference latency
- Tokens per second
- Memory usage

## Future Improvements

- Compare multiple models
- Add Apple MPS GPU support
- Add visualization
- Add batch inference testing