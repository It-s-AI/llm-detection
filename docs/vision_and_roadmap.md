# Vision & Roadmap

At the moment, many subnets have tasks for which they have implemented SOTA models in their miner codes to instantly achieve high quality. For such tasks, implementing better solutions could give miners only basis points of improvement. Almost no room to grow.

But our subnet is different. AI detection is a hard task to achieve high quality. That's why we aimed not just make "marketplace for inference SOTA models" as other subnets did but rather to create a constantly evolving environment where miners have to get better over time and not just run the same models for months.

In order to implement such an environment, we need to do the following.

## Incentive Mechanism

For the current baseline version, we've implemented [incentive mechanism](https://github.com/It-s-AI/llm-detection/tree/main?tab=readme-ov-file#reward-counting) in which we take responses from miners, count 3 metrics (ap score, f1 score and fp score) and average them. These metrics represent quality of miners' predictions, but we are going to improve rewards mechanism, so that miners will be more interested in improving solution:

1. Using SoftMax function for rewards normalization (instead of linear normalization) increase the difference in emission between solution with different scores and will stimulate miners to achieve better quality.
2. Penalty for miners, which didn't respond - they have to build a stable solution, which work without crashing and always uptime.
3. Time limit component: while improving the model, miners should keep an eye on their time for response.

## Validators

Currently, validators use one large dataset with human data and two models (mistral and vicuna) to generate AI texts. What could be done to improve that:

1. Add more models. By increasing the number and diversity of models, we will improve the overall quality of detection
2. Add more languages
3. Paraphrasing of AI texts
4. Make it resilient to tricks and attacks
5. Various types of text: differentiate articles/comments/posts/etc., in order to improve quality on each distinct type

## Miners

Generally speaking, improving miners is not our task. Miners should get better themselves. But there are a few things we can do to help them:

1. Host testnet validators so miners can start without wasting TAO.
2. Make leaderboard and local dataset: we will list miners' metrics and allow people who want to start mining to evaluate their solution on a local dataset to compare them with existing ones before going to the mainnet.
3. Create Kaggle competition to introduce some of the best ML engineers to our subnet and make them run their top solution on-chain.

## Commerce

One of the important tasks for us as subnet owners is to apply the subnet for real usage. Given the relevance of the problem, there is clearly a request for such solutions. That’s what we’re going to do:

1. Create an API Service: It could be used for integrations by developers and for AI engineers to clean up their datasets.
2. Web service: Will be used for regular users to determine single pieces of text.
3. Browser extension: that way our tool will be easily accessible at any time. Users will only need to highlight text.
4. Telegram bot: In the form of the bot our tool can be used within the Telegram app so users will only need to forward a post/message/comment to the bot to get an instant result.

We're going to implement a subscription with Tiers. We're still trying to figure out what the tiers will be based on, but we have something in mind:

1. Amount of words per months
2. Batch file scanning
3. Quality of detection models

By commercializing our product, we will become less reliant on emissions and start gaining real usage. Also, by the time when dynamic tao is introduced and validators' emission becomes zero, our token will already have great utility, and validators will be earning from the mentioned services.