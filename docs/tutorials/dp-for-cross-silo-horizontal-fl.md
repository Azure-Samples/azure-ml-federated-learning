# Differential privacy for cross-silo horizontal federated learning

## Background
Differential privacy (DP) is a popular and powerful technique that could add another layer of privacy protection for sensitive private data, with rigorous mathematical guarantee. DP technique can be applied to both Federated Learning (FL) applications as well as non-FL applications. There are different DP algorithms available in the literature, and among them DP-SGD algorithm is a well-known one that achieves State-Of-The-Art performance in terms of model utility-privacy tradeoff. This tutorial focuses on the scenario about using DP-SGD for cross-silo horizontal FL.

## Objective and contents
This tutorial will go through a high level intuitive explanation of DP concept, show you how to apply DP-SGD in cross-silo horizontal FL experiments, summarize some key tips and pitfalls, and share helpful resources in case people want to go deeper.

### DP concept
The theory for DP is complex, but the main idea is quite simple -- it clips datum gradient and adds Gaussian noise before model update to avoid privacy leakage. DP-SGD is an algorithm described in this [paper](https://arxiv.org/pdf/1607.00133.pdf); [Opacus](https://github.com/pytorch/opacus/blob/main/docs/faq.md) is its Pytorch implementation. Please refer to [this blog post](https://medium.com/pytorch/differential-privacy-series-part-1-dp-sgd-algorithm-explained-12512c3959a3) to read more about DP-SGD. 

### How to apply DP in the MNIST example

1. Follow the instructions given in the [quickstart](../quickstart.md).
2. Set the `dp` parameter to `true` while submitting the [pipeline](../quickstart.md#launch-the-demo-experiment).

### How to apply DP in FL experiments
When applying DP to cross-silo horizontal FL experiments, we just need to add DP-SGD to the silos training step, and the implementation using Opacus DP library is quite elegant. The example FL experiments in the current repo provide straightforward and clear demonstration of using DP for cross-silo FL experiments, which you could easily mimic/adapt to your own cross-silo horizontal FL scenarios. Taking the fl_cross_silo_literal experiment using MNIST dataset as an example, briefly speaking, to implement DP-SGD, what you need are:
1. Adding DP libarary opacus dependencies to the [conda env yaml file](../../examples/components/MNIST/traininsilo/conda.yaml)
2. Adding DP related args in the corresponding silo [training component yaml file](../../examples/components/MNIST/traininsilo/spec.yaml), like below:
    ```yaml
    dp: # dp, dp_target_epsilon, dp_target_delta, dp_max_grad_norm, and total_num_of_iterations are defined for the only purpose of DP and can be ignored when users don't want to use Differential Privacy
        type: boolean
        description: differential privacy
        default: false
        optional: true
    dp_target_epsilon:
        type: number
        description: DP target epsilon
        default: 50.0
        optional: true
    dp_target_delta:
        type: number
        description: DP target delta
        default: 1e-5
        optional: true
    dp_max_grad_norm:
        type: number
        description: DP max gradient norm
        default: 1.0
        optional: true
    total_num_of_iterations:
        type: integer
        description: Total num of iterations
        default: 1
        optional: true
    ```
3. Define privacy engine and wrap model, optimizer and data_loader with DP in the corresponding silo [training component run.py](../../examples/components/MNIST/traininsilo/run.py) file, like below. 
    > **Note 1**: Compared with standard non-FL DP training, there is one thing noteworthy about `privacy_engine.make_private_with_epsilon()`. Cross-silo FL experiments would include multiple iterations of silo training, and we need to specify `epochs=total_num_of_iterations * epochs` to correctly apply the right level of noise for a given privacy budget (dp_target_epsilon) for the final DP-trained FL model. 

    ```python
    if dp:
            # Define privacy_engine
            privacy_engine = PrivacyEngine(secure_mode=False)
            # Wrap model, optimizer and data_loader with DP
            (
                self.model_,
                self.optimizer_,
                self.train_loader_,
            ) = privacy_engine.make_private_with_epsilon(
                module=self.model_,
                optimizer=self.optimizer_,
                data_loader=self.train_loader_,
                epochs=total_num_of_iterations * epochs,
                target_epsilon=dp_target_epsilon,
                target_delta=dp_target_delta,
                max_grad_norm=dp_max_grad_norm,
            )
    ```
    > **Note 2**: Alternatively, instead of passing targeted epsilon and delta, you can also achieve DP training by directly passing the noise multiplier, like below. This might be helpful if you don't have a targeted privacy budget epsilon in mind and prefer to play with different level of DP noise multiplier.

    ```python
    if dp:
            # Define privacy_engine
            privacy_engine = PrivacyEngine(secure_mode=False)
            # Wrap model, optimizer and data_loader with DP
            privacy_engine.make_private(
                module=self.model_,
                optimizer=self.optimizer_,
                data_loader=self.train_loader_,
                noise_multiplier=dp_noise_multiplier,
                max_grad_norm=dp_max_grad_norm,
            )
    ```

    > **Note 3**: Some model architecture layers might be incompatible with Opacus DP library, and Opacus ModuleValidator is helpful to validate the model architecture compatiblity and help fix/replace the incompatible layers, shown as below.

    ```python
    if dp:
        # Validate and fix/replace incompatible layers
        if not ModuleValidator.is_valid(self.model_):
            self.model_ = ModuleValidator.fix(self.model_)
    ```
4. In the corresponding FL experiment [config file](../../examples/pipelines/fl_cross_silo_literal/config.yaml), add the DP relevant parameters, like below:
    ```yaml
     # Differential privacy
     dp: true # Flag to enable/disable differential privacy
     dp_target_epsilon: 50.0 # Smaller epsilon means more privacy, more noise
     dp_target_delta: 1e-5 # The target δ of the (ϵ,δ)-differential privacy guarantee. Generally, it should be set to be less than the inverse of the size of the training dataset. 
     dp_max_grad_norm: 1.0 # Clip per-sample gradients to this norm (DP)
    ```
5. In the corresponding FL experiment [submit script](../../examples/pipelines/fl_cross_silo_literal/submit.py), add the DP relevant args, like below:
    ```python
    # Differential Privacy
    dp=YAML_CONFIG.training_parameters.dp,
    # DP target epsilon
    dp_target_epsilon=YAML_CONFIG.training_parameters.dp_target_epsilon,
    # DP target delta
    dp_target_delta=YAML_CONFIG.training_parameters.dp_target_delta,
    # DP max gradient norm
    dp_max_grad_norm=YAML_CONFIG.training_parameters.dp_max_grad_norm,
    # Total num of iterations
    total_num_of_iterations=YAML_CONFIG.training_parameters.num_of_iterations,
    ```
It is also worthy to point out that, the provided DP cross-silo FL experiments in the current repo could be used for both DP-training and non-DP training. You can quickly turn DP off by setting the corresponding FL experiment `config.yaml` file's `dp` parameter to `false`, and turn DP on by setting `dp` parameter to `true`.

### Tips and pitfalls about DP
1. **DP privacy protection doesn't come without cost**
    There is a model utility-privacy tradeoff. Epsilon is the DP privacy budget, and smaller epsilon means more privacy protection, more noises -- and hence worse model utility. For very sensitive scenarios, e.g., large language generative modeling tasks on very sensitive private data, DP might be desired to ensure approprite level of privacy protection. For less sensitive scenarios (e.g., standard classification modeling tasks), DP might be an overkill. Users should be aware of such model utility-privacy trade-off for DP, and evaluate their scenarios carefully to decide whether to apply DP, and if applying DP, with how much privacy budget.
2. **Convergence of DP model training is not trivial** 
    It might be challenging to tuning DP model training to converge to desired performance. Here are some tips that might help: 
    - Generally speaking, DP training is usually a sufficient regularizer by itself. Adding any more regularization (such as dropouts or data augmentation) is unnecessary and typically hurts performance.
    - Tuning max_grad_norm is important. Start with a low noise multiplier like 0.1 (alternatively, use a large Epsilon like 2000), which should give you compatible performance to a non-DP model. Then do a grid search (typical grid range [0.1, 10]) to find the optimal max_grad_norm.
    - You can play around the level of privacy, epsilon, and get a better idea about the model utility-privacy tradeoff for your scenario.
    - It is usually quite useful to pre-train a model on public (non-private) data, before completing the DP training on the private training data. 
    - It is important to tune the learning rate appropriately for smooth DP training convergence. Compared with a non-DP training, DP-trained models usually converge with a smaller learning rate (each gradient update is noisier, thus we want to take smaller steps). 
    - Pay attentions to the secure_mode arg when defining PrivacyEngine(). If the DP-trained model targets for shipping to production, users are encouraged to set secure_mode to True to achieve cryptographically strong DP guarantee, at the cost of slower DP training. For helpful context, secure_mode=True uses secure random number generator for noise and shuffling (as opposed to pseudo-rng in vanilla PyTorch) and prevents certain floating-point arithmetic-based attacks.
3. **DP should be the last line of privacy preservation**
    One should also keep in mind that DP should be the last line of privacy preservation and users should not rely only on DP to address all privacy leakage risks. If we know what information is highly sensitive and cannot be leaked, e.g., password or phone number, data with the information should be removed or masked.

## Additional resources
- [FAQ of DP library Opacus](https://github.com/pytorch/opacus/blob/main/docs/faq.md).
- [DP-SGD algorithm paper](https://arxiv.org/pdf/1607.00133.pdf). 
- PyTorch blog post series about DP: [blog post part 1](https://medium.com/pytorch/differential-privacy-series-part-1-dp-sgd-algorithm-explained-12512c3959a3), [blog post part 2](https://medium.com/pytorch/differential-privacy-series-part-2-efficient-per-sample-gradient-computation-in-opacus-5bf4031d9e22), [blog post part 3](https://pytorch.medium.com/differential-privacy-series-part-3-efficient-per-sample-gradient-computation-for-more-layers-in-39bd25df237).