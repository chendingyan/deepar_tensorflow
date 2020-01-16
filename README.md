Tensorflow 2 implementation of Amazon DeepAR Time Series Forecasting algorithm (https://arxiv.org/abs/1704.04110).

Influenced by these two open-source implementations: https://github.com/arrigonialberto86/deepar and https://github.com/zhykoties/TimeSeries (pytorch).

**deepar/dataset**: 

1. **time_series.py**: contains *TimeSeries* and *TimeSeriesTest* objects that perform covariate augmentation, grouping, scaling, and standardization according to Salinas et al. The objects are also easy to integrate with the **D3M** AutoML DARPA primitive and piepline infrastructure (https://docs.datadrivendiscovery.org/).

**deepar/model**: 

1. **learner.py**: contains a *DeepARLearner* class, which creates the model structure and implements a custom training loop. The model learns a categorical embedding for each unique time series group. It also performs ancestral sampling during inference (for arbitrary horizons into the future) and generates *n* samples at each timestep. Ancestral sampling can be conditioned with the whole training time series or just the final window.

2. **layers.py**: contains custom *LSTMResetStateful* layer and *GaussianLayer* layer (the latter is from https://github.com/arrigonialberto86/deepar and unused in current codebase)

3. **loss.py**: contains custom *GaussianLogLikelihood* loss for real data and *NegativeBinomialLogLikelihood* loss for positive count data. Both losses support masking and inverse scaling per Salinas et al. 

<!-- ## TODO
    -CI / Weights
    -ACLED
    -PHEM window size nan training 
    -Multiple Targets (VAR)
    -DAR: 
        lower range on early stopping delta?
        clip gradients
        multiple lstm layers
        batch size smaller for testing + training (sequential preds, concatenate, timing?)
        multiple targets
        performance review - what are bottlenecks
        lr scheduling

        added lagged value (multiple?) according to frequency of the data 
        multiple embeddings!
        attention? -->


