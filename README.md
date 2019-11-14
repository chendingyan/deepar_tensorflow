Tensorflow 2 implementation of Amazon DeepAR Time Series Forecasting algorithm (https://arxiv.org/abs/1704.04110).

Influenced by these two open-source implementations: https://github.com/arrigonialberto86/deepar and https://github.com/zhykoties/TimeSeries (pytorch).

**deepar/dataset**: 

1. **time_series.py**: contains *TimeSeries* and *TimeSeriesTest* objects that perform covariate augmentation, grouping, scaling, and standardization according to Salinas et al. The objects are also easy to integrate with the **D3M** AutoML DARPA primitive and piepline infrastructure (https://docs.datadrivendiscovery.org/).

**deepar/model**: 

1. **learner.py**: contains a *DeepARLearner* class, which creates the model structure and implements a custom training loop. The model 

2. **layers.py**: contains custom *LSTMResetStateful* layer and *GaussianLayer* layer (the latter is from https://github.com/arrigonialberto86/deepar and unused in current codebase)

3. **loss.py**: contains custom *GaussianLogLikelihood* loss for real data and *NegativeBinomialLogLikelihood* loss for positive count data. Both losses support masking and inverse scaling per Salinas et al. 




