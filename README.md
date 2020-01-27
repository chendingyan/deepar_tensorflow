Tensorflow 2 implementation of Amazon DeepAR Time Series Forecasting algorithm (https://arxiv.org/abs/1704.04110).

Influenced by these two open-source implementations: https://github.com/arrigonialberto86/deepar and https://github.com/zhykoties/TimeSeries (pytorch).

**deepar/dataset**: 

1. **time_series.py**: contains *TimeSeriesTrain* and *TimeSeriesTest* objects that perform covariate augmentation, grouping, scaling, and standardization according to Salinas et al. The objects are also easy to integrate with the **D3M** AutoML DARPA primitive and piepline infrastructure (https://docs.datadrivendiscovery.org/).

**deepar/model**: 

1. **learner.py**: contains a *DeepARLearner* class, which creates the model structure and implements a custom training loop. The model learns a categorical embedding for each unique time series group. It also performs ancestral sampling during inference (for arbitrary horizons into the future) and generates *n* samples at each timestep. Ancestral sampling can be conditioned with the whole training time series or just the final window.

2. **layers.py**: contains custom *LSTMResetStateful* layer and *GaussianLayer* layer (the latter is from https://github.com/arrigonialberto86/deepar and unused in current codebase)

3. **loss.py**: contains custom *GaussianLogLikelihood* loss for real data and *NegativeBinomialLogLikelihood* loss for positive count data. Both losses support masking and inverse scaling per Salinas et al. 

<!-- ## TODO
    -DAR: 

        EASY
        clip gradients (Gluon - 10)
        multiple lstm layers
        lr scheduling (10^-3 halve after 300 batches if no improvement or exponential)
        multivariate targets + categoricals

        TRY TO REPRODUCE DeepAR / Gluon experiments on public datasets

        EXPERIMENT
        multiple embeddings, constraint on embeddings to respect hierarchical ordering!

        Compare DeepAR 1 embedding, multiple embeddings, multiple constrained to Optimal Reconciliation approach on OR public datasets -> PAPER
        If improved -> compare to multiple ts models. If not is there a way to use as constraint

        NICE LIBRARY ADDITIONS
        add multiple lagged seasonal values according to frequency of the data    
        a) automatically select distribution that minimizes validation loss (include student's t)
        b) loss term that increases variance to account for model misspecification
        box-cox / differencing transform of z before modeling?

        MORE EXPERIMENTS
        generative loss component for p(x), non-targets, only if significant x

        comparisons:
        NPTS 
        Deep state space
        S2S (unknown horizon?)
        same datasets from GluonTS paper

    -ACLED    
    -->


    


