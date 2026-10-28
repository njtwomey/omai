# Time series (part 7, application domains)

Series modelled and mined: foundations (stationarity, autocorrelation, decomposition), ARMA and ARIMA (identification, fitting, forecasting), exponential smoothing, state-space models (the Kalman filter and smoother, structural models), volatility models (ARCH, GARCH), multivariate series (VAR, cointegration), forecasting practice (backtesting, horizons), change-point detection, the matrix-profile family (motifs, discords, chains, segmentation, snippets), time-series classification (shapelets, kernels, dictionaries) and neural forecasting, as sessions over `signal`, `stats`, `dynamics`, `nn` and the series algorithms moving to `lib`.

## Capabilities

- Foundations: ACF and PACF as a series is generated, differencing and decomposition (STL) as knobs.
- ARIMA: orders as knobs, the fit by conditional or exact likelihood stepped, the forecast with intervals rolling forward.
- Exponential smoothing: Holt-Winters' components updating observation by observation.
- State space: the Kalman filter and smoother on a structural model, the states as the series arrives (the Kalman tracking method exists).
- GARCH: the conditional variance tracking a volatile series.
- Change points: PELT, binary segmentation and Bayesian online change-point detection on a planted series, the run-length posterior as a probe.
- Matrix profile: STOMP's profile filling in, motifs and discords found; time-series classification by shapelets with the candidate shapelets scored.
- Neural forecasting: a small sequence model or N-BEATS-style network, the backtest on the tape.

## Examples to make

- The ACF and PACF of an AR(2) against the truth as n grows, with the coefficients as knobs.
- Holt-Winters' level, trend and season components updating step by step.
- Bayesian online change-point detection: the run-length posterior as a heatmap over time with the planted change.
- The matrix profile filling in, the best motif pair drawn on the series.

## Rendering

- Series with bands and events, heatmaps over time, forecast fans, components stacked, motif overlays (the `sequences` views).
- Missing: a stacked-components panel; a forecast-fan panel that follows the cursor (the origin moving).

## Shared

- `shared/`: the series slot (generators with truth: planted changes, motifs, seasonality), the stream base shared with `online-learning`.

## Problems

- `timeseries/` moves here with the Kalman tracking of `dynamics/` (or `control-theory`; decide). Matrix profile and change-point algorithms go to `lib` first (`lib/timeseries`).
