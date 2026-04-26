[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
![License](https://img.shields.io/badge/License-MIT-blue.svg)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/shako-stats)

# Awesome Marketing Science

A curated list of awesome machine learning libraries for marketing, including media mix models, multi touch attribution, causal inference and more [shakostats.com](https://shakostats.com).

Star ⭐ the repo if it helps you, and feel free to contribute your own favorite resources

## Start Here / Must Read
If you're new to the space or building a measurement stack from scratch, start with these before going deep into the longer lists below. This shortlist is meant to build intuition across incrementality, experimentation, MMM, causal inference, and Bayesian thinking.

### Geo Incrementality & Matched Markets
*   [GeoLift](https://github.com/facebookincubator/GeoLift) - Open-source geo experimentation package from Meta.
*   [A Time-Based Regression Matched Markets Approach for Designing Geo Experiments](https://research.google/pubs/a-time-based-regression-matched-markets-approach-for-designing-geo-experiments/) - Canonical matched-markets design reference from Google.
*   [Trimmed Match Design for Randomized Paired Geo Experiments](https://research.google/pubs/trimmed-match-design-for-randomized-paired-geo-experiments/) - Strong reference for paired geo design and iROAS estimation.

### Platform Incrementality & Ghost Ads
*   [Ghost Ads: Improving the Economics of Measuring Online Ad Effectiveness](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2620078) - Foundational ghost ads paper.
*   [How DoorDash Measures Restaurant Ad Incrementality with Ghost Ads](https://advertising.doordash.com/en-us/resources/measuring-incrementality-with-ghost-ads-at-doordash) - Practical ghost-ad implementation notes from a real production measurement workflow.

### Measurement Strategy & MMM
*   [Unified Marketing Measurement: The Power of Blending Methodologies (PDF)](https://www.thinkwithgoogle.com/_qs/documents/15401/UnifiedMarketingMeasurement.pdf) - Still one of the clearest blended measurement explainers.
*   [Media Effect Estimation with PyMC: Adstock, Saturation & Diminishing Returns](https://juanitorduz.github.io/pymc_mmm/) - Practical walkthrough of key MMM transformations.

### A/B Testing & Experiment Quality
*   [Trustworthy Online Controlled Experiments](https://www.cambridge.org/core/books/trustworthy-online-controlled-experiments/D97B26382EB0EB2DC2019A7A7B518F59) - Canonical A/B testing book.
*   [Experiment Guide - Trustworthy Online Controlled Experiments](https://experimentguide.com/) - Free entry point for modern experimentation practice.
*   [Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment Data](https://exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf) - The original CUPED paper.

### Causal Inference Foundations
*   [Causal Inference: The Mixtape](https://mixtape.scunning.com/) - One of the most accessible introductions to modern causal inference and research design.
*   [Causal Inference for the Brave and True](https://matheusfacure.github.io/python-causality-handbook/landing-page.html) - Practical causal-inference guide with approachable intuition and Python examples.

### Bayesian Modeling Foundations
*   [Statistical Rethinking](https://github.com/pymc-devs/resources/tree/master/Rethinking) - Strong applied Bayesian modeling foundation with a large teaching and practice community around it.
*   [Probabilistic Programming & Bayesian Methods for Hackers](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers) - Friendly first exposure to Bayesian thinking and probabilistic programming.

### Marketing Science Breadth
*   [Introduction to Algorithmic Marketing](https://www.algorithmicmarketingbook.com/) - Broad practitioner-oriented guide to applying machine learning, economics, pricing, search, recommendations, and decisioning in marketing.

## Open Source Libraries
A collection of open source repositories and libraries.

### Attribution
*   [Marketing-Attribution-Models](https://github.com/DP6/Marketing-Attribution-Models) ![Github Stars](https://img.shields.io/github/stars/DP6/Marketing-Attribution-Models.svg?style=social) - Python Class created to address problems regarding Digital Marketing Attribution.
*   [mta](https://github.com/eeghor/mta) ![Github Stars](https://img.shields.io/github/stars/eeghor/mta.svg?style=social) - Multi-Touch Attribution
*   [markov-chain-attribution](https://github.com/jerednel/markov-chain-attribution) ![Github Stars](https://img.shields.io/github/stars/jerednel/markov-chain-attribution.svg?style=social) - A PoC for a fractional attribution model leveraging first order Markov Chains.
*   [pychattr](https://github.com/jmwoloso/pychattr) ![Github Stars](https://img.shields.io/github/stars/jmwoloso/pychattr.svg?style=social) - Python Channel Attribution (pychattr) - A Python implementation of the excellent R ChannelAttribution library
*   [fractribution](https://github.com/google/fractribution) ![Github Stars](https://img.shields.io/github/stars/google/fractribution.svg?style=social)
*   [shapley-attribution-model-zhao-naive](https://github.com/ianchute/shapley-attribution-model-zhao-naive) ![Github Stars](https://img.shields.io/github/stars/ianchute/shapley-attribution-model-zhao-naive.svg?style=social) - Shapley Attribution: A scikit-learn compatible library that uses Shapley values to measure each marketing channel's true marginal contribution to conversions
*   [shapley](https://github.com/hartmann-lars/shapley) ![Github Stars](https://img.shields.io/github/stars/hartmann-lars/shapley.svg?style=social) - Attribution modelling using Shapley Values
*   [ChannelAttribution](https://github.com/DavideAltomare/ChannelAttribution) ![Github Stars](https://img.shields.io/github/stars/DavideAltomare/ChannelAttribution.svg?style=social) - Python and R library that uses Markov-chain approaches for channel attribution and journey analysis.
*   [ChannelAttribution](https://pypi.org/project/ChannelAttribution/) - PyPI package for channel attribution workflows in Python.
*   [ChannelAttribution R package manual](https://cran.r-project.org/web/packages/ChannelAttribution/ChannelAttribution.pdf)
*   [Deep Conversion Attribution](https://github.com/rk2900/deep-conv-attr) ![Github Stars](https://img.shields.io/github/stars/rk2900/deep-conv-attr.svg?style=social) - Research implementation of deep-learning-based conversion attribution for online advertising.
*   [Regression Based Attribution](https://github.com/google/rba) ![Github Stars](https://img.shields.io/github/stars/google/rba.svg?style=social) - Google's regression-based attribution approach for measuring incremental channel contribution.

### Marketing Mix Models (MMM)
*   [Robyn](https://github.com/facebookexperimental/Robyn) ![Github Stars](https://img.shields.io/github/stars/facebookexperimental/Robyn.svg?style=social) - Robyn is an experimental, AI/ML-powered and open sourced Marketing Mix Modeling (MMM) package from Meta Marketing Science. Our mission is to democratise modeling knowledge, inspire the industry through innovation, red...
*   [pymc-marketing](https://github.com/pymc-labs/pymc-marketing) ![Github Stars](https://img.shields.io/github/stars/pymc-labs/pymc-marketing.svg?style=social) - Bayesian marketing toolbox in PyMC. Media Mix (MMM), customer lifetime value (CLV), buy-till-you-die (BTYD) models and more.
*   [lightweight-mmm](https://github.com/google/lightweight_mmm) ![Github Stars](https://img.shields.io/github/stars/google/lightweight_mmm.svg?style=social) - LightweightMMM 🦇 is a lightweight Bayesian Marketing Mix Modeling (MMM) library that allows users to easily train MMMs and obtain channel attribution information.
*   [mmm-stan](https://github.com/sibylhe/mmm_stan) ![Github Stars](https://img.shields.io/github/stars/sibylhe/mmm_stan.svg?style=social) - Python/STAN Implementation of Multiplicative Marketing Mix Model, with deep dive into Adstock (carry-over effect), ROAS, and mROAS
*   [mamimo](https://github.com/Garve/mamimo) ![Github Stars](https://img.shields.io/github/stars/Garve/mamimo.svg?style=social) - A package to compute a marketing mix model.
*   [BayesianMMM](https://github.com/leopoldavezac/BayesianMMM) ![Github Stars](https://img.shields.io/github/stars/leopoldavezac/BayesianMMM.svg?style=social) - (ml) - python implementation of bayesian media mix modelling with shape and carryover effect
*   [dammmdatagen](https://github.com/DoktorMike/dammmdatagen) ![Github Stars](https://img.shields.io/github/stars/DoktorMike/dammmdatagen.svg?style=social) - Marketing Mix Modeling Data Generator
*   [Bayesian Time Varying Coefficients \ (python package)](https://orbit-ml.readthedocs.io/en/latest/about.html)
*   [Ecommerce Marketing Spend Optimization](https://github.com/Morphl-AI/Ecommerce-Marketing-Spend-Optimization) ![Github Stars](https://img.shields.io/github/stars/Morphl-AI/Ecommerce-Marketing-Spend-Optimization.svg?style=social) - Machine-learning workflow for optimizing ecommerce marketing budget allocation.
*   [Meridian](https://github.com/google/meridian) ![Github Stars](https://img.shields.io/github/stars/google/meridian.svg?style=social) - Google's open-source Bayesian MMM framework and successor to LightweightMMM.
*   [MMM Prior Elicitation](https://github.com/louismagowan/mmm-prior-elicitation) ![Github Stars](https://img.shields.io/github/stars/louismagowan/mmm-prior-elicitation.svg?style=social) - Practical tools for prior elicitation and Bayesian MMM calibration.
*   [Robyn \ (R Library)](https://facebookexperimental.github.io/Robyn/)

### Geo Experimentation & Lift Testing
*   [GeoLift](https://github.com/facebookincubator/GeoLift) ![Github Stars](https://img.shields.io/github/stars/facebookincubator/GeoLift.svg?style=social) - GeoLift is an end-to-end geo-experimental methodology based on Synthetic Control Methods used to measure the true incremental effect (Lift) of ad campaign.
*   [GeoexperimentsResearch](https://github.com/google/GeoexperimentsResearch) ![Github Stars](https://img.shields.io/github/stars/google/GeoexperimentsResearch.svg?style=social) - An open-source implementation of the geo experiment analysis methodology developed at Google. Disclaimer: This is not an official Google product.
*   [matched_markets](https://github.com/google/matched_markets) ![Github Stars](https://img.shields.io/github/stars/google/matched_markets.svg?style=social) - Matched Markets is a Python library for design and analysis of Geo experiments using Matched Markets and Time Based Regression.
*   [trimmed_match](https://github.com/google/trimmed_match) ![Github Stars](https://img.shields.io/github/stars/google/trimmed_match.svg?style=social) - This Python library implements Trimmed Match for analyzing randomized paired geo experiments and also implements Trimmed Match Design for designing randomized paired geo experiments.
*   [CausalImpact \ (R library)](https://cran.r-project.org/web/packages/CausalImpact/index.html)
*   [Geo RCT Methodology](https://github.com/rickcentralcontrolcom/geo-rct-methodology) ![Github Stars](https://img.shields.io/github/stars/rickcentralcontrolcom/geo-rct-methodology.svg?style=social) - Open methodology notes and code for geo randomized controlled trials.
*   [MarketMatching](https://github.com/klarsen1/MarketMatching) ![Github Stars](https://img.shields.io/github/stars/klarsen1/MarketMatching.svg?style=social) - Market matching and causal impact tooling for geo-based tests and comparisons.
*   [Meta Geolift \ (R Library)](https://facebookincubator.github.io/GeoLift/)
*   [Murray \ (Python Library)](https://github.com/entropyx/murray) ![Github Stars](https://img.shields.io/github/stars/entropyx/murray.svg?style=social)

### Causal Inference & Bayesian Analysis
*   [statsmodels](https://github.com/statsmodels/statsmodels) ![Github Stars](https://img.shields.io/github/stars/statsmodels/statsmodels.svg?style=social) - Statsmodels: statistical modeling and econometrics in Python
*   [dowhy](https://github.com/py-why/dowhy) ![Github Stars](https://img.shields.io/github/stars/py-why/dowhy.svg?style=social) - DoWhy is a Python library for causal inference that supports explicit modeling and testing of causal assumptions. DoWhy is based on a unified language for causal inference, combining causal graphical models and potent...
*   [causalml](https://github.com/uber/causalml) ![Github Stars](https://img.shields.io/github/stars/uber/causalml.svg?style=social) - Uplift modeling and causal inference with machine learning algorithms
*   [EconML](https://github.com/py-why/EconML) ![Github Stars](https://img.shields.io/github/stars/py-why/EconML.svg?style=social) - ALICE (Automated Learning and Intelligence for Causation and Economics) is a Microsoft Research project aimed at applying Artificial Intelligence concepts to economic decision making. One of its goals is to build a to...
*   [External Resource](http://github.com/SheffieldML/GPy) ![Github Stars](https://img.shields.io/github/stars/SheffieldML/GPy.svg?style=social) - Gaussian processes framework in python
*   [CausalImpact](https://github.com/google/CausalImpact) ![Github Stars](https://img.shields.io/github/stars/google/CausalImpact.svg?style=social) - An R package for causal inference in time series
*   [CausalPy](https://github.com/pymc-labs/CausalPy) ![Github Stars](https://img.shields.io/github/stars/pymc-labs/CausalPy.svg?style=social) - A Python package for causal inference in quasi-experimental settings
*   [scikit-uplift](https://github.com/maks-sh/scikit-uplift) ![Github Stars](https://img.shields.io/github/stars/maks-sh/scikit-uplift.svg?style=social) - exclamation: uplift modeling in scikit-learn style in python :snake:
*   [tfcausalimpact](https://github.com/WillianFuks/tfcausalimpact) ![Github Stars](https://img.shields.io/github/stars/WillianFuks/tfcausalimpact.svg?style=social) - Python Causal Impact Implementation Based on Google's R Package. Built using TensorFlow Probability.
*   [External Resource](https://github.com/pymc-devs/pymc-examples) ![Github Stars](https://img.shields.io/github/stars/pymc-devs/pymc-examples.svg?style=social) - Examples of PyMC models, including a library of Jupyter notebooks.
*   [upliftml](https://github.com/bookingcom/upliftml) ![Github Stars](https://img.shields.io/github/stars/bookingcom/upliftml.svg?style=social) - UpliftML: A Python Package for Scalable Uplift Modeling
*   [SyntheticControlMethods](https://github.com/OscarEngelbrektson/SyntheticControlMethods) ![Github Stars](https://img.shields.io/github/stars/OscarEngelbrektson/SyntheticControlMethods.svg?style=social) - A Python package for causal inference using Synthetic Controls
*   [CausalLift](https://github.com/Minyus/causallift) ![Github Stars](https://img.shields.io/github/stars/Minyus/causallift.svg?style=social) - Uplift-modeling package for causal inference and treatment-effect ranking.
*   [Causmos](https://github.com/google-marketing-solutions/causmos) ![Github Stars](https://img.shields.io/github/stars/google-marketing-solutions/causmos.svg?style=social) - Open-source web application for CausalImpact-style analysis from Google Marketing Solutions.
*   [diff-diff](https://github.com/igerber/diff-diff) ![Github Stars](https://img.shields.io/github/stars/igerber/diff-diff.svg?style=social) - Python Difference-in-Differences library with sklearn-like ergonomics, modern staggered-adoption estimators, event studies, diagnostics, and sensitivity analysis.
*   [MatchIt](https://github.com/kosukeimai/MatchIt) ![Github Stars](https://img.shields.io/github/stars/kosukeimai/MatchIt.svg?style=social) - Widely used R package for matching and preprocessing before causal estimation.
*   [mlsynth](https://github.com/jgreathouse9/mlsynth) ![Github Stars](https://img.shields.io/github/stars/jgreathouse9/mlsynth.svg?style=social) - A Python library for doing policy evaluation using panel data estimators.
*   [pysmatch](https://github.com/miaohancheng/pysmatch) ![Github Stars](https://img.shields.io/github/stars/miaohancheng/pysmatch.svg?style=social) - Python propensity score matching library with logistic, KNN, and CatBoost scoring, balance diagnostics, exhaustive matching, and reproducible workflows.
*   [pysyncon](https://github.com/sdfordham/pysyncon) ![Github Stars](https://img.shields.io/github/stars/sdfordham/pysyncon.svg?style=social)
*   [ScidesignR](https://github.com/scidesign/scidesignR) ![Github Stars](https://img.shields.io/github/stars/scidesign/scidesignR.svg?style=social) - R package for scientific and experimental design work.
*   [scpi](https://github.com/nppackages/scpi) ![Github Stars](https://img.shields.io/github/stars/nppackages/scpi.svg?style=social)
*   [SparseSC](https://github.com/microsoft/SparseSC) ![Github Stars](https://img.shields.io/github/stars/microsoft/SparseSC.svg?style=social)
*   [TensorFlow CausalImpact](https://github.com/google/tfp-causalimpact) ![Github Stars](https://img.shields.io/github/stars/google/tfp-causalimpact.svg?style=social) - TensorFlow Probability implementation of CausalImpact.

### Customer Analytics (CLV, Segmentation, Uplift)
*   [lifetimes](https://github.com/CamDavidsonPilon/lifetimes) ![Github Stars](https://img.shields.io/github/stars/CamDavidsonPilon/lifetimes.svg?style=social) - Lifetime value in Python
*   [retentioneering-tools](https://github.com/retentioneering/retentioneering-tools) ![Github Stars](https://img.shields.io/github/stars/retentioneering/retentioneering-tools.svg?style=social) - Retentioneering: product analytics, data-driven CJM optimization, marketing analytics, web analytics, transaction analytics, graph visualization, process mining, and behavioral segmentation in Python. Predictive analy...
*   [ecommercetools](https://github.com/practical-data-science/ecommercetools) ![Github Stars](https://img.shields.io/github/stars/practical-data-science/ecommercetools.svg?style=social) - EcommerceTools is a Python data science toolkit for ecommerce, marketing science, and technical SEO analysis and modelling and was created by Matt Clarke.
*   [btyd](https://github.com/ColtAllen/btyd) ![Github Stars](https://img.shields.io/github/stars/ColtAllen/btyd.svg?style=social) - Buy Till You Die and Customer Lifetime Value statistical models in Python.
*   [amazon-denseclus](https://github.com/awslabs/amazon-denseclus) ![Github Stars](https://img.shields.io/github/stars/awslabs/amazon-denseclus.svg?style=social) - Clustering for mixed-type data
*   [rfm](https://github.com/sonwanesuresh95/rfm) ![Github Stars](https://img.shields.io/github/stars/sonwanesuresh95/rfm.svg?style=social) - Python Package for RFM Analysis and Customer Segmentation
*   [lucius-ltv](https://github.com/plexagon/lucius-ltv) ![Github Stars](https://img.shields.io/github/stars/plexagon/lucius-ltv.svg?style=social) - A simple multicohort LTV calculator for subscriptions
*   [BTYD](https://github.com/ghuiber/BTYD) ![Github Stars](https://img.shields.io/github/stars/ghuiber/BTYD.svg?style=social) - R package for Buy 'Til You Die customer-base analysis and CLV modeling.
*   [BTYDplus](https://github.com/mplatzer/BTYDplus) ![Github Stars](https://img.shields.io/github/stars/mplatzer/BTYDplus.svg?style=social) - Extended BTYD models in R for customer-base analysis.
*   [CLV (python) \ - Lifetimes](https://lifetimes.readthedocs.io/en/latest/)
*   [lifelines](https://github.com/CamDavidsonPilon/lifelines) ![Github Stars](https://img.shields.io/github/stars/CamDavidsonPilon/lifelines.svg?style=social)
*   [pysurvival](https://github.com/square/pysurvival) ![Github Stars](https://img.shields.io/github/stars/square/pysurvival.svg?style=social)
*   [scikit-survival](https://github.com/sebp/scikit-survival) ![Github Stars](https://img.shields.io/github/stars/sebp/scikit-survival.svg?style=social)
*   [Survival Models (python) \ - Lifelines](https://lifelines.readthedocs.io/en/latest/)

### Customer Response Modeling
*   [mr-uplift](https://github.com/Ibotta/mr_uplift) ![Github Stars](https://img.shields.io/github/stars/Ibotta/mr_uplift.svg?style=social) - Open-source uplift-modeling package for multiple treatments and responses.
*   [Uplift Modeling with Multiple Treatments/Responses \ (Python Package)](https://pypi.org/project/mr-uplift/)

### Forecasting
*   [darts](https://github.com/unit8co/darts) ![Github Stars](https://img.shields.io/github/stars/unit8co/darts.svg?style=social)
*   [gluonts](https://github.com/awslabs/gluonts) ![Github Stars](https://img.shields.io/github/stars/awslabs/gluonts.svg?style=social)
*   [Macroeconomic Data (US) Python \ - pandas-datareader](https://pandas-datareader.readthedocs.io/en/latest/)
*   [MLForecast](https://github.com/Nixtla/mlforecast) ![Github Stars](https://img.shields.io/github/stars/Nixtla/mlforecast.svg?style=social) - Scalable machine-learning forecasting library from Nixtla.
*   [neural_prophet](https://github.com/ourownstory/neural_prophet) ![Github Stars](https://img.shields.io/github/stars/ourownstory/neural_prophet.svg?style=social)
*   [NeuralForecast](https://github.com/Nixtla/neuralforecast) ![Github Stars](https://img.shields.io/github/stars/Nixtla/neuralforecast.svg?style=social) - Neural forecasting library for large-scale time series modeling.
*   [Nixtla](https://github.com/Nixtla/nixtla) ![Github Stars](https://img.shields.io/github/stars/Nixtla/nixtla.svg?style=social) - TimeGPT and forecasting foundation-model toolkit from Nixtla.
*   [orbit](https://github.com/uber/orbit) ![Github Stars](https://img.shields.io/github/stars/uber/orbit.svg?style=social)
*   [pmdarima](https://github.com/alkaline-ml/pmdarima) ![Github Stars](https://img.shields.io/github/stars/alkaline-ml/pmdarima.svg?style=social)
*   [prophet](https://github.com/facebook/prophet) ![Github Stars](https://img.shields.io/github/stars/facebook/prophet.svg?style=social)
*   [sktime](https://github.com/sktime/sktime) ![Github Stars](https://img.shields.io/github/stars/sktime/sktime.svg?style=social)
*   [statsforecast](https://github.com/Nixtla/statsforecast) ![Github Stars](https://img.shields.io/github/stars/Nixtla/statsforecast.svg?style=social)
*   [stumpy](https://github.com/TDAmeritrade/stumpy) ![Github Stars](https://img.shields.io/github/stars/TDAmeritrade/stumpy.svg?style=social)
*   [tbats](https://github.com/intive-DataScience/tbats) ![Github Stars](https://img.shields.io/github/stars/intive-DataScience/tbats.svg?style=social)
*   [temporian](https://github.com/google/temporian) ![Github Stars](https://img.shields.io/github/stars/google/temporian.svg?style=social)
*   [tsfresh](https://github.com/blue-yonder/tsfresh) ![Github Stars](https://img.shields.io/github/stars/blue-yonder/tsfresh.svg?style=social)
*   [tslearn](https://github.com/tslearn-team/tslearn) ![Github Stars](https://img.shields.io/github/stars/tslearn-team/tslearn.svg?style=social)

### Product Affinity/Association
*   [Association Rules (apriori, eclat) \ - R Package](https://cran.r-project.org/web/packages/arules/index.html)

### Recommender Systems
*   [lightfm](https://github.com/lyst/lightfm) ![Github Stars](https://img.shields.io/github/stars/lyst/lightfm.svg?style=social) - A Python implementation of LightFM, a hybrid recommendation algorithm.
*   [recmetrics](https://github.com/statisticianinstilettos/recmetrics) ![Github Stars](https://img.shields.io/github/stars/statisticianinstilettos/recmetrics.svg?style=social) - A library of metrics for evaluating recommender systems
*   [openrec](https://github.com/ylongqi/openrec) ![Github Stars](https://img.shields.io/github/stars/ylongqi/openrec.svg?style=social) - OpenRec is an open-source and modular library for neural network-inspired recommendation algorithms
*   [recommenders](https://github.com/microsoft/recommenders) ![Github Stars](https://img.shields.io/github/stars/microsoft/recommenders.svg?style=social)
*   [Surprise](https://github.com/NicolasHug/Surprise) ![Github Stars](https://img.shields.io/github/stars/NicolasHug/Surprise.svg?style=social)

### Data & Utilities
*   [gapandas4](https://github.com/practical-data-science/gapandas4) ![Github Stars](https://img.shields.io/github/stars/practical-data-science/gapandas4.svg?style=social) - GAPandas4 is a Python package for querying the Google Analytics Data API for GA4 and displaying the results in a Pandas dataframe.
*   [Decoy](https://github.com/EqualExperts/decoy) ![Github Stars](https://img.shields.io/github/stars/EqualExperts/decoy.svg?style=social)
*   [SDV](https://github.com/sdv-dev/SDV) ![Github Stars](https://img.shields.io/github/stars/sdv-dev/SDV.svg?style=social)

## Papers, Blogs, & Resources
Articles, papers, and other resources organized by topic.

### Geo Experimentation & Lift Testing
*   [CausalImpact \ (paper)](https://research.google/pubs/pub41854/) - An important problem in econometrics and marketing is to infer the causal impact that a designed market intervention has exerted on an outcome metric over time. In order to allocate a given budget optimally, for examp...
*   [Meta Geolift \ (paper)](https://eml.berkeley.edu/~jrothst/workingpapers/BMFR_Synth_Nov_2018.pdf)
*   [Online A/B Testing \ - Trustworthy Online Controlled Experiments](https://experimentguide.com/)

### Platform Incrementality & Lift Testing
*   [Ghost Ads: Improving the Economics of Measuring Online Ad Effectiveness](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2620078) - Foundational paper on ghost ads and platform-based incrementality measurement.
*   [How DoorDash Measures Restaurant Ad Incrementality with Ghost Ads](https://advertising.doordash.com/en-us/resources/measuring-incrementality-with-ghost-ads-at-doordash) - Implementation-oriented writeup on using ghost ads in practice.
*   [How Instacart Measures the True Value of Advertising: The Methodology of Ad Incrementality](https://www.instacart.com/company/how-its-made/how-instacart-measures-the-true-value-of-advertising-the-methodology-of-ad-incrementality/) - Useful retail media perspective on randomized incrementality measurement.

### Experimentation & A/B Testing
*   [Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment Data](https://exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf) - Original CUPED paper.
*   [Trustworthy Online Controlled Experiments](https://www.cambridge.org/core/books/trustworthy-online-controlled-experiments/D97B26382EB0EB2DC2019A7A7B518F59) - Canonical book on large-scale A/B testing practice.
*   [Diagnosing Sample Ratio Mismatch in A/B Testing](https://www.microsoft.com/en-us/research/group/experimentation-platform-exp/articles/diagnosing-sample-ratio-mismatch-in-a-b-testing/) - Practical SRM debugging guide from Microsoft.

### MMM Calibration & Tuning
*   [Media Effect Estimation with PyMC: Adstock, Saturation & Diminishing Returns](https://juanitorduz.github.io/pymc_mmm/) - Practical walkthrough of core MMM transformations and modeling choices.

### Segmentation & Personas
*   [Data-driven Buyer Personas: What They Are and How to Build Them](https://clay.global/blog/data-driven-personas) - Practical guide to turning persona work into something more evidence-based and operational.

### Causal Inference & Bayesian Analysis
*   [scdata](https://github.com/nppackages/scpi/blob/HEAD/stata/scdata.pdf) - Prediction and inference procedures for synthetic control methods with multiple treated units and staggered adoption.
*   [scest](https://github.com/nppackages/scpi/blob/HEAD/stata/scest.pdf) - Prediction and inference procedures for synthetic control methods with multiple treated units and staggered adoption.
*   [scpi](https://github.com/nppackages/scpi/blob/HEAD/stata/scpi.pdf) - Prediction and inference procedures for synthetic control methods with multiple treated units and staggered adoption.
*   [scplot](https://github.com/nppackages/scpi/blob/HEAD/stata/scplot.pdf) - Prediction and inference procedures for synthetic control methods with multiple treated units and staggered adoption.
*   [A More Credible Approach to Parallel Trends](https://doi.org/10.1093/restud/rdad018) - Honest DiD sensitivity analysis for violations of parallel trends.
*   [Augmented Difference-in-Differences](https://doi.org/10.1287/mksc.2022.1406)
*   [Bayesian](https://arxiv.org/pdf/2503.06454) - The challenges posed by high-dimensional data and use of the simplex constraint are two major concerns in the empirical application of the synthetic control method (SCM) in econometric studies. To address both issues...
*   [CausalML: Python package for causal machine learning](https://arxiv.org/abs/2002.11631)
*   [Difference-in-Differences with Multiple Time Periods](https://doi.org/10.1016/j.jeconom.2020.12.001) - Callaway and Sant'Anna's reference paper for staggered DiD estimation.
*   [Estimating Ad Effectiveness Using Geo Experiments in a Time-Based Regression Framework](https://research.google.com/pubs/pub45950.html) - Two previously published papers (Vaver and Koehler, 2011, 2012) describe<br>a model for analyzing geo experiments. This model was designed to measure<br>advertising effectiveness using the rigor of a randomized experi...
*   [Estimating Dynamic Treatment Effects in Event Studies with Heterogeneous Treatment Effects](https://doi.org/10.1016/j.jeconom.2020.09.006) - Sun and Abraham's event-study correction for staggered treatment timing.
*   [External Resource](https://arxiv.org/abs/2004.11408) - Gaussian processes are powerful non-parametric probabilistic models for stochastic functions. However, the direct implementation entails a complexity that is computationally intractable when the number of observations...
*   [External Resource](https://arxiv.org/abs/2105.07060) - How to measure the incremental Return On Ad Spend (iROAS) is a fundamental problem for the online advertising industry. A standard modern tool is to run randomized geo experiments, where experimental units are non-ove...
*   [Feature Selection Methods for Uplift Modeling](https://arxiv.org/abs/2005.03447)
*   [Forward Difference-in-Differences](https://doi.org/10.1287/mksc.2022.0212)
*   [GeoX paper](https://research.google/pubs/pub38355/) - Advertisers have a fundamental need to quantify the effectiveness of their advertising. For search ad spend, this information provides a basis for formulating strategies related to bidding, budgeting, and campaign des...
*   [infernce for simplex weights](https://arxiv.org/pdf/2501.15692) - In many applications, the parameter of interest involves a simplex-valued weight which is identified as a solution to an optimization problem. Examples include synthetic control methods with group-level weights and va...
*   [L1-INF Synthetic Control](https://arxiv.org/abs/2510.26053) - This paper reinterprets the Synthetic Control (SC) framework through the lens of weighting philosophy, arguing that the contrast between traditional SC and Difference-in-Differences (DID) reflects two distinct modelin...
*   [Matched Markets paper](https://research.google/pubs/pub48983/) - Although randomized controlled trials are regarded as the &quot;gold standard&quot; for causal inference, advertisers have been hesitant to embrace them as their primary method of experimental design and analysis due...
*   [Measuring Ad Effectiveness Using Geo Experiments](http://static.googleusercontent.com/media/research.google.com/en//pubs/archive/38355.pdf)
*   [Multivariate and propensity score matching software with automated balance optimization: The Matching package for R](https://www.jstatsoft.org/article/view/v042i07) - Practical matching and balance-diagnostics reference cited in `pysmatch`.
*   [N-HiTS paper](https://arxiv.org/abs/2201.12886) - Recent progress in neural forecasting accelerated improvements in the performance of large-scale forecasting systems. Yet, long-horizon forecasting remains a very difficult task. Two common challenges afflicting the t...
*   [Prediction Intervals for Synthetic Control Methods](https://nppackages.github.io/references/Cattaneo-Feng-Titiunik_2021_JASA.pdf)
*   [Pretest with Caution: Event-Study Estimates after Testing for Parallel Trends](https://doi.org/10.1257/aeri.20210236) - Roth on why pre-trends tests can have low power and distort inference.
*   [Proximal Causal Inference for SCM (Surrogates)](https://arxiv.org/abs/2308.09527) - The synthetic control method (SCM) has become a popular tool for estimating causal effects in policy evaluation, where a single treated unit is observed, and a heterogeneous set of untreated units with pre- and post-p...
*   [Proximal SCM Framework](https://arxiv.org/abs/2108.13935) - Synthetic control (SC) methods are commonly used to estimate the treatment effect on a single treated unit in panel data settings. An SC is a weighted average of control units built to match the treated unit, with wei...
*   [Relaxed Balanced Synthetic Control](https://arxiv.org/abs/2508.01793) - The synthetic control method (SCM) is widely used for constructing the counterfactual of a treated unit based on data from control units in a donor pool. Allowing the donor pool contains more control units than time p...
*   [scpi: Uncertainty Quantification for Synthetic Control Methods](https://nppackages.github.io/references/Cattaneo-Feng-Palomba-Titiunik_2025_JSS.pdf)
*   [Stacked Difference-in-Differences](https://www.nber.org/papers/w32054) - Wing, Freedman, and Hollingsworth on stacked DiD for staggered adoption designs.
*   [synthetic business cycles](https://arxiv.org/pdf/2505.22388) - This paper investigates the use of synthetic control methods for causal inference in macroeconomic settings when dealing with possibly nonstationary data. While the synthetic control approach has gained popularity for...
*   [Synthetic Control Method (Vanilla SCM)](https://doi.org/10.1198/jasa.2009.ap08746)
*   [Synthetic Control Method with Nonlinear Outcomes](https://arxiv.org/abs/2306.01967) - The synthetic control estimator (Abadie et al., 2010) is asymptotically unbiased assuming that the outcome is a linear function of the underlying predictors and that the treated unit can be well approximated by the sy...
*   [Synthetic Control with Multiple Outcomes (TLP and SBMF)](https://arxiv.org/abs/2304.02272) - We generalize the synthetic control (SC) method to a multiple-outcome framework, where the conventional pre-treatment time dimension is supplemented with the extra dimension of related outcomes in computing the SC wei...
*   [Synthetic Controls for Experimental Design](https://arxiv.org/abs/2108.02196) - This article studies experimental design in settings where the experimental units are large aggregate entities (e.g., markets), and only one or a small number of units can be exposed to the treatment. In such settings...
*   [TBR paper](https://research.google/pubs/pub45950/) - Two previously published papers (Vaver and Koehler, 2011, 2012) describe<br>a model for analyzing geo experiments. This model was designed to measure<br>advertising effectiveness using the rigor of a randomized experi...
*   [TCN paper](https://arxiv.org/abs/1803.01271) - For most deep learning practitioners, sequence modeling is synonymous with recurrent networks. Yet recent results indicate that convolutional architectures can outperform recurrent networks on tasks such as audio synt...
*   [The central role of the propensity score in observational studies for causal effects](https://stat.cmu.edu/~ryantibs/journalclub/rosenbaum_1983.pdf) - Foundational propensity score paper by Rosenbaum and Rubin.
*   [Treatment Effects with Instruments paper](https://arxiv.org/pdf/1905.10176.pdf)
*   [Two Step Synthetic Control](https://doi.org/10.1287/mnsc.2023.4878)
*   [Two-stage differences in differences](https://arxiv.org/abs/2207.05943) - Gardner's two-stage DiD estimator for staggered treatment timing.
*   [Uncertainty Quantification in Synthetic Controls with Staggered Treatment Adoption](https://nppackages.github.io/references/Cattaneo-Feng-Palomba-Titiunik_2025_RESTAT.pdf)
*   [xDeepFM](https://arxiv.org/abs/1803.05170) - Combinatorial features are essential for the success of many commercial models. Manually crafting these features usually comes with high cost due to the variety, volume and velocity of raw data in web-scale systems. F...
*   [External Resource](https://github.com/stan-dev/stan/wiki/prior-choice-recommendations#prior-for-degrees-of-freedom-in-students-t-distribution) - Stan development repository. The master branch contains the current release. The develop branch contains the latest stable development. See the Developer Process Wiki for details.
*   [External Resource](https://github.com/pymc-devs/pymc-examples/blob/main/examples/gaussian_processes/HSGP-Advanced.ipynb) - Examples of PyMC models, including a library of Jupyter notebooks.
*   [External Resource](https://github.com/pymc-devs/pymc-examples/blob/main/examples/gaussian_processes/HSGP-Basic.ipynb) - Examples of PyMC models, including a library of Jupyter notebooks.
*   [External Resource](https://github.com/pymc-devs/pymc-examples/blob/main/LICENSE) - Examples of PyMC models, including a library of Jupyter notebooks.
*   [External Resource](https://github.com/pymc-devs/pymc-examples/edit/main/examples/gaussian_processes/HSGP-Advanced.ipynb) - Examples of PyMC models, including a library of Jupyter notebooks.
*   [External Resource](https://github.com/pymc-devs/pymc-examples/edit/main/examples/gaussian_processes/HSGP-Basic.ipynb) - Examples of PyMC models, including a library of Jupyter notebooks.
*   [External Resource](https://github.com/pymc-devs/pymc-examples/pull/647) - Examples of PyMC models, including a library of Jupyter notebooks.
*   [External Resource](https://github.com/pymc-devs/pymc-examples/pull/668) - Examples of PyMC models, including a library of Jupyter notebooks.
*   [2021 Conference on Digital Experimentation @ MIT (CODE@MIT)](https://ide.mit.edu/events/2021-conference-on-digital-experimentation-mit-codemit/)
*   [Be Careful When Interpreting Predictive Models in Search of Causal Insights](https://towardsdatascience.com/be-careful-when-interpreting-predictive-models-in-search-of-causal-insights-e68626e664b6)
*   [diff-diff Tutorials](https://github.com/igerber/diff-diff/tree/main/docs/tutorials) - Notebook collection covering basic DiD, staggered adoption, synthetic DiD, parallel trends checks, Honest DiD, power analysis, and survey-aware DiD workflows.
*   [Gaussian Processes: HSGP Advanced Usage](https://www.pymc.io/projects/examples/en/latest/gaussian_processes/HSGP-Advanced.html)
*   [Gaussian Processes: HSGP Reference & First Steps](https://www.pymc.io/projects/examples/en/latest/gaussian_processes/HSGP-Basic.html)
*   [The Kernel Cookbook: Advice on Covariance functions](https://www.cs.toronto.edu/~duvenaud/cookbook/)

### Attribution
*   ["Shapley Value Methods for Attribution Modeling in Online Advertising" by Zhao, et al.](https://arxiv.org/abs/1804.05327) - This paper re-examines the Shapley value methods for attribution analysis in the area of online advertising. As a credit allocation solution in cooperative game theory, Shapley value method directly quantifies the con...
*   [Attribution modeling paper (arXiv)](https://arxiv.org/pdf/1502.06657.pdf)
*   [Attribution paper collection (Southampton ePrints)](https://eprints.soton.ac.uk/380534/1/GHLEFMG_FGMJHM_VJ1QM9QF.pdf)
*   [Attribution survival model paper](http://www0.cs.ucl.ac.uk/staff/w.zhang/rtb-papers/attr-survival.pdf)
*   [Causally motivated attribution in online advertising](https://dstillery.com/wp-content/uploads/2016/07/CAUSALLY-MOTIVATED-ATTRIBUTION.pdf)
*   [Data-driven conversion attribution paper](http://www0.cs.ucl.ac.uk/staff/w.zhang/rtb-papers/data-conv-att.pdf)
*   [Shapley Value Methods for Attribution Modeling in Online Advertising (PDF)](https://arxiv.org/pdf/1808.03737.pdf)
*   [ChannelAttribution](https://github.com/DavideAltomare/ChannelAttribution/tree/master) - Markov Model for Online Multi-Channel Attribution
*   [Attribution model comparison paper (Recercat)](https://www.recercat.cat/bitstream/handle/2072/290758/201702.pdf?sequence=1)

### Customer Analytics (CLV, Segmentation, Uplift)
*   ["Counting Your Customers" the Easy Way: An Alternative to the Pareto/NBD Model](http://brucehardie.com/papers/018/fader_et_al_mksc_05.pdf) - Seminal BG/NBD reference for practical non-contractual CLV modeling.
*   [Counting Your Customers: Who Are They and What Will They Do Next?](https://www.jstor.org/stable/2631608?seq=1#page_scan_tab_contents) - Foundational customer-base-analysis paper on CLV, repeat buying, and retention behavior.
*   [The Gamma-Gamma Model of Monetary Value](http://www.brucehardie.com/notes/025/gamma_gamma.pdf) - Practical companion for modeling customer spend in CLV workflows.
*   [Bruce Hardie: Notes](https://www.brucehardie.com/notes/) - Deep collection of CLV, retention, BG/NBD, Gamma-Gamma, and customer-base-analysis notes and worked examples.
*   [BTYD models overview and intuition - Peter Fader Talk](https://www.youtube.com/watch?v=guj2gVEEx4s)

### Multi Armed Bandits
*   [Applications \ - Pricing](https://www.sciencedirect.com/science/article/pii/S0888613X17303821)
*   [Multi-Armed Bandits Overview](https://arxiv.org/pdf/1904.07272.pdf)
*   [Applications \ - Amazon Causal Marketing](https://www.amazon.science/publications/contextual-multi-armed-bandits-for-causal-marketing)
*   [Applications \ - Application to Performance Marketing](https://assets.researchsquare.com/files/rs-2684616/v1/743a3c4b9360ea574bea92dd.pdf?c=1678945781)
*   [Applications \ - Meta Ad Placement](https://research.facebook.com/blog/2021/4/auto-placement-of-ad-campaigns-using-multi-armed-bandits/)
*   [Applications \ - Stitchfix Experimentation](https://multithreaded.stitchfix.com/blog/2020/08/05/bandits/)

### Recommender Systems
*   [Tech Company Implementations \ - Alibaba](https://arxiv.org/pdf/1905.06874v1.pdf)
*   [Tech Company Implementations \ - LinkedIn](https://arxiv.org/pdf/1809.06481.pdf)
*   [Tech Company Implementations \ - Pinterest](https://cs.stanford.edu/people/jure/pubs/pixie-www18.pdf)
*   [Tech Company Implementations \ - TikTok](https://arxiv.org/pdf/2007.07203.pdf)
*   [Tech Company Implementations \ - Youtube](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45530.pdf)
*   [Tech Company Implementations \ - DoorDash](https://doordash.news/company/powering-search-recommendations-at-doordash/)
*   [Tech Company Implementations \ - Etsy](https://www.etsy.com/codeascraft/how-we-built-a-multi-task-canonical-ranker-for-recommendations-at-etsy)
*   [Tech Company Implementations \ - Netflix](https://dl.acm.org/doi/pdf/10.1145/2843948)

## Key Researchers
*   [Bruce Hardie](https://www.london.edu/faculty-and-research/faculty-profiles/h/hardie-bgs) - Customer analytics and CLV researcher known for probability models for customer-base analysis, retention, and valuation.
*   [Byron Sharp](https://marketingscience.info/staff/professor-byron-sharp/) - Professor of Marketing Science and Director of the Ehrenberg-Bass Institute. Author of [How Brands Grow](https://www.amazon.com/How-Brands-Grow-Marketers-Know/dp/0195573560).
*   [Catherine Tucker](https://mitsloan.mit.edu/faculty/directory/catherine-tucker) - Sloan Distinguished Professor of Management at MIT Sloan. Expert in digital marketing, privacy, and online advertising.
*   [Dominique Hanssens](https://www.anderson.ucla.edu/faculty-and-research/marketing/faculty/hanssens) - Distinguished Research Professor of Marketing at UCLA Anderson. Known for Long-Term Impact of Marketing.
*   [Garrett Johnson](https://www.bu.edu/questrom/profiles/garrett-johnson/) - Associate Professor of Marketing at Boston University. Co-author of "Ghost Ads" and research on privacy/GDPR.
*   [Guido Imbens](https://www.gsb.stanford.edu/faculty-research/faculty/guido-w-imbens) - Applied Econometrics Professor and Professor of Economics at Stanford Graduate School of Business. Nobel Laureate (2021) for methodological contributions to the analysis of causal relationships.
*   [Hema Yoganarasimhan](https://faculty.washington.edu/hemay/) - Quantitative marketing researcher focused on digital marketing, online advertising, experimentation, pricing, and machine learning for large-scale marketing decisions.
*   [Koen Pauwels](https://damore-mckim.northeastern.edu/people/koen-pauwels/) - Marketing effectiveness and marketing-mix-modeling scholar focused on attribution, field experiments, ROI measurement, and long-term brand impact.
*   [Peter Fader](https://marketing.wharton.upenn.edu/profile/faderp/) - Frances and Pei-Yuan Chia Professor of Marketing at The Wharton School. Author of [Customer Centricity](https://www.amazon.com/Customer-Centricity-Customers-Strategic-Essentials/dp/1613631022).
*   [Randall Lewis](http://econinformatics.com/blog/about/) - Economic Research Scientist at Netflix. Known for work on "Ghost Ads" and measuring advertising effectiveness.
*   [Ron Berman](https://marketing.wharton.upenn.edu/profile/ronber/) - Associate Professor of Marketing at The Wharton School. Focuses on online marketing, marketing analytics, and game theory.
*   [Stefan Wager](https://web.stanford.edu/~swager/index.html) - Associate Professor of Operations, Information & Technology at Stanford GSB. Research on causal inference and statistical learning.
*   [Susan Athey](https://www.gsb.stanford.edu/faculty-research/faculty/susan-athey) - The Economics of Technology Professor at Stanford Graduate School of Business. Leading researcher in the intersection of machine learning and causal inference.

## Books & Courses
*   [Bayesian Analysis with Python](https://github.com/aloctavodia/BAP3) ![Github Stars](https://img.shields.io/github/stars/aloctavodia/BAP3.svg?style=social) - Book code and examples for applied Bayesian modeling in Python.
*   [Causal Inference for the Brave and True](https://matheusfacure.github.io/python-causality-handbook/landing-page.html) - Practical open-source guide to causal inference with intuitive explanations and Python examples.
*   [Causal Inference: The Mixtape](https://mixtape.scunning.com/) - Accessible, example-driven introduction to causal inference methods including DiD, IV, panel designs, and research design.
*   [Causality](https://cambridge.org/9780521895606) - Technical causal-inference reference by Judea Pearl.
*   [Doing Bayesian Data Analysis](https://github.com/cluhmann/DBDA-python) ![Github Stars](https://img.shields.io/github/stars/cluhmann/DBDA-python.svg?style=social) - Python port of a well-known applied Bayesian statistics text.
*   [Forecasting: Principles and Practice](https://otexts.com/fpp3/) - Strong practical forecasting text for analysts who need foundational time-series intuition.
*   [Introduction to Algorithmic Marketing](https://www.algorithmicmarketingbook.com/) - Broad practitioner-oriented guide to advanced marketing automation, search, recommendations, pricing, promotions, and measurement.
*   [Market Segmentation Analysis](https://link.springer.com/book/10.1007/978-981-10-8818-6) - Book-length treatment of segmentation methods and strategy.
*   [Probabilistic Programming & Bayesian Methods for Hackers](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers) ![Github Stars](https://img.shields.io/github/stars/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers.svg?style=social) - Friendly introduction to Bayesian thinking and probabilistic programming.
*   [Statistical Rethinking](https://github.com/pymc-devs/resources/tree/master/Rethinking) - Strong applied Bayesian modeling foundation with a large teaching and practice community around it.
*   [The Book of Why](https://www.amazon.com/Book-Why-Science-Cause-Effect/dp/046509760X) - Popular but useful introduction to causal thinking and causal diagrams.

## Blogs
*   [An Analyst's Guide to MMM | Robyn](https://facebookexperimental.github.io/Robyn/docs/analysts-guide-to-MMM/)
*   [Bayesian Media Mix Modeling for Marketing Optimization](https://www.pymc-labs.io/blog-posts/bayesian-media-mix-modeling-for-marketing-optimization/)
*   [Causal Analysis with PyMC: Answering "What If?" with the New do Operator](https://www.pymc-labs.com/blog-posts/causal-analysis-with-pymc-answering-what-if-with-the-new-do-operator/)
*   [Causal Sales Analytics: Are my sales incremental or cannibalistic?](https://www.pymc-labs.com/blog-posts/causal-sales-analytics-are-my-sales-incremental-or-cannibalistic)
*   [Decision Making Processes in Marketing Mix Modelling (PDF)](https://github.com/facebookexperimental/Robyn/blob/main/publication/Decision_Making_Processes_in_Marketing_Mix_Modelling_A_Survey_of_MMM_Providers.pdf)
*   [How Wayfair Uses Geo Experiments to Measure Incrementality](https://www.aboutwayfair.com/careers/tech-blog/how-wayfair-uses-geo-experiments-to-measure-incrementality)
*   [Juan Orduz's Blog](https://juanitorduz.github.io/) - High-signal walkthroughs on MMM, Bayesian modeling, forecasting, and causal inference.
*   [Marketing and Metrics](https://www.marketingandmetrics.com/) - Koen Pauwels on marketing effectiveness, attribution, MMM, dashboards, and return on marketing investment.
*   [Matheus Facure's Blog](https://matheusfacure.github.io/) - Practical writing on causal inference and applied econometrics.
*   [Modelling Changes in Marketing Effectiveness Over Time](https://www.pymc-labs.com/blog-posts/modelling-changes-marketing-effectiveness-over-time/)
*   [Occam’s Razor by Avinash Kaushik](https://www.kaushik.net/avinash/) - Long-running independent blog on digital analytics, measurement, experimentation, and marketing strategy.
*   [PyMC Labs Blog](https://www.pymc-labs.io/blog/) - Broader archive of Bayesian modeling, MMM, causal inference, and applied case studies.
*   [Python/STAN Implementation of Multiplicative Marketing Mix Model](https://towardsdatascience.com/python-stan-implementation-of-multiplicative-marketing-mix-model-with-deep-dive-into-adstock-a7320865b334/)
*   [Recast Blog](https://getrecast.com/blog/) - Industry-facing writing on MMM, incrementality, and measurement systems.
*   [Recast Technical Documentation](https://docs.getrecast.com/docs/recast-model-technical-documentation) - Public technical documentation for Recast's Bayesian MMM, covering model specification, priors, and validation methodology. A useful reference point for how a commercial MMM is structured relative to the open-source frameworks listed elsewhere in this repo.
*   [Reducing Customer Acquisition Costs: How we helped optimizing HelloFresh's marketing budget](https://www.pymc-labs.com/blog-posts/reducing-customer-acquisition-costs-how-we-helped-optimizing-hellofreshs-marketing-budget/)
*   [The Future is Modeled: A How-to Guide for Advanced Marketing Mix Models](https://www.deloitte.com/content/dam/assets-zone2/es/es/docs/services/consulting/2023/Deloitte-es-estrategia-y-operaciones-guide-advanced-marketing-mix-models.pdf)
*   [Unified Marketing Measurement: The Power of Blending Methodologies (PDF)](https://www.thinkwithgoogle.com/_qs/documents/15401/UnifiedMarketingMeasurement.pdf)
*   [Unified online marketing measurement - Think with Google](https://www.thinkwithgoogle.com/intl/en-emea/marketing-strategies/data-and-measurement/unified-online-marketing-measurement/)
*   [Using Geographic Splitting & Optimization Techniques to Measure Marketing Performance](https://www.aboutwayfair.com/tech-innovation/using-geographic-splitting-optimization-techniques-to-measure-marketing-performance)

## Resources
*   [A Time-Based Regression Matched Markets Approach for Designing Geo Experiments](https://research.google/pubs/a-time-based-regression-matched-markets-approach-for-designing-geo-experiments/) - Although randomized controlled trials are regarded as the &quot;gold standard&quot; for causal inference, advertisers have been hesitant to embrace them as their primary method of experimental design and analysis due...
*   [Ebay Hybrid Geo/User experiment](https://dl.acm.org/doi/10.1145/2940716.2940717) - Hybrid experimental design reference for combining geographic and user-level thinking.
*   [Estimating Ad Effectiveness using Geo Experiments in a Time-Based Regression Framework](https://research.google/pubs/estimating-ad-effectiveness-using-geo-experiments-in-a-time-based-regression-framework/) - Two previously published papers (Vaver and Koehler, 2011, 2012) describe<br>a model for analyzing geo experiments. This model was designed to measure<br>advertising effectiveness using the rigor of a randomized experi...
*   [Measuring Ad Effectiveness Using Geo Experiments](https://research.google/pubs/measuring-ad-effectiveness-using-geo-experiments/) - Advertisers have a fundamental need to quantify the effectiveness of their advertising. For search ad spend, this information provides a basis for formulating strategies related to bidding, budgeting, and campaign des...
*   [Share of Search as a Predictive Metric](https://ipa.co.uk/knowledge/publications-reports/share-of-search-thinkbox) - Widely cited brand-measurement resource from Les Binet and Thinkbox.
*   [Switchback tests](https://doordash.engineering/2018/02/13/switchback-tests-and-randomized-experimentation-under-network-effects-at-doordash/) - Useful primer on experimentation under interference and network effects.
*   [The Modern Measurement Playbook](https://www.thinkwithgoogle.com/_qs/documents/18393/For_pub_on_TwG___External_Playbook_Modern_Measurement.pdf) - Google's practical measurement framework connecting MMM, attribution, and incrementality.
*   [Trimmed Match Design for Randomized Paired Geo Experiments](https://research.google/pubs/trimmed-match-design-for-randomized-paired-geo-experiments/) - How to measure the incremental return on Ad spend (iROAS) is a fundamental problem for the online advertising industry. A standard modern tool is to run randomized geo experiments, where experimental units are non-ove...

## About

Feel free to submit an issue or pull request with any suggestions!

This list is maintained by [Shako Stats](https://shakostats.com).

Connect with me on [LinkedIn](https://www.linkedin.com/in/shako-stats).
