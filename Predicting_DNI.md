# Predicting Direct Normal Irradiance (DNI)

### Background

While Photovoltaic (PV) power plants were the first form of solar powered energy to be implemented across the globe, Concentrated Solar Power, otherwise known as CSP, has seen a rise in deployment in recent years.

![Ivanpah Solar Electric](images/ivanpah_2.png)
173,000 Heliostats focus the Sun's energy on thee Power Tower's at Ivanpah Solar Electric in California.

In contrast to PV power plants, CSP technology has the ability to efficiently and inexpensively store the Sun's energy in the form of high temperature fluid (usually molten salt, however [new research](https://www.energy.gov/sites/prod/files/2016/08/f33/05-Ho_falling_particle_receiver_CSPSummit2016_0.pdf) is experimenting with other particles), which can be used to power a turbine when the the Sun isn't shining, given enough storage.

![](images/csp_diagram.jpg)

### The Problem

Knowing how much energy a plant will be able to produce is clearly highly dependent on the amount of DNI (measured in $\frac{Watts}{Meter^2}$ here) the heliostats receive, which, as we will see below, is highly irregular.

The goal of this analysis will be to see how accurately different models are able to predict DNI 15 minutes into the future from any given minute during the day.

![](images/correlation_plot.png)

![](images/avg_monthly_irradiance.png)

![](images/avg_hourly_irradiance.png)

![](images/irradiance_20170704.png)

![](images/irradiance_20170705.png)

### Data Preview

| Time Stamp  | DNI | DNI 15 Minutes from Now
| ------------- | ------------- | ------------- |
| 2015-06-11 11:00:00 | 889.945 | **892.412**
| 2015-06-11 11:01:00 | 891.852 | **892.833**
| 2015-06-11 11:02:00 | 893.199 | **893.499**
| 2015-06-11 11:03:00 | 892.430 | **892.421**
| 2015-06-11 11:04:00 | 890.426 | **892.787**
| 2015-06-11 11:05:00 | 890.070 | **893.600**
| 2015-06-11 11:06:00 | 890.360 | **895.570**
| 2015-06-11 11:15:00 | **892.412** | 896.047
| 2015-06-11 11:16:00 | **892.833** | 898.870
| 2015-06-11 11:17:00 | **893.499** | 900.519
| 2015-06-11 11:18:00 | **892.421** | 901.682
| 2015-06-11 11:19:00 | **892.787** | 898.668
| 2015-06-11 11:20:00 | **893.600** | 896.846
| 2015-06-11 11:21:00 | **895.570** | 898.221

### My Base Model vs The Benchmark

According to a [2013 article ](https://ac-els-cdn-com.www2.lib.ku.edu/S1364032113004334/1-s2.0-S1364032113004334-main.pdf?_tid=41f83cfe-de21-4d94-803f-a7470d8e51df&acdnat=1523992118_8198b37af15a4d0e24f139dfcd721a9d) that reviewed the current statistical models used to predict global irradiance, the benchmark model, called the *Persistence Model*, predicts that irradiance at time step $t$ is equal to irradiance at time step $t-1$. That is to say,

$$\hat{y_{t}} = y_{t-1}$$

I decided to use a Random Forest Regressor as my base model to compare against this benchmark. Using Scikit-Learn defaults, the errors for the Random Forest Regressor as well as the Persistence Model are show below. The Random Forest was trained over a 90 day period and then tested on the 30 days immediately thereafter.

| Train Start  | Train End | Test Start | Test End | Test RMSE | Persistence Model RMSE |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| 2016-08-29 | 2016-11-27 | 2016-11-28 | 2016-12-28 | 106.086681 | 104.662225
| 2006-02-09 | 2006-05-10 | 2006-05-11 | 2006-06-10 | 299.232761 | 130.591648
| 2016-01-05 | 2016-04-04 | 2016-04-05 | 2016-05-05 | 140.317356 | 134.845834
| 2012-05-19 | 2012-08-17 | 2012-08-18 | 2012-09-17 | 161.868311 | 141.924265
| 2015-09-15 | 2015-12-14 | 2015-12-15 | 2016-01-14 | 127.654941 | 103.548290
| 2011-07-30 | 2011-10-28 | 2011-10-29 | 2011-11-28 | 146.043241 | 116.224239
| 2010-09-19 | 2010-12-18 | 2010-12-19 | 2011-01-18 | 139.110381 | 113.714270
| 2014-10-24 | 2015-01-22 | 2015-01-23 | 2015-02-22 | 137.170199 | 107.731116
| 2010-05-03 | 2010-08-01 | 2010-08-02 | 2010-09-01 | 108.796504 | 104.764394
| 2013-11-14 | 2014-02-12 | 2014-02-13 | 2014-03-15 | 140.121192 | 110.903093


A MLP was developed to predict irradiance 24 hours in advance for PV plant in Italy ([article](https://ac-els-cdn-com.www2.lib.ku.edu/S0038092X10000782/1-s2.0-S0038092X10000782-main.pdf?_tid=85616b05-995e-48d0-bfa8-9fd7fae6cf27&acdnat=1523992062_3fc582bfafa044fee8fcabd7275d202b)). This MLP accepted as input mean daily irradiance and mean daily air temperature, which resulted in a "...correlation coefficient of more than 98% for sunny days and slightly less than 95% for cloudy days."[$^{[1]}$](https://ac-els-cdn-com.www2.lib.ku.edu/S0038092X10000782/1-s2.0-S0038092X10000782-main.pdf?_tid=85616b05-995e-48d0-bfa8-9fd7fae6cf27&acdnat=1523992062_3fc582bfafa044fee8fcabd7275d202b)
