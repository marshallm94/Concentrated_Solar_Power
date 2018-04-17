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
