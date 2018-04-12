### Module 2 Capstone Proposal - Marshall McQuillen

#### The Question

How accurately can Direct Normal Irradiance (DNI) be predicted at a granular level (30 minute intervals)?

#### The Data

NREL has an API where I can obtain DNI measurements in half hour intervals for a given latitude and longitude, for years dating back to ~2010. In addition, they have data on CSP and PV power plant locations in the United States, as well as other locations around the world (see MVP++).

Most of the latitude and longitude values are populated, however there are a few missing values. Some of this information could be scraped from an accompanying webpage to where the data set is located, however if that doesn't work I plan on using [Google's Geolocation API](https://developers.google.com/maps/documentation/geolocation/intro) to supplement the existing data.

##### MVP

For all solar power plants in the USA...

Use weather data (temp, cloud cover, humidity, etc,), time (30 min. intervals) and location (latitude, longitude) data to predict DNI.

* Compare all complex models (I want to train random forests, boosting and a regression Neural Network) against multiple linear regression

##### MVP+

Forecast DNI $X$ days out (with confidence intervals) for all locations in US

##### MVP++

* Do the above for solar power plants outside of the United States.

A few barplots showing the current status of various Concentrated Solar Power projects...

![](images/operational_csp_technologies.png)

![](images/under_construction_csp_technologies.png)

![](images/under_development_csp_technologies.png)
