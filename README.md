# Time series prediction using RNNs

## Introduction

The goal was to develop a recurrent network model that achieves the closest possible prediction of average daily temperature (expressed in degrees Fahrenheit) to the real values observed by measuring stations. 

## Data

I used the dataset called Global Surface Summary of the Day Data (https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ncdc:C00516). The data comes from 2010 records obtained in Japan. The training set comes from 90 weather stations, the test set from 10. The record from each station has a set of weather information pertaining to 365 days of the year. The data are saved as Excel sheets.

Each dataset contains information describing the station (station_id), the time of measurement (year, month, day), and detailed data provided by the station (other, unmentioned columns - e.g., average temperature for the day, or wind speed). The goal is to forecast the average daily temperature (mean temperature [deg F]) based on the full station record for the last two weeks. This means that, for example, when forecasting the average daily temperature for January 29, 2010, the recurrent network will take as its input 14 values from January 15, 2010 to January 28, 2010 (including historical temperature data).

## Installation

To install all required packages, just use:

```
pip install -r requirements.txt
```
