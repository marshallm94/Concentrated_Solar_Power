import numpy as np
import pandas as pd
import os
import sys
import json
import matplotlib.pyplot as plt
plt.style.use('ggplot')

with open("/Users/marsh/api_keys/nrel_api.json", 'r') as f:
    data = json.load(f)
    api = data['api']

lat, lon, year = 35.5566, -115.4709, 2010

# attr = 'ghi,dhi,dni,wind_speed_10m_nwp,surface_air_temperature_nwp,solar_zenith_angle'

# attr = 'air_temperature,clearsky_dni,clearskycloud_type'

attr = 'air_temperature,clearsky_dhi,clearsky_dni,clearsky_ghi,cloud_type,dew_point,dhi,dni,fill_flag,ghi,relative_humidity,solar_zenith_angle,surface_albedo,surface_pressure,total_precipitable_water,wind_direction,wind_speed'


year = '2010'
leap = 'false'
interval = '30'
# Specify Coordinated Universal Time (UTC), 'true' will use UTC, 'false' will use the local time zone of the data.
# NOTE: In order to use the NSRDB data in SAM, you must specify UTC as 'false'. SAM requires the data to be in the
# local time zone.
utc = 'false'
name = 'Marshall+McQuillen'
# Your reason for using the NSRDB.
reason = 'education'
affiliation = 'Galvanzie'
email = 'marshallm94@gmail.com'
mailing_list = 'false'

# df = pd.read_csv(f'http://developer.nrel.gov/api/solar/nsrdb_0512_download.csv?wkt=POINT({lon}%20{lat})&names={year}&leap_day={leap}&interval={interval}&utc={utc}&full_name={name}&email={email}&affiliation={affiliation}&mailing_list={mailing_list}&reason={reason}&api_key={api}&attributes={attr}', skiprows=2)

df = pd.read_csv(f'http://developer.nrel.gov//api/solar/nsrdb_psm3_download.csv?wkt=POINT({lon}%20{lat})&names={year}&leap_day={leap}&interval={interval}&utc={utc}&full_name={name}&email={email}&affiliation={affiliation}&mailing_list={mailing_list}&reason={reason}&api_key={api}&attributes={attr}', skiprows=2)

# Set the time index in the pandas dataframe:
df = df.set_index(pd.date_range('1/1/{yr}'.format(yr=year), freq=interval+'Min', periods=525600/int(interval)))

# take a look
print(df.shape)
