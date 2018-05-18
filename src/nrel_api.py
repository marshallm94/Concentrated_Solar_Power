import pandas as pd
import json
import os

def get_nrel_data(api_key, latitude, longitude, year, interval, leap_year=False, utc=False):
    '''
    Uses the Physical Solar Model (PSM) v3 from the National Solar Radiation
    Database (NSRDB) API provided by the National Renewable Energy Laboratory
    (NREL).

    Parameters:
    ----------
    api_key : (str)
    latitude : (int)
    longitude : (int)
    year : (int)
    interval : (str)
        Time interval. Options are "30" or "60".
    leap_year : bool, default = False
        Whether the year specified is a leap year or not.
    utc : bool, default = False
        Whether the timestamps should be in Universal Time Coordinated (True)
        or not (False). If False, local time is used.


    Returns:
    ----------
    df : (Pandas DataFrame)
        Index of DataFrame set to the timestamps of the data retrieved.
    '''
    if utc:
        utc = 'true'
    if not utc:
        utc = 'false'

    if leap_year:
        leap = 'true'
    if not leap_year:
        leap = 'false'

    attr = 'air_temperature,clearsky_dhi,clearsky_dni,clearsky_ghi,cloud_type,dew_point,dhi,dni,fill_flag,ghi,relative_humidity,solar_zenith_angle,surface_albedo,surface_pressure,total_precipitable_water,wind_direction,wind_speed'

    df = pd.read_csv(f'http://developer.nrel.gov//api/solar/nsrdb_psm3_download.csv?wkt=POINT({lon}%20{lat})&names={year}&leap_day={leap}&interval={interval}&utc={utc}&full_name={name}&email={email}&affiliation={affiliation}&mailing_list={mailing_list}&reason={reason}&api_key={api}&attributes={attr}', skiprows=2)

    if leap_year:
        df = df.set_index(pd.date_range('1/1/{yr}'.format(yr=year), freq=interval+'Min', periods=527040/int(interval)))
    if not leap_year:
        df = df.set_index(pd.date_range('1/1/{yr}'.format(yr=year), freq=interval+'Min', periods=525600/int(interval)))

    return df

def aggregate_dataframes(directory, filepath):
    '''
    Concatenates all files in directory into one Pandas DataFrame and saves that
    DataFrame to filepath as a csv.

    Parameters:
    ----------
    directory : (str)
        The relative or absolute path to the directory containing the csv files
        to be concatenated into one DataFrame.
    filepath : (str)
        The filepath where the final DataFrame should be saved.

    Returns:
    ----------
    None
    '''
    frames = []
    for file in os.listdir(directory):
        frames.append(pd.read_csv(directory + file))
        print(f"\n{file} added to cache...")
    df = pd.concat(frames)
    df.to_csv(filepath)
    print(f"\nFinal DataFrame saved to {filepath}")
    return None


if __name__ == "__main__":
    with open("/Users/marsh/api_keys/nrel_api.json", 'r') as f:
        data = json.load(f)
        api = data['api']

    lat = 35.5566
    lon = -115.4709
    interval = '30'
    utc = 'false'
    name = 'Marshall+McQuillen'
    reason = 'education'
    affiliation = 'Galvanzie'
    email = 'marshallm94@gmail.com'
    mailing_list = 'false'

    for year in range(2003, 2017):
        if year % 4 == 0:
            df = get_nrel_data(api_key=api, latitude=lat, longitude=lon, year=year, leap_year=True, interval=interval)
        else:
            df = get_nrel_data(api_key=api, latitude=lat, longitude=lon, year=year, leap_year=False, interval=interval)
        df.to_csv(f"../data/yearly/{year}.csv")
        print(f"\n{year} DataFrame created and saved to file...")

    aggregate_dataframes("/Users/marsh/galvanize/dsi/projects/csp_capstone/data/yearly/", '/Users/marsh/galvanize/dsi/projects/csp_capstone/data/2003_2016.csv')
