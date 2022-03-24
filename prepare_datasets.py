# Library imports
from   collections import Counter
from   time import sleep
from   tqdm import tqdm
import pandas as pd
import numpy as np
import tarfile,os
import zlib
import sys

# Setting the state of the rng
np.random.seed(42)

# Screen clearance
if os.name == 'nt':
    os.system('cls')
else:
    os.system('clear')

# The input dataset for processing is:
# https://www.kaggle.com/noaa/noaa-global-surface-summary-of-the-day?select=gsod_all_years.zip

# ------------------------------------------------------------------------------------
# Basic procesing procedures:
# ------------------------------------------------------------------------------------

# A data loader procedure for the dataset
def read_dataset_for_year(infile):
    name_core,_ = os.path.splitext(infile)
    name_core = os.path.basename(name_core)
    output_fname = f"converted_{name_core}.hdf"
    
    # The decompression is carried out on the fly.
    if not os.path.isfile(output_fname):

        tar = tarfile.open(infile)

        print(f"Data are being loaded from {infile}")

        dataset_df      = []
        first_iteration = True
        
        # Documentation of the dataset specifies which letters of partial reports
        # contain data relevant to each measure of the dataset.
        for member in tqdm(tar.getmembers()):
            
            # decompression
            f=tar.extractfile(member)
            if f is None: continue
            content=f.read()
            content = zlib.decompress(content, 16+zlib.MAX_WBITS).decode('utf-8')
            
            # split file into rows
            data_rows = content.split('\n')
            
            # for the rest of data - gather them from the rest of rows
            for i in range(1,len(data_rows)):
                data_row = data_rows[i]
                
                if len(data_row) != 138: continue
                
                content_row = {}
                content_row.update({'station_id':str(data_row[0:6])})
                content_row.update({'WBAN':int(data_row[7:12])})
                content_row.update({'year':int(data_row[14:18])})
                content_row.update({'month':int(data_row[18:20])})
                content_row.update({'day':int(data_row[20:22])})
                content_row.update({'mean temperature [deg F]':float(data_row[24:30])})
                content_row.update({'mean dew point [deg F]':float(data_row[35:41])})
                content_row.update({'mean pressure (sea level) [Pa]':float(data_row[46:52])})
                content_row.update({'mean pressure (station) [Pa]':float(data_row[57:63])})
                content_row.update({'mean visibility [mile]':float(data_row[68:73])})
                content_row.update({'mean wind speed [knot]':float(data_row[78:83])})
                content_row.update({'max wind gust [knot]':float(data_row[68:73])})
                content_row.update({'max wind speed [knot]':float(data_row[95:100])})
                content_row.update({'max temperature [deg F]':float(data_row[102:108])})
                content_row.update({'min temperature [deg F]':float(data_row[110:116])})
                content_row.update({'total precipitation [inch]':float(data_row[118:123])})
                content_row.update({'snow depth [inch]':float(data_row[68:73])})
                
                for measure_name, measure_values in content_row.items():
                    if measure_name in ['station_id', 'year','month','day']: continue
                    if int(measure_values) % 9 == 0:
                        content_row[measure_name] = None
                
                if content_row['station_id'] == '999999':
                    content_row['station_id'] = None
                
                dataset_df.append(content_row)
        tar.close()

        dataset_df = pd.DataFrame(dataset_df)
        
        # Some measures are assumed to be equal to zero if there is no value provided.
        dataset_df['total precipitation [inch]'] = dataset_df['total precipitation [inch]'].fillna(0)
        dataset_df['snow depth [inch]'] = dataset_df['snow depth [inch]'].fillna(0)
        
        # Save cache file for later use
        dataset_df.to_hdf(output_fname, key="dataset_df")
    else:
        # Load data from cache if one is available.
        dataset_df = pd.read_hdf(output_fname, key="dataset_df")
    
    return dataset_df

# A diagnostic procedure for quick estimation of number of stations
# provided for each country.
def print_station_counts(dataset_df, isd_history, min_cnt):
    print("Number of stations for each country:")
    
    for country_name, freq in Counter(isd_history['CTRY']).most_common():
        if freq < min_cnt: continue
        print(f"\t- country: {country_name}, count: {freq}")

def obtain_station_numbers_for_country(isd_history, country_id):
    country_df  = isd_history[isd_history['CTRY']==country_id]
    station_ids = country_df['USAF'].to_numpy()
    
    # Numbers may appear double in the metadata file, thus we have to remove duplicates:
    station_ids = np.array(list(set(station_ids)))
    
    return station_ids

# A writing procedure. It also performs train-evaluation split.
def write_train_and_eval_data(data_df, station_ids, fname_core, fillna=True, train_prop=0.9):
    training_path = f"{fname_core}_training.xlsx"
    eval_path     = f"{fname_core}_evaluation.xlsx"
    print(f"Saving training data to: {training_path} and evaluation data to {eval_path}")
    
    id_mask   = np.in1d(data_df['station_id'].to_numpy(),station_ids)
    output_df = data_df[id_mask]
    
    # NA values removal:
    if fillna:
        output_df = output_df.groupby('station_id').apply(lambda x:x.ffill().bfill())
    
    # train-evaluation split
    np.random.shuffle(station_ids)
    
    split_idx = int(train_prop*len(station_ids))
    training_station_ids   = station_ids[0:split_idx]
    evaluation_station_ids = station_ids[split_idx::]
    
    print(f"Training proportion is {train_prop}. From {len(station_ids)} stations, {len(training_station_ids)} stations are selected for training, and {len(evaluation_station_ids)} ones are selected for evaluation.")
    
    training_set_df   = output_df[np.in1d(output_df['station_id'],training_station_ids)]
    evaluation_set_df = output_df[np.in1d(output_df['station_id'],evaluation_station_ids)]
    
    training_set_df   = training_set_df.drop(columns=['WBAN'])
    evaluation_set_df = evaluation_set_df.drop(columns=['WBAN'])
    
    training_set_df.to_excel(training_path, index=None)
    evaluation_set_df.to_excel(eval_path, index=None)

# Reads the metadata file, which contains association between station numbers and
# the country in which each station is located.
def read_stations_metadata(dataset_df, metadata_file, metadata_cache_file):
    if not os.path.isfile(metadata_cache_file):
        print("Reading and filtering metadata")
        isd_history           = pd.read_csv(metadata_file)
        available_station_ids = dataset_df['station_id'].unique()
        
        print(isd_history)
        
        print()
        print("Checking present station ids:")
        is_present_in_current_dataset = []
        for station_id in tqdm(isd_history['USAF']):
            if station_id in available_station_ids:
                is_present_in_current_dataset.append(True)
            else:
                is_present_in_current_dataset.append(False)
        
        available_station_ids = np.array(available_station_ids)
        isd_history = isd_history.assign(is_present_in_current_dataset=is_present_in_current_dataset)
        isd_history.to_excel(metadata_cache_file, index=False)
    else:
        print(f"Reading metadata from cache file (\"{metadata_cache_file}\")")
        isd_history = pd.read_excel(metadata_cache_file)
    
    return isd_history

# A filtering procedure for finding out stations providing data with least
# missing values - useful to minimize necessity to interpolate/fill data
# at a final stage of data cleaning.
def select_with_least_missing_records(dataset_df, station_ids, num_selected):
    print("Performing the not-available (na) status reduction:")
    station_na_counts = []
    
    for station_id in tqdm(station_ids):
        station_df = dataset_df[dataset_df['station_id']==station_id]
        
        # Avoid including columns containing no data at all.
        column_length = station_df.shape[0]
        data_frame_contains_empty_columns = False
        for colname in station_df.columns:
            if colname in ['WBAN', 'station_id']: continue
            if station_df[colname].isna().sum() == column_length:
                data_frame_contains_empty_columns = True
                continue
        
        if not data_frame_contains_empty_columns:
            # Count how many na values are in the data frame.
            na_count   = station_df.isna().sum().sum()
            station_na_counts.append({'station_id':station_id,'na_count':na_count})
    
    # Sort the list of frames and choose ones with least count of na values
    station_na_counts = sorted(station_na_counts, key=lambda x:x['na_count'])
    station_na_counts = pd.DataFrame(station_na_counts)
    selected_ids      = station_na_counts['station_id'][0:num_selected]
    
    if len(selected_ids) < num_selected:
        print(f"Warning: {num_selected} are specified to be selected, but only {len(selected_ids)} meet criteria for doing so, user action is required")
    
    return selected_ids.to_numpy()

def select_dataset_for_country(dataset_df, isd_history, country_id, num_selected):
    print(f"Generating dataset for country: {country_id}")
    station_ids  = obtain_station_numbers_for_country(isd_history,country_id)
    selected_ids = select_with_least_missing_records(dataset_df, station_ids, num_selected)
    return selected_ids

# ------------------------------------------------------------------------------------
# Script execution:
# ------------------------------------------------------------------------------------

print('Reading input data:')
dataset_df  = read_dataset_for_year('indata/gsod_2010.tar')
isd_history = read_stations_metadata(dataset_df, 'indata/NOAA Global Surface Summary of the Day/isd-history.csv', 'dataset_metadata.cache.xlsx')

# Removal of columns containing near to no data:
dataset_df = dataset_df.drop(columns=['max wind speed [knot]'])

print()
print("Leaving only the present station ids...")
isd_history = isd_history[isd_history['is_present_in_current_dataset']==True]

print()
print_station_counts(dataset_df, isd_history, min_cnt = 80)

# Number of "densest" weather data to be selected for each
# country
num_selected  = 100

# FIPS codes are used:
# https://en.wikipedia.org/wiki/List_of_FIPS_country_codes
sel_countries = ['UK','FR','JA']

for country_id in sel_countries:
    print()
    station_ids = select_dataset_for_country(dataset_df, isd_history, country_id, num_selected)
    write_train_and_eval_data(dataset_df, station_ids, f'data_{country_id}_df')

print()

