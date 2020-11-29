import yellow_pages
import csv
from tqdm import tqdm
import pandas as pd
import typer
from pathlib import Path
import json
from geopy.geocoders import Nominatim
from collections import defaultdict
import numpy as np
from shapely.geometry import shape, Point
import scipy.stats
from shapely.ops import transform
from functools import partial

app = typer.Typer()

def get_ny_zipcodes():
    zips = []
    with open('ny_zipcodes.csv') as f:
        r = csv.reader(f)
        next(r)
        
        for row in r:
            zips.append(int(row[0]))

    return zips

@app.command()
def scrape_type(place_type: str):
    # Makes assumption each zip code has less than 30 synagogues
    all_results = set()
    for zipcode in tqdm(get_ny_zipcodes()):
        r = yellow_pages.parse_listing(place_type, str(zipcode))
        
        if r is None:
            continue
        
        results = set(r)
        
        all_results |= results
    
    df = pd.DataFrame(all_results, columns=yellow_pages.listing_keys)
    


    df.to_csv(f'{place_type.replace(" ", "_")}_ny.csv', index=False)

@app.command()
def compute_lat_long(csv_path: Path):
    df = pd.read_csv(csv_path)

    lats = []
    longs = []
    
    geo = Nominatim(user_agent='14.33_project')
    for _, row in tqdm(df.iterrows(), total=len(df.index)):
        query = f'{row.street}, {row.locality} {row.region} {row.zipcode}'
        result = geo.geocode(query)
        
        if result is None:
            query = f'{row.locality}, {row.region} {row.zipcode}'
            result = geo.geocode(query)
        
        if result is None:
            query = f'{row.street}, {row.region} {row.zipcode}'
            result = geo.geocode(query)
        
        if result is None:
            query = f'{row.region} {row.zipcode}'
            result = geo.geocode(query)
        
        if result is None:
            print(row)
            lats.append(-1)
            longs.append(-1)
            continue

        lats.append(result.latitude)
        longs.append(result.longitude)
    
    df['latitude'] = lats
    df['longitude'] = longs

    df.to_csv(csv_path, index=False)
        
@app.command()
def make_ny_tract_zip_dict():
    data = {}
    ny_zips = set(get_ny_zipcodes())
    with open('tract_zip.csv') as f:
        r = csv.reader(f)
        next(r)
        
        for row in tqdm(r):
            tract, zipcode = int(row[0]), int(row[1])
            if zipcode in ny_zips:
                data[tract] = (zipcode, float(row[2])) # use res ratio
    
    with open('ny_tract_zip.json', 'w') as f:
        json.dump(data, f)

@app.command()
def make_nyc_census_zips():
    with open('ny_tract_zip.json') as f:
        tract_zip = json.load(f)

    start_key = 3

    zip_data_raw = defaultdict(lambda: [])
    with open('nyc_census.csv') as f:
        r = csv.reader(f)
        keys = next(r)[start_key:]
        
        for row in r:
            census_tract = row[0]

            fmt_row_keys = np.zeros(len(keys))

            for i, row_el in enumerate(row):
                if i < start_key:
                    continue
                
                if row_el == '':
                    fmt_row_keys[i-start_key] = np.NaN
                else:
                    if i in (4, 5, 11): # percentage:
                        if float(row[3]) != 0:
                            fmt_row_keys[i-start_key] = 100*float(row_el)/float(row[3])
                    elif i == 3:
                        fmt_row_keys[i-start_key] = int(row_el)
                    else:
                        fmt_row_keys[i-start_key] = float(row_el)
                    
            if not census_tract in tract_zip:
                continue

            zipcode, weight = tract_zip[census_tract]

            zip_data_raw[zipcode].append((weight, fmt_row_keys))
    
    zip_data_compiled = []
    for zipcode, tracts in zip_data_raw.items():
        weights, tract_data = zip(*tracts)
        tract_data = np.stack(tract_data)

        masked_data = np.ma.masked_array(tract_data, np.isnan(tract_data))
        average = np.ma.average(masked_data[:, 1:], axis=0, weights=weights)
        pop = np.ma.sum(masked_data[:, 0], axis=0)
        zip_data_compiled.append([zipcode,int(pop)]+list(average.filled(np.nan)))
    
    zip_data_df = pd.DataFrame(zip_data_compiled, columns=['Zipcode']+keys).set_index('Zipcode').dropna()
    
    zip_data_df.to_csv('nyc_census_zips.csv')


def get_zip_kde(kde):
    import pyproj
    N_SAMPLES_PER_ZIP = 50
    
    def get_latlong_polygon_area(s):
        proj = partial(pyproj.transform, pyproj.Proj('epsg:4326'),
                   pyproj.Proj('epsg:3857'))

        return transform(proj, s).area


    with open('nyc-zip-code-tabulation-areas-polygons.geojson') as f:
        nyc_zips_geojson = json.load(f)['features']

    def get_random_point_in_polygon(poly):
        minx, miny, maxx, maxy = poly.bounds
        while True:
            p = Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy))
            if poly.contains(p):
                return p

    ret = {}
    for zipcode_item in tqdm(nyc_zips_geojson):
        zipcode = int(zipcode_item['properties']['postalCode'])
        zip_shape = shape(zipcode_item['geometry'])
        avg = 0
        for i in range(N_SAMPLES_PER_ZIP):
            avg += kde(get_random_point_in_polygon(zip_shape))[0]
        avg /= N_SAMPLES_PER_ZIP

        ret[zipcode] = {'kde':avg, 'area':get_latlong_polygon_area(zip_shape)}
    
    return ret

KDE_CONSTANT = 0.0025#0.01

def build_kde_model():
    synagogue_df = pd.read_csv('synagogue_ny.csv')
    yeshiva_df = pd.read_csv('yeshiva_ny_cleaned.csv')
    kosher_df = pd.read_csv('kosher_restaurant_ny.csv')

    synagogue_latlongs = np.array(list(zip(synagogue_df['longitude'], synagogue_df['latitude'])))
    yeshiva_latlongs = np.array(list(zip(yeshiva_df['longitude'], yeshiva_df['latitude'])))
    kosher_latlongs = np.array(list(zip(kosher_df['longitude'], kosher_df['latitude'])))

    weights = [0.5,2,1]

    latlongs = np.concatenate((synagogue_latlongs, yeshiva_latlongs, kosher_latlongs))

    total_weights = len(synagogue_latlongs)*[weights[0]] + len(yeshiva_latlongs)*[weights[1]] + len(kosher_latlongs)*[weights[2]]
    kde = scipy.stats.gaussian_kde(latlongs.T, bw_method=KDE_CONSTANT, weights=total_weights)

    return kde, (synagogue_latlongs, yeshiva_latlongs, kosher_latlongs)

@app.command()
def assemble_final_data():
    kde, _ = build_kde_model()
    kde_data = get_zip_kde(kde)
    
    df = pd.read_csv('nyc_census_zips.csv')
    covid_df = pd.read_csv('last7days-by-modzcta.csv').set_index('modzcta')
    
    covid_map = covid_df['percentpositivity_7day'].to_dict()

    df['Covid'] = df['Zipcode'].map(covid_map)

    areas = {int(zipcode):data['area'] for zipcode, data in kde_data.items()}
    kdes = {int(zipcode):data['kde'] for zipcode, data in kde_data.items()}
    
    df['Area'] = df['Zipcode'].map(areas)
    df['KDE'] = df['Zipcode'].map(kdes)
    df['Density'] = df['TotalPop']/df['Area']
    df['Employed'] = df['Employed']/df['Area']
    df = df.drop(['Area', 'TotalPop', 'Women'], axis=1)

    df = df.set_index('Zipcode')
    
    df.to_csv('total_data.csv')

if __name__ == "__main__":
    app()
