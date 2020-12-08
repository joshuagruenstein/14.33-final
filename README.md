### Covid-19 Orthodox NYC Analysis

Analysis files:

1. `data_collection.py`: contains all methods for scraping and manipulating data.  Run `python3 data_collection.py --help` to see available commands in the data processing pipeline.
2. `regressions.ipynb`: contains double post-lasso regression.
3. `geo_explore.ipynb`: contains heatmap visualization of feature maps.

Data files:

- `total_data.csv`: all aggregated data (Census, Covid, Orthodox) by NYC zip code 
- `kosher_restaurant_ny.csv`, `yeshiva_ny_cleaned.csv`, and `synagogue_ny.csv`: scraped (and cleaned) tables of Orthodox institutions in New York State, from the Yellow Pages.
- `nyc_census.csv`: census data for NYC census tracts.
- `nyc_census_zips.csv`: census data for NYC zip codes, generated from the above.
- `ny_zipcodes.csv`: list of NY zipcodes.
- `last7days-by-modzcta.csv`: official NYC covid testing rates by zip code.
- `tract_zip.csv`: HUD crosswalk file for converting census blocks to zip codes
- `nyc-zip-code-tabulation-areas-polygons.geojson`: shape information for sampling average heatmap values over zip code polygons
- `yellow_pages.py`: script for scraping the yellow pages, from [scrapehero/yellowpages-scraper](https://github.com/scrapehero/yellowpages-scraper)