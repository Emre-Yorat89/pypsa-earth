# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText:  PyPSA-Earth and PyPSA-Eur Authors
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# -*- coding: utf-8 -*-
"""
Creates electric demand profile csv.

Relevant Settings
-----------------

.. code:: yaml

    load:
        scale:
        ssp:
        weather_year:
        prediction_year:
        region_load:

Inputs
------

- ``networks/base.nc``: confer :ref:`base`, a base PyPSA Network
- ``resources/bus_regions/regions_onshore.geojson``: confer :mod:`build_bus_regions`
- ``load_data_paths``: paths to load profiles, e.g. hourly country load profiles produced by GEGIS
- ``resources/shapes/gadm_shapes.geojson``: confer :ref:`shapes`, file containing the gadm shapes

Outputs
-------

- ``resources/demand_profiles.csv``: the content of the file is the electric demand profile associated to each bus. The file has the snapshots as rows and the buses of the network as columns.

Description
-----------

The rule :mod:`build_demand` creates load demand profiles in correspondence of the buses of the network.
It creates the load paths for GEGIS outputs by combining the input parameters of the countries, weather year, prediction year, and SSP scenario.
Then with a function that takes in the PyPSA network "base.nc", region and gadm shape data, the countries of interest, a scale factor, and the snapshots,
it returns a csv file called "demand_profiles.csv", that allocates the load to the buses of the network according to GDP and population.
"""
import os
import os.path
from itertools import product

import geopandas as gpd
import numpy as np
import pandas as pd
import pypsa
import scipy.sparse as sparse
import xarray as xr
from _helpers import (
    BASE_DIR,
    configure_logging,
    create_logger,
    read_csv_nafix,
    read_osm_config,
)
from shapely.prepared import prep
from shapely.validation import make_valid

logger = create_logger(__name__)


def normed(s):
    if s.sum() == 0:
        return s
    return s / s.sum()


def get_gegis_regions(countries):
    """
    Get the GEGIS region from the config file.

    Parameters
    ----------
    region : str
        The region of the bus

    Returns
    -------
    str
        The GEGIS region
    """
    gegis_dict, world_iso = read_osm_config("gegis_regions", "world_iso")

    regions = []

    for d_region in [gegis_dict, world_iso]:
        for key, value in d_region.items():
            # ignore if the key is already in the regions list
            if key not in regions:
                # if a country is in the regions values, then load it
                cintersect = set(countries).intersection(set(value.keys()))
                if cintersect:
                    regions.append(key)
    return regions


def get_load_paths_gegis(ssp_parentfolder, config):
    """
    Create load paths for GEGIS outputs.

    The paths are created automatically according to included country,
    weather year, prediction year and ssp scenario

    Example
    -------
    ["/data/ssp2-2.6/2030/era5_2013/Africa.nc", "/data/ssp2-2.6/2030/era5_2013/Africa.nc"]
    """
    countries = config.get("countries")
    region_load = get_gegis_regions(countries)
    weather_year = config.get("load_options")["weather_year"]
    prediction_year = config.get("load_options")["prediction_year"]
    ssp = config.get("load_options")["ssp"]

    scenario_path = os.path.join(ssp_parentfolder, ssp)

    load_paths = []
    load_dir = os.path.join(
        ssp_parentfolder,
        str(ssp),
        str(prediction_year),
        "era5_" + str(weather_year),
    )

    file_names = []
    for continent in region_load:
        sel_ext = ".nc"
        for ext in [".nc", ".csv"]:
            load_path = os.path.join(BASE_DIR, str(load_dir), str(continent) + str(ext))
            if os.path.exists(load_path):
                sel_ext = ext
                break
        file_name = str(continent) + str(sel_ext)
        load_path = os.path.join(str(load_dir), file_name)
        load_paths.append(load_path)
        file_names.append(file_name)

    logger.info(
        f"Demand data folder: {load_dir}, load path is {load_paths}.\n"
        + f"Expected files: "
        + "; ".join(file_names)
    )

    return load_paths


def shapes_to_shapes(orig, dest):
    """
    Adopted from vresutils.transfer.Shapes2Shapes()
    """
    orig_prepped = list(map(prep, orig))
    transfer = sparse.lil_matrix((len(dest), len(orig)), dtype=float)

    for i, j in product(range(len(dest)), range(len(orig))):
        if orig_prepped[j].intersects(dest[i]):
            area = orig[j].intersection(dest[i]).area
            transfer[i, j] = area / dest[i].area

    return transfer


def load_demand_csv(path):
    df = read_csv_nafix(path, sep=";")
    df.time = pd.to_datetime(df.time, format="%Y-%m-%d %H:%M:%S")
    load_regions = {c: n for c, n in zip(df.region_code, df.region_name)}

    gegis_load = df.set_index(["region_code", "time"]).to_xarray()
    gegis_load = gegis_load.assign_coords(
        {
            "region_name": (
                "region_code",
                [name for (code, name) in load_regions.items()],
            )
        }
    )
    return gegis_load


def build_demand_profiles(
    n,
    load_paths,
    regions,
    admin_shapes,
    countries,
    scale,
    start_date,
    end_date,
    out_path,
):
    """
    Create csv file of electric demand time series.

    Parameters
    ----------
    n : pypsa network
    load_paths: paths of the load files
    regions : .geojson
        Contains bus_id of low voltage substations and
        bus region shapes (voronoi cells)
    admin_shapes : .geojson
        contains subregional gdp, population and shape data
    countries : list
        List of countries that is config input
    scale : float
        The scale factor is multiplied with the load (1.3 = 30% more load)
    start_date: parameter
        The start_date is the first hour of the first day of the snapshots
    end_date: parameter
        The end_date is the last hour of the last day of the snapshots

    Returns
    -------
    demand_profiles.csv : csv file containing the electric demand time series
    """
    substation_lv_i = n.buses.index[n.buses["substation_lv"]]
    regions = gpd.read_file(regions).set_index("name").reindex(substation_lv_i)
    load_paths = load_paths

    gegis_load_list = []

    for path in load_paths:
        if str(path).endswith(".csv"):
            gegis_load_xr = load_demand_csv(path)
        else:
            # Merge load .nc files: https://stackoverflow.com/questions/47226429/join-merge-multiple-netcdf-files-using-xarray
            gegis_load_xr = xr.open_mfdataset(path, combine="nested")
        gegis_load_list.append(gegis_load_xr)

    logger.info(f"Merging demand data from paths {load_paths} into the load data frame")
    gegis_load = xr.merge(gegis_load_list)
    gegis_load = gegis_load.to_dataframe().reset_index().set_index("time")

    # filter load for analysed countries
    gegis_load = gegis_load.loc[gegis_load.region_code.isin(countries)]

    if isinstance(scale, dict):
        logger.info(f"Using custom scaling factor for load data.")
        DEFAULT_VAL = scale.get("DEFAULT", 1.0)
        for country in countries:
            scale.setdefault(country, DEFAULT_VAL)

        for country, scale_country in scale.items():
            gegis_load.loc[
                gegis_load.region_code == country, "Electricity demand"
            ] *= scale_country

    elif isinstance(scale, (int, float)):
        logger.info(f"Load data scaled with scaling factor {scale}.")
        gegis_load["Electricity demand"] *= scale

    shapes = gpd.read_file(admin_shapes).set_index("GADM_ID")
    shapes["geometry"] = shapes["geometry"].apply(lambda x: make_valid(x))

    def upsample(cntry, group):
        """
        Distributes load in country according to population and gdp.
        """
        l = gegis_load.loc[gegis_load.region_code == cntry]["Electricity demand"]
        if len(group) == 1:
            return pd.DataFrame({group.index[0]: l})
        else:
            shapes_cntry = shapes.loc[shapes.country == cntry]
            transfer = shapes_to_shapes(group, shapes_cntry.geometry).T.tocsr()
            gdp_n = pd.Series(
                transfer.dot(shapes_cntry["gdp"].fillna(1.0).values), index=group.index
            )
            pop_n = pd.Series(
                transfer.dot(shapes_cntry["pop"].fillna(1.0).values), index=group.index
            )

            # relative factors 0.6 and 0.4 have been determined from a linear
            # regression on the country to EU continent load data
            # (refer to vresutils.load._upsampling_weights)
            # TODO: require adjustment for Africa
            factors = normed(0.6 * normed(gdp_n) + 0.4 * normed(pop_n))
            if factors.sum() == 0:
                logger.warning(
                    f"Upsampling factors for {cntry} are all zero, returning uniform distribution across {len(factors)} shapes."
                )
                factors = pd.Series(
                    np.ones(len(factors)) / len(factors), index=factors.index
                )
            return pd.DataFrame(
                factors.values * l.values[:, np.newaxis],
                index=l.index,
                columns=factors.index,
            )

    demand_profiles = pd.concat(
        [
            upsample(cntry, group)
            for cntry, group in regions.geometry.groupby(regions.country)
        ],
        axis=1,
    )

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date) - pd.Timedelta(hours=1)
    demand_profiles = demand_profiles.loc[start_date:end_date]
    demand_profiles.to_csv(out_path, header=True)

    logger.info(f"Demand_profiles csv file created for the corresponding snapshots.")

def get_WorldPop_data(
    country_code,
    year,
    update=False,
    out_logging=False,
    size_min=300,
):
    """
    Download tiff file for each country code using the standard method from worldpop datastore with 1kmx1km resolution.

    Parameters
    ----------
    country_code : str
        Two letter country codes of the downloaded files.
        Files downloaded from https://data.worldpop.org/ datasets WorldPop UN adjusted
    year : int
        Year of the data to download
    update : bool
        Update = true, forces re-download of files
    size_min : int
        Minimum size of each file to download
    Returns
    -------
    WorldPop_inputfile : str
        Path of the file
    """

    three_digits_code = two_2_three_digits_country(country_code)

    if out_logging:
        _logger.info("Get WorldPop datasets")

    if country_code == "XK":
        WorldPop_filename = f"srb_ppp_{year}_UNadj_constrained.tif"
        WorldPop_urls = [
            f"https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/2020/BSGM/SRB/{WorldPop_filename}",
            f"https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/2020/maxar_v1/SRB/{WorldPop_filename}",
        ]
    else:
        WorldPop_filename = (
            f"{three_digits_code.lower()}_ppp_{year}_UNadj_constrained.tif"
        )
        # Urls used to possibly download the file
        WorldPop_urls = [
            f"https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/2020/BSGM/{two_2_three_digits_country(country_code).upper()}/{WorldPop_filename}",
            f"https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/2020/maxar_v1/{two_2_three_digits_country(country_code).upper()}/{WorldPop_filename}",
        ]

    WorldPop_inputfile = os.path.join(
        os.getcwd(),
        "pypsa-earth",
        "data",
        "WorldPop",
        WorldPop_filename,
    )  # Input filepath tif

    if not os.path.exists(WorldPop_inputfile) or update is True:
        if out_logging:
            _logger.warning(
                f"{WorldPop_filename} does not exist, downloading to {WorldPop_inputfile}"
            )
        #  create data/osm directory
        os.makedirs(os.path.dirname(WorldPop_inputfile), exist_ok=True)

        loaded = False
        for WorldPop_url in WorldPop_urls:
            with requests.get(WorldPop_url, stream=True) as r:
                with open(WorldPop_inputfile, "wb") as f:
                    if float(r.headers["Content-length"]) > size_min:
                        shutil.copyfileobj(r.raw, f)
                        loaded = True
                        break
        if not loaded:
            _logger.error(f"Impossible to download {WorldPop_filename}")

    return WorldPop_inputfile, WorldPop_filename


def estimate_microgrid_population(raster_path, shapes_path, output_file):
    """
    Estimates the population within each microgrid by using raster data and shape geometries.
    The function processes population density raster data and calculates the total population
    for each microgrid by masking the raster data using the corresponding geometries from a
    GeoJSON file. The population estimates are saved as a CSV file.

    Parameters
    ----------
    raster_path : str
        Path to the population density raster file (GeoTIFF format).
    shapes_path : str
        Path to the GeoJSON file containing the microgrid geometries.
    output_file : str
        Path to the CSV file where the population estimates will be saved.
    Returns
    -------
    pd.DataFrame
        A DataFrame containing the names of microgrids and their corresponding population estimates.
    """
    # Dictionary to store the population data for each microgrid
    population_data = {}
    # Load the GeoJSON file containing microgrid geometries
    shapes = gpd.read_file(shapes_path)
    # Iterate through each microgrid geometry
    for i, shape in shapes.iterrows():
        name = shape["name"]  # Extract the name of the microgrid
        # Open the raster file and mask it using the microgrid geometry
        with rasterio.open(raster_path) as src:
            # Mask the raster data to only include the area within the microgrid
            masked, out_transform = rasterio.mask.mask(src, [shape.geometry], crop=True)
            # Update the raster metadata for the masked area
            out_meta = src.meta.copy()
            out_meta.update(
                {
                    "driver": "GTiff",
                    "height": masked.shape[1],
                    "width": masked.shape[2],
                    "transform": out_transform,
                }
            )
        # Calculate the total population within the microgrid by summing non-negative raster values
        pop_microgrid = masked[masked >= 0].sum()
        population_data[name] = pop_microgrid
    # Convert the population data dictionary to a DataFrame
    population_df = pd.DataFrame(
        list(population_data.items()), columns=["Microgrid_Name", "Population"]
    )
    # Save the population estimates to a CSV file
    # population_df.to_csv(output_file, index=False)

    return population_df


def calculate_load(
    p,
    raster_path,
    shapes_path,
    sample_profile,
    output_file,
    input_path,
    microgrids_list,
    start_date,
    end_date,
    inclusive,
):
    """
    Calculate the microgrid demand based on a load profile provided as input,
    appropriately scaled according to the population calculated for each cluster.
    The output includes a time-indexed DataFrame containing the load for each bus in the microgrid
    and is saved as a CSV file.

    Parameters
    ----------
    n : object
        PyPSA network object containing snapshots.
    p : int or float
        Scaling factor for the per-unit load.
    raster_path : str
        Path to the raster file containing population density data.
    shapes_path : str
        Path to the GeoJSON file containing the geometries of the microgrids.
    sample_profile : str
        Path to the CSV file containing the sample load profile.
    output_file : str
        Path where the resulting load profile CSV file will be saved.
    input_path : str
        Path to the CSV file containing building classifications.
    microgrids_list : dict
        Dictionary with microgrid names as keys and their cluster information as values.
    start_date : str
        Start date for filtering the time series data.
    end_date : str
        End date for filtering the time series data.
    inclusive : str
        Specifies whether the filtering is inclusive of the start or end date. Possible values: "left" or "right".

    Returns
    -------
    pd.DataFrame
        DataFrame containing the calculated load profile for all microgrids.

    """
    # Estimate the population for the two microgrids
    pop_microgrid = estimate_microgrid_population(raster_path, shapes_path, output_file)
    # Load the building classification data
    building_class = pd.read_csv(input_path)
    # Dictionary to store the load profiles for each microgrid
    microgrid_dataframes = {}
    # Load the sample load profile and create the time index
    df = pd.read_csv(sample_profile)
    per_unit_load = df["0"] / p  # Scale the load using the provided factor `p`
    df["per_unit_load"] = per_unit_load
    time_index = pd.date_range(start="2013-01-01", end="2013-12-31 23:00:00", freq="h")
    df = df.set_index(time_index)

    # Generate the snapshots range for filtering
    snapshots_range = pd.date_range(
        start=start_date, end=end_date, freq="h", inclusive="both"
    )

    # Filter the DataFrame based on the specified time range
    df_filtered = df.loc[snapshots_range]
    per_unit_load = df_filtered["per_unit_load"].values

    # Loop over each microgrid
    for grid_name in microgrids_list.keys():
        # Filter buildings belonging to the current microgrid
        total_buildings = building_class[building_class["name_microgrid"] == grid_name]
        total_buildings = total_buildings["count"].sum()
        # Group buildings by cluster and count the number of buildings per cluster
        building_for_cluster = pd.DataFrame(
            building_class[building_class["name_microgrid"] == grid_name]
            .groupby("cluster_id")
            .sum()["count"]
        )
        # Retrieve the population for the current microgrid
        pop_for_microgrid = pop_microgrid.loc[
            pop_microgrid["Microgrid_Name"] == grid_name, "Population"
        ].values[0]
        # Calculate the population per building and per cluster
        population_per_building = pop_for_microgrid / total_buildings
        population_per_cluster = building_for_cluster * population_per_building
        # Calculate the load for each cluster
        load_per_cluster = pd.DataFrame(
            np.outer(population_per_cluster["count"].values, per_unit_load)
        )
        load_per_cluster = load_per_cluster.T  # Transpose for time indexing
        # Rename columns to represent the buses of the microgrid
        new_column_names = {
            i: f"{grid_name}_bus_{i}" for i in range(load_per_cluster.shape[1])
        }
        load_per_cluster.rename(columns=new_column_names, inplace=True)
        # Add the DataFrame for the microgrid to the dictionary
        microgrid_dataframes[grid_name] = load_per_cluster

    # Concatenate all microgrid DataFrames horizontally
    all_load_per_cluster = pd.concat(microgrid_dataframes.values(), axis=1)
    all_load_per_cluster.index = snapshots_range

    # Save the cumulative results to a CSV file with time index as the first column
    all_load_per_cluster.to_csv(output_file, index_label="Time")
    return all_load_per_cluster


def calculate_load_ramp(
    input_file_buildings,
    p,
    raster_path,
    shapes_path,
    sample_profile,
    output_file,
    input_file_profile_tier1,
    input_file_profile_tier2,
    input_file_profile_tier3,
    input_file_profile_tier4,
    input_file_profile_tier5,
    output_path_csv,
    tier_percent,
    date_start,
    date_end,
    inclusive,
    microgrid_list,
    std,
):
    # Upload of buildings and data demand for each tier
    cleaned_buildings = gpd.read_file(input_file_buildings)
    demand_files = [
        input_file_profile_tier1,
        input_file_profile_tier2,
        input_file_profile_tier3,
        input_file_profile_tier4,
        input_file_profile_tier5,
    ]

    mean_demand_tier_df = pd.DataFrame(
        {
            f"tier_{i+1}": pd.read_excel(file)["mean"]
            for i, file in enumerate(demand_files)
        }
    )
    std_demand_tier_df = pd.DataFrame(
        {
            f"tier_{i+1}": pd.read_excel(file)["std"]
            for i, file in enumerate(demand_files)
        }
    )
    mean_demand_tier_df.insert(0, "tier_0", np.zeros(len(mean_demand_tier_df)))
    std_demand_tier_df.insert(0, "tier_0", np.zeros(len(mean_demand_tier_df)))
    mean_demand_tier_df.index = pd.date_range(
        "00:00:00", periods=len(mean_demand_tier_df), freq="H"
    ).time
    std_demand_tier_df.index = pd.date_range(
        "00:00:00", periods=len(mean_demand_tier_df), freq="H"
    ).time

    pop = estimate_microgrid_population(raster_path, shapes_path, output_file)

    all_microgrid_loads = pd.DataFrame()

    for grid_name, grid_data in microgrid_list.items():
        microgrid_buildings = cleaned_buildings[
            cleaned_buildings["name_microgrid"] == grid_name
        ]
        # Calculate the population density for the current microgrid based only on house buildings
        house = microgrid_buildings[microgrid_buildings["tags_building"] == "house"]
        pop_microgrid = pop.loc[
            pop["Microgrid_Name"] == grid_name, "Population"
        ].values[0]
        density = pop_microgrid / house["area_m2"].sum()

        # Calculate population per cluster
        grouped_buildings = microgrid_buildings.groupby("cluster_id")
        clusters = np.sort(microgrid_buildings["cluster_id"].unique())
        house_area_for_cluster = [
            grouped_buildings.get_group(cluster)[
                grouped_buildings.get_group(cluster)["tags_building"] == "house"
            ]["area_m2"].sum()
            for cluster in clusters
        ]
        population_df = pd.DataFrame(
            {"cluster": clusters, "house_area_for_cluster": house_area_for_cluster}
        ).set_index("cluster")
        population_df["people_for_cluster"] = (
            population_df["house_area_for_cluster"] * density
        ).round()
        tier_pop_df = pd.DataFrame(
            np.outer(population_df["people_for_cluster"], tier_percent),
            index=population_df.index.astype(int),
        )

        if inclusive == "left":
            date_range = pd.date_range(start=date_start, end=date_end, freq="D")[:-1]
        else:
            date_range = pd.date_range(start=date_start, end=date_end, freq="D")

        mean_demand_tier_df_extended = pd.concat(
            [mean_demand_tier_df] * len(date_range), ignore_index=True
        )
        std_demand_tier_df_extended = pd.concat(
            [std_demand_tier_df] * len(date_range), ignore_index=True
        )

        # Calculate load for each cluster and tier
        if std == "on":
            result_dict = {}
            for k, pop_cluster in tier_pop_df.iterrows():
                load_df = pd.DataFrame()
                for j, n_person in enumerate(pop_cluster / 7):  # Scale by family size
                    mean_load = mean_demand_tier_df_extended.iloc[:, j] * n_person
                    std_load = np.random.normal(
                        mean_demand_tier_df_extended.iloc[:, j],
                        std_demand_tier_df_extended.iloc[:, j],
                    ) * np.sqrt(n_person)
                    total_load = (mean_load + std_load) / 1e6
                    load_df[f"tier_{j}"] = total_load
                result_dict[f"{grid_name}_bus_{k}"] = load_df
        elif std == "off":
            result_dict = {}
            for k, pop_cluster in tier_pop_df.iterrows():
                load_df = pd.DataFrame()
                for j, n_person in enumerate(pop_cluster / 7):  # Scale by family size
                    mean_load = mean_demand_tier_df_extended.iloc[:, j] * n_person
                    total_load = (mean_load) / 1e6
                    load_df[f"tier_{j}"] = total_load
                result_dict[f"{grid_name}_bus_{k}"] = load_df

        # Aggregate total load per cluster
        tot_result_dict = {
            f"{k}": df.sum(axis=1).rename(f"{k}") for k, df in result_dict.items()
        }
        tot_loads_df = pd.concat(tot_result_dict.values(), axis=1)
        if inclusive == "left":
            date_range_tot = pd.date_range(start=date_start, end=date_end, freq="H")[
                :-1
            ]
        else:
            date_range_tot = pd.date_range(start=date_start, end=date_end, freq="H")
        tot_loads_df.index = date_range_tot

        # Replace zero values with a small value just for avoid problem with plotting
        small_value = 1e-26
        tot_loads_df.loc[:, (tot_loads_df == 0).all()] = small_value

        all_microgrid_loads = pd.concat([all_microgrid_loads, tot_loads_df], axis=1)

    # Extracting the final dataframe
    all_microgrid_loads.to_csv(output_path_csv)

if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("build_demand_profiles")

    configure_logging(snakemake)

    n = pypsa.Network(snakemake.input.base_network)

    # Snakemake imports:
    regions = snakemake.input.regions
    load_paths = snakemake.input["load"]
    countries = snakemake.params.countries
    admin_shapes = snakemake.input.gadm_shapes
    scale = snakemake.params.load_options.get("scale", 1.0)
    start_date = snakemake.params.snapshots["start"]
    end_date = snakemake.params.snapshots["end"]
    out_path = snakemake.output[0]

    build_demand_profiles(
        n,
        load_paths,
        regions,
        admin_shapes,
        countries,
        scale,
        start_date,
        end_date,
        out_path,
    )
