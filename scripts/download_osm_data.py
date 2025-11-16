# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText:  PyPSA-Earth and PyPSA-Eur Authors
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# -*- coding: utf-8 -*-
"""
Python interface to download OpenStreetMap data
Documented at https://github.com/pypsa-meets-earth/earth-osm

Relevant Settings
-----------------

*None*  # multiprocessing & infrastructure selection can be an option in future

Inputs
------

*None*

Outputs
-------

- ``data/osm/pbf``: Raw OpenStreetMap data as .pbf files per country
- ``data/osm/power``: Filtered power data as .json files per country
- ``data/osm/out``:  Prepared power data as .geojson and .csv files per country
- ``resources/osm/raw``: Prepared and per type (e.g. cable/lines) aggregated power data as .geojson and .csv files
"""
import os
import json
import shutil
from pathlib import Path

from _helpers import BASE_DIR, configure_logging, create_logger, read_osm_config
from earth_osm import eo
import requests

logger = create_logger(__name__)


def country_list_to_geofk(country_list):
    """
    Convert the requested country list into geofk norm.

    Parameters
    ----------
    input : str
        Any two-letter country name or aggregation of countries given in the regions config file
        Country name duplications won't distort the result.
        Examples are:
        ["NG","ZA"], downloading osm data for Nigeria and South Africa
        ["SNGM"], downloading data for Senegal&Gambia shape
        ["NG","ZA","NG"], won't distort result.

    Returns
    -------
    full_codes_list : list
        Example ["NG","ZA"]
    """
    full_codes_list = [convert_iso_to_geofk(c_code) for c_code in set(country_list)]

    return full_codes_list


def convert_iso_to_geofk(
    iso_code, iso_coding=True, convert_dict=read_osm_config("iso_to_geofk_dict")
):
    """
    Function to convert the iso code name of a country into the corresponding
    geofabrik In Geofabrik, some countries are aggregated, thus if a single
    country is requested, then all the agglomeration shall be downloaded For
    example, Senegal (SN) and Gambia (GM) cannot be found alone in geofabrik,
    but they can be downloaded as a whole SNGM.

    The conversion directory, initialized to iso_to_geofk_dict is used to perform such conversion
    When a two-letter code country is found in convert_dict, and iso_coding is enabled,
    then that two-letter code is converted into the corresponding value of the dictionary

    Parameters
    ----------
    iso_code : str
        Two-code country code to be converted
    iso_coding : bool
        When true, the iso to geofk is performed
    convert_dict : dict
        Dictionary used to apply the conversion iso to geofk
        The keys correspond to the countries iso codes that need a different region to be downloaded
    """
    if iso_coding and iso_code in convert_dict:
        return convert_dict[iso_code]
    else:
        return iso_code
        
def retrieve_osm_data_geojson(microgrids_list, feature_name, url, path):
    """
    The buildings inside the specified coordinates are retrieved by using overpass API.
    The region coordinates should be defined in the config.yaml file.
    Parameters
    ----------
    microgrids_list : dict
        Dictionary containing the microgrid names and their bounding box coordinates (lat_min, lon_min, lat_max, lon_max).
    features : str
        The feature that is searched in the osm database
    url : str
        osm query address
    path : str
        Directory where the GeoJSON file will be saved.
    """
    # Collect all features from all microgrids
    geojson_features = []

    for grid_name, grid_data in microgrids_list.items():
        # Extract the bounding box coordinates for the current microgrid to construct the query
        lat_min = grid_data["lat_min"]
        lon_min = grid_data["lon_min"]
        lat_max = grid_data["lat_max"]
        lon_max = grid_data["lon_max"]

        # Construct the Overpass API query for the specified feature
        overpass_query = f"""
        [out:json];
        way["{feature_name}"]({lat_min},{lon_min},{lat_max},{lon_max});
        (._;>;);
        out body;
        """
        try:
            logger.info(
                f"Querying Overpass API for microgrid: {grid_name}"
            )  # Log the current query
            response = requests.get(
                url, params={"data": overpass_query}
            )  # Send the query to Overpass API
            response.raise_for_status()  # Raise an error if the request fails
            data = response.json()  # Parse the JSON response

            # Check if the response contains any elements
            if "elements" not in data:
                logger.error(f"No elements found for microgrid: {grid_name}")
                continue
            # Extract node coordinates from the response
            node_coordinates = {
                node["id"]: [node["lon"], node["lat"]]
                for node in data["elements"]
                if node["type"] == "node"
            }
            # Process "way" elements to construct polygon geometries
            for element in data["elements"]:
                if element["type"] == "way" and "nodes" in element:
                    # Get the coordinates of the nodes that form the way
                    coordinates = [
                        node_coordinates[node_id]
                        for node_id in element["nodes"]
                        if node_id in node_coordinates
                    ]
                    if not coordinates:
                        continue

                    # Add properties for the feature, including the microgrid name and element ID
                    properties = {"name_microgrid": grid_name, "id": element["id"]}
                    if "tags" in element:  # Include additional tags if available
                        properties.update(element["tags"])

                    # Create a GeoJSON feature for the way
                    feature = {
                        "type": "Feature",
                        "properties": properties,
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [coordinates],
                        },
                    }
                    # Serialize each feature as a compact JSON string and add it to the list
                    geojson_features.append(json.dumps(feature, separators=(",", ":")))

        except json.JSONDecodeError:
            # Handle JSON parsing errors
            logger.error(f"JSON decoding error for microgrid: {grid_name}")
        except requests.exceptions.RequestException as e:
            # Handle request-related errors
            logger.error(f"Request error for microgrid: {grid_name}: {e}")

            # Save all features to a single GeoJSON file
        try:
            outpath = Path(path) / "all_raw_buildings.geojson"
            outpath.parent.mkdir(parents=True, exist_ok=True)

            with open(outpath, "w") as f:
                f.write('{"type":"FeatureCollection","features":[\n')
                f.write(
                    ",\n".join(geojson_features)
                )  # Write features in one-line format
                f.write("\n]}\n")

            logger.info(f"Combined GeoJSON saved to {outpath}")

        except IOError as e:
            logger.error(f"Error saving GeoJSON file: {e}")


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("download_osm_data")
    configure_logging(snakemake)

    run = snakemake.config.get("run", {})
    RDIR = run["name"] + "/" if run.get("name") else ""
    store_path_resources = Path.joinpath(
        Path(BASE_DIR), "resources", RDIR, "osm", "raw"
    )
    store_path_data = Path.joinpath(Path(BASE_DIR), "data", "osm")
    country_list = country_list_to_geofk(snakemake.params.countries)

    eo.save_osm_data(
        primary_name="power",
        region_list=country_list,
        feature_list=["substation", "line", "cable", "generator"],
        update=False,
        mp=True,
        data_dir=store_path_data,
        out_dir=store_path_resources,
        out_format=["csv", "geojson"],
        out_aggregate=True,
        progress_bar=snakemake.config["enable"]["progress_bar"],
    )

    out_path = Path.joinpath(store_path_resources, "out")
    names = ["generator", "cable", "line", "substation"]
    out_formats = ["csv", "geojson"]
    new_files = os.listdir(out_path)  # list downloaded osm files

    # earth-osm (eo) only outputs files with content
    # If the file is empty, it is not created
    # This is a workaround to create empty files for the workflow

    # Rename and move osm files to the resources folder output
    for name in names:
        for f in out_formats:
            new_file_name = Path.joinpath(store_path_resources, f"all_raw_{name}s.{f}")
            old_files = list(Path(out_path).glob(f"*{name}.{f}"))
            # if file is missing, create empty file, otherwise rename it an move it
            if not old_files:
                with open(new_file_name, "w") as f:
                    pass
            else:
                logger.info(f"Move {old_files[0]} to {new_file_name}")
                shutil.move(old_files[0], new_file_name)
    if snakemake.config["enable"]["download_osm_method"] == "overpass" and snakemake.params.run_distribution == True:
        microgrids_list = snakemake.config["microgrids_list"]
        features = "building"
        overpass_url = "https://overpass-api.de/api/interpreter"
        output_file = Path.cwd() / "resources" / RDIR / "osm" / "raw"
        retrieve_osm_data_geojson(microgrids_list, features, overpass_url, output_file)
