#!/usr/bin/python
"""
Created on Wed 6th Feb 2019

@author: adeacon
"""

import geocoder
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.DEBUG
)

addresses = ["google london office", "1-13 St Giles High St", "WC2H 8AG"]


def get_geocoded_results(address):

    g = geocoder.osm(address)

    # if there's no results or an error, return empty results.
    if g.ok:
        output = {
            "formatted_address": g.address,
            "latitude": g.lat,
            "longitude": g.lng,
            "postcode": g.postal,
            "status": "OK",
        }
    else:
        output = {
            "formatted_address": None,
            "latitude": None,
            "longitude": None,
            "postcode": None,
            "status": "FAILED",
        }

    # Append some other details:
    output["input_string"] = address
    output["number_of_results"] = 1

    return output


# Create a list to hold results
results = []
# Go through each address in turn
for address in addresses:
    # While the address geocoding is not finished:
    geocoded = False
    while geocoded is not True:
        # Geocode the address with google
        try:
            geocode_result = get_geocoded_results(address)
        except Exception as e:
            logging.exception(e)
            logging.error("Major error with {}".format(address))
            logging.error("Skipping!")
            geocoded = True

        if geocode_result["status"] != "OK":
            logging.warning(
                "Error geocoding {}: {}".format(address, geocode_result["status"])
            )
        logging.debug("Geocoded: {}: {}".format(address, geocode_result["status"]))
        results.append(geocode_result)
        geocoded = True

    # Print status every 100 addresses
    if len(results) % 100 == 0:
        logging.info("Completed {} of {} address".format(len(results), len(addresses)))

# All done
logging.info("Finished geocoding all addresses")

print(results)
