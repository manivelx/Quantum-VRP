"""Generate a deterministic customer fixture for CSV loading tests."""

import csv
from pathlib import Path


OUTPUT = Path(__file__).parent / "data" / "customers_1000.csv"
DEPOT_LATITUDE = 18.5362
DEPOT_LONGITUDE = 73.8958
GRID_ROWS = 250
GRID_COLUMNS = 400
LOCATION_POOL_SIZE = GRID_ROWS * GRID_COLUMNS
CUSTOMER_COUNT = 1000


with OUTPUT.open("w", newline="", encoding="utf-8") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["id", "type", "latitude", "longitude", "demand"])
    writer.writerow(["DEPOT_01", "depot", DEPOT_LATITUDE, DEPOT_LONGITUDE, 0])

    for customer_number in range(1, CUSTOMER_COUNT + 1):
        pool_index = (customer_number * 97) % LOCATION_POOL_SIZE
        grid_row, grid_column = divmod(pool_index, GRID_COLUMNS)
        latitude = 18.35 + grid_row * 0.0016
        longitude = 73.70 + grid_column * 0.001
        demand = 10 + ((customer_number * 17) % 41)
        writer.writerow([
            f"LOC_{customer_number:06d}",
            "customer",
            f"{latitude:.6f}",
            f"{longitude:.6f}",
            demand,
        ])

print(f"Wrote {CUSTOMER_COUNT} customers from a {LOCATION_POOL_SIZE}-location pool to {OUTPUT}")
