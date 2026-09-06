import sys
import os
import time
import math
from typing import Callable, Dict, Any

import streamlit as st
import pandas as pd
import folium
import osmnx as ox
import networkx as nx
import numpy as np

from streamlit_folium import st_folium


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.abspath(__file__)
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# IMPORT PROJECT MODULES
# ============================================================

from graph.osm_loader import load_city_graph
from graph.graph_builder import describe_graph, validate_graph
from graph.location_snapper import snap_locations

from algorithms.qpso import QPSO_VRP
from algorithms.pso import PSO_VRP
from algorithms.ga import GeneticAlgorithmVRP


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="BLACK ROCK VRP",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 1rem;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
        padding-bottom: 2rem;
    }

    .main-title {
        font-size: 2.3rem;
        font-weight: 900;
        letter-spacing: -1px;
        margin-bottom: 0;
    }

    .main-subtitle {
        color: #64748b;
        font-size: 0.95rem;
        margin-bottom: 1rem;
    }

    section[data-testid="stSidebar"] {
        border-right: 1px solid #e2e8f0;
    }

    .status-online {
        color: #16a34a;
        font-weight: 700;
    }

    .section-title {
        font-size: 1.25rem;
        font-weight: 800;
        margin-top: 1rem;
        margin-bottom: 0.6rem;
    }

    .route-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 15px;
        margin-bottom: 10px;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.05);
    }

    .route-title {
        font-size: 1rem;
        font-weight: 800;
    }

    .route-info {
        color: #475569;
        font-size: 0.88rem;
        margin-top: 6px;
    }

    .footer {
        text-align: center;
        color: #64748b;
        font-size: 0.8rem;
        padding-top: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# BENCHMARK ENGINE
# ============================================================

class BenchmarkResult:
    """
    Stores the result of one optimization algorithm.
    """

    def __init__(
        self,
        algorithm_name: str,
        routes,
        distance: float,
        runtime: float,
        iterations: int = 0
    ):
        self.algorithm_name = algorithm_name
        self.routes = routes
        self.distance = float(distance)
        self.runtime = float(runtime)
        self.iterations = int(iterations)

    def to_dict(self) -> Dict[str, Any]:

        return {
            "Algorithm": self.algorithm_name,
            "Total Distance (km)": round(
                self.distance,
                2
            ),
            "Runtime (seconds)": round(
                self.runtime,
                4
            ),
            "Iterations": self.iterations,
            "Vehicles Used": len(self.routes)
        }


class BenchmarkEngine:
    """
    Runs multiple optimization algorithms
    on the same VRP problem.
    """

    def __init__(self):

        self.results = []


    def run_algorithm(
        self,
        algorithm_name: str,
        optimizer_function: Callable
    ):

        start_time = time.time()

        result = optimizer_function()

        runtime = time.time() - start_time


        if not isinstance(result, tuple):

            raise ValueError(
                "Optimizer function must return "
                "a tuple."
            )


        if len(result) == 3:

            routes, distance, iterations = result

        elif len(result) == 2:

            routes, distance = result
            iterations = 0

        else:

            raise ValueError(
                "Optimizer must return "
                "(routes, distance) or "
                "(routes, distance, iterations)."
            )


        benchmark_result = BenchmarkResult(

            algorithm_name=algorithm_name,

            routes=routes,

            distance=distance,

            runtime=runtime,

            iterations=iterations

        )


        self.results.append(
            benchmark_result
        )


        return benchmark_result


    def get_results(self):

        return self.results


    def results_dataframe(self):

        if not self.results:

            return pd.DataFrame(
                columns=[
                    "Algorithm",
                    "Total Distance (km)",
                    "Runtime (seconds)",
                    "Iterations",
                    "Vehicles Used"
                ]
            )


        return pd.DataFrame(
            [
                result.to_dict()
                for result in self.results
            ]
        )


    def best_result(self):

        if not self.results:

            return None


        return min(
            self.results,
            key=lambda result: result.distance
        )


    def clear(self):

        self.results.clear()


# ============================================================
# ROAD DISTANCE MATRIX
# ============================================================

def build_distance_matrix(
    G,
    location_nodes
):
    """
    Build a road-distance matrix using
    Dijkstra shortest-path distances.

    Distances are returned in kilometres.
    """

    n = len(location_nodes)

    distance_matrix = np.full(
        (n, n),
        1e9,
        dtype=float
    )

    np.fill_diagonal(
        distance_matrix,
        0.0
    )


    # --------------------------------------------------------
    # Directed network
    # --------------------------------------------------------

    for i, source_node in enumerate(
        location_nodes
    ):

        try:

            lengths = nx.single_source_dijkstra_path_length(
                G,
                source_node,
                weight="length"
            )

        except Exception:

            lengths = {}


        for j, target_node in enumerate(
            location_nodes
        ):

            if target_node in lengths:

                distance_matrix[i, j] = (
                    float(
                        lengths[target_node]
                    ) / 1000.0
                )


    # --------------------------------------------------------
    # Undirected fallback
    # --------------------------------------------------------

    if np.any(
        distance_matrix >= 1e9
    ):

        try:

            undirected_graph = G.to_undirected()

            for i, source_node in enumerate(
                location_nodes
            ):

                try:

                    lengths = (
                        nx.single_source_dijkstra_path_length(
                            undirected_graph,
                            source_node,
                            weight="length"
                        )
                    )

                except Exception:

                    lengths = {}


                for j, target_node in enumerate(
                    location_nodes
                ):

                    if (
                        distance_matrix[i, j] >= 1e9
                        and target_node in lengths
                    ):

                        distance_matrix[i, j] = (
                            float(
                                lengths[target_node]
                            ) / 1000.0
                        )

        except Exception:
            pass


    return distance_matrix


# ============================================================
# SAFE NAVIGATION PATH
# ============================================================

def get_safe_navigation_path(
    G,
    start_node,
    end_node
):
    """
    Find the actual road path between
    two snapped OSM nodes.
    """

    if start_node == end_node:

        return [start_node], "directed"


    # --------------------------------------------------------
    # Directed
    # --------------------------------------------------------

    try:

        path = nx.shortest_path(
            G,
            start_node,
            end_node,
            weight="length"
        )

        return path, "directed"

    except (
        nx.NetworkXNoPath,
        nx.NodeNotFound
    ):

        pass


    # --------------------------------------------------------
    # Undirected fallback
    # --------------------------------------------------------

    try:

        undirected_graph = (
            G.to_undirected()
        )

        path = nx.shortest_path(
            undirected_graph,
            start_node,
            end_node,
            weight="length"
        )

        return path, "undirected"

    except (
        nx.NetworkXNoPath,
        nx.NodeNotFound
    ):

        return [], "none"


# ============================================================
# GET PATH COORDINATES
# ============================================================

def get_path_coordinates(
    G,
    path
):

    coordinates = []

    for node in path:

        try:

            latitude = float(
                G.nodes[node]["y"]
            )

            longitude = float(
                G.nodes[node]["x"]
            )

            coordinates.append(
                [latitude, longitude]
            )

        except Exception:

            continue


    return coordinates


# ============================================================
# GET PATH DISTANCE
# ============================================================

def get_path_distance(
    G,
    path
):

    if not path or len(path) < 2:

        return 0.0


    total_meters = 0.0


    for i in range(
        len(path) - 1
    ):

        u = path[i]
        v = path[i + 1]

        edge_length = get_edge_length(
            G,
            u,
            v
        )

        if math.isfinite(
            edge_length
        ):

            total_meters += edge_length


    return total_meters / 1000.0


# ============================================================
# EDGE LENGTH
# ============================================================

def get_edge_length(
    G,
    u,
    v,
    key=None
):

    try:

        if isinstance(
            G,
            (nx.MultiGraph, nx.MultiDiGraph)
        ):

            data = G.get_edge_data(
                u,
                v
            )

            if data is None:

                return float("inf")


            if key is not None:

                edge_data = data.get(
                    key
                )

                if edge_data is None:

                    return float("inf")

                return float(
                    edge_data.get(
                        "length",
                        0
                    )
                )


            lengths = []


            for edge_key, edge_data in data.items():

                length = edge_data.get(
                    "length"
                )

                if length is not None:

                    lengths.append(
                        float(length)
                    )


            if lengths:

                return min(
                    lengths
                )


        else:

            data = G.get_edge_data(
                u,
                v
            )

            if data is None:

                return float("inf")


            return float(
                data.get(
                    "length",
                    0
                )
            )


    except Exception:

        return float("inf")


    return float("inf")


# ============================================================
# ETA
# ============================================================

def calculate_eta(
    distance_km,
    average_speed_kmh=30.0
):

    if distance_km <= 0:

        return 0.0


    hours = (
        distance_km
        / average_speed_kmh
    )


    return hours * 60.0


def format_eta(
    minutes
):

    minutes = float(
        minutes
    )


    if minutes < 60:

        return f"{minutes:.0f} min"


    hours = int(
        minutes // 60
    )


    remaining_minutes = int(
        round(minutes % 60)
    )


    if remaining_minutes == 60:

        hours += 1
        remaining_minutes = 0


    if remaining_minutes == 0:

        return f"{hours} hr"


    return (
        f"{hours} hr "
        f"{remaining_minutes} min"
    )


# ============================================================
# SESSION STATE
# ============================================================

defaults = {

    "graph": None,

    "routes": None,

    "route_stats": None,

    "location_nodes": None,

    "location_names": None,

    "distance_matrix": None,

    "optimization_result": None,

    "benchmark_results": None

}


for key, value in defaults.items():

    if key not in st.session_state:

        st.session_state[key] = value


# ============================================================
# HEADER
# ============================================================

header_col1, header_col2 = st.columns(
    [5, 1]
)


with header_col1:

    st.markdown(
        '<div class="main-title">'
        '🚚 BLACK ROCK'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="main-subtitle">'
        'Quantum-Inspired Vehicle Routing Optimization Platform'
        '</div>',
        unsafe_allow_html=True
    )


with header_col2:

    st.markdown(
        '<div style="text-align:right; padding-top:12px;">'
        '<span class="status-online">'
        '● SYSTEM ONLINE'
        '</span>'
        '</div>',
        unsafe_allow_html=True
    )


st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "🗺️ Route Planner"
)

st.sidebar.markdown(
    '<span class="status-online">'
    '● System Online'
    '</span>',
    unsafe_allow_html=True
)

st.sidebar.divider()


# ============================================================
# NETWORK SETTINGS
# ============================================================

st.sidebar.subheader(
    "🌍 Network"
)


place = st.sidebar.text_input(
    "Map Location",
    value="Koregaon Park, Pune, India"
)


load_button = st.sidebar.button(
    "🌐 Load Road Network",
    use_container_width=True,
    type="primary"
)


# ============================================================
# LOAD ROAD NETWORK
# ============================================================

if load_button:

    with st.spinner(
        "Loading OpenStreetMap road network..."
    ):

        try:

            G = load_city_graph(
                place
            )

            validate_graph(
                G
            )

            st.session_state["graph"] = G

            st.session_state["routes"] = None

            st.session_state[
                "route_stats"
            ] = None

            st.session_state[
                "location_nodes"
            ] = None

            st.session_state[
                "location_names"
            ] = None

            st.session_state[
                "distance_matrix"
            ] = None

            st.session_state[
                "optimization_result"
            ] = None

            st.session_state[
                "benchmark_results"
            ] = None

            st.success(
                "Road network loaded successfully."
            )

        except Exception as e:

            st.error(
                f"Unable to load road network: {e}"
            )


# ============================================================
# CUSTOMER DATA
# ============================================================

st.sidebar.divider()

st.sidebar.subheader(
    "📦 Orders"
)


uploaded_file = st.sidebar.file_uploader(
    "Upload customer CSV",
    type=["csv"]
)


if uploaded_file is not None:

    customers_df = pd.read_csv(
        uploaded_file
    )

else:

    try:

        customers_df = pd.read_csv(
            os.path.join(
                PROJECT_ROOT,
                "data",
                "customers.csv"
            )
        )

    except FileNotFoundError:

        st.error(
            "data/customers.csv not found."
        )

        st.stop()


# ============================================================
# CUSTOMER VALIDATION
# ============================================================

required_columns = [
    "id",
    "type",
    "latitude",
    "longitude",
    "demand"
]


missing = [
    column
    for column in required_columns
    if column not in customers_df.columns
]


if missing:

    st.error(
        f"Missing columns: {missing}"
    )

    st.stop()


# ============================================================
# NORMALIZE CUSTOMER DATA
# ============================================================

customers_df["type"] = (
    customers_df["type"]
    .astype(str)
    .str.strip()
)


customers_df["latitude"] = pd.to_numeric(
    customers_df["latitude"],
    errors="coerce"
)


customers_df["longitude"] = pd.to_numeric(
    customers_df["longitude"],
    errors="coerce"
)


customers_df["demand"] = pd.to_numeric(
    customers_df["demand"],
    errors="coerce"
)


if customers_df[
    [
        "latitude",
        "longitude",
        "demand"
    ]
].isnull().any().any():

    st.error(
        "Customer CSV contains invalid "
        "latitude, longitude or demand values."
    )

    st.stop()


# ============================================================
# ORDER STATISTICS
# ============================================================

customer_mask = (
    customers_df["type"]
    .str.lower()
    != "depot"
)


depot_mask = (
    customers_df["type"]
    .str.lower()
    == "depot"
)


customer_count = len(
    customers_df[
        customer_mask
    ]
)


depot_count = len(
    customers_df[
        depot_mask
    ]
)


total_demand = customers_df[
    customer_mask
]["demand"].sum()


# ============================================================
# SIDEBAR METRICS
# ============================================================

st.sidebar.metric(
    "Customers",
    customer_count
)


st.sidebar.metric(
    "Depots",
    depot_count
)


st.sidebar.metric(
    "Total Demand",
    f"{total_demand:.0f} kg"
)


# ============================================================
# LOCATION DICTIONARY
# ============================================================

locations = {}


for _, row in customers_df.iterrows():

    locations[row["id"]] = {

        "type": row["type"],

        "latitude": float(
            row["latitude"]
        ),

        "longitude": float(
            row["longitude"]
        ),

        "demand": float(
            row["demand"]
        )

    }


# ============================================================
# CHECK NETWORK
# ============================================================

if st.session_state["graph"] is None:

    st.info(
        "👈 Enter a location and click "
        "**Load Road Network** to begin."
    )

    st.stop()


G = st.session_state["graph"]


# ============================================================
# GRAPH INFORMATION
# ============================================================

try:

    info = describe_graph(
        G
    )

except Exception:

    info = {

        "nodes": len(
            G.nodes
        ),

        "edges": len(
            G.edges
        )

    }


metric1, metric2, metric3, metric4 = st.columns(4)


with metric1:

    st.metric(
        "Road Nodes",
        f"{info['nodes']:,}"
    )


with metric2:

    st.metric(
        "Road Edges",
        f"{info['edges']:,}"
    )


with metric3:

    st.metric(
        "Customers",
        customer_count
    )


with metric4:

    st.metric(
        "Demand",
        f"{total_demand:.0f} kg"
    )


# ============================================================
# SNAP LOCATIONS
# ============================================================

with st.spinner(
    "Snapping orders to road network..."
):

    try:

        snapped_locations = snap_locations(
            G,
            locations
        )

    except Exception as e:

        st.error(
            f"Unable to snap locations: {e}"
        )

        st.stop()


# ============================================================
# SNAP STATUS
# ============================================================

unsnapped_locations = [

    location_id

    for location_id in locations

    if location_id not in snapped_locations

]


if unsnapped_locations:

    st.warning(
        "Some locations could not be snapped: "
        + ", ".join(
            str(x)
            for x in unsnapped_locations
        )
    )


st.sidebar.success(
    f"{len(snapped_locations)} locations snapped"
)


# ============================================================
# MAP CENTER
# ============================================================

first_node = list(
    G.nodes
)[0]


center_lat = float(
    G.nodes[first_node]["y"]
)


center_lon = float(
    G.nodes[first_node]["x"]
)


# ============================================================
# CREATE MAP
# ============================================================

m = folium.Map(

    location=[
        center_lat,
        center_lon
    ],

    zoom_start=14,

    tiles=None,

    control_scale=True

)


# ============================================================
# TILE LAYERS
# ============================================================

folium.TileLayer(
    "OpenStreetMap",
    name="Street Map"
).add_to(m)


folium.TileLayer(
    "CartoDB positron",
    name="Light Map"
).add_to(m)


folium.TileLayer(
    "CartoDB dark_matter",
    name="Dark Map"
).add_to(m)


# ============================================================
# DRAW ROAD NETWORK
# ============================================================

try:

    edges = ox.graph_to_gdfs(
        G,
        nodes=False
    )


    folium.GeoJson(

        edges.to_json(),

        name="Road Network",

        style_function=lambda feature: {

            "color": "#64748b",

            "weight": 2,

            "opacity": 0.55

        }

    ).add_to(m)


except Exception as e:

    st.warning(
        f"Unable to draw road network: {e}"
    )


# ============================================================
# ROUTE COLORS
# ============================================================

route_colors = [

    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#9333ea",
    "#ea580c",
    "#0891b2",
    "#db2777",
    "#65a30d",
    "#7c3aed",
    "#ca8a04"

]


# ============================================================
# DRAW OPTIMIZED ROUTES
# ============================================================

if st.session_state["routes"] is not None:

    routes = st.session_state["routes"]

    location_nodes = (
        st.session_state[
            "location_nodes"
        ]
    )

    location_names = (
        st.session_state[
            "location_names"
        ]
    )


    for vehicle_number, route in enumerate(
        routes
    ):

        color = route_colors[
            vehicle_number
            % len(route_colors)
        ]


        complete_route = (
            [0]
            + list(route)
            + [0]
        )


        for i in range(
            len(complete_route) - 1
        ):

            start_index = (
                complete_route[i]
            )

            end_index = (
                complete_route[i + 1]
            )


            start_node = (
                location_nodes[
                    start_index
                ]
            )


            end_node = (
                location_nodes[
                    end_index
                ]
            )


            navigation_path, path_type = (
                get_safe_navigation_path(
                    G,
                    start_node,
                    end_node
                )
            )


            if not navigation_path:

                continue


            coordinates = (
                get_path_coordinates(
                    G,
                    navigation_path
                )
            )


            if not coordinates:

                continue


            road_distance = (
                get_path_distance(
                    G,
                    navigation_path
                )
            )


            if path_type == "directed":

                path_label = (
                    "Actual OSM direction"
                )

            elif path_type == "undirected":

                path_label = (
                    "OSM road fallback"
                )

            else:

                path_label = (
                    "No road path"
                )


            folium.PolyLine(

                locations=coordinates,

                color=color,

                weight=7,

                opacity=0.90,

                tooltip=(
                    f"🚚 Vehicle "
                    f"{vehicle_number + 1}"
                ),

                popup=folium.Popup(

                    f"""
                    <div style="
                        width:230px;
                        font-family:Arial;
                    ">

                        <h4>
                            🚚 Vehicle
                            {vehicle_number + 1}
                        </h4>

                        <b>From:</b>
                        {location_names[start_index]}

                        <br><br>

                        <b>To:</b>
                        {location_names[end_index]}

                        <br><br>

                        <b>Road Distance:</b>
                        {road_distance:.2f} km

                        <br><br>

                        <b>Routing:</b>
                        {path_label}

                    </div>
                    """,

                    max_width=300

                )

            ).add_to(m)


# ============================================================
# DEPOT + CUSTOMER MARKERS
# ============================================================

customer_number = 1


for location_id, data in (
    snapped_locations.items()
):

    lat = data["latitude"]
    lon = data["longitude"]


    # --------------------------------------------------------
    # DEPOT
    # --------------------------------------------------------

    if (
        str(data["type"]).lower()
        == "depot"
    ):

        folium.Marker(

            location=[
                lat,
                lon
            ],

            popup=folium.Popup(

                f"""
                <div style="
                    width:190px;
                    font-family:Arial;
                ">

                    <h4>
                        🏭 Depot
                    </h4>

                    <b>ID:</b>
                    {location_id}

                    <br>

                    <b>Road Node:</b>
                    {data["node"]}

                </div>
                """,

                max_width=250

            ),

            tooltip="Depot",

            icon=folium.Icon(

                icon="home",

                prefix="fa",

                color="red"

            )

        ).add_to(m)


    # --------------------------------------------------------
    # CUSTOMER
    # --------------------------------------------------------

    else:

        folium.Marker(

            location=[
                lat,
                lon
            ],

            popup=folium.Popup(

                f"""
                <div style="
                    width:200px;
                    font-family:Arial;
                ">

                    <h4>
                        📦 Customer
                        {customer_number}
                    </h4>

                    <b>ID:</b>
                    {location_id}

                    <br>

                    <b>Demand:</b>
                    {data["demand"]:.1f} kg

                    <br>

                    <b>Road Node:</b>
                    {data["node"]}

                </div>
                """,

                max_width=250

            ),

            tooltip=(
                f"Customer "
                f"{customer_number}"
            ),

            icon=folium.DivIcon(

                html=f"""
                <div style="
                    background:#2563eb;
                    color:white;
                    border-radius:50%;
                    width:32px;
                    height:32px;
                    text-align:center;
                    line-height:32px;
                    font-weight:bold;
                    font-family:Arial;
                    border:2px solid white;
                    box-shadow:
                        0 2px 8px
                        rgba(0,0,0,0.35);
                ">
                    {customer_number}
                </div>
                """

            )

        ).add_to(m)


        customer_number += 1


# ============================================================
# MAP LEGEND
# ============================================================

legend_html = """

<div style="
    position: fixed;
    bottom: 25px;
    left: 25px;
    z-index: 9999;
    background:white;
    padding:13px;
    border-radius:10px;
    border:1px solid #ddd;
    box-shadow:0 2px 10px rgba(0,0,0,0.2);
    font-family:Arial;
    font-size:13px;
">

<b>🗺️ Map Legend</b>

<br><br>

<span style="
    display:inline-block;
    width:14px;
    height:14px;
    background:#2563eb;
    border-radius:50%;
"></span>

Customer

<br><br>

<span style="
    display:inline-block;
    width:14px;
    height:14px;
    background:#dc2626;
    border-radius:50%;
"></span>

Depot

<br><br>

<span style="
    display:inline-block;
    width:28px;
    height:6px;
    background:#2563eb;
"></span>

Optimized Route

<br><br>

<span style="
    display:inline-block;
    width:28px;
    height:3px;
    background:#64748b;
"></span>

Road Network

</div>

"""


m.get_root().html.add_child(
    folium.Element(
        legend_html
    )
)


# ============================================================
# MAP LAYER CONTROL
# ============================================================

folium.LayerControl(
    position="topright"
).add_to(m)


# ============================================================
# MAP DISPLAY
# ============================================================

st.markdown(
    '<div class="section-title">'
    '🗺️ Live Network Map'
    '</div>',
    unsafe_allow_html=True
)


map_result = st_folium(

    m,

    width=None,

    height=680,

    returned_objects=[
        "last_object_clicked"
    ]

)


if map_result:

    clicked = map_result.get(
        "last_object_clicked"
    )

    if clicked:

        st.info(
            "📍 Map location selected."
        )


# ============================================================
# ROUTE PLANNING PANEL
# ============================================================

st.markdown(
    '<div class="section-title">'
    '🚚 Route Planning'
    '</div>',
    unsafe_allow_html=True
)


route_col1, route_col2, route_col3 = st.columns(3)


with route_col1:

    algorithm = st.selectbox(

        "Optimization Algorithm",

        [
            "QPSO — Quantum-Inspired PSO",
            "PSO — Particle Swarm Optimization",
            "GA — Genetic Algorithm"
        ]

    )


with route_col2:

    num_vehicles = st.number_input(

        "Number of Vehicles",

        min_value=1,

        max_value=100,

        value=2,

        step=1

    )


with route_col3:

    vehicle_capacity = st.number_input(

        "Vehicle Capacity (kg)",

        min_value=1,

        max_value=10000,

        value=200,

        step=1

    )


# ============================================================
# DEFAULT PARAMETERS
# ============================================================

num_particles = 30
max_iterations = 100
beta = 0.75

inertia = 0.7
cognitive = 1.5
social = 1.5

population_size = 30
mutation_rate = 0.1


# ============================================================
# ALGORITHM CONFIGURATION
# ============================================================

if algorithm == "QPSO — Quantum-Inspired PSO":

    with st.expander(
        "⚛️ QPSO Configuration",
        expanded=True
    ):

        qpso_col1, qpso_col2, qpso_col3 = st.columns(3)


        with qpso_col1:

            num_particles = st.number_input(

                "Particles",

                min_value=5,

                max_value=200,

                value=30,

                step=5

            )


        with qpso_col2:

            max_iterations = st.number_input(

                "Iterations",

                min_value=10,

                max_value=1000,

                value=100,

                step=10

            )


        with qpso_col3:

            beta = st.slider(

                "Contraction-Expansion β",

                min_value=0.1,

                max_value=1.5,

                value=0.75,

                step=0.05

            )


elif algorithm == "PSO — Particle Swarm Optimization":

    with st.expander(
        "🌀 PSO Configuration",
        expanded=True
    ):

        pso_col1, pso_col2, pso_col3 = st.columns(3)


        with pso_col1:

            num_particles = st.number_input(

                "Particles",

                min_value=5,

                max_value=200,

                value=30,

                step=5

            )


        with pso_col2:

            max_iterations = st.number_input(

                "Iterations",

                min_value=10,

                max_value=1000,

                value=100,

                step=10

            )


        with pso_col3:

            inertia = st.slider(

                "Inertia",

                min_value=0.1,

                max_value=1.5,

                value=0.7,

                step=0.05

            )


        cognitive = st.slider(

            "Cognitive coefficient",

            min_value=0.1,

            max_value=3.0,

            value=1.5,

            step=0.1

        )


        social = st.slider(

            "Social coefficient",

            min_value=0.1,

            max_value=3.0,

            value=1.5,

            step=0.1

        )


else:

    with st.expander(

        "🧬 Genetic Algorithm Configuration",

        expanded=True

    ):

        ga_col1, ga_col2, ga_col3 = st.columns(3)


        with ga_col1:

            population_size = st.number_input(

                "Population Size",

                min_value=5,

                max_value=200,

                value=30,

                step=5

            )


        with ga_col2:

            max_iterations = st.number_input(

                "Iterations",

                min_value=10,

                max_value=1000,

                value=100,

                step=10

            )


        with ga_col3:

            mutation_rate = st.slider(

                "Mutation Rate",

                min_value=0.01,

                max_value=1.0,

                value=0.1,

                step=0.01

            )


# ============================================================
# ACTION BUTTONS
# ============================================================

button_col1, button_col2 = st.columns(
    [4, 1]
)


with button_col1:

    optimize_button = st.button(

        "🚀 Generate Optimized Routes",

        use_container_width=True,

        type="primary"

    )


with button_col2:

    clear_button = st.button(

        "🗑️ Clear",

        use_container_width=True

    )


# ============================================================
# CLEAR RESULTS
# ============================================================

if clear_button:

    st.session_state["routes"] = None

    st.session_state["route_stats"] = None

    st.session_state["optimization_result"] = None

    st.session_state["benchmark_results"] = None

    st.rerun()


# ============================================================
# OPTIMIZATION
# ============================================================

if optimize_button:

    try:

        # ----------------------------------------------------
        # DEPOT
        # ----------------------------------------------------

        depot_rows = customers_df[
            customers_df["type"]
            .astype(str)
            .str.lower()
            == "depot"
        ]


        if len(depot_rows) == 0:

            st.error(
                "❌ No depot found."
            )

            st.stop()


        depot_id = (
            depot_rows.iloc[0]["id"]
        )


        # ----------------------------------------------------
        # CUSTOMERS
        # ----------------------------------------------------

        customer_rows = customers_df[
            customers_df["type"]
            .astype(str)
            .str.lower()
            != "depot"
        ]


        if len(customer_rows) == 0:

            st.error(
                "❌ No customers found."
            )

            st.stop()


        # ----------------------------------------------------
        # LOCATION NAMES
        # ----------------------------------------------------

        location_names = [
            depot_id
        ]


        for _, row in customer_rows.iterrows():

            location_names.append(
                row["id"]
            )


        # ----------------------------------------------------
        # LOCATION NODES
        # ----------------------------------------------------

        location_nodes = []

        missing_snaps = []


        for name in location_names:

            if name not in snapped_locations:

                missing_snaps.append(
                    name
                )

            else:

                location_nodes.append(
                    snapped_locations[
                        name
                    ]["node"]
                )


        if missing_snaps:

            st.error(
                "Locations not snapped: "
                + ", ".join(
                    str(x)
                    for x in missing_snaps
                )
            )

            st.stop()


        # ----------------------------------------------------
        # DEMANDS
        # ----------------------------------------------------

        demands = {}


        for index, name in enumerate(
            location_names
        ):

            if index == 0:

                demands[index] = 0.0

            else:

                demands[index] = float(
                    snapped_locations[
                        name
                    ]["demand"]
                )


        # ----------------------------------------------------
        # CAPACITY CHECK
        # ----------------------------------------------------

        if total_demand > (
            num_vehicles
            * vehicle_capacity
        ):

            st.error(

                "❌ Problem is infeasible.\n\n"

                f"Total demand: "
                f"{total_demand:.1f} kg\n\n"

                f"Total capacity: "
                f"{num_vehicles * vehicle_capacity:.1f} kg"

            )

            st.stop()


        # ----------------------------------------------------
        # INDIVIDUAL CUSTOMER CHECK
        # ----------------------------------------------------

        oversized_customers = []


        for name in location_names[1:]:

            demand = float(
                snapped_locations[
                    name
                ]["demand"]
            )


            if demand > vehicle_capacity:

                oversized_customers.append(
                    (name, demand)
                )


        if oversized_customers:

            details = ", ".join(

                f"{name} ({demand:.1f} kg)"

                for name, demand
                in oversized_customers

            )


            st.error(

                "❌ Customer demand exceeds "
                "vehicle capacity: "
                + details

            )

            st.stop()


        # ----------------------------------------------------
        # DISTANCE MATRIX
        # ----------------------------------------------------

        progress = st.progress(
            0,
            text="Preparing road network..."
        )


        start_time = time.time()


        distance_matrix = (
            build_distance_matrix(
                G,
                location_nodes
            )
        )


        progress.progress(
            25,
            text="Road distance matrix ready."
        )


        # ----------------------------------------------------
        # DISCONNECTED CHECK
        # ----------------------------------------------------

        if np.any(
            distance_matrix >= 1e9
        ):

            st.error(
                "❌ Some locations are disconnected "
                "from the road network."
            )

            st.stop()


        # ----------------------------------------------------
        # CUSTOMER INDICES
        # ----------------------------------------------------

        customer_indices = list(
            range(
                1,
                len(location_names)
            )
        )


        # ----------------------------------------------------
        # CREATE OPTIMIZER
        # ----------------------------------------------------

        if algorithm == "QPSO — Quantum-Inspired PSO":

            optimizer = QPSO_VRP(

                distance_matrix=distance_matrix,

                demands=demands,

                customer_indices=customer_indices,

                depot_index=0,

                vehicle_capacity=int(
                    vehicle_capacity
                ),

                num_vehicles=int(
                    num_vehicles
                ),

                num_particles=int(
                    num_particles
                ),

                max_iterations=int(
                    max_iterations
                ),

                beta=float(
                    beta
                ),

                seed=42

            )


        elif algorithm == "PSO — Particle Swarm Optimization":

            optimizer = PSO_VRP(

                distance_matrix=distance_matrix,

                demands=demands,

                customer_indices=customer_indices,

                depot_index=0,

                vehicle_capacity=int(
                    vehicle_capacity
                ),

                num_vehicles=int(
                    num_vehicles
                ),

                num_particles=int(
                    num_particles
                ),

                max_iterations=int(
                    max_iterations
                ),

                inertia=float(
                    inertia
                ),

                cognitive=float(
                    cognitive
                ),

                social=float(
                    social
                ),

                seed=42

            )


        else:

            optimizer = GeneticAlgorithmVRP(

                distance_matrix=distance_matrix,

                demands=demands,

                customer_indices=customer_indices,

                depot_index=0,

                vehicle_capacity=int(
                    vehicle_capacity
                ),

                num_vehicles=int(
                    num_vehicles
                ),

                population_size=int(
                    population_size
                ),

                max_iterations=int(
                    max_iterations
                ),

                mutation_rate=float(
                    mutation_rate
                ),

                seed=42

            )


        # ----------------------------------------------------
        # RUN OPTIMIZATION
        # ----------------------------------------------------

        progress.progress(
            40,
            text=f"Running {algorithm}..."
        )


        with st.spinner(
            f"🚀 {algorithm} is optimizing..."
        ):

            result = optimizer.optimize()


        # ----------------------------------------------------
        # SUPPORT 2 OR 3 RETURN VALUES
        # ----------------------------------------------------

        if not isinstance(result, tuple):

            raise ValueError(
                "Optimizer.optimize() must return "
                "a tuple."
            )


        if len(result) == 3:

            routes, best_distance, iterations = result

        elif len(result) == 2:

            routes, best_distance = result

            iterations = max_iterations

        else:

            raise ValueError(
                "Optimizer.optimize() must return "
                "(routes, distance) or "
                "(routes, distance, iterations)."
            )


        progress.progress(
            80,
            text="Calculating route statistics..."
        )


        # ----------------------------------------------------
        # VALIDATE RESULT
        # ----------------------------------------------------

        if routes is None:

            st.error(
                "❌ Optimizer did not generate routes."
            )

            st.stop()


        if len(routes) > num_vehicles:

            st.error(
                "❌ Optimizer generated more routes "
                "than available vehicles."
            )

            st.stop()


        # ----------------------------------------------------
        # SAVE OPTIMIZATION RESULT
        # ----------------------------------------------------

        st.session_state["routes"] = routes

        st.session_state[
            "location_nodes"
        ] = location_nodes

        st.session_state[
            "location_names"
        ] = location_names

        st.session_state[
            "distance_matrix"
        ] = distance_matrix


        st.session_state[
            "optimization_result"
        ] = {

            "algorithm": algorithm,

            "distance": float(
                best_distance
            ),

            "runtime": time.time() - start_time,

            "iterations": int(
                iterations
            )

        }


        # ----------------------------------------------------
        # ROUTE STATISTICS
        # ----------------------------------------------------

        route_stats = []


        for vehicle_number, route in enumerate(
            routes
        ):

            route = list(route)


            # LOAD

            load = sum(
                demands.get(
                    int(customer),
                    0.0
                )
                for customer in route
            )


            # DISTANCE

            try:

                distance = float(
                    optimizer.route_distance(
                        route
                    )
                )

            except Exception:

                complete_route = (
                    [0]
                    + route
                    + [0]
                )

                distance = 0.0


                for i in range(
                    len(complete_route) - 1
                ):

                    distance += (
                        distance_matrix[
                            complete_route[i],
                            complete_route[i + 1]
                        ]
                    )


            # CUSTOMER NAMES

            route_customer_names = [

                str(
                    location_names[int(c)]
                )

                for c in route

            ]


            # ETA

            eta_minutes = calculate_eta(
                distance
            )

            eta_text = format_eta(
                eta_minutes
            )


            # UTILIZATION

            if vehicle_capacity > 0:

                utilization = (
                    load
                    / vehicle_capacity
                ) * 100

            else:

                utilization = 0.0


            route_stats.append({

                "Vehicle":
                    f"Vehicle {vehicle_number + 1}",

                "Customers":
                    len(route),

                "Load (kg)":
                    round(
                        load,
                        2
                    ),

                "Capacity (kg)":
                    vehicle_capacity,

                "Utilization (%)":
                    round(
                        utilization,
                        1
                    ),

                "Distance (km)":
                    round(
                        distance,
                        2
                    ),

                "ETA":
                    eta_text,

                "Route":
                    "DEPOT → "
                    + " → ".join(
                        route_customer_names
                    )
                    + " → DEPOT"

            })


        st.session_state[
            "route_stats"
        ] = pd.DataFrame(
            route_stats
        )


        progress.progress(
            100,
            text="Optimization completed."
        )


        st.success(

            f"✅ {algorithm} completed successfully "
            f"in {time.time() - start_time:.2f} seconds."

        )


        st.rerun()


    except Exception as e:

        st.error(
            f"❌ Route optimization failed: {e}"
        )


# ============================================================
# OPTIMIZATION RESULTS
# ============================================================

if st.session_state[
    "route_stats"
] is not None:

    stats_df = (
        st.session_state[
            "route_stats"
        ]
    )


    result_info = (
        st.session_state[
            "optimization_result"
        ]
    )


    st.markdown(

        '<div class="section-title">'
        '📊 Optimization Dashboard'
        '</div>',

        unsafe_allow_html=True

    )


    # --------------------------------------------------------
    # KPIs
    # --------------------------------------------------------

    total_distance = stats_df[
        "Distance (km)"
    ].sum()


    total_load = stats_df[
        "Load (kg)"
    ].sum()


    if len(stats_df) > 0:

        average_utilization = stats_df[
            "Utilization (%)"
        ].mean()

    else:

        average_utilization = 0.0


    k1, k2, k3, k4 = st.columns(4)


    with k1:

        st.metric(

            "Algorithm",

            result_info[
                "algorithm"
            ].split(" — ")[0]

        )


    with k2:

        st.metric(

            "Total Distance",

            f"{total_distance:.2f} km"

        )


    with k3:

        st.metric(

            "Vehicles Used",

            len(stats_df)

        )


    with k4:

        st.metric(

            "Avg. Utilization",

            f"{average_utilization:.1f}%"

        )


    # --------------------------------------------------------
    # SECONDARY KPIs
    # --------------------------------------------------------

    k5, k6, k7 = st.columns(3)


    with k5:

        st.metric(

            "Total Load",

            f"{total_load:.1f} kg"

        )


    with k6:

        st.metric(

            "Runtime",

            f"{result_info['runtime']:.4f} sec"

        )


    with k7:

        st.metric(

            "Iterations",

            result_info[
                "iterations"
            ]

        )


    # --------------------------------------------------------
    # VEHICLE PERFORMANCE
    # --------------------------------------------------------

    st.markdown(
        "### 🚚 Vehicle Performance"
    )


    st.dataframe(

        stats_df,

        use_container_width=True,

        hide_index=True

    )


    # --------------------------------------------------------
    # CAPACITY UTILIZATION
    # --------------------------------------------------------

    st.markdown(
        "### 📦 Vehicle Capacity Utilization"
    )


    for _, row in stats_df.iterrows():

        st.markdown(

            f"**{row['Vehicle']}** — "
            f"{row['Load (kg)']:.1f} / "
            f"{row['Capacity (kg)']:.1f} kg "
            f"({row['Utilization (%)']:.1f}%)"

        )


        st.progress(

            min(
                row["Utilization (%)"] / 100,
                1.0
            )

        )


    # --------------------------------------------------------
    # ROUTE CARDS
    # --------------------------------------------------------

    st.markdown(
        "### 🛣️ Optimized Routes"
    )


    for _, row in stats_df.iterrows():

        st.markdown(

            f"""
            <div class="route-card">

                <div class="route-title">
                    🚚 {row["Vehicle"]}
                </div>

                <div class="route-info">

                    📦 Customers:
                    <b>{row["Customers"]}</b>

                    &nbsp;&nbsp; | &nbsp;&nbsp;

                    ⚖️ Load:
                    <b>{row["Load (kg)"]:.1f} kg</b>

                    &nbsp;&nbsp; | &nbsp;&nbsp;

                    📊 Utilization:
                    <b>{row["Utilization (%)"]:.1f}%</b>

                    &nbsp;&nbsp; | &nbsp;&nbsp;

                    🛣️ Distance:
                    <b>{row["Distance (km)"]:.2f} km</b>

                    &nbsp;&nbsp; | &nbsp;&nbsp;

                    ⏱️ ETA:
                    <b>{row["ETA"]}</b>

                </div>

                <br>

                <b>Route:</b>

                <br>

                {row["Route"]}

            </div>
            """,

            unsafe_allow_html=True

        )


    # --------------------------------------------------------
    # EXPORT
    # --------------------------------------------------------

    st.markdown(
        "### 📥 Export Results"
    )


    download_col1, download_col2 = st.columns(2)


    with download_col1:

        csv_data = stats_df.to_csv(
            index=False
        )


        st.download_button(

            label="⬇️ Download Route Results",

            data=csv_data,

            file_name="black_rock_routes.csv",

            mime="text/csv",

            use_container_width=True

        )


    with download_col2:

        summary_df = pd.DataFrame({

            "Metric": [

                "Algorithm",

                "Total Distance (km)",

                "Vehicles Used",

                "Runtime (seconds)",

                "Iterations",

                "Average Utilization (%)"

            ],

            "Value": [

                result_info[
                    "algorithm"
                ],

                round(
                    total_distance,
                    3
                ),

                len(stats_df),

                round(
                    result_info[
                        "runtime"
                    ],
                    4
                ),

                result_info[
                    "iterations"
                ],

                round(
                    average_utilization,
                    2
                )

            ]

        })


        summary_csv = (
            summary_df.to_csv(
                index=False
            )
        )


        st.download_button(

            label="⬇️ Download Optimization Summary",

            data=summary_csv,

            file_name=(
                "black_rock_optimization_summary.csv"
            ),

            mime="text/csv",

            use_container_width=True

        )


# ============================================================
# ALGORITHM BENCHMARK
# ============================================================

st.markdown(
    '<div class="section-title">'
    '⚖️ Algorithm Comparison'
    '</div>',
    unsafe_allow_html=True
)


st.write(
    "Compare QPSO, PSO and Genetic Algorithm "
    "on the same VRP problem."
)


benchmark_button = st.button(

    "⚖️ Run Algorithm Benchmark",

    use_container_width=True,

    type="primary"

)


# ============================================================
# RUN BENCHMARK
# ============================================================

if benchmark_button:

    try:

        if st.session_state["graph"] is None:

            st.error(
                "❌ Please load the road network first."
            )

            st.stop()


        if not snapped_locations:

            st.error(
                "❌ No locations are available."
            )

            st.stop()


        # ----------------------------------------------------
        # DEPOT
        # ----------------------------------------------------

        depot_rows = customers_df[
            customers_df["type"]
            .astype(str)
            .str.lower()
            == "depot"
        ]


        if len(depot_rows) == 0:

            st.error(
                "❌ No depot found."
            )

            st.stop()


        depot_id = (
            depot_rows.iloc[0]["id"]
        )


        # ----------------------------------------------------
        # CUSTOMERS
        # ----------------------------------------------------

        customer_rows = customers_df[
            customers_df["type"]
            .astype(str)
            .str.lower()
            != "depot"
        ]


        if len(customer_rows) == 0:

            st.error(
                "❌ No customers found."
            )

            st.stop()


        # ----------------------------------------------------
        # LOCATION NAMES
        # ----------------------------------------------------

        location_names = [
            depot_id
        ]


        for _, row in customer_rows.iterrows():

            location_names.append(
                row["id"]
            )


        # ----------------------------------------------------
        # LOCATION NODES
        # ----------------------------------------------------

        location_nodes = []

        missing_snaps = []


        for name in location_names:

            if name not in snapped_locations:

                missing_snaps.append(
                    name
                )

            else:

                location_nodes.append(
                    snapped_locations[
                        name
                    ]["node"]
                )


        if missing_snaps:

            st.error(
                "❌ Locations not snapped: "
                + ", ".join(
                    str(x)
                    for x in missing_snaps
                )
            )

            st.stop()


        # ----------------------------------------------------
        # DEMANDS
        # ----------------------------------------------------

        demands = {}


        for index, name in enumerate(
            location_names
        ):

            if index == 0:

                demands[index] = 0.0

            else:

                demands[index] = float(
                    snapped_locations[
                        name
                    ]["demand"]
                )


        # ----------------------------------------------------
        # CAPACITY CHECK
        # ----------------------------------------------------

        if total_demand > (
            num_vehicles
            * vehicle_capacity
        ):

            st.error(

                "❌ Problem is infeasible.\n\n"

                f"Total demand: "
                f"{total_demand:.1f} kg\n\n"

                f"Total vehicle capacity: "
                f"{num_vehicles * vehicle_capacity:.1f} kg"

            )

            st.stop()


        # ----------------------------------------------------
        # CUSTOMER CAPACITY
        # ----------------------------------------------------

        oversized_customers = []


        for name in location_names[1:]:

            demand = float(
                snapped_locations[
                    name
                ]["demand"]
            )


            if demand > vehicle_capacity:

                oversized_customers.append(
                    (name, demand)
                )


        if oversized_customers:

            details = ", ".join(

                f"{name} ({demand:.1f} kg)"

                for name, demand
                in oversized_customers

            )


            st.error(

                "❌ Customer demand exceeds "
                "vehicle capacity: "
                + details

            )

            st.stop()


        # ----------------------------------------------------
        # DISTANCE MATRIX
        # ----------------------------------------------------

        with st.spinner(
            "Building road distance matrix..."
        ):

            distance_matrix = (
                build_distance_matrix(
                    G,
                    location_nodes
                )
            )


        if np.any(
            distance_matrix >= 1e9
        ):

            st.error(
                "❌ Some locations are disconnected "
                "from the road network."
            )

            st.stop()


        # ----------------------------------------------------
        # CUSTOMER INDICES
        # ----------------------------------------------------

        customer_indices = list(
            range(
                1,
                len(location_names)
            )
        )


        # ----------------------------------------------------
        # BENCHMARK ENGINE
        # ----------------------------------------------------

        benchmark_engine = BenchmarkEngine()


        # ----------------------------------------------------
        # QPSO
        # ----------------------------------------------------

        def run_qpso():

            optimizer = QPSO_VRP(

                distance_matrix=distance_matrix,

                demands=demands,

                customer_indices=customer_indices,

                depot_index=0,

                vehicle_capacity=int(
                    vehicle_capacity
                ),

                num_vehicles=int(
                    num_vehicles
                ),

                num_particles=int(
                    num_particles
                ),

                max_iterations=int(
                    max_iterations
                ),

                beta=float(
                    beta
                ),

                seed=42

            )


            return optimizer.optimize()


        # ----------------------------------------------------
        # PSO
        # ----------------------------------------------------

        def run_pso():

            optimizer = PSO_VRP(

                distance_matrix=distance_matrix,

                demands=demands,

                customer_indices=customer_indices,

                depot_index=0,

                vehicle_capacity=int(
                    vehicle_capacity
                ),

                num_vehicles=int(
                    num_vehicles
                ),

                num_particles=int(
                    num_particles
                ),

                max_iterations=int(
                    max_iterations
                ),

                inertia=float(
                    inertia
                ),

                cognitive=float(
                    cognitive
                ),

                social=float(
                    social
                ),

                seed=42

            )


            return optimizer.optimize()


        # ----------------------------------------------------
        # GENETIC ALGORITHM
        # ----------------------------------------------------

        def run_ga():

            optimizer = GeneticAlgorithmVRP(

                distance_matrix=distance_matrix,

                demands=demands,

                customer_indices=customer_indices,

                depot_index=0,

                vehicle_capacity=int(
                    vehicle_capacity
                ),

                num_vehicles=int(
                    num_vehicles
                ),

                population_size=int(
                    population_size
                ),

                max_iterations=int(
                    max_iterations
                ),

                mutation_rate=float(
                    mutation_rate
                ),

                seed=42

            )


            return optimizer.optimize()


        # ----------------------------------------------------
        # RUN BENCHMARK
        # ----------------------------------------------------

        progress = st.progress(
            0,
            text="Running benchmark..."
        )


        progress.progress(
            10,
            text="Running QPSO..."
        )


        benchmark_engine.run_algorithm(
            "QPSO",
            run_qpso
        )


        progress.progress(
            40,
            text="Running PSO..."
        )


        benchmark_engine.run_algorithm(
            "PSO",
            run_pso
        )


        progress.progress(
            70,
            text="Running Genetic Algorithm..."
        )


        benchmark_engine.run_algorithm(
            "GA",
            run_ga
        )


        progress.progress(
            100,
            text="Benchmark completed."
        )


        # ----------------------------------------------------
        # RESULT TABLE
        # ----------------------------------------------------

        benchmark_df = (
            benchmark_engine.results_dataframe()
        )


        # ----------------------------------------------------
        # BEST ALGORITHM
        # ----------------------------------------------------

        best = (
            benchmark_engine.best_result()
        )


        if best is not None:

            st.session_state[
                "benchmark_results"
            ] = benchmark_df


            st.success(

                f"🏆 Best Algorithm: "
                f"{best.algorithm_name} "
                f"— {best.distance:.2f} km"

            )


        else:

            st.error(
                "❌ Benchmark did not produce results."
            )


    except Exception as e:

        st.error(
            f"❌ Benchmark failed: {e}"
        )


# ============================================================
# DISPLAY BENCHMARK RESULTS
# ============================================================

if st.session_state[
    "benchmark_results"
] is not None:

    benchmark_df = (
        st.session_state[
            "benchmark_results"
        ]
    )


    st.markdown(
        "### 📊 Benchmark Results"
    )


    st.dataframe(

        benchmark_df,

        use_container_width=True,

        hide_index=True

    )


    # --------------------------------------------------------
    # BEST ALGORITHM
    # --------------------------------------------------------

    valid_results = benchmark_df[
        benchmark_df[
            "Total Distance (km)"
        ].notna()
    ]


    if not valid_results.empty:

        best_row = valid_results.loc[
            valid_results[
                "Total Distance (km)"
            ].idxmin()
        ]


        b1, b2, b3 = st.columns(3)


        with b1:

            st.metric(
                "🏆 Best Algorithm",
                best_row["Algorithm"]
            )


        with b2:

            st.metric(
                "Shortest Distance",
                f"{best_row['Total Distance (km)']:.2f} km"
            )


        with b3:

            st.metric(
                "Runtime",
                f"{best_row['Runtime (seconds)']:.4f} sec"
            )


# ============================================================
# ORDERS & LOCATIONS
# ============================================================

st.markdown(

    '<div class="section-title">'
    '📍 Orders & Network Locations'
    '</div>',

    unsafe_allow_html=True

)


display_df = customers_df.copy()


display_df["road_node"] = [

    snapped_locations[
        row["id"]
    ]["node"]

    if row["id"] in snapped_locations

    else None

    for _, row in customers_df.iterrows()

]


display_df["status"] = [

    "Ready"

    if row["id"] in snapped_locations

    else "Not Snapped"

    for _, row in customers_df.iterrows()

]


st.dataframe(

    display_df,

    use_container_width=True,

    hide_index=True

)


# ============================================================
# NETWORK INFORMATION
# ============================================================

with st.expander(
    "🔎 Network Information"
):

    network_col1, network_col2 = st.columns(2)


    with network_col1:

        st.write(
            "**Map Location**"
        )

        st.write(
            place
        )


        st.write(
            "**Road Nodes**"
        )

        st.write(
            f"{info['nodes']:,}"
        )


        st.write(
            "**Road Edges**"
        )

        st.write(
            f"{info['edges']:,}"
        )


    with network_col2:

        st.write(
            "**Graph Type**"
        )

        st.write(

            "Directed"

            if G.is_directed()

            else "Undirected"

        )


        st.write(
            "**Customers**"
        )

        st.write(
            customer_count
        )


        st.write(
            "**Locations Snapped**"
        )

        st.write(
            len(snapped_locations)
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()


st.markdown(

    """
    <div class="footer">

        🚚 <b>BLACK ROCK VRP</b>
        &nbsp;•&nbsp;
        OpenStreetMap Road Network
        &nbsp;•&nbsp;
        Quantum-Inspired Optimization
        &nbsp;•&nbsp;
        Vehicle Routing Platform

    </div>
    """,

    unsafe_allow_html=True

)