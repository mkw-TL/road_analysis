# Author: Joe Harrison
# Date: Oct 27, 2023
# License: MIT
# NOTE: DearPyGui is not officially supported on Windows 11

import geopandas as gpd
import shapely
import pandas as pd
import numpy as np
import timeit
import math

# pip install icecream
from icecream import ic

# pip install tqdm
from tqdm import tqdm

# pip install networkx
import networkx as nx
import time
import copy

from operator import itemgetter

# pip install Amazon-DenseClus

import matplotlib.pyplot as plt

plt.switch_backend("Agg")  # so no issues with GUI backend
# as per this https://stackoverflow.com/questions/52839758/matplotlib-and-runtimeerror-main-thread-is-not-in-main-loop
import seaborn as sns

sns.set_style("darkgrid", {"axes.facecolor": ".9"})
sns.set(rc={"figure.figsize": (8, 6)})

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import matplotlib.colors as mcolors
import os
from numpy import abs, min, subtract


def get_neighbors_and_update(this_dict, this_name, this_line, tree, data):
    nearest, dist = tree.query_nearest(
        this_line, return_distance=True, max_distance=0.001
    )
    if dist[0] == 0:
        for _, id in enumerate(nearest):
            # nearest from .query_nearest gives the ids, not the names
            if data["FULLNAME"][id] == this_name:
                this_dict["self_connections"] += 1
            # append all connected roads that are not this road
            if data["FULLNAME"][id] not in this_dict["connections_names"]:
                this_dict["connections_names"].append(data["FULLNAME"][id])
            this_dict["number_of_connector_roads"] += 1
    return this_dict


def create_data_for_road(i, set_of_looked_at_roads, list_of_dicts, tree, data):
    this_dict = {
        "road_name": "",
        "connections_names": [],
        "is_named": 1,
        "self_connections": 0,
        "number_of_connector_roads": 0,
        "length": 0,
    }
    this_line = data["geometry"][i]
    this_name = data["FULLNAME"][i]

    # first and last character of road are numeric
    if (this_name[0] in {"1", "2", "3", "4", "5", "6", "7", "8", "9", "0"}) and (
        this_name[-1] in {"1", "2", "3", "4", "5", "6", "7", "8", "9", "0"}
    ):
        this_dict["is_named"] = 0.9  # arbitrary value for false

    if this_name not in set_of_looked_at_roads:
        # ignore it if the road has been looked at
        set_of_looked_at_roads.add(this_name)
        this_dict["road_name"] = this_name
        this_dict["length"] = shapely.length(this_line)
        this_dict["number_of_parts"] = 1
        # main update step
        this_dict = get_neighbors_and_update(
            this_dict, this_name, this_line, tree, data
        )
        list_of_dicts.append(this_dict)
    else:  # if you have seen the road before
        orig_ind = 100000000
        # find the original road, so that you can append to it later
        # can't use this_dict anymore
        for ind in range(len(list_of_dicts)):
            if this_name == list_of_dicts[ind]["road_name"]:
                orig_ind = ind
                break
        this_dict = list_of_dicts[orig_ind]

        this_dict["length"] += shapely.length(this_line)
        this_dict["number_of_parts"] += 1
        this_dict = get_neighbors_and_update(
            this_dict, this_name, this_line, tree, data
        )
        list_of_dicts[orig_ind] = this_dict  # the modifications go back in
    yield i


def loop_through_roads(set_of_looked_at_roads, list_of_dicts, tree, data, num_roads):
    i = 0
    while True and i % 100 == 0:
        try:
            print(
                next(
                    create_data_for_road(
                        i, set_of_looked_at_roads, list_of_dicts, tree, data
                    )
                ),
                end=" ",
            )
            print("/", str(num_roads))
            i = i + 1
        except:
            return list_of_dicts, set_of_looked_at_roads


def graph_analysis(list_of_dicts, start_time, shape_files):
    # Create a graph
    G = nx.Graph()

    # Add nodes (roads) to the graph
    for road_data in list_of_dicts:
        # some roads don't have a road_name field?? Not sure how that happens..
        # OH Thats what's causing our failed import
        if "road_name" in road_data:
            G.add_node(road_data["road_name"], data=road_data)

    # once all the nodes are created
    # Add edges (connections between roads) to the graph
    for road_data in list_of_dicts:
        if "road_name" in road_data and "connections_names" in road_data:
            road_name = road_data["road_name"]
            connections = road_data["connections_names"]
            for connection in connections:
                G.add_edge(road_name, connection)

    ic("Time is", timeit.default_timer() - start_time)
    ic("Number of roads in G is", len(G))

    depth = 400
    if nx.is_connected(G):
        ic("connected!")
        ic("number of nodes is", str(G.number_of_nodes()))
        ic("number of edges is", str(G.number_of_edges()))
        ic("degree centrality is", str(nx.degree_centrality(G))[:depth])
        ic("pagerank is", str(nx.pagerank(G))[:depth])
        ic(list_of_connected_dicts=G.nodes)

        centr = nx.degree_centrality(G)
        pge_rnk = nx.pagerank(G)
    else:
        ic("not connected. Analysis on largest connected subset below")
        list_of_connected_dicts = []
        orig_idxs = []
        G3 = nx.Graph()

        G2 = copy.deepcopy(G)
        # It looks like copy was depreciated as an argument to max() or connected_components
        # And since connected_components removes attribute labels, we need to keep these labels floating around.
        G2 = max(nx.connected_components(G2), key=len)
        # G2 gives a list of road names that are connected
        for i, this_road in enumerate(list_of_dicts):
            # enumerate on list_of_dicts rather than G. G is our graph, and it yields the name
            if "road_name" in this_road:
                # check if this road is part of the set
                if this_road["road_name"] in G2:
                    list_of_connected_dicts.append(this_road)
                    orig_idxs.append(i)
                    G3.add_node(this_road["road_name"], data=this_road)

        for i, conn_road in enumerate(list_of_connected_dicts):
            # loop through again because edges have to come after nodes
            if conn_road["road_name"] in G2:
                road_name = conn_road["road_name"]
                connections = conn_road["connections_names"]
                for connection in connections:
                    G3.add_edge(road_name, connection)
        ic("number of nodes is", str(G3.number_of_nodes()))
        ic("number of edges is", str(G3.number_of_edges()))
        centr = nx.degree_centrality(G3)
        centr_list = list(centr.values())
        pge_rnk = nx.pagerank(G3)
        pge_rnk_list = list(pge_rnk.values())
        ic(
            "the difference between roads before and after cleaning is",
            str(G.number_of_nodes()),
            "to",
            str(G3.number_of_nodes()),
        )

    connected_sfs = []

    for i in range(len(list_of_connected_dicts)):
        list_of_connected_dicts[i]["degree_centrality"] = centr_list[i]
        list_of_connected_dicts[i]["page_rank"] = pge_rnk_list[i]
        connected_sfs.append(shape_files[orig_idxs[i]])

    return list_of_connected_dicts, connected_sfs


def show_results(list_of_connected_dicts):
    ic("this might take some time")
    df = pd.DataFrame(list_of_connected_dicts)
    # takes a while to put back into queriable format
    df = df.drop(columns=["road_name", "connections_names", "is_named"])
    pd.set_option("display.max_columns", None)
    ic(df.columns)

    normalized_df = df.copy()
    eps = 0.000001

    for i, col in enumerate(df.columns):
        normalized_df[col] = df[col] / df[col].max()
        normalized_df[col] = np.log(normalized_df[col] + eps)
        # normalize. Make sure not all vals are the same. If two values then they will be as far apart as possible
        # log scale, 0 undefined

    kmeans = KMeans(n_clusters=6)
    kmeans.fit(normalized_df)
    orig_dimen_centers = [
        kmeans.cluster_centers_[i] * df[col].max() for i, col in enumerate(df.columns)
    ]
    # kmeans first and then pca later -- provides the most seperation

    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(normalized_df)
    print(
        "our variance explained by the first two principle components are:",
        pca.explained_variance_ratio_,
    )

    transformed_center = pca.transform(kmeans.cluster_centers_)

    tabl_cols = list(mcolors.BASE_COLORS.values())
    cols = [tabl_cols[int(val)] for val in kmeans.labels_]

    k_cols = [tabl_cols[i] for i in range(6)]

    # now combining results together
    df["kmeans_result"] = kmeans.labels_

    for j in range(6):
        ic(df[df["kmeans_result"] == j].loc[:, "length"].mean())
        ic(len(df[df["kmeans_result"] == j].loc[:, "length"]))
        ic(df[df["kmeans_result"] == j].loc[:, "length"].median())

    # https://stackoverflow.com/questions/39216897/plot-pca-loadings-and-loading-in-biplot-in-sklearn-like-rs-autoplot
    # from Qiyun Zhu

    # coordinates of features (i.e., loadings; note the transpose)
    loadings = pca.components_[:2].T

    # proportions of variance explained by axes
    pvars = pca.explained_variance_ratio_[:2] * 100

    arrows = loadings * abs(reduced_data).max(axis=0)

    # empirical formula to determine arrow width
    width = -0.0075 * min([subtract(*plt.xlim()), subtract(*plt.ylim())])

    col_names = [
        "Connections to Self",
        "Num of Connections",
        "Length",
        "Segments",
        "Centrality",
        "Page Rank",
    ]

    # features as arrows
    for i, arrow in enumerate(arrows):
        plt.arrow(
            0,
            0,
            *arrow,
            color="k",
            alpha=0.3,
            width=width,
            ec="none",
            length_includes_head=True,
        )
        plt.text(*(arrow * 1.1), col_names[i], ha="center", va="center", fontsize=14)

    centers = pd.DataFrame(transformed_center)

    # axis labels
    for i, axis in enumerate("xy"):
        getattr(plt, f"{axis}ticks")([])
        getattr(plt, f"{axis}label")(f"Principle Component {i + 1} ({pvars[i]:.2f}%)")

    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=cols)
    # plt.scatter(centers.iloc[:, 0], centers.iloc[:, 1], c=k_cols, s=300)

    # for k in range(6):
    #     new_str = ""
    #     for val in range(6):
    #         str_seg = float(orig_dimen_centers[val][k])
    #         new_str += f"{str_seg:.2f}, "
    #     plt.text(
    #         centers.iloc[k, 0],
    #         centers.iloc[k, 1],
    #         s=new_str,
    #         horizontalalignment="center",
    #     )

    # center with label ^
    plt.savefig("pca_w_loadings_2.png", bbox_inches="tight")
    plt.close()

    # Put the result into a color plot (saved as png, not interactive)
    fig, ax = plt.subplots()
    ax.scatter(reduced_data[:, 0], reduced_data[:, 1], c=cols)
    # ax.scatter(centers.iloc[:, 0], centers.iloc[:, 1], c=k_cols, s=300)
    # center ^
    ax.set_title("K-means clustering on the roads (projected via PCA)")
    ax.set_xticks(())
    ax.set_yticks(())
    ax.plot()
    plt.savefig(
        "road_pca_2.png", bbox_inches="tight"
    )  # puts it in the working directory -- where this app is located and executed from
    plt.close(fig)

    return cols


def plot(shape_files):
    gpd_files = gpd.GeoSeries(shape_files)  # passed in shape_files
    fig, ax = plt.subplots()
    gpd_files.plot(ax=ax)

    plt.savefig(
        "orig_roads_2.png", bbox_inches="tight"
    )  # puts it in the working directory
    plt.close()


def setup(filepath):
    start_time = timeit.default_timer()

    # Set filepath -- automate download of shapefile, and see if it exists on the OS already
    # (Done via GUI)
    fp_test = "~/Downloads/tl_rd22_51001_roads/tl_rd22_51001_roads.shp"
    fp_test2 = (
        "~/Downloads/tl_rd22_06059_roads/tl_rd22_06059_roads.shp"  # LA roads. ~3 mins
    )

    data = gpd.read_file(filepath)
    try:
        os.remove("road_pca_2.png")  # removes any existing PCA visualization
    except:
        pass

    num_roads = len(data["geometry"])
    shape_files = data["geometry"]

    for j in range(num_roads):
        if str(data["FULLNAME"][j]) == "nan":
            # the primary cleaning step. Converts roads without names and gives them numeric labels
            data["FULLNAME"][j] = str(data["LINEARID"][j])

    tree = shapely.STRtree(data["geometry"])
    # this is crucial for performance. Creates distances

    list_of_dicts = []
    set_of_looked_at_roads = set()

    return (
        start_time,
        data,
        tree,
        list_of_dicts,
        set_of_looked_at_roads,
        num_roads,
        shape_files,
    )


def validate(data, set_of_looked_at_roads):
    for i in range(len(data)):
        if data["FULLNAME"][i] in set_of_looked_at_roads:
            pass
        else:
            ic("Our i is", i, "and our road_name is", data["FULLNAME"][i])


def main(filepath):
    (
        start_time,
        data,
        tree,
        list_of_dicts,
        set_of_looked_at_roads,
        num_roads,
        shape_files,
    ) = setup(filepath)

    # in the old formulation, this would give us the progress bar
    # for i in tqdm(range(size)):
    list_of_dicts, set_of_looked_at_roads = loop_through_roads(
        set_of_looked_at_roads, list_of_dicts, tree, data, num_roads
    )

    list_of_connected_dicts, shape_files_connected = graph_analysis(
        list_of_dicts, start_time, shape_files
    )

    show_results(list_of_connected_dicts)

    validate(data, set_of_looked_at_roads)


if __name__ == "__main__":
    main(filepath="tl_2022_06047_roads.zip")
