# Author: Joe Harrison
# Date: Oct 27, 2023
# License: MIT
# NOTE: DearPyGui is not officially supported on Windows 11

import geopandas as gpd
import shapely
import pandas as pd
import timeit

# pip install icecream
from icecream import ic

# pip install tqdm
from tqdm import tqdm

# pip install networkx
import networkx as nx
import time
import copy

# pip install Amazon-DenseClus

import matplotlib.pyplot as plt

plt.switch_backend("Agg")  # so no issues with GUI backend
# as per this https://stackoverflow.com/questions/52839758/matplotlib-and-runtimeerror-main-thread-is-not-in-main-loop
import seaborn as sns

sns.set_style("darkgrid", {"axes.facecolor": ".9"})
sns.set(rc={"figure.figsize": (10, 8)})

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import matplotlib.colors as mcolors
import os


def get_neighbors_and_update(this_dict, this_name, this_line, tree, data):
    nearest, dist = tree.query_nearest(this_line, return_distance=True)
    if dist[0] == 0:
        for _, id in enumerate(nearest):
            # nearest from .query_nearest gives the ids, not the names
            if data["FULLNAME"][id] == this_name:
                this_dict["self_connections"] += 1
            # append all connected roads that are not this road
            else:
                this_dict["connections_names"].append(data["FULLNAME"][id])
                this_dict["number_of_connector_roads"] += 1
    return this_dict


def create_data_for_road(i, set_of_looked_at_roads, list_of_dicts, tree, data):
    this_dict = {
        "road_name": "",
        "connections_names": [],
        "is_named": True,
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
        this_dict["is_named"] = False

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
        this_dict["number_of_parts"] += 1
        this_dict["length"] += shapely.length(this_line)
        this_dict = get_neighbors_and_update(
            this_dict, this_name, this_line, tree, data
        )
        list_of_dicts[orig_ind] = this_dict  # the modifications go back in
    yield i


def loop_through_roads(set_of_looked_at_roads, list_of_dicts, tree, data, num_roads):
    i = 0
    while True:
        try:
            i = i + 1
            print(
                next(
                    create_data_for_road(
                        i, set_of_looked_at_roads, list_of_dicts, tree, data
                    )
                ),
                end=" ",
            )
            print("/", str(num_roads))
        except:
            return list_of_dicts


def graph_analysis(list_of_dicts, start_time, shape_files):
    # Create a graph
    G = nx.Graph()

    # Add nodes (roads) to the graph
    for road_data in list_of_dicts:
        # some roads don't have a road_name field?? Not sure how that happens..
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
        list_of_unconnected_dicts = []
        orig_idxs = []
        G3 = nx.Graph()

        G2 = copy.deepcopy(G)
        # MAN this sucks. It looks like copy was depreciated as an argument to max() or connected_components
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
                else:
                    list_of_unconnected_dicts.append(this_road)

        for i, conn_road in enumerate(list_of_connected_dicts):
            # loop through again because edges have to come after nodes
            if conn_road["road_name"] in G2:
                road_name = conn_road["road_name"]
                connections = conn_road["connections_names"]
                for connection in connections:
                    G3.add_edge(road_name, connection)
        ic("number of nodes is", str(G3.number_of_nodes()))
        ic("number of edges is", str(G3.number_of_edges()))
        ic("degree centrality is", str(nx.degree_centrality(G3))[:depth])
        ic("pagerank is", str(nx.pagerank(G3))[:depth])
        centr = nx.degree_centrality(G3)
        centr_list = list(centr.values())
        pge_rnk = nx.pagerank(G3)
        pge_rnk_list = list(pge_rnk.values())
        # ic("barycenter is", str(nx.barycenter(G3))[:depth])
        # too slow
        # ic("information centrality is", str(nx.information_centrality(G3))[:depth])
        # also too slow
        print(
            "the difference between roads before and after cleaning is",
            str(G.number_of_nodes()),
            "to",
            str(G3.number_of_nodes()),
        )

    # need to now collect the information back out of G into something in order to classify
    # only currently works for G3.

    connected_sfs = []

    for i in range(len(list_of_connected_dicts)):
        list_of_connected_dicts[i]["degree_centrality"] = centr_list[i]
        list_of_connected_dicts[i]["page_rank"] = pge_rnk_list[i]
        connected_sfs.append(shape_files[orig_idxs[i]])

    return list_of_connected_dicts, connected_sfs

    # classify(list_of_connected_dicts, k=6)

    # from denseclus import DenseClus

    # clf = DenseClus(
    #     cluster_selection_method="leaf",
    #     umap_combine_method="intersection_union_mapper",
    # )


def show_results(list_of_connected_dicts):
    ic("this might take some time")
    df = pd.DataFrame(list_of_connected_dicts)
    # takes a while to put back into queriable format
    df = df.drop(columns=["road_name", "connections_names"])
    df["is_named"] = df["is_named"].astype(int)  # kmeans requires all to be numeric
    pd.set_option("display.max_columns", None)

    print(df.head())

    pd.reset_option("all")

    kmeans = KMeans(n_clusters=6)
    kmeans.fit(df)
    # kmeans first and then pca later

    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(df)
    print(pca.explained_variance_ratio_)

    tabl_cols = list(mcolors.TABLEAU_COLORS.values())
    cols = [tabl_cols[int(val)] for val in kmeans.labels_]

    # Put the result into a color plot (saved as png, not interactive)
    fig, ax = plt.subplots()
    ax.scatter(reduced_data[:, 0], reduced_data[:, 1], c=cols)
    ax.set_title("K-means clustering on the roads (projected via PCA)")
    ax.set_xticks(())
    ax.set_yticks(())
    ax.plot()
    plt.savefig(
        "road_pca.png", bbox_inches="tight"
    )  # puts it in the working directory -- where this app is located and executed from
    plt.close(fig)

    return cols


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
        os.remove("road_pca.png")  # removes any existing PCA visualization
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
    list_of_dicts = loop_through_roads(
        set_of_looked_at_roads, list_of_dicts, tree, data, num_roads
    )

    list_of_connected_dicts, _ = graph_analysis(list_of_dicts, start_time, shape_files)

    show_results(list_of_connected_dicts)

    #######

    # way too slow -- was going to deal with mixed data
    # clf.fit(df)

    # for i in range(clf.n_components):
    #     sns.kdeplot(clf.mapper_.embedding_[:, i], shade=True)

    # _ = sns.jointplot(
    #     x=clf.mapper_.embedding_[:, 0], y=clf.mapper_.embedding_[:, -1], kind="kde"
    # )

    # labels = clf.score()

    # print(labels, "\n")
    # print(pd.DataFrame(labels).value_counts(normalize=True))

    # _ = sns.jointplot(
    #     x=clf.mapper_.embedding_[:, 0],
    #     y=clf.mapper_.embedding_[:, 1],
    #     hue=labels,
    #     kind="kde",
    # )

    # _ = clf.hdbscan_.condensed_tree_.plot(
    #     select_clusters=True,
    #     selection_palette=sns.color_palette("deep", np.unique(labels).shape[0]),
    # )

    # df["segment"] = clf.score()

    # numerics = (
    #     df.select_dtypes(include=[int, float]).drop(["segment"], 1).columns.tolist()
    # )

    # df[numerics + ["segment"]].groupby(["segment"]).median()


if __name__ == "__main__":
    main(filepath="~/Downloads/tl_rd22_51001_roads.zip")  # Lexington, VA by default