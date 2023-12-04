# Author: Joe Harrison
# Date: Nov 11, 2023
# License: MIT
# NOTE: DearPyGui is not officially supported on Windows 11

import dearpygui.dearpygui as dpg
import webbrowser

from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QApplication

import capstone_roads  # our python file. Should be in the same working directory as this python script
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
from icecream import ic
import time

from selenium import webdriver  # update requirements.txt
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from seleniumwire import webdriver  # needed to see GET requests

plt.switch_backend("Agg")  # so no issues with GUI backend
import seaborn as sns
import project

sns.set_style("darkgrid", {"axes.facecolor": ".9"})
sns.set(rc={"figure.figsize": (8, 6)})


def main():
    dpg.create_context()

    def visualize(list_of_connected_dicts, shape_files, cols_list, county, state):
        # project.get_vmt_county(county, state)

        time.sleep(0.02)

        dpg.hide_item(slow_warning)

        gpd_files = gpd.GeoSeries(shape_files, crs=4326)  # passed in shape_files
        fig, ax = plt.subplots()
        gpd_files.plot(ax=ax, color=cols_list)

        # df = pd.DataFrame(list_of_connected_dicts)
        # new_df = df.loc[df["road_name"] == "Kendall Dr"]
        # new_df = df.loc[df["road_name"] == "Sunset Dr"]
        # print(new_df)
        # ic(new_df.shape)

        plt.savefig(
            "road_clustering_2.png", bbox_inches="tight"
        )  # puts it in the working directory
        plt.close()

        # needs time to savefig first
        time.sleep(0.02)

        width1, height1, channels, data2 = dpg.load_image("road_clustering_2.png")
        width2, height2, channels, data_loadings = dpg.load_image(
            "pca_w_loadings_2.png"
        )
        # width3, height3, channels, AV_pred_dat = dpg.load_image("AV_pred.png")
        # width3, height3, channels, data = dpg.load_image("road_pca_2.png")

        with dpg.texture_registry(show=False):
            dpg.add_static_texture(
                width=width1,
                height=height1,
                default_value=data2,
                tag="road_clustering",
            )
        with dpg.texture_registry(show=False):
            dpg.add_static_texture(
                width=width2,
                height=height2,
                default_value=data_loadings,
                tag="pca_w_loadings",
            )

        # with dpg.texture_registry(show=False):
        #     dpg.add_static_texture(
        #         width=width3,
        #         height=height3,
        #         default_value=AV_pred_dat,
        #         tag="AV_pred",
        #     )

        # with dpg.texture_registry(show=False):
        #     dpg.add_static_texture(
        #         width=width3, height=height3, default_value=data, tag="road_pca"
        #     )

        with dpg.window(label="Clustering Results"):
            dpg.add_image("road_clustering")
            dpg.add_image("pca_w_loadings")
            # dpg.add_image("AV_pred")
            # dpg.add_image("road_pca")

    def run_script():
        filepath = dpg.get_value(fp)
        county = dpg.get_item_user_data(button1)[0]
        state = dpg.get_item_user_data(button1)[1]
        print("our filepath is", filepath)
        dpg.hide_item(run_button)
        dpg.show_item(slow_warning)

        # dpg.show_item(progress_bar)

        (
            start_time,
            data,
            tree,
            list_of_dicts,
            set_of_looked_at_roads,
            num_roads,
            shape_files,
        ) = capstone_roads.setup(filepath)

        # in the old formulation, this would give us the progress bar
        # for i in tqdm(range(size)):

        i = 0
        while True:
            try:
                i = i + 1
                val = next(
                    capstone_roads.create_data_for_road(
                        i, set_of_looked_at_roads, list_of_dicts, tree, data
                    )
                )
                dpg.set_value(
                    "viewer",
                    f"{val + 1} out of {num_roads} ({round(val*100/num_roads, 1)}%)",
                )
            except:
                dpg.hide_item(progress_bar)
                break

        list_of_connected_dicts, shape_files = capstone_roads.graph_analysis(
            list_of_dicts, start_time, shape_files
        )

        cols_list = capstone_roads.show_results(list_of_connected_dicts)
        dpg.hide_item(text1)
        dpg.hide_item(button1)
        dpg.hide_item(button2)
        dpg.hide_item(zip)
        dpg.hide_item(text0)
        dpg.show_item(viz)
        visualize(list_of_connected_dicts, shape_files, cols_list, county, state)

    def get_user_link():
        app = QApplication([])  # Create a QApplication instance if not already created
        directory = QFileDialog.getOpenFileName(
            None, "Select Directory", "", "Zip Files (*.zip);;SHP Files (*.shp)"
        )
        print(directory)
        dir = str(directory[0])
        dpg.set_value(fp, dir)
        dpg.hide_item(zip)
        dpg.show_item(run_button)
        app.quit()

    def download(sender, data, user_data):
        dpg.show_item(text1_5)
        driver = webdriver.Firefox()
        driver.get(
            "https://www.census.gov/cgi-bin/geo/shapefiles/index.php?year=2022&layergroup=Roads"
        )
        request = driver.wait_for_request(".zip", 60)
        time.sleep(.5)
        request = str(request)
        zip_label = request.split("2022_")[1]
        zip_label = zip_label.split("_roads")[0]
        print("zip label is", zip_label)
        driver.implicitly_wait(5)
        element = driver.find_elements(By.TAG_NAME, "option")
        associated_county = "na"
        for e in element:
            if str(e.get_attribute("value")) == zip_label:
                associated_county = e.text
            if str(e.get_attribute("value")) == zip_label[:2]:
                associated_state = e.text

        driver.close()

        time.sleep(1)

        dpg.hide_item(text1_5)
        dpg.hide_item(text1)
        dpg.hide_item(button1)
        dpg.hide_item(text0)
        dpg.hide_item(button2)
        dpg.hide_item(spacing)

        dpg.show_item(downloaded)
        dpg.show_item(zip)
        dpg.show_item(fp)

        ic(associated_county)
        ic(associated_state)

        dpg.set_item_user_data(sender, [associated_county, associated_state])

    def city_finder():
        webbrowser.open_new(
            "https://www.statsamerica.org/CityCountyFinder/Default.aspx"
        )

    with dpg.theme() as global_theme:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_style(
                dpg.mvStyleVar_FrameRounding, 6, category=dpg.mvThemeCat_Core
            )
            dpg.add_theme_style(
                dpg.mvStyleVar_FrameBorderSize, 2, category=dpg.mvThemeCat_Core
            )
            dpg.add_theme_style(
                dpg.mvStyleVar_ScrollbarSize, 4, category=dpg.mvThemeCat_Core
            )

    with dpg.window(label="Chose Road Network to Analyze", width=600, height=350):
        text0 = dpg.add_text(
            "Find County associated with City (Does not work with Cities)"
        )
        button2 = dpg.add_button(
            label="Find County",
            callback=city_finder,
        )
        spacing = dpg.add_text("")
        text1 = dpg.add_text(
            ">> Select the County to download your roads from. Choose from 'All Roads'"
        )
        button1 = dpg.add_button(label="Download Roads", callback=download)
        text1_5 = dpg.add_text("Loading")
        dpg.hide_item(text1_5)
        downloaded = dpg.add_text("Your file has been downloaded!")
        dpg.hide_item(downloaded)
        dpg.add_text("")
        zip = dpg.add_button(label="Locate File", callback=get_user_link)
        dpg.hide_item(zip)
        fp = dpg.add_text(label="file_path", default_value="no path defined yet")
        dpg.hide_item(fp)

        run_button = dpg.add_button(
            label="Run", callback=run_script, user_data=dpg.get_value(fp)
        )
        dpg.hide_item(run_button)  # only show once the user puts in the zip file

        progress_bar = dpg.add_progress_bar(label="progress")
        dpg.hide_item(progress_bar)

        slow_warning = dpg.add_loading_indicator()
        dpg.hide_item(slow_warning)

        viewer = dpg.add_text(label="Text", tag="viewer")
        viz = dpg.add_text("Visualizing...")
        dpg.hide_item(viz)
        # dpg.hide_item(viewer)

    dpg.bind_theme(global_theme)

    dpg.create_viewport(title="Analyze Roads", width=800, height=600)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()


if __name__ == "__main__":
    main()
