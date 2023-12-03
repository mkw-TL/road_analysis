import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from statsmodels.tsa.arima.model import ARIMA

from pmdarima import auto_arima
from pmdarima.arima.utils import ndiffs, nsdiffs
import warnings

warnings.filterwarnings("ignore")


def get_vmt_change(state):
    # Load in population data
    county = pd.read_csv("data/county_pop.csv", header=4)
    county.columns = ["County", "Estimates Base", "Pop_2020", "Pop_2021", "Pop_2022"]
    county = county.iloc[:-5, :]
    county["County"] = county["County"].str[1:]
    county[["County", "State"]] = county["County"].str.split(", ", expand=True)
    county = county[
        ["State", "County", "Estimates Base", "Pop_2020", "Pop_2021", "Pop_2022"]
    ]
    county.replace(",", "", regex=True, inplace=True)
    county[["Estimates Base", "Pop_2020", "Pop_2021", "Pop_2022"]] = county[
        ["Estimates Base", "Pop_2020", "Pop_2021", "Pop_2022"]
    ].astype(int)
    sc = county.groupby(["State", "County"])
    jj = 0
    county_pop = {}
    for x, y in sc:
        county_pop[x[0] + ", " + x[1]] = y["Estimates Base"].values[0]

    registrations = pd.read_csv(
        "data/Motor_Vehicle_Registrations__by_vehicle_type_and_state_20231109.csv"
    )
    registrations = registrations[["year", "state", "Auto"]]
    recentReg = registrations[registrations["year"] > 2009]
    state_averages = recentReg.groupby("state")["Auto"].mean()

    # Function to replace NaN with state average
    def replace_nan_with_state_average(row):
        if pd.isna(row["Auto"]):
            return state_averages[row["state"]]
        else:
            return row["Auto"]

    # Apply the function to replace NaN values
    recentReg["Auto"] = recentReg.apply(replace_nan_with_state_average, axis=1)
    recentReg

    vmts = pd.read_excel("data/10315_vmt_traveled_7-1-22.xlsx", header=2)
    vmts.drop(vmts.tail(14).index, inplace=True)
    vmts.drop(vmts.head(39).index, inplace=True)
    vmts = vmts[["Year*", "Million Miles"]]

    state_pops = pd.read_excel("data/nst-est2019-01.xlsx", header=3, nrows=56)
    state_pops = state_pops[5:56]

    recentReg["state"] = recentReg["state"].apply(
        lambda x: re.sub(r"\([^)]*\)", "", x).strip()
    )

    # Convert 'year' and 'Auto' columns to numeric if they are not already
    recentReg["year"] = pd.to_numeric(recentReg["year"], errors="coerce")
    recentReg["Auto"] = pd.to_numeric(recentReg["Auto"], errors="coerce")

    # Filter the DataFrame for the years 2013 to 2018
    filtered_data = recentReg[(recentReg["year"] >= 2013) & (recentReg["year"] <= 2018)]

    # Calculate the difference in registration amounts for each state
    filtered_data["diff"] = filtered_data.groupby("state")["Auto"].diff()
    filtered_data = filtered_data[filtered_data["diff"] >= 0]

    # Group by state and calculate the average difference
    avg_difference = filtered_data.groupby("state")["diff"].mean()

    vmt_perCap = pd.DataFrame()
    vmt_perState = pd.DataFrame()

    columns_to_use = state_pops.columns[3:13]
    column_sum = state_pops[columns_to_use].sum(axis=1)
    vmt_perCap[columns_to_use] = state_pops[columns_to_use].div(column_sum, axis=0)
    vmt_perState[columns_to_use] = vmt_perCap[columns_to_use].multiply(
        np.array(vmts["Million Miles"])
    )
    vmt_perState.index = state_pops.iloc[:, 0]
    vmt_perState.index = vmt_perState.index.str.lstrip(".")
    # Calculate the difference between each consecutive year
    yearly_diff = vmt_perState.diff(axis=1)

    # Calculate the average difference for each state
    average_diff = yearly_diff.mean(axis=1)

    # Resulting Series
    vmt_perStateDiff = pd.Series(average_diff)

    # Initialize an empty list to store VMT per capita for each county
    vmt_per_capita_list = []

    # Loop through each county in county_pop
    for county, population in county_pop.items():
        # Extract state from the county name
        state = county.split(", ")[0]

        # Get the VMT for the current state
        state_vmt = vmt_perStateDiff.loc[state]

        # Calculate VMT per Capita for the current county
        vmt_per_capita_county = state_vmt * (
            population
            / sum(
                county_pop[state_county]
                for state_county in county_pop
                if state in state_county
            )
        )

        # Append the results to the list
        vmt_per_capita_list.append((state, county, vmt_per_capita_county))

    # Create a DataFrame from the results
    vmt_per_capita_diff = pd.DataFrame(
        vmt_per_capita_list, columns=["State", "County", "VMT per Capita"]
    )

    # Initialize an empty list to store Registration per capita for each county
    reg_per_capita_list = []

    # Loop through each county in county_pop
    for county, population in county_pop.items():
        # Extract state from the county name
        state = county.split(", ")[0]

        # Get the registration amount for the current state or use the average if the state is missing
        if state in avg_difference.index:
            state_reg = avg_difference.loc[state]
        else:
            state_reg = avg_difference.mean()

        # Calculate Registration per Capita for the current county
        reg_per_capita_county = state_reg * (
            population
            / sum(
                county_pop[state_county]
                for state_county in county_pop
                if state in state_county
            )
        )

        # Append the results to the list
        reg_per_capita_list.append((state, county, reg_per_capita_county))

    # Create a DataFrame from the results
    reg_per_capita_diff = pd.DataFrame(
        reg_per_capita_list, columns=["State", "County", "Reg per Capita"]
    )

    # Initialize an empty list to store the results
    vmt_per_500_cars_list = []

    # Loop through each row in vmt_per_capita_diff
    for vmt_index, vmt_row in vmt_per_capita_diff.iterrows():
        # Find the corresponding row in reg_per_capita_diff based on State and County
        reg_row = reg_per_capita_diff[
            (reg_per_capita_diff["State"] == vmt_row["State"])
            & (reg_per_capita_diff["County"] == vmt_row["County"])
        ]

        # Calculate VMT per 500 cars added
        vmt_per_500_cars = (vmt_row["VMT per Capita"] / reg_row["Reg per Capita"]) * 500

        # Append the results to the list
        vmt_per_500_cars_list.append(
            {
                "State": vmt_row["State"],
                "County": vmt_row["County"],
                "VMT per 500 Cars Added": vmt_per_500_cars.values[0],
            }
        )

    # Create a DataFrame from the results
    vmt_per_500_cars_df = pd.DataFrame(vmt_per_500_cars_list)
    matching_row = vmt_per_500_cars_df[vmt_per_500_cars_df["State"] == state]

    # Return the resulting increase
    return matching_row.iloc[0]["VMT per 500 Cars Added"] / 12


def get_vmt_county(
    county, state, years=3, road_types=[100, 100, 100, 100, 100, 100], num_avs=500
):
    # Reading in and cleaning of vmt dataset 'vmt_state.csv'
    dat = pd.read_csv("data/vmt_state.csv")
    dat = dat.iloc[:-3].T
    dat.columns = dat.iloc[0, :]
    dat = dat.iloc[1:, :]
    for col in dat:
        dat[col] = dat[col].map(lambda x: int(x.replace(",", "")))
    dat.index = pd.to_datetime(dat.index, format="%d %m %Y")

    # Cleaning of population dataset 2010-2019 'pop_10_19.csv'¶
    pop_old = pd.read_csv("data/pop_10_19.csv", encoding="ISO-8859-1")
    pop_old = pop_old.iloc[:, 5:]
    pop_old = pop_old[pop_old["STNAME"] != pop_old["CTYNAME"]]
    pop_old.columns = [
        "State",
        "County",
        "2010",
        "2011",
        "2012",
        "2013",
        "2014",
        "2015",
        "2016",
        "2017",
        "2018",
        "2019",
    ]
    pop_old["County"] = pop_old["County"] + ", " + pop_old["State"]

    # Cleaning of population dataset 2020-2022 'pop_20_22.csv'¶

    pop_cur = pd.read_csv("data/pop_20_22.csv", header=4)
    pop_cur = pop_cur.iloc[:-5, :]
    pop_cur.columns = ["County", "2020", "2021", "2022"]
    pop_cur["County"] = pop_cur["County"].map(lambda x: x.replace(".", ""))
    for col in pop_cur.iloc[:, 1:]:
        pop_cur[col] = pop_cur[col].map(lambda x: int(x.replace(",", "")))

    # Merging population dataframes¶
    pop = pop_old.merge(pop_cur, on="County")
    pop.set_index("County", inplace=True, drop=True)

    vmt_df = dat

    pop_df = pop

    # Reading in and cleaning of AV dataset '2021_22_AV_CA.csv'¶
    # already slightly pre-cleaned earlier
    av_data = pd.read_csv("data/2021_22_AV_CA.csv")

    # only considering AVs with more than 5000 miles driven over the year

    av_data = av_data[av_data["ANNUAL TOTAL"] > 5000]

    # average vmt per av
    avg_vmt_per_car = (
        av_data["ANNUAL TOTAL"].sum() / len(av_data["ANNUAL TOTAL"]) / (1000000 * 12)
    )

    # Query the state
    vmt_df = vmt_df[state]

    # Query the county
    county_df = pop_df.loc[county + ", " + state].iloc[1:]

    # State population
    state_pop = pop_df.groupby("State").sum().loc[state]

    vmt_county = []

    # Finding vmt of each county per capita:
    # vmt(county) (approximately equal to) == vmt(state) / population(state) * population(county)
    j = 0
    for i in range(len(vmt_df)):
        vmt_county.append(vmt_df[i] / state_pop[j] * county_df[j])

        if j < len(state_pop) - 1:
            j += 1
        else:
            j = 0

    data = pd.Series(vmt_county, index=vmt_df.index)

    # ARIMA Model

    d = ndiffs(data)
    D = nsdiffs(data, m=12)  # Assuming monthly data with a yearly seasonality (m=12)

    stepwise_fit = auto_arima(
        vmt_county, d=d, D=D, seasonal=True, m=12, error_action="ignore"
    )

    print(stepwise_fit.summary())
    fit = stepwise_fit.order

    fit_seas = list(stepwise_fit.order)
    fit_seas.append(12)
    #     print(fit_seas)

    model = ARIMA(
        data, order=fit, seasonal_order=fit_seas
    )  # Example order, you might need to tune this

    model_fit = model.fit()

    forecast = model_fit.forecast(steps=years * 12)

    state_estimate = get_vmt_change(state)

    # Taking into account AVs
    pred = [x + state_estimate for x in forecast]

    coef = [
        -0.366482332,
        -0.285876052,
        0.290216209,
        0.206422313,
        0.292864002,
        -0.151311717,
    ]
    with_avs = []
    j = 0
    r = 1.05
    y = 1
    for i in forecast:
        with_avs.append((i + abs(coef[j] * road_types[j])) * r)
        if y % 12 == 0:
            r += 0.05
        if j > 4:
            j = 0

    # Visualizations
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data, label="Original Data")
    plt.plot(
        pd.date_range(start="2023-01-01", periods=years * 12, freq="m"),
        forecast,
        label="Forecast",
        color="red",
    )
    plt.plot(
        pd.date_range(start="2023-01-01", periods=years * 12, freq="m"),
        with_avs,
        label="Forecast with County Roads for " + str(num_avs) + " AVs",
        color="purple",
    )
    plt.plot(
        pd.date_range(start="2023-01-01", periods=years * 12, freq="m"),
        pred,
        label="Forecast with State Traffic for " + str(num_avs) + " AVs",
        color="orange",
    )
    plt.xlabel("Year")
    plt.ylabel("VMT in Millions")
    plt.title(
        "ARIMA Model: Monthly VMT of " + county + ", " + state + " From 2010 to 2026"
    )
    plt.legend()
    #     plt.show()
    plt.savefig("AV_pred.png")


def added_road_cost(miles):
    road_degradation = 1.42 * (miles / 1000)
    crash_cost = 52.99 * (miles / 1000)
    return road_degradation + crash_cost


def rubber_polution(miles):
    rubber = 0.00032 * miles
    return rubber
