import pygmt
import shapefile as shp
import matplotlib.pyplot as plt
import pandas as pd

with pygmt.clib.Session() as session:
    session.call_module('gmtset', 'FONT 12p,Helvetica,black')

fig = pygmt.Figure()
# Using the origin projection pole
fig.coast(
    projection="Oc34/34/-90/50/20c",
    frame="afg",
    # Set bottom left and top right coordinates of the figure with "+r"
    region="36/35.8/38.9/38.5+r",
    land="lightgrey",
    shorelines="1/thin",
    water="lightblue",
)
for fn in ["../ThirdParty/Turkey_Emergency_EQ_Data/simple_fault_2023-02-17/simple_fault_2023-2-17.shp"]:
    sf = shp.Reader(fn)
    for sr in sf.shapeRecords():
        listx = []
        listy = []
        for xNew, yNew in sr.shape.points:
            listx.append(xNew)
            listy.append(yNew)
        fig.plot(x = listx, y = listy)#, width=2.0)

stations2plot = [
    "4404",
    "0213",
    "4611",
    "2712",
    "4615",
    "4625",
    "4616",
    "2718",
    "3138",
    "3139",
    "3141",
    "3136",
]


fn = "../ThirdParty/stations.csv"
cols = ["Code", "Longitude", "Latitude"]
df = pd.read_csv(fn)

df["waveform_to_plot"] = [row["Code"] in stations2plot for index, row in df.iterrows()]
df_no = df.loc[df["waveform_to_plot"] == False]
for index, row in df_no.iterrows():
    is_inside = (35.5 <= row["Longitude"] <= 39.5) and (35.8 <= row["Latitude"] <= 38.2)
    c = "b" if row["Code"] in stations2plot else "k"
    if row["DeviceCode"] == "N" and is_inside:
        fig.plot(
            x=row["Longitude"],
            y=row["Latitude"],
            style="c",
            fill='black',
            size=[0.125],
        )

df_yes = df.loc[df["waveform_to_plot"]]
for index, row in df_yes.iterrows():
    is_inside = (35.5 <= row["Longitude"] <= 39.5) and (35.8 <= row["Latitude"] <= 38.2)
    if row["DeviceCode"] == "N" and is_inside:
        fig.plot(
            x=row["Longitude"],
            y=row["Latitude"],
            style="c",
            fill='blue',
            size=[0.25],
        )
        fig.text(text=row["Code"], x=row["Longitude"] + 0.07, y=row["Latitude"], font="12p,Helvetica,black")

fig.savefig('oblique_station.png')
