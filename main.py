"""
Supposed to be in a folder together with umap_metrics.py
Reads features csvs from folder in upper level ../features_csv
"""
import os
from pathlib import Path
import time
from datetime import datetime
import sys
import copy
from functools import partial
import itertools
import random
from functools import partial

import numpy as np
import pandas as pd
from sklearn import metrics
import umap
import hdbscan
from bokeh.plotting import figure
from bokeh.io import curdoc
from bokeh.palettes import Category10, Category20, Colorblind
from bokeh.models import CheckboxGroup, Button, Slider, TextInput, Div, OpenURL
from bokeh.models import Legend, Select, RadioButtonGroup, Paragraph, TapTool
from bokeh.models.widgets import Tabs, Panel
from bokeh.layouts import row, column, layout
from bokeh.resources import CDN
from bokeh.embed import file_html

# from clusim import sim
# from clusim.clustering import Clustering



from umap_metrics import tanimoto_gower, tanimoto_gower_b
from features_description import features_descr
# from clustering_inference_test import clusterInfoTest
# from clusters_summary import cluster_distro_summary
from utils import pretty_dict, time_now, update_widget_text
from utils import unsupervised_clu_met
from plotting_utils import generate_plot


def select_cbk(attr, old, new, source):
    pid = source.data["patient_name"][new]
    df_label = "F7"  # borda 20
    pid_mask = (features[df_label]["patient_name"].values == pid)
    try:
        pat_text = features[df_label].iloc[pid_mask, :].transpose().to_html()
    except AttributeError:
        pat_text = features[df_label].iloc[pid_mask, :].to_frame().to_html()

    select_info_div.text = pat_text

def prova(attr, old, new):
    print("ciao ", new)

def change_color(attr, old, new):
    print("Updating colors ...")
    u = savings["umap_2d"]
    cluster_lbls = savings["cluster_lbls"]
    main_scatter = generate_plot(
        u, cluster_lbls,
        target_input.value, marker_input.value, highlighted_input.value,
        color_mode_button.active, features)
    explorer_layout.children[1].children[1] = main_scatter



def generate_umap_embeddings(ds_in_use, target_id):

    target_lbl = None  # array of labels for given target
    X = None
    X_col = 0

    # assume all features dataset in the same order!
    # so we don't have to merge
    for i, ds_id in enumerate(ds_in_use):
        ds = features[ds_id].copy()
        ds = ds.loc[:, ~ds.columns.str.startswith(name_col)]
        if exclude_trgt_button.active == 0:
            # exlude target considering widget spec
            if target_id in ds.columns:
                # eventually remove everything similar?
                ds.drop(columns=[target_id], inplace=True)
        X = ds if X is None else pd.concat([X, ds], axis=1)

    # umap embedding for visualization
    np.random.seed(42)
    umapper_2d = umap.UMAP(
        n_neighbors=n_neighbour,
        metric="l2",
        min_dist=0.2,
        #metric_kwds=metric_kwds,
        random_state=42,
        n_components=2
    )

    # umap embedding for clustering
    umapper_clu = umap.UMAP(
        n_neighbors=20,
        metric="l2",
        min_dist=0.0,
        #metric_kwds=metric_kwds,
        random_state=42,
        n_components=5
    )

    print(str(datetime.now()) + " " + "Start dim red...")
    umap_2d = umapper_2d.fit_transform(X.values)
    umap_clu = umapper_clu.fit_transform(X.values)
    print(str(datetime.now()) + " " + "Completed dim red...")

    return X, umap_2d, umap_clu


def generate_cluster_labels(data, **kwargs):
    """
    Parameters
    ----------
    data: numpy.ndarray

    Returns
    -------
    cluster.labels_ array_like int [n_samples]
    metrics_dict dictionary (float, float) [n_measures]
    """

    print("Clustering...")
    clu_params = dict(
        min_cluster_size=5, min_samples=5, cluster_selection_epsilon=0.0,
        metric="euclidean", alpha=1.0, cluster_selection_method='eom',
        allow_single_cluster=True
    )

    # main clustering
    clusterer = hdbscan.HDBSCAN(**clu_params)
    clusterer.fit(data)

    return clusterer.labels_


# scatter callback
def update_scatterplot():

    # log start
    print(time_now() + ": " + "Starting scatterplot update...")

    # list of datasets in use
    ds_in_use = [feat_list_ordered[x] for x in df_choice.active]
    # target id and dataset from which is taken

    target_id, target_df_id = target_input.value.split(" ")
    target_lbls = features[target_df_id][target_id]

    # create various embeddings
    data, umap_2d, umap_clu = generate_umap_embeddings(
        ds_in_use, target_id
    )

    # perform clustering
    cluster_lbls = generate_cluster_labels(umap_clu)

    # compute internal metrics using target it labels or cluster labels
    # clustering_met = unsupervised_clu_met(umap_clu, cluster_lbls)
    # target_met = unsupervised_clu_met(umap_clu, target_lbls)

    # update_widget_text(
    #     metrics_div,   # object with text attribute
    #     pretty_dict(clustering_met, "<br>", "Clustering score"),  # string
    #     "w"
    # )
    # update_widget_text(
    #     metrics_b_div,
    #     pretty_dict(target_met, "<br>", "Target labels score"),
    #     "w"
    # )

    # Update plot title
    ds_used = "_".join(ds_in_use)  # use as title and save name
    mytitle = f"{ds_used}_target_{target_id}.html"

    # generate main plot
    main_scatter, sources = generate_plot(
        umap_2d, cluster_lbls,
        target_input.value, marker_input.value, highlighted_input.value,
        color_mode_button.active, features)
    
    for i, elem in enumerate(sources):
        elem.selected.on_change("indices", partial(select_cbk, source=elem))
   
    explorer_layout.children[1].children[1] = main_scatter
    update_widget_text(
        scatter_title_div,
        mytitle,
        "w"
    )
    update_widget_text(
        log_div,
        time_now() + r"<br>" + mytitle + " completed",
        "a"
    )

    # store data to be reused in other operations
    savings["umap_2d"] = umap_2d
    savings["cluster_lbls"] = cluster_lbls
    savings["w_data"] = data

    print(time_now() + "Scatter plot update completed.")


def save_html():
    filename = savings["title"]
    with open(filename, "w") as f:
        f.write(savings["plot_html"])
    print("saved : ", filename)
    os.system(f"open {filename}")
    # log in the ui
    update_widget_text(log_div, time_now() + " " + filename + " saved")


############################################
# set the path where we are working
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
FILE_ROOT = Path(__file__).parent.resolve()
FEATURES_CSV_DIR = FILE_ROOT / "features"

# collect possibly multiple feature files
features = {
    x.split(".")[0]: pd.read_csv(FEATURES_CSV_DIR / x)
    for x in os.listdir(FEATURES_CSV_DIR)
    if ".csv" in x
}

# obtain the shape of the feature files
shapes = {}
for k, v in features.items():
    print(f"features {k} : shape: {v.shape}")
    shapes[k] = v.shape

# search for possible file for dashboard description
with open(FILE_ROOT / "description.html", "r") as f:
    description_html_text = f.read()


name_col = "id"
study_col = "study_uid"
n_neighbour = 10


# select variables tipe for tanimoto-gower
feature_types = {
    'bools': [],
    'cat': []
}

# dictionary to store temporary info/data
savings = {"title": None, "plot_html": None,
           "w_data": None, "umap_2d": None,
           "cluster_lbls":None, "target_lbls": None}

# set empty scatter plot and tools
TOOLS = """hover,crosshair,pan,wheel_zoom,zoom_in,zoom_out,
         box_zoom,undo,redo,reset,tap,save,box_select,
         poly_select,lasso_select,"""

TOOLTIPS = [
    ("index", "$index"),
    ("(x,y)", "($x, $y)"),
    # ("study_UID", "@study_uid"),
    #("patient_name", "@patient_name"),
]

PLOT_HEIGHT = 600
PLOT_WIDTH = PLOT_HEIGHT + 200
# figure
p = figure(
    plot_width=PLOT_WIDTH,
    plot_height=PLOT_HEIGHT,
    toolbar_location="above",
    tools=TOOLS,
    tooltips=TOOLTIPS,
)
p.toolbar.logo = None

# widgets
button_submit = Button(label="Submit!", button_type="primary")
button_save_open = Button(label="Save html (local)", button_type="success")
color_mode_button = RadioButtonGroup(labels=["Target", "Clustering"], active=1)
exclude_trgt_button = RadioButtonGroup(labels=["Exclude", "Include"], active=0)
slider_weights = Slider(
    start=0, end=1, value=0.5, step=0.05, title="Categorical weight"
)





# distinguish displayed features option and real features options

feat_list_ordered = sorted(list(features.keys()))
feat_display = [" - ".join([x, features_descr[x]]) for x in feat_list_ordered]  # features groups to display with descr



df_choice = CheckboxGroup(labels=feat_display, active=[0])
# target selection after dataset selection
#  target and color selection
options_target = [
    f"{x} {feat_list_ordered[0]}"
    for x
    in features["sample_data"].columns[21:]
    if x not in [name_col, study_col]
]

def_target = options_target[0]
target_input = Select(title="Target:", value=def_target, options=options_target)


#   annotation display selection
options = [
    (f"{x} {feat_list_ordered[0]}", x)
    for x 
    in features["sample_data"].columns[21:]
    if x not in [name_col, study_col, "old_cluster_id",
                 "train_set_mask", "test_set_mask"]]
options.insert(0, ("- -", "None"))

# null initial options in UI

def_one = "- -"
def_two = "- -"
marker_input = Select(title="Highlight 1:", value=def_one, options=options)
highlighted_input = Select(title="Highlight 2:", value=def_two, options=options)


log_div = Paragraph(text="Logs:\n", width=600, height=200)
description_div = Div(text=description_html_text, width=600, height=200)
scatter_title_div = Div(text="Please select one or more datasets and press submit", width=600, height=40)
metrics_div = Div(text="", width=250, height=100)
metrics_b_div =  Div(text="", width=250, height=100)
metrics_sample_div =  Div(text="", width=250, height=100)
select_info_div = Div(text="", width=400, height=600)

target_inclusion_div = Div(
    width=500, height=20,
    text="Include / exclude target variable"
    )
color_mode_div = Div(
    width=500, height=20,
    text="Color with respect to Target or Clustering"
    )

# Cluster summary tab
run_button_sum = Button(label="Create Cluster Summary", button_type="primary")

# Inference test TAB
run_button = Button(label="Run Inference Test", button_type="primary")
inference_test_div = Div(text="", width=700, height=600)

# callbacks
# tab explorer 1
button_submit.on_click(update_scatterplot)
button_save_open.on_click(save_html)
color_mode_button.on_change('active', change_color)

# tab inference test 2
# run_button_sum.on_click(cluster_summary)
# tab inference test 3
# run_button.on_click(inference_test)



# set up layout
widget_width = 300
widget_col = column(
    target_input,
    target_inclusion_div,
    exclude_trgt_button,
    color_mode_div,
    color_mode_button,
    marker_input,
    highlighted_input,
    df_choice,
    slider_weights,
    row(button_submit, button_save_open, width=widget_width),
    log_div,
    width = widget_width
)

# set tabs
explorer_layout = row(
    widget_col,
    column(scatter_title_div,
           row([]),
           row(
            metrics_div, metrics_b_div, metrics_sample_div),
            select_info_div,
            width = PLOT_WIDTH)
)

description_layout = description_div

clu_summ_layout = layout([run_button_sum, row()])
inf_test_layout = column(run_button, inference_test_div)


explorer = Panel(child=explorer_layout, title="Scatter explorer")
# cluster_summary = Panel(child=clu_summ_layout, title="Clustering summary")
description = Panel(child=description_layout, title="About")
# inference_test = Panel(child=inf_test_layout, title="Inference Test")
#tabs = Tabs(tabs=[explorer, cluster_summary, inference_test, description])
tabs = Tabs(tabs=[explorer, description])
# set document
curdoc().add_root(tabs)
