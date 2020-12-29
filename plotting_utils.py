import numpy as np
import pandas as pd
from bokeh.plotting import figure
from bokeh.io import curdoc
from bokeh.palettes import Category10, Category20, Colorblind
from bokeh.models import CheckboxGroup, Button, Slider, TextInput, Div
from bokeh.models import Legend, Select, RadioButtonGroup, TapTool, ColumnDataSource
from bokeh.models.widgets import Tabs, Panel
from bokeh.layouts import row, column, layout
from bokeh.resources import CDN
from bokeh.embed import file_html


def generate_plot(
    u,
    cluster_labels,
    target_input,
    marker_input,
    highlighted_input,
    color_mode,
    features
    ):

    PLOT_HEIGHT = 600
    PLOT_WIDTH = PLOT_HEIGHT + 200
    TOOLS = "hover,crosshair,pan,wheel_zoom,zoom_in,zoom_out,\
         box_zoom,undo,redo,reset,tap,save,box_select,\
         poly_select,lasso_select,"
    TOOLTIPS = [
        ("index", "$index"),
        ("(x,y)", "($x, $y)"),
        # ("study_UID", "@study_uid"),
        ("patient_name", "@patient_name"),
    ]
    target_id, target_df_id = target_input.split(" ")
    target_mark_id, marker_df_id = marker_input.split(" ")
    target_high_id, highlight_df_id = highlighted_input.split(" ")

    # target_lbl = features[target_df_id][target_id]
    # fixed on online clustering labels
    color_target = {0: features[target_df_id][target_id],
                    1: cluster_labels}
    # target_lbl = cluster_labels
    target_lbl = color_target[color_mode]

    pu = figure(
        plot_width=PLOT_WIDTH,
        plot_height=PLOT_HEIGHT,
        toolbar_location="above",
        tools=TOOLS,
        tooltips=TOOLTIPS
    )
    pu.toolbar.logo = None

    # useful variables for colors etc
    unique_lbl = np.unique(target_lbl)

    # map colors
    # depending on how much levels choose the palette
    if len(unique_lbl) <= 8:
        col_palette = Colorblind[max(3, len(unique_lbl))]
    elif len(unique_lbl) <= 10:
        col_palette = Category10[len(unique_lbl)]
    else:
        col_palette = Category20[len(unique_lbl)]

    col_dict = {k: v for k, v in zip(unique_lbl, col_palette)}

    # if present -1 set to black ( non assigned points hdbscan)
    try:
        col_dict[-1] = "black"
    except KeyError:
        pass

    # collect legend items
    legend_items = []
    sources = []

    # plot glyph for each level of the target for interactive legend
    for lbl in unique_lbl:
        # mask for given target level
        mask = target_lbl == lbl
        data_lbl = u[mask, :]
        # build plot source
        plot_source = features[target_df_id].loc[mask, ["id"]]

        plot_source["x"] = pd.Series(data_lbl[:, 0], index=plot_source.index)
        plot_source["y"] = pd.Series(data_lbl[:, 1], index=plot_source.index)
        plot_source["lbl"] = pd.Series(
            [str(lbl) for _ in range(sum(mask))], index=plot_source.index
        )
        plot_source["color"] = pd.Series(
            [col_dict[lbl] for _ in range(sum(mask))], index=plot_source.index
        )
        plot_source["fill_alpha"] = pd.Series(
            [(0 if lbl == -1 else 0.8) for _ in range(sum(mask))], index=plot_source.index
        )
        plot_source["markers"] = pd.Series(
            [("cross" if lbl == -1 else "circle") for _ in range(sum(mask))], index=plot_source.index
        )
        plot_source["size"] = pd.Series(
            [(6 if lbl == -1 else 5) for _ in range(sum(mask))], index=plot_source.index
        )
        plot_source = ColumnDataSource(plot_source)

        tmp = pu.scatter(
            "x", "y", color="color", fill_alpha="fill_alpha", marker='markers',
            size='size',
            source=plot_source)

        sources.append(plot_source)
        legend_items.append(
            (str(lbl), [tmp])
        )


    # other annotations as plot elements
    if target_high_id != "-":
        #to be highlighted first 
        target_high_lbl = features[target_high_df][target_high_id]
        high_mask = (features["F0"][target_high_id] == 1)
        data_high = u[high_mask, :]
        tmp = pu.scatter(
            x=data_high[:, 0], y=data_high[:, 1],
            line_width=1,  fill_alpha=0, line_color="red", size=20,
            line_alpha=0.5, marker="circle")
        legend_items.append(
                (str(target_high_id), [tmp])
            )
    if target_mark_id != "-":
        # to be highlighted second 

        target_mark_lbl = features[target_mark_df][target_mark_id]

        mark_mask = (features["F0"][target_mark_id] == 1)
        data_mark = u[mark_mask, :]
        tmp = pu.scatter(
            x=data_mark[:, 0], y=data_mark[:, 1],
            line_width=1,  fill_alpha=0, line_color="blue", size=20,
            line_alpha=0.5, marker="triangle")
        legend_items.append(
                (str(target_mark_id), [tmp])
            )

    # custom legend
    legend = Legend(items=legend_items, location="center")
    legend.click_policy = "hide"
    legend.title = target_id if color_mode==0 else "Clustering"
    pu.add_layout(legend, 'right')

    # update scatter plot as children of layout element
    # explorer_layout.children[0].children[1] = pu
    # store info for saving of the plot
    # savings["plot_html"] = file_html(pu, CDN, "my plot")
    # savings["title"] = mytitle

    #explorer_layout.children[0].children[1] = pu
    return row(pu), sources
