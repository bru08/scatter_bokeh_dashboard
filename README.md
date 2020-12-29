
## brief description

Example of dashboard with bokeh.
Now run on synthetic datasets inside `bokeh_dashboard/features`

The dataset are created using `bokeh_dashboard/data_generator`, using scikit learn `make_blobs` function


## synthetic dataset included (sample_data)
4 gaussian blobs with 20 features (x_0 ... x_19)    
y_0 contains the label representing membership to one of 4 blob   
y_1 trough y_10 are random binary labels using different probabilities of extraction on {0, 1}

## get started
Use environment.yml to create conda env  
run with `bokeh serve --show main.py`

Select one of the datasets: `sample_data` or `sample_data2`
Press the **submit** button to render the visualization


If everything is ok, should return something like this..

<img src="./static/example.png" alt="example">