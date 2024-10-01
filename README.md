conda create -n chaos python==3.9
conda activate chaos
pip install -r requirements.txt
torch_geometric
torch_scatter -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
torch_sparse -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
torch_cluster -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.1+cu118.html