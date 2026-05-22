1. Install the requirements from `requirements.txt` and the `cosmos-transfer1-diffusion-renderer`. 
   
   I did that in this notebook: https://colab.research.google.com/drive/17KGk4DGgmPixbzqMdbtshNsAe6A-sQc1?usp=sharing#scrollTo=16092402
2. Run `download_and_process.py`. It downloads four datasets and puts the images that we want to transform into `{name}_images`, inside the `data` folder.
   
   You probably need to download the `lego` and `armadillo` datatsets manually, because they are hosted as zip files on zenodo. But everything should be explained in the console output of `download_and_process.py`.
3. Run `run_cosmos.py`. It takes the four folders with the images, runs the inverse rendering and puts them in `data_resutls`.