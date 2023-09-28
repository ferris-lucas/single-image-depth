# Single Image Depth Prediction

This repository contains a TensorFlow implementation of the paper by Eigen, David, Christian Puhrsch, and Rob Fergus, titled "Depth map prediction from a single image using a multi-scale deep network" (Advances in Neural Information Processing Systems 27, 2014).

## How to Use

1. **Get the Data:**
   Execute `download_data.sh` by running `./download_data.sh`. If needed, update permissions with `chmod +x download_data.sh`. This script will download the labeled dataset (2.8GB) into the `data` folder.

2. **Training:**
   Train the model using either the provided Jupyter notebook or the `train.py` file. If using Singularity for containerization, follow these steps:
   - Load Singularity module: `module load singularity`
   - Log in to a distributed server: `srun -p gpu_intr --gres=gpu:1 --pty bash`
   - Pull the TensorFlow GPU image: `singularity pull docker://tensorflow/tensorflow:latest-gpu`
   - Run the TensorFlow GPU image: `singularity run --nv tensorflow_latest-gpu.sif`
   - Execute the training script: `python train.py`
   
   If there are missing dependencies, you can install them within the container using `pip install`. If not using Singularity, run the code directly with `python train.py`, ensuring appropriate adjustments for GPU usage.

3. **Work in Progress:**
   This project is still a work in progress. Planned features include:
   - [ ] Implement data augmentation for the dataset
   - [ ] Implement validation and comparison with other methods as described in the paper
   - [ ] Implement the use of the large NYUv2 dataset (acquire Kinect raw data, process, and add to training)

---

Feel free to contribute to this project and open issues for any suggestions or improvements.

## License

This project is licensed under the [MIT License](LICENSE).
