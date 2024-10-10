## MB-Net: A Network for Accurately Identifying Creeping Landslides from Wrapped Interferograms 

### Data Acquisition
ISSLIDE dataset can be accessed from IEEE Dataport  [here](https://ieee-dataport.org/documents/isslide-insardataset-slow-sliding-area-detection-machine-learning).


the data used to verify the model's generalization ability can be obtained from the COMET-LiCS Sentinel-1 InSAR portal  [here](https://comet.nerc.ac.uk/COMET-LiCS-portal/).


The data used to train MB-Net can be obtained from the cloud storage

```bash
Link: https://pan.baidu.com/s/1D7yj3SkyBuepr0AyiDMbhA?pwd=6666
Password: 6666 
```


### Environment Setup

To set up the conda environment for this project, please follow these steps:

  **Install Anaconda**: Ensure that you have Anaconda or Miniconda installed on your system. You can download it from [here](https://www.anaconda.com/products/distribution).


  **Create a New Conda Environment**: Use the `requirements.txt` file to create a new conda environment. Run the following command in your terminal:
   ```bash
   conda create --name <env> --file requirements.txt
   ```
   
  **Activate the Environment**: Activate the newly created conda environment with the following command:
   ```bash
   conda activate <env>
   ```


### Training
To train the model, please run `main.py`. You can execute it with the default parameters. Of course, you can also specify some parameters and paths as needed.
   ```bash
   python main.py --train
   python main.py --train --save_path=<pth> --model=<name> --workers=<num1> --epochs=<num2> --batch_size=<num3>
   ```
### Testing
To test the model, please run `main.py`. You can execute it with the default parameters. Of course, you can also specify some parameters and paths as needed.
   ```bash
   python main.py --test
   python main.py --test --save_path=<pth> --model=<name> --workers=<num1> --epochs=<num2> --batch_size=<num3>
   ```

Training and testing can also be executed together.
```bash
   python main.py --train --test
   python main.py --train --test --save_path=<pth> --model=<name> --workers=<num1> --epochs=<num2> --batch_size=<num3>
   ```
### Citation Notice
When utilizing this code and data, please ensure to appropriately cite the relevant literature. Thank you for your understanding and cooperation!
