# Documentation for building OpenCV with CUDA support

This documentation refers to the OpenCV build located on `Desktop/OpenCV`

## Folder structure
`build` contains the build files from CMake
`build/bin/Release/bin` contains the built opencv module with CUDA support
`build/bin/Release/lib/python3` contains the built python bindings for the cuda support
`opencv` contains the git repository
`opencv_contrib` contains the additional git repository

## Check installation
In command window, start python.
- `import cv2`
- `cv2.__version__` should return 4.5.3
- `print(cv2.cuda.getCudaEnabledDeviceCount())`should return 1

## Repairing installation
Open command window and write
- `pip uninstall opencv-python`

- Open `Visual Studio` and select `OpenCV.sln`
- On the right side, right click on `INSTALL` and select `build`. It will show some errors, but after 20 seconds you can close Visual Studio
- Go to `Check installation again from above`

## Repairing venv
Make sure that the installation is successful (see Check installation above)
- Go to `Gazeclassify` and delete the folders `venv` and `.idea`
- Open `Pycharm` and go `File, open` and select the `GazeClassify` folder
- Add a new venv and make sure to tick `include site packages`
- Configure the python interpreter
- Run ``
- Alternatively, run `pip install --upgrade git+https://github.com/footballdaniel/gazeclassify.git[dev]`
- Run the script `test_if_cuda_is_supported.py`


## How to add the built opencv to python
[Understanding the build](https://answers.opencv.org/question/235491/error-importing-cv2-after-compiling-opencv-from-source-python/)
[Adding cv2 to python after building, see section Including python bindings, point 4](https://jamesbowley.co.uk/accelerate-opencv-4-3-0-build-with-cuda-and-python-bindings/),
	- It turns out the files `opencv_world430.dll and opencv_img_hash430.dll` have to be on the path.
	- The files are built under `%openCvBuild%\install\x64\vc16\bin;%path%`

- The python bindings from `build/bin/Release/lib/python3/*.pyd` have to be added to the site packages `C:\Users\eyetracking\AppData\Local\Programs\Python\Python39\Lib\site-packages`
	- No folder with `cv2` should be available in the site packages folder.