<div align="center">
  <p>
      <img width="100%" src="https://github.com/user-attachments/assets/92c331ba-827a-49f6-a3fb-338523cea8f4"></a></p>   
</div>
  
## <div align="center">VEHICLE MOVEMENT ANALYSIS AND INSIGHT GENERATION USING EDGE AI - TEAM TECHSMASHERS  🚀</div>

### <div align="center">Intel Unnati Industrial Training Program 2024 </div>


### <div align="center">Deparment of CSE-AIML, Sreenidhi Insitute of Science and Technology, Hyderabad </div>

### A powerful edge AI based tool for vehicle entry and exit checking, parking lot occupancy evaluation and insight generation. 

## <div align="center">Documentation 📄</div>
<details open>
<summary>Install</summary>

Clone repo and install [requirements.txt](https://github.com/mopasha1/IntelUnnati2024TTS/blob/main/requirements.txt) 

```bash
git clone https://https://github.com/mopasha1/IntelUnnati2024TTS  # clone
cd IntelUnnati2024TTS
pip install -r requirements.txt  # install
```
</details>

  
<details open>
<summary>Running the application</summary>
To test out the entire project after installation, run the following:
 
```
cd streamlit
```
In the MainDashboard and Parking Data Insights generation file, replace the key with your Google Gemini API Key.

Running the Streamlit app
```
python -m streamlit run MainDashboard.py
```
To test out the CLI mode for the model inference, run the following code

```
cd src
python model_inference.py [arguments_list]

usage: model_inference.py [-h] (-i IMAGE | -v VIDEO) [-show SHOW] [--output OUTPUT]
```

CLI for the parking space detection can also be found in the same directory: 
```
python parking_inference.py [arguments_list]

usage: parking_inference.py [-h] (-i IMAGE | -v VIDEO) [-show SHOW] [--output OUTPUT]
```

Code for insights generation is present in the "streamlit" folder
Please refer to the README files in each of the folders for more information.

</details>

### Details 🔎

The project contains the following:
1. **data** - Links to the datasets used for training, validation and testing purposes
2. **models** - The trained models for parking space detection, license plate detection, and license plates OCR. Also contains the model formats optimized for Edge AI inference, in ONNX, OpenVINO and NCNN formats
3. **notebooks** - Contains the notebooks used for model training and export.
4. **scripts** - Various miscellaneous scripts for preprocessing, data generation, test data sampling for export, etc.
5. **src** - The CLI based code for model inference of the parking space and license plate models.
6. **streamlit** - The code for running the model inference, parking space detection and insights generation modules. Part of the main source code of the project. 
7. **test_data** - Test images and videos for testing the license plate and parking detection models
8. **requirements.txt** - The requirements file, containing all dependencies for the project.

### Features 🧠:
1. **Hyper-fast edge device inference:** Our models for License Plate processing and parking lot detection achive a combined average inference speed of just 59.38 ms on a Ryzen 5 CPU due to the use of specialized LPRNet and quantized YOLOv8 models, making them extremly fast compared to other alternatives. 
2. **Modular design:** The license plate detection models, the parking occupancy monitoring models and the insights engine are modular and can be used separately or in conjunction with one another
3. **Large variety of insights:** Our project delivers many different types of insights, including visualizations in the form of bar charts, line charts, etc., metrics for comparision and LLM-powered insights for interactive chat as well.
4. **Robust:** The models were all trained on large and diverse datasets, each with upwards of ~20,000 images, making them robust and accurate in detecting even outliers.

### Objectives achieved ✅: 

1. Near-real time monitoring of parking occupancy, and vehicle movement capture and inference.
2. Identification of authorized and unauthorized vehicles coming into the campus
3. Frequency analysis and pattern identification of both parking occupancy and incoming/outgoing vehicles

### For complete details about implementation, methodology and results, along with visualizations, please refer the project report [here](https://github.com/mopasha1/IntelUnnati2024TTS/blob/main/reports/Project_Report_and_Appendix.pdf)


Example of License plate evaluation

![license_plate_new-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/ea30c052-ee6c-44ca-80d1-deb7fa5d5011)

Example of insight generation

![record_gif-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/510720a5-94f3-4d45-8738-f3e00f5e322e)

Example of parking lot occupancy monitoring:

![parking_lot_gif-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/f973a823-1929-4f82-818c-227c56b39307)

### The Team ✨

1. Mohammed Moiz Pasha - [21311A6601@sreenidhi.edu.in](mailto:21311A6601@sreenidhi.edu.in), CSE-AI&ML

2. Gurram Bharath - [21311A6603@sreenidhi.edu.in](mailto:21311A6603@sreenidhi.edu.in), CSE-AI&ML

3. Ramachandruni Sumedha - [21311A6662@sreenidhi.edu.in](mailto:21311A6662@sreenidhi.edu.in), CSE-AI&ML

4. Mendi Pavan Kumar - [21311A6639@sreenidhi.edu.in](mailto:21311A6639@sreenidhi.edu.in), CSE-AI&ML

5. Gugulothu Vijaya - [21311A6643@sreenidhi.edu.in](mailto:21311A6643@sreenidhi.edu.in), CSE-AI&ML

### Credits 🎓
Huge thanks to our internal mentor, **Dr. T.V. Rajinkanth sir** , Professor & Head, Department of CSE-AI&ML,SNIST, for his support in this project.

This project also would not have been possible without the support of our external mentor, **Archana Vaidheeswaran**, who guided us throughout the duration of this project. 

Thanks to the Intel Unnati team for providing this opportunity for developing a solution to such an interesting and unique problem statement
