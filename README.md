# Football Analysis System

## Overview

The **Football Analysis System** is a project designed to analyze football matches using advanced computer vision and machine learning techniques. The system focuses on detecting, tracking, and analyzing players, referees, and the ball to provide actionable insights for performance evaluation and match dynamics understanding.

## Key Features

1. **Player and Ball Detection**:
   - Utilizes state-of-the-art object detection models, such as YOLO (You Only Look Once), to identify and track players, referees, and the ball across video frames.

2. **Team Assignment**:
   - Leverages K-Means clustering to differentiate and assign players to teams based on jersey colors.

3. **Ball Possession Calculation**:
   - Tracks interactions between players and the ball to compute ball possession percentages for each team, offering insights into match control.

4. **Player Movement Analysis**:
   - Implements optical flow techniques to estimate camera movement between frames, enabling accurate tracking of player movements even with panning and zooming.
   - Applies perspective transformations to convert pixel measurements into real-world distances.

5. **Speed and Distance Metrics**:
   - Calculates player speeds and total distances covered during the match, providing critical data for performance evaluation.

6. **Video Annotation**:
   - Annotates video frames with player tracks, ball trajectories, speed, distance, and possession data, creating an enriched video output for analysis.

## Repository Structure

```
football_analysis_system/
├── config/                    # Configuration files
├── data/                      # Contains input videos, output videos, models, and stubs
├── development_and_analysis/  # Notebooks and scripts for model analysis
├── src/                       # Source code for core functionalities
├── .gitignore                 # Ignored files and directories
├── README.md                  # Project documentation
```

## Getting Started

### Prerequisites

- Anaconda environment setup:
  - Use the provided `config/environment.yml` file to create the environment.
    ```bash
    conda env create -f config/environment.yml
    ```
  - Activate the environment:
    ```bash
    conda activate football_analysis_env
    ```

This will install all necessary dependencies, including Python 3.10 and libraries such as OpenCV, PyTorch, NumPy, Matplotlib, scikit-learn, and supervision, along with the required YOLO pre-trained models.

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/miquelarisa/football_analysis_system.git
   ```
2. Navigate to the project directory:
   ```bash
   cd football_analysis_system
   ```
3. Ensure dependencies are installed using the Anaconda environment setup instructions provided above.

### Usage

1. **Training the Model**:
   - To train YOLO for player detection:
     ```bash
     yolo task=detect mode=train model=yolov8x.pt data=/path/to/data.yaml epochs=100 imgsz=640
     ```
2. **Running the Analysis**:
3. Execute the main script to process a video. Ensure the video is located in the input_videos directory and that its name is correctly specified in the code:
     ```bash
     python src/main.py
     ```
3. **Visualizing Results**:
   - Navigate to the output_videos directory and open the generated output video to review the analysis results.

## Evaluation Metrics

The project uses COCO-like evaluation metrics to assess the performance of the detection and tracking models:
- **Precision**
- **Recall**
- **mAP@50**
- **mAP@50-95**

Additionally, confusion matrices, F1-score curves, and other statistical charts are generated during the evaluation process.

## Contributions

Contributions are welcome! Feel free to submit pull requests or raise issues to help improve the project.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contact

For any inquiries or collaboration opportunities, please contact:
- **Miquel Arisa**
- [GitHub Profile](https://github.com/miquelarisa)
- [LinkedIn Profile](https://www.linkedin.com/in/miquel-arisa-fuente/)