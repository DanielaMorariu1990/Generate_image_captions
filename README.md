# Generate Image Captions

<img src="https://github.com/DanielaMorariu1990/Generate_image_captions/blob/main/demo.gif" width="700" height="350">

## Description

The program takes an image from the Flicker8 image set and outputs a description of the picture it "sees". Currently it only works with Flicker8 images for the following reasons:

1. The image feature vector extracted from InceptionV3 (please see arhitecture below) is pre-extracted for all Flicker8 images and stored on this repo. This could be easly converted in real time extractions for new images, on a better computer.

2. The image descriptions (text generations) have been trained on the Flicker8 captions. This could be extended on a larger data set, like Flicker30. The steps are however the same, as in this repo.

I have used the following model arhitecture, as depicted in the picture below.
<img src="https://github.com/DanielaMorariu1990/Generate_image_captions/blob/main/model_arhitecture.PNG" width="700" height="350">

The interface is simple, and built in Flask (see gif).

For this pupose I have used the following tools:

- Python
- Keras: InceptionV3, LSTM and Dense layers
- word2vec, for pre-training word emedding
- HTML, CSS, Flask

## How to use

In a terminal:

1. Clone this repo: `git clone https://github.com/lorenanda/movie-recommender.git`
2. Install the necessary libraries: `pip install -r requirements.txt`
3. Make sure you are in the main directory Generate_image_captions
4. Download the Flicker8 image set.
5. Run these three commands:
   - `export FLASK_APP=application.py`
   - `export FLASK_DEBUG=1`
   - `flask run`
6. Open the listed localhost http://127.0.0.1:5000/ in a browser.
7. Try out some phothos!

_The project was tested on Chrome, Firefox and Opera!_
