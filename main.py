from estimator import *
from test import *
from utils import load_data, split_data, create_model
import warnings
warnings.filterwarnings('ignore')

# change the input audio file, which should be a \\\\\\\\\\.wav file with 2 channels.//////////
'''
                 Here are some pre-set input wav file, you can change the name to 
    'Dancing With A Stranger.wav'  'Love Story.wav' 'Someone Like You.wav' 'Thinking Out Loud.wav'      
                                                                                                        '''
input = './wav file/Dancing with stranger.wav'
separate_output = './output'
# check the human voice file and the BGM file in the directory above
filename = estimate(input,separate_output)
# construct the model
model = create_model()
# load the saved/trained weights
model.load_weights("./model.h5")
# load the human voice file
fea,sr = sf.read(filename,-1,2500000,2800000)
# take a fragment and rewrite a temporary audio file
sf.write('./temp.wav',fea,sr)
# analyze the feature of the temporary file
features = extract_feature('./temp.wav', mel=True).reshape(1, -1)
# predict the gender
male_prob = model.predict(features)[0][0]
female_prob = 1 - male_prob
gender = "male" if male_prob > female_prob else "female"
# show the result
print("Result:", gender)
print(f"Probabilities:     Male: {male_prob*100:.2f}%    Female: {female_prob*100:.2f}%")