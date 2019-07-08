# Aspect-Based Sentiment Classification Explanation Methods (ABSCEM)
Code for two kind of explanation models, local interpretable models, and diagnostic classifiers, for the Neural Rotatory Attention model, which is a Aspect-Based Sentiment Classifier.

All software is written in PYTHON3 (https://www.python.org/) and makes use of the TensorFlow framework (https://www.tensorflow.org/).


## Installation Instructions:
### Dowload required files and add them to the data/external_data folder:
1. Download ontology: https://github.com/KSchouten/Heracles/tree/master/src/main/resources/externalData
2. Download SemEval2016 Dataset: http://alt.qcri.org/semeval2016/task5/index.php?id=data-and-tools
3. Download Glove Embeddings: http://nlp.stanford.edu/data/glove.42B.300d.zip
4. Download Stanford CoreNLP:https://stanfordnlp.github.io/CoreNLP/download.html

### Setup Environment
1. Make sure that pip is installed and use pip to install the following packages: setuptools and virtualenv (http://docs.python-guide.org/en/latest/dev/virtualenvs/#virtualenvironments-ref).
2. Create a virtual environment in a desired location by running the following command: `code(virtualenv ENV_NAME)`
3. Direct to the virtual environment source directory. 
4. Unzip the ABSCEM_software.zip file in the virtual environment directory. 
5. Activate the virtual environment.
6. Install the required packages from the requirements.txt file by running the following command: `code(pip install -r requirements.txt)`.

### Run Software
1. Please set up all your preferences in config.py before running the code
2. Configure the main.py by mark True the program(s) you want to run
3. Run main.py from your command line by `code(python main.py)`, or from your python editor

## Software explanation:
#### main environment:
- main.py: program to run . Each method can be activated by setting its corresponding boolean to True e.g. to run the lcr_Rot method set lcr_Rot = True.
- config.py: contains parameter configurations that can be changed such as: dataset_year, batch_size, iterations.
#### Aspect-Based sentiment classifiers:
- abs_classifiers/lcr_rot.py:  implementation for the LCR-Rot algorithm, a subclass of neural language model
- abs_classifiers/lcr_rot_inverse.py:  implementation for the LCR-Rot-inverse algorithm, a subclass of neural language model
- abs_classifiers/lcr_rot_hop.py:  implementation for the LCR-Rot-hop algorithm, a subclass of neural language model
- abs_classifiers/neural_language_model.py: implementation of neural language model algorithm, main class
- abs_classifiers/ontology_reasoner.py:  implementation for the ontology reasoner
#### data pre-processing steps:
- data_setup/external_data_loader.py: loads the raw xml data, and transform it to the a JSON formats to be used by the models
- data_setup/internal_data_loader.py: loads the pre-processed JSON files into the environment
- data_setup/ontology_tagging.py: implementation of the 
#### diagnostic classifiers:
- diagnostic_classifier/classifier.py: implementation of a TensorFlow Neural Network
- diagnostic_classifier/diagnostic_classifier.py: implementation of the diagnostic classifiers
#### opinion explanation:
- explanation/prediction_explanations.py: implementation of the in-depth analysis of the neural language model
- explanation/sentence_explanation.py: program to obtain the sentence explanation for a single sentence 
- explanation/sentence_explanation_plot.py: program to obtain plots of the sentence explanation
#### local interpretable model:
- local_interpretable_model/contribution_evaluators.py: implementation of getting the contributions by LETA, A-LACE, and A-LIME
- local_interpretable_model/decision_tree.py: implementation of the decision tree to obtain the word combinations
- local_interpretable_model/linear_model.py: implementation of the linear model we use in A-LIME, and LETA
- local_interpretable_model/local_interpretable_model.py: full implementation of the local interpretable models LETA, A-LACE, and A-LIME
- local_interpretable_model/locality_algorithms.py: program to obtain the sample in the locality of an opinion 
- local_interpretable_model/plots_set_up.py: program to get the plots of the local interpretable model
#### layers for the neural language models:
- model_layers/attention_layers.py: implementation of the attention function
- model_layers/nn_layers.py: implementation of the Bi-LSTM, and softmax layer

## Directory explanation:
- results: please create the following folders
    - abs_classifiers: accuracy results of the abs_classifiers
    - diagnostic_classifiers: results of all diagnostic_classifiers
    - local_interpretable_models: results of the local interpretable models
    - sentence_explanations: sentence explanations
- data:
	- external_data: location for the external data required by the methods
	- internal_data: location for the internal data required by the methods
	- hidden_layers: location for the hidden layers of the models
	- indices: location for the indices of correctly predicted opinions, incorrectly predicted indices, etc.

## Related Work: ##
This code uses ideas and code of the following related papers:
- Zheng, S. and Xia, R. (2018). Left-center-right separated neural network for aspect-based sentiment analysis with rotatory attention. arXiv preprint arXiv:1802.00892.
- Schouten, K. and Frasincar, F. (2018). Ontology-driven sentiment analysis of product and service aspects. In Proceedings of the 15th Extended Semantic Web Conference (ESWC 2018), pages 608–623.
- Wallaart, O. and Frasincar, F. (2019). A hybrid approach for aspect-based sentiment analysis using a lexicalized domain ontology and attentional neural models. In The Semantic Web - 16th Extended Semantic Web Conference (ESWC 2019), pages 363–378.
- Ribeiro, M. T., Singh, S., and Guestrin, C. (2016). ”why should I trust you?”: Explaining the predictions of any classifier. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD 2016), pages 1135–1144. ACM.
- Hupkes, D., Veldhoen, S., and Zuidema, W. H. (2018). Visualisation and ’diagnostic classifiers’ reveal how recurrent and recursive neural networks process hierarchical structure. J. Artif. Intell. Res., 61:907–926.

