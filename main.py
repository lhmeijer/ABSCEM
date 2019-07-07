from config import Config, OntologyConfig, LCR_RotConfig, LCR_RotInverseConfig, LCR_RotHopConfig, \
    LocalInterpretableConfig
from data_setup.external_data_loader import ExternalDataLoader
from data_setup.internal_data_loader import InternalDataLoader
from abs_classifiers.ontology_reasoner import OntologyReasoner
from abs_classifiers.lcr_rot import LCRRot
from abs_classifiers.lcr_rot_inverse import LCRRotInverse
from abs_classifiers.lcr_rot_hop import LCRRotHopModel
from diagnostic_classifier.diagnostic_classifier import DiagnosticClassifier
from local_interpretable_model.local_interpretable_model import LocalInterpretableModel
from explanation.sentence_explanation import SentenceExplaining
from local_interpretable_model.plots_set_up import SingleSentencePlot
from explanation.prediction_explanations import get_indices_of_correct_wrong_predictions
from explanation.sentence_explanation_plot import SentenceExplanationPlot
import os, json


def main():

    # Load external data if the internal files are not available
    if not os.path.isfile(Config.internal_train_data) or not os.path.isfile(Config.internal_test_data):
        external_data_loader = ExternalDataLoader(Config)
        external_data_loader.load_external_data(load_external_file_name=Config.external_train_data,
                                                write_internal_file_name=Config.internal_train_data)
        external_data_loader.load_external_data(load_external_file_name=Config.external_test_data,
                                                write_internal_file_name=Config.internal_test_data)

    # Run internal data loader to set up all different data sets, you always have to load these files
    internal_data_loader = InternalDataLoader(Config)
    internal_data_loader.load_internal_training_data(load_internal_file_name=Config.internal_train_data)
    internal_data_loader.load_internal_test_data(load_internal_file_name=Config.internal_test_data)

    # Aspect-Based Sentiment Classifiers, which do you want on or off
    ontology = False
    lcr_rot = False
    lcr_rot_inverse = False
    lcr_rot_hop = False

    # Do you want to run a hybrid model, the ontology reasoner always runs the hybrid form to set up the
    # remaining data file. Therefore after running ontology reasoner once, you do not need to run it again
    Config.hybrid_method = False

    # Do you want to run cross-validation rounds for abs classifiers,
    # please specify the cross_validation_round settings in config.py
    Config.cross_validation = False

    # Which Diagnostic Classifier, do you want to switch on or off, you can choose more than one
    # Please specify in config.py the specification per classifier
    diagnostic_classifier_for_word_sentiments = False
    diagnostic_classifiers_for_ontology_mentions = False
    diagnostic_classifier_for_part_of_speech_tagging = False
    diagnostic_classifier_for_aspect_relations = False
    diagnostic_classifier_for_aspect_sentiments = False
    diagnostic_classifier_for_full_aspect_sentiment = False

    diagnostic_classifiers = {
        'word_sentiments': diagnostic_classifier_for_word_sentiments,
        'aspect_sentiments': diagnostic_classifier_for_aspect_sentiments,
        'relations': diagnostic_classifier_for_aspect_relations,
        'pos_tags': diagnostic_classifier_for_part_of_speech_tagging,
        'mentions': diagnostic_classifiers_for_ontology_mentions,
        'full_aspect_sentiment': diagnostic_classifier_for_full_aspect_sentiment
    }

    # Local Interpretable model, do you want it on or off, please set up the configuration in config.py
    local_interpretable_model = False

    # Visualising the n_of_relevance_words of a given sentence_id from the local interpretable models
    n_of_relevance_words = 5
    visualize_single_sentence_word_relevance = False

    # Full Explanation model for a given sentence_id: combination of both the diagnostic classifiers, and the local
    # interpretable models
    full_explanation_model = False
    sentence_id = '1041457:3'


    if ontology:
        # Running the ontology reasoner. Cannot use an explanation model for the ontology reasoner
        ontology_reasoner = OntologyReasoner(OntologyConfig, internal_data_loader)
        ontology_reasoner.run()

    if lcr_rot:
        # Running the the LCR Rot model.
        lcr_rot_model = LCRRot(LCR_RotConfig, internal_data_loader)

        # Check whether LCR Rot is already trained. Otherwise train the model.
        if not os.path.isfile(lcr_rot_model.config.file_to_save_model + ".index"):
            lcr_rot_model.run()

        # Check whether we already obtain the indices for incorrectly and correctly predicted opinions.
        if not os.path.isfile(lcr_rot_model.config.file_of_indices):
            get_indices_of_correct_wrong_predictions(lcr_rot_model)

        # Running Diagnostic Classifiers on the LCR Rot model.
        if True in diagnostic_classifiers.values():
            diagnostic_classifier = DiagnosticClassifier(lcr_rot_model, diagnostic_classifiers)
            diagnostic_classifier.run()

        # Running Local Interpretable model on the LCR Rot model.
        if local_interpretable_model:
            local_interpretable_model_config = LocalInterpretableConfig
            results_file = local_interpretable_model_config.get_file_of_results(lcr_rot_model.config.name_of_model)

            # Check whether we already have the results from the local interpretable model.
            if not os.path.isfile(results_file):
                local_interpretable_model = LocalInterpretableModel(LocalInterpretableConfig, lcr_rot_model)
                local_interpretable_model.run()

            # Creating the plots of a single prediction
            if visualize_single_sentence_word_relevance:
                single_sentence_plot = SingleSentencePlot(local_interpretable_model_config, lcr_rot_model)
                single_sentence_plot.plot(n_of_relevance_words, sentence_id)

        # Running the full explanation model, a combination of the diagnostic classifiers and the local
        # interpretable models
        if full_explanation_model:
            file = lcr_rot_model.config.get_explanation_file(lcr_rot_model.config.name_of_model, sentence_id)

            # Check whether we have obtained all the information the create the plots
            if not os.path.isfile(file):
                diagnostic_classifier = DiagnosticClassifier(lcr_rot_model, diagnostic_classifiers)
                local_interpretable_model = LocalInterpretableModel(LocalInterpretableConfig, lcr_rot_model)
                sentence_explanation = SentenceExplaining(lcr_rot_model, diagnostic_classifier,
                                                          local_interpretable_model)
                sentence_explanation.explain_sentence(sentence_id)

            sentence_plot = SentenceExplanationPlot(lcr_rot_model)
            sentence_plot.run(sentence_id)

    if lcr_rot_inverse:
        # Run Diagnostic classifiers on the LCR Rot inverse model
        lcr_rot_inverse_model = LCRRotInverse(LCR_RotInverseConfig, internal_data_loader)

        # Check whether LCR Rot inverse is already trained. Otherwise train the model.
        if not os.path.isfile(lcr_rot_inverse_model.config.file_to_save_model+".index"):
            lcr_rot_inverse_model.run()

        # Check whether we already obtain the indices for incorrectly and correctly predicted opinions.
        if not os.path.isfile(lcr_rot_inverse_model.config.file_of_indices):
            get_indices_of_correct_wrong_predictions(lcr_rot_inverse_model)

        # Running Diagnostic Classifiers on the LCR Rot inverse model.
        if True in diagnostic_classifiers.values():
            diagnostic_classifier = DiagnosticClassifier(lcr_rot_inverse_model, diagnostic_classifiers)
            diagnostic_classifier.run()

        # Running Local Interpretable model on the LCR Rot inverse model.
        if local_interpretable_model:
            local_interpretable_model_config = LocalInterpretableConfig
            results_file = local_interpretable_model_config.get_file_of_results(lcr_rot_inverse_model.config.name_of_model)

            # Check whether we already have the results from the local interpretable model.
            if not os.path.isfile(results_file):
                local_interpretable_model = LocalInterpretableModel(LocalInterpretableConfig, lcr_rot_inverse_model)
                local_interpretable_model.run()

            # Creating the plots of a single prediction
            if visualize_single_sentence_word_relevance:
                single_sentence_plot = SingleSentencePlot(local_interpretable_model_config, lcr_rot_inverse_model)
                single_sentence_plot.plot(n_of_relevance_words, sentence_id)

        # Running the full explanation model, a combination of the diagnostic classifiers and the local
        # interpretable models
        if full_explanation_model:
            file = lcr_rot_inverse_model.config.get_explanation_file(lcr_rot_inverse_model.config.name_of_model,
                                                                     sentence_id)
            # Check whether we have obtained all the information the create the plots
            if not os.path.isfile(file):
                diagnostic_classifier = DiagnosticClassifier(lcr_rot_inverse_model, diagnostic_classifiers)
                local_interpretable_model = LocalInterpretableModel(LocalInterpretableConfig, lcr_rot_inverse_model)
                sentence_explanation = SentenceExplaining(lcr_rot_inverse_model, diagnostic_classifier,
                                                          local_interpretable_model)
                sentence_explanation.explain_sentence(sentence_id)

            sentence_plot = SentenceExplanationPlot(lcr_rot_inverse_model)
            sentence_plot.run(sentence_id)

    if lcr_rot_hop:
        # Run Diagnostic classifiers on the LCR Rot hop model
        lcr_rot_hop_model = LCRRotHopModel(LCR_RotHopConfig, internal_data_loader)

        # Check whether LCR Rot hop is already trained. Otherwise train the model.
        if not os.path.isfile(lcr_rot_hop_model.config.file_to_save_model+".index"):
            lcr_rot_hop_model.run()

        # Check whether we already obtain the indices for incorrectly and correctly predicted opinions.
        if not os.path.isfile(lcr_rot_hop_model.config.file_of_indices):
            get_indices_of_correct_wrong_predictions(lcr_rot_hop_model)

        # Running Diagnostic Classifiers on the LCR Rot hop model.
        if True in diagnostic_classifiers.values():
            diagnostic_classifier = DiagnosticClassifier(lcr_rot_hop_model, diagnostic_classifiers)
            diagnostic_classifier.run()

        # Running Local Interpretable model on the LCR Rot hop model.
        if local_interpretable_model:
            local_interpretable_model_config = LocalInterpretableConfig
            results_file = local_interpretable_model_config.get_file_of_results(lcr_rot_hop_model.config.name_of_model)

            # Check whether we already have the results from the local interpretable model.
            if not os.path.isfile(results_file):
                local_interpretable_model = LocalInterpretableModel(LocalInterpretableConfig, lcr_rot_hop_model)
                local_interpretable_model.run()

            # Creating the plots of a single prediction
            if visualize_single_sentence_word_relevance:
                single_sentence_plot = SingleSentencePlot(local_interpretable_model_config, lcr_rot_hop_model)
                single_sentence_plot.plot(n_of_relevance_words, sentence_id)

        # Running the full explanation model, a combination of the diagnostic classifiers and the local
        # interpretable models
        if full_explanation_model:
            file = lcr_rot_hop_model.config.get_explanation_file(lcr_rot_hop_model.config.name_of_model, sentence_id)

            # Check whether we have obtained all the information the create the plots
            if not os.path.isfile(file):
                diagnostic_classifier = DiagnosticClassifier(lcr_rot_hop_model, diagnostic_classifiers)
                local_interpretable_model = LocalInterpretableModel(LocalInterpretableConfig, lcr_rot_hop_model)
                sentence_explanation = SentenceExplaining(lcr_rot_hop_model, diagnostic_classifier,
                                                          local_interpretable_model)
                sentence_explanation.explain_sentence(sentence_id)

            sentence_plot = SentenceExplanationPlot(lcr_rot_hop_model)
            sentence_plot.run(sentence_id)


if __name__ == '__main__':
    main()
