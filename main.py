from config import Config, OntologyConfig, CabascConfig, LCR_RotConfig, LCR_RotInverseConfig, LCR_RotHopConfig, \
    LocalInterpretableConfig, SVMConfig
from data_setup.external_data_loader import ExternalDataLoader
from data_setup.internal_data_loader import InternalDataLoader
from abs_classifiers.ontology_reasoner import OntologyReasoner
from abs_classifiers.cabasc import CABASCModel
from abs_classifiers.lcr_rot import LCRRot
from abs_classifiers.lcr_rot_inverse import LCRRotInverse
from abs_classifiers.lcr_rot_hop import LCRRotHopModel
from abs_classifiers.svm import SVM
from diagnostic_classifier.diagnostic_classifier import DiagnosticClassifier
from local_interpretable_model.local_interpretable_model import LocalInterpretableModel
from local_interpretable_model.plots_set_up import SingleSentencePlot
import os


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
    svm = False
    cabasc = False
    lcr_rot = False
    lcr_rot_inverse = False
    lcr_rot_hop = True

    # Do you want to run a hybrid model, the ontology reasoner always runs the hybrid form to set up the
    # remaining data file. Therefore after running ontology reasoner once, you do not need to run it again
    Config.hybrid_method = False

    # Do you want to run cross-validation rounds, please specify the cross_validation_round settings in config.py
    Config.cross_validation = False

    # Which Diagnostic Classifier, do you want to switch on or off, you can choose more than one
    # Please specify in config.py the specification per classifier
    diagnostic_classifier_for_ontology_mention = False
    diagnostic_classifier_for_part_of_speech_tagging = False
    diagnostic_classifier_for_aspect_relations = False
    diagnostic_classifier_for_aspect_polarities = False

    diagnostic_classifiers = {
        'mentions': diagnostic_classifier_for_ontology_mention,
        'polarities': diagnostic_classifier_for_aspect_polarities,
        'relations': diagnostic_classifier_for_aspect_relations,
        'part_of_speech': diagnostic_classifier_for_part_of_speech_tagging
    }

    # Local Interpretable model, do you want it on or off, please set up the configuration in config.py
    local_interpretable_model = False
    n_of_relevance_words = 10

    if ontology:
        # Running the ontology reasoner. Cannot use an explanation model for the ontology reasoner
        ontology_reasoner = OntologyReasoner(OntologyConfig, internal_data_loader)
        ontology_reasoner.run()

    if svm:
        # Running the support vector machine (SVM). Cannot use an explanation model for SVM
        svm_model = SVM(SVMConfig, internal_data_loader)
        svm_model.run()

    if cabasc:
        # Running the the CABASC model.
        cabasc_model = CABASCModel(CabascConfig, internal_data_loader)
        cabasc_model.run()

        if True in diagnostic_classifiers:
            diagnostic_classifier = DiagnosticClassifier(cabasc_model, diagnostic_classifiers)
            diagnostic_classifier.run()

        if local_interpretable_model:
            local_interpretable_model = LocalInterpretableModel(LocalInterpretableConfig, cabasc_model)
            local_interpretable_model.run()

    if lcr_rot:
        # Running the the LCR Rot model.
        lcr_rot_model = LCRRot(LCR_RotConfig, internal_data_loader)

        # Check whether LCR Rot is already trained. Otherwise train the model.
        if not os.path.isfile(lcr_rot_model.config.file_to_save_model+".index"):
            lcr_rot_model.run()

        # Running Diagnostic Classifiers on the LCR Rot model.
        if True in diagnostic_classifiers.values():
            diagnostic_classifier = DiagnosticClassifier(lcr_rot_model, diagnostic_classifiers)
            diagnostic_classifier.run()

        # Running Local Interpretable model on the LCR Rot model.
        if local_interpretable_model:
            local_interpretable_model_config = LocalInterpretableConfig
            results_file = local_interpretable_model_config.get_file_of_results(lcr_rot_model.config.name_of_model)

            if not os.path.isfile(results_file):
                local_interpretable_model = LocalInterpretableModel(local_interpretable_model_config, lcr_rot_model)
                local_interpretable_model.run()

            single_sentence_plot = SingleSentencePlot(local_interpretable_model_config,
                                                      lcr_rot_model.config.name_of_model)
            single_sentence_plot.plot(n_of_relevance_words)

    if lcr_rot_inverse:
        # Run Diagnostic classifiers on the LCR Rot inverse model
        lcr_rot_inverse_model = LCRRotInverse(LCR_RotInverseConfig, internal_data_loader)

        # Check whether LCR Rot inverse is already trained. Otherwise train the model.
        if not os.path.isfile(lcr_rot_inverse_model.config.file_to_save_model+".index"):
            lcr_rot_inverse_model.run()

        # Running Diagnostic Classifiers on the LCR Rot inverse model.
        if True in diagnostic_classifiers:
            diagnostic_classifier = DiagnosticClassifier(lcr_rot_inverse_model, diagnostic_classifiers)
            diagnostic_classifier.run()

        # Running Local Interpretable model on the LCR Rot inverse model.
        if local_interpretable_model:
            local_interpretable_model = LocalInterpretableModel(LocalInterpretableConfig, lcr_rot_inverse_model)
            local_interpretable_model.run()

    if lcr_rot_hop:
        # Run Diagnostic classifiers on the LCR Rot hop model
        lcr_rot_hop_model = LCRRotHopModel(LCR_RotHopConfig, internal_data_loader)

        # Check whether LCR Rot hop is already trained. Otherwise train the model.
        if not os.path.isfile(lcr_rot_hop_model.config.file_to_save_model+".index"):
            lcr_rot_hop_model.run()

        # Running Diagnostic Classifiers on the LCR Rot hop model.
        if True in diagnostic_classifiers:
            diagnostic_classifier = DiagnosticClassifier(lcr_rot_hop_model, diagnostic_classifiers)
            diagnostic_classifier.run()

        # Running Local Interpretable model on the LCR Rot hop model.
        if local_interpretable_model:
            local_interpretable_model = LocalInterpretableModel(LocalInterpretableConfig, lcr_rot_hop_model)
            local_interpretable_model.run()


if __name__ == '__main__':
    main()
