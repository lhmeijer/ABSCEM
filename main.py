from config import Config, OntologyConfig, CabascConfig, LCR_RotConfig, LCR_RotInverseConfig, LCR_RotHopConfig, \
    DiagnosticClassifierConfig, LocalInterpretableConfig, SVMConfig
from data_setup.external_data_loader import ExternalDataLoader
from data_setup.internal_data_loader import InternalDataLoader
from abs_classifiers.ontology_reasoner import OntologyReasoner
from abs_classifiers.cabasc import CABASCModel
from abs_classifiers.lcr_rot import LCRRot
from abs_classifiers.lcr_rot_inverse import LCRRotInverse
from abs_classifiers.lcr_rot_hop import LCRRotHopModel
from abs_classifiers.svm import SVM
from diagnostic_classifier.diagnostic_classifier import DiagnosticClassifier
import os

def main():

    # Load external data if the internal files are not available
    if not os.path.isfile(Config.internal_train_data) and not os.path.isfile(Config.internal_test_data):
        external_data_loader = ExternalDataLoader(Config)
        external_data_loader.load_external_data(load_external_file_name=Config.external_train_data,
                                                write_internal_file_name=Config.internal_train_data)
        external_data_loader.load_external_data(load_external_file_name=Config.external_test_data,
                                                write_internal_file_name=Config.internal_test_data)

    # Aspect-Based Sentiment Classifiers, which do you want on or off
    ontology = True
    svm = True
    cabasc = True
    lcr_rot = True
    lcr_rot_inverse = True
    lcr_rot_hop = True

    # Do you want to run a hybrid model
    Config.hybrid = True

    # Do you want to run cross-validation rounds, please specify the cross_validation_rounds in config.py
    Config.cross_validation = True

    # Diagnostic Classifier, do you want it on or off, please specify in config.py the ..
    diagnostic_classifier = True

    # Local Interpretable model, do you want it on or off, please specify in config.py ..
    local_interpretable_model = True

    if ontology:

        ontology_reasoner = OntologyReasoner(OntologyConfig)
        ontology_reasoner.run()

    if svm:

        svm_model = SVM(SVMConfig)
        svm_model.run()


    if cabasc:
        cabasc_model = CABASCModel(CabascConfig)
        cabasc_model.run()

        if diagnostic_classifier:
            diagnostic_classifier = DiagnosticClassifier(DiagnosticClassifierConfig)


        if local_interpretable_model:


    if lcr_rot:
        lcr_rot_model = LCRRot(LCR_RotConfig)
        lcr_rot_model.run()

        if diagnostic_classifier:

        if local_interpretable_model


    if lcr_rot_inverse:
        lcr_rot_inverse_model = LCRRotInverse(LCR_RotInverseConfig)
        lcr_rot_inverse_model.run()

        if diagnostic_classifier:

        if local_interpretable_model:


    if lcr_rot_hop:
        lcr_rot_hop_model = LCRRotHopModel(LCR_RotHopConfig)
        lcr_rot_hop_model.run()


if __name__ == '__main__':
    main()