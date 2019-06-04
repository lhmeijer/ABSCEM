

class LocalInterpretableModel:

    def __init__(self, config, neural_language_model):
        self.config = config
        self.neural_language_model = neural_language_model