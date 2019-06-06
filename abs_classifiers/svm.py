

class SVM:

    def __init__(self, config, internal_data_loader):
        self.config = config
        self.internal_data_loader = internal_data_loader

    def run(self):
        cvec = CountVectorizer(lowercase=True, binary=True)
        wm = cvec.fit_transform(review_vector)
        bow = np.array(wm.A)