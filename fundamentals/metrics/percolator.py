from metric import Metric


class Percolator(Metric):

    def fit_loess(self):
        """
        Use IRT value here
        """
        pass

    @staticmethod
    def get_scannr(rawfile, scannr):
        s = "{}{}".format(rawfile, scannr).encode()
        return int(hashlib.sha224(s).hexdigest()[:12], 16)

    def calc(self):
        """
        Here we should calculate all metrics
        """
        pass
