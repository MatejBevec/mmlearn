import unittest

from mmlearn import data
from mmlearn.fe import image_fe, text_fe, audio_fe
from mmlearn.models import base_models, image_models, mm_models, text_models, audio_models
from mmlearn import eval

# SYSTEM LEVEL TESTS

class SystemRunsWithNoError(unittest.TestCase):

    def test_system_runs(self):
        try:
            dataset = data.TastyRecipes(shuffle=True)

            model= base_models.MajorityClassifier()
            model2 = text_models.TextSkClassifier(fe=text_fe.NGrams(), clf="lr_best")
            latef = mm_models.LateFusion(combine="stack", stacking_clf="lr")
            earlyf = mm_models.NaiveEarlyFusion(text_fe=text_fe.NGrams(), clf="lr_best")
            models = {"majority": model, "text": model2, "late fusion": latef}
            results = eval.holdout_many(dataset, models, dataframe=True)

        except Exception as e:
            self.fail(f"Exception {e} was raised.")

        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()