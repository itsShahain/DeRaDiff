import os
import statistics
import torch
from typing import Union

from src.utils.pickscore_utils import Selector as PickScoreSelector
from src.utils.clip_utils import Selector as CLIPScoreSelector
from src.utils.hps_utils import Selector as HPSScoreSelector

CLIP = "CLIP"
HPS = "HPS"
PICKSCORE = "PickScore"

METRICS = [CLIP, HPS, PICKSCORE]


class Evaluator:
    def __init__(self, device: Union[str, torch.device], output_dir: str = "."):
        self.scorers = {
            CLIP: CLIPScoreSelector(device),
            HPS: HPSScoreSelector(device),
            PICKSCORE: PickScoreSelector(device),
        }
        self.all_scores = {m: [] for m in METRICS}
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def score_image(self, image, prompt):
        scores = {}
        for metric, scorer in self.scorers.items():
            s = scorer.score([image], prompt)[0]
            self.all_scores[metric].append(s)
            scores[metric] = s
            print(f"{metric}: {s}")
        return scores

    def save_results(self, model_type: str, lambda_val: float, beta_val: int):
        for metric in METRICS:
            self.all_scores[metric].sort()
            result_dir = os.path.join(self.output_dir, f"{model_type}_results_{metric}")
            os.makedirs(result_dir, exist_ok=True)
            filename = os.path.join(
                result_dir, f"{model_type}_{lambda_val}_Beta{beta_val}_{metric}.txt"
            )
            scores = self.all_scores[metric]
            with open(filename, "a") as f:
                print(f"Beta anchor: {beta_val}", file=f)
                if lambda_val != 0:
                    print(f"Beta approx: {beta_val / lambda_val}", file=f)
                print(f"Lambda: {lambda_val}", file=f)
                print(f"Median: {statistics.median(scores)}", file=f)
                print(f"Mean:   {statistics.mean(scores)}", file=f)
                print(file=f)
                print(file=f)
