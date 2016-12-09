#tests: mypy, lint

from typing import Optional, List

from nltk.translate.bleu_score import corpus_bleu


class NLTKBleu(object):

    def __init__(self, n: int=4, name: Optional[str]=None) -> None:
        self.n = n
        if name is not None:
            self.name = name
        else:
            self.name = "NLTK-BLEU-{}".format(n)

    def __call__(self,
                 decoded: List[List[str]],
                 references: List[List[str]]) -> float:

        weights = [1/self.n for _ in range(self.n)]
        listed_references = [[s] for s in references]
        return 100 * corpus_bleu(listed_references, decoded, weights)
