from __future__ import annotations

import random
from abc import ABC, abstractmethod

import numpy as np
import torch
from typing_extensions import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .ssdg_dataset import SSDGDataset

Mode = Literal["fourier", "hist"]


class StyleSampler(ABC):
    """Sampler for style reference.

    Attributes:
        mode: style augmentation mode
        kwargs: extra arguments for style augmentation
        datasets: where to sample the style reference
    """

    def __init__(self, mode: Mode, **kwargs):
        self.mode: Mode = mode
        self.kwargs = kwargs
        self.datasets = []
        self.cutmix_prob = 0.5

    def set_cutmix_prob(self, cutmix_prob):
        self.cutmix_prob = cutmix_prob
        return self

    def bind(self, dataset: SSDGDataset):
        self.datasets = dataset.datasets
        return self

    @property
    def bound(self):
        return len(self.datasets) > 0

    @property
    def n_domains(self) -> int:
        return len(self.datasets)

    @property
    def n_samples(self) -> int:
        return sum(len(d) for d in self.datasets)

    @abstractmethod
    def _sample_index(self,
                      domain: int,
                      **kwargs) -> tuple[int, int]:
        """The inner implementation of ref sampling.

        Returns:
            (domain_id, sample_index relative to the current domain)
        """

    def sample(self,
               domain: int) -> tuple[np.ndarray, int, int]:
        """Sample a style reference from a domain.

        Returns:
            (image, domain_id, sample_index global)
        """
        domain_id, index = self._sample_index(domain)
        image = self.datasets[domain_id][index][0]
        return image, domain_id, index + sum(
            len(d) for d in self.datasets[:domain_id])

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass


class SameDomainStyleSampler(StyleSampler):
    """Sample a reference from the same domain."""

    def _sample_index(self, domain: int, **kwargs) -> tuple[int, int]:
        ref_dataset = self.datasets[domain]
        return domain, random.randint(0, len(ref_dataset) - 1)


class RandomStyleSampler(StyleSampler):
    """Random sample a reference.

    Attributes:
        exclude_self: exclude the same domain, only for balanced=True
        balanced: whether fairly sample from each domain
    """

    def __init__(
        self,
        mode: Mode,
        exclude_self: bool = False,
        balanced: bool = True,
        **kwargs,
    ):
        StyleSampler.__init__(self, mode, **kwargs)

        self.exclude_self = exclude_self
        self.balanced = balanced

    def _sample_index(self, domain: int, **kwargs) -> tuple[int, int]:
        if not self.balanced:
            # sample from all domains
            global_id = np.random.choice(self.n_samples)
            cum_sum = 0
            domain_id = -1
            ref_id = -1
            for domain_id, dataset in enumerate(self.datasets):
                cum_sum += len(dataset)
                if cum_sum > global_id:
                    ref_id = global_id - (cum_sum - len(dataset))
                    break
            return domain_id, ref_id

        other_domains = [
            i for i in range(self.n_domains)
            if i != domain or not self.exclude_self
        ]
        ref_domain = random.choice(other_domains)
        ref_dataset = self.datasets[ref_domain]
        return ref_domain, random.randint(0, len(ref_dataset) - 1)


class DomainScoreStyleSampler(StyleSampler):
    """Sample a reference based on inter-domain score.

    Attributes:
        decay: moving average decay of the score
        t: sharpening temperature
        exclude_self: exclude the same domain
    """

    def __init__(
        self,
        mode: Mode,
        decay: float = 0.99,
        t: float = 50.0,
        exclude_self: bool = False,
        **kwargs,
    ):
        StyleSampler.__init__(self, mode, **kwargs)
        # score matrix (src, ref)
        self.score: np.ndarray = np.zeros((self.n_domains, self.n_domains))
        self.decay = decay
        self.t = t
        self.exclude_self = exclude_self

    def bind(self, dataset: SSDGDataset):
        StyleSampler.bind(self, dataset)
        self.score = np.zeros((self.n_domains, self.n_domains))
        return self

    def _sample_index(
        self,
        domain: int,
        **kwargs,
    ) -> tuple[int, int]:
        ref_prob = self.get_current_prob(domain, kwargs.get("negative", False))
        ref_domain = np.random.choice(self.n_domains, p=ref_prob)
        ref_dataset = self.datasets[ref_domain]
        return ref_domain, random.randint(0, len(ref_dataset) - 1)

    def update(
        self,
        src: list[int],
        ref: list[int],
        score: list[float],
    ):
        """Update the score matrix.

        Args:
            src: source domain
            ref: reference domain
            score: score of each sample
        """
        if len(src) != len(ref) or len(src) != len(score):
            raise ValueError("inconsistent length")

        for s, r, sc in zip(src, ref, score):
            self.score[s, r] = (self.score[s, r] * self.decay + sc *
                                (1 - self.decay))

    def get_current_prob(
        self,
        domain: int,
        negative: bool = False,
    ) -> np.ndarray:
        ref_score = self.score[domain]
        if negative:
            ref_score = -ref_score
        # convert to probability (score can be negative)
        ref_prob = np.exp(self.t * ref_score)
        ref_prob /= ref_prob.sum()
        if self.exclude_self:
            ref_prob[domain] = 0
            ref_prob /= ref_prob.sum()
        return ref_prob

    def get_domain_score(self) -> np.ndarray:
        # source-to-ref source indicates the style difficulty of the ref domain
        # if a ref domain makes all other domains worse, it is a hard domain
        # average over all source domains
        score = self.score.mean(axis=0)
        score = np.exp(self.t * score)
        score /= score.sum()
        return score

    def state_dict(self):
        return {
            "score": torch.tensor(self.score),
        }

    def load_state_dict(self, state_dict):
        self.score = state_dict["score"].numpy()


class SampleScoreStyleSampler(StyleSampler):
    """Sample a reference based on sample-to-domain score.

    Attributes:
        decay: moving average decay of the score
        t: sharpening temperature
        top_k: clip the top k% samples
    """

    def __init__(
        self,
        mode: Mode,
        decay: float = 0.99,
        t: float = 50.0,
        top_k: float = 1.0,
        **kwargs,
    ):
        StyleSampler.__init__(self, mode, **kwargs)

        self.score: np.ndarray = np.zeros((self.n_domains, self.n_samples))
        self.decay = decay
        self.t = t
        self.top_k = top_k

    def bind(self, dataset: SSDGDataset):
        StyleSampler.bind(self, dataset)
        self.score = np.zeros((self.n_domains, self.n_samples))
        return self

    def update(
        self,
        src: list[int],
        ref_sample_indices: list[int],
        score: list[float],
    ):
        """Update the score matrix.

        Args:
            src: source domain
            ref_sample_indices: reference sample indices
            score: score of each sample
        """
        if len(src) != len(ref_sample_indices) or len(src) != len(score):
            raise ValueError("inconsistent length")

        for s, r, sc in zip(src, ref_sample_indices, score):
            self.score[s, r] = (self.score[s, r] * self.decay + sc *
                                (1 - self.decay))

    def _sample_index(
        self,
        domain: int,
        **kwargs,
    ) -> tuple[int, int]:
        ref_prob = self.get_current_prob(domain, kwargs.get("negative", False))
        ref_sample = np.random.choice(self.n_samples, p=ref_prob)

        cum_sum = 0
        domain_id = -1
        ref_id = -1
        for domain_id, dataset in enumerate(self.datasets):
            cum_sum += len(dataset)
            if cum_sum > ref_sample:
                ref_id = ref_sample - (cum_sum - len(dataset))
                break
        return domain_id, ref_id

    def get_current_prob(
        self,
        domain: int,
        negative: bool = False,
    ) -> np.ndarray:
        ref_score = self.score[domain]
        k = int(self.top_k * len(ref_score))
        mask = np.zeros_like(ref_score, dtype=np.bool_)
        if negative:
            ref_score = -ref_score
            # min k
            mask[np.argsort(ref_score)[:k]] = True
        else:
            # max k
            mask[np.argsort(ref_score)[-k:]] = True
        ref_prob = np.exp(self.t * ref_score)
        ref_prob[~mask] = 0
        ref_prob /= ref_prob.sum()
        return ref_prob

    def state_dict(self):
        return {
            "score": torch.tensor(self.score),
        }

    def load_state_dict(self, state_dict):
        self.score = state_dict["score"].numpy()


class NegStyleSampler(StyleSampler):
    """Choose the negative result of an existing sampler."""

    def __init__(self, sampler: StyleSampler):
        StyleSampler.__init__(self, sampler.mode, **sampler.kwargs)

        self.sampler = sampler

    def _sample_index(
        self,
        domain: int,
    ) -> tuple[int, int]:
        return self.sampler._sample_index(domain, negative=True)
