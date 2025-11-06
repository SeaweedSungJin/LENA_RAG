from mmengine.registry import DATA_SAMPLERS
from mmengine.dist import get_dist_info
from torch.utils.data import Sampler, WeightedRandomSampler
import numpy as np, torch, math

@DATA_SAMPLERS.register_module()
class BalancedRAGSampler(Sampler):
    """
    rag_labels (0=NO_RAG, 1=USE_RAG)를 활용해 배치에서 NO_RAG 비율을 r로 맞추는 샘플러.
    - dataset이 ConcatDataset인 경우, 내부 datasets[*].rag_labels를 합쳐 사용.
    """
    def __init__(self,
                 dataset,
                 desired_no_ratio: float = 0.45,
                 seed: int = 3407,
                 samples_per_epoch: int | None = None):
        super().__init__()
        self.dataset = dataset
        self.r = float(desired_no_ratio)
        self.seed = seed
        self.epoch = 0

        # ---- rag_labels 수집 ----
        if hasattr(dataset, "rag_labels"):              # 단일 dataset
            labels = np.array(dataset.rag_labels, dtype=np.int64)
        elif hasattr(dataset, "datasets"):              # ConcatDataset
            labels = []
            for ds in dataset.datasets:
                if hasattr(ds, "rag_labels"):
                    labels.extend(ds.rag_labels)
                else:
                    raise AttributeError(f"Sub-dataset {type(ds)} has no rag_labels")
            labels = np.array(labels, dtype=np.int64)
        else:
            raise AttributeError("Dataset has no rag_labels or sub-datasets")

        # ---- 가중치 계산 ----
        n0 = max(1, int((labels == 0).sum()))
        n1 = max(1, int((labels == 1).sum()))
        w0 = self.r / n0
        w1 = (1.0 - self.r) / n1
        self.weights = np.where(labels == 0, w0, w1).astype(np.float64)

        # ---- 분산 환경 설정 ----
        self.rank, self.world_size = get_dist_info()
        N = len(labels) if samples_per_epoch is None else samples_per_epoch
        self.num_samples = int(math.ceil(N / self.world_size))
        self.total_size = self.num_samples * self.world_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.rank + self.epoch * 1009)
        base_sampler = WeightedRandomSampler(
            weights=torch.as_tensor(self.weights, dtype=torch.double),
            num_samples=self.total_size,
            replacement=True,
            generator=g
        )
        it = iter(base_sampler)
        for _ in range(self.rank * self.num_samples):  # rank offset
            next(it)
        for _ in range(self.num_samples):
            yield next(it)

    def __len__(self):
        return self.num_samples
