# # FILE: src/training/models/mlp_ranker.py
# from __future__ import annotations

# from typing import Iterable, List
# import torch
# import torch.nn as nn


# class Ranker(nn.Module):
#     """
#     MLP → scalar score. Используется как "скорер" для ранжирования пилотов внутри гонки.

#     Args:
#         in_dim:   размерность входного признакового вектора
#         hidden:   список скрытых слоёв, например [256, 128]
#         dropout:  dropout после каждого скрытого слоя
#     """
#     def __init__(self, in_dim: int, hidden: Iterable[int] = (256, 128), dropout: float = 0.10):
#         super().__init__()
#         hidden = list(hidden) if hidden is not None else []
#         dims: List[int] = [in_dim] + hidden + [1]

#         layers: List[nn.Module] = []
#         for i in range(len(dims) - 2):
#             layers.append(nn.Linear(dims[i], dims[i + 1]))
#             layers.append(nn.ReLU())
#             if dropout and dropout > 0:
#                 layers.append(nn.Dropout(dropout))
#         layers.append(nn.Linear(dims[-2], dims[-1]))  # финальный скор

#         self.net = nn.Sequential(*layers)
#         self.apply(self._init_weights)

#     @staticmethod
#     def _init_weights(m: nn.Module):
#         if isinstance(m, nn.Linear):
#             nn.init.kaiming_uniform_(m.weight, a=0.01, nonlinearity="relu")
#             if m.bias is not None:
#                 nn.init.zeros_(m.bias)

#     def forward(self, X: torch.Tensor) -> torch.Tensor:
#         """
#         X: [N, F] → scores: [N]
#         """
#         s = self.net(X).squeeze(-1)
#         return s


# def make_model(in_dim: int, hidden: Iterable[int] = (256, 128), dropout: float = 0.10) -> Ranker:
#     """
#     Фабричная функция для совместимости с конфигом/скриптами.
#     """
#     return Ranker(in_dim=in_dim, hidden=list(hidden) if hidden is not None else [], dropout=dropout)


# __all__ = ["Ranker", "make_model"]


# FILE: src/training/models/advanced_ranker.py
from __future__ import annotations
from typing import Iterable, List, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class StochasticDepth(nn.Module):
    """
    DropPath для резидуальных блоков: во время обучения случайно "обнуляет"
    резидуальную ветку, масштабируя сохранённые проходы (inverted rule).
    """
    def __init__(self, p: float = 0.0):
        super().__init__()
        if not (0.0 <= p < 1.0):
            raise ValueError("StochasticDepth p must be in [0,1)")
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        # Бернулли по батчу (broadcast по фичам)
        mask = torch.rand((x.shape[0],) + (1,) * (x.ndim - 1), device=x.device, dtype=x.dtype) < keep
        return x * mask / keep


class GEGLU(nn.Module):
    """
    Gated Linear Unit с GELU: proj(x)->[a,b]; out = a * GELU(b)
    Хорошо работает на табличных фичах.
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.proj(x).chunk(2, dim=-1)
        return a * F.gelu(b)


class ResidualBlock(nn.Module):
    """
    PreNorm Residual: x + DropPath( FF( LayerNorm(x) ) )
      FF: GEGLU -> Dropout -> Linear
      Skip-proj, если изменяется размерность.
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.1,
        use_layernorm: bool = True,
        droppath: float = 0.0,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim) if use_layernorm else nn.Identity()
        self.ff = nn.Sequential(
            GEGLU(in_dim, out_dim),
            nn.Dropout(dropout) if (dropout and dropout > 0.0) else nn.Identity(),
            nn.Linear(out_dim, out_dim),
        )
        self.skip = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim, bias=False)
        self.drop = StochasticDepth(droppath)

        self._init_weights()

    def _init_weights(self):
        # Инициализация, устойчивая к SiLU/GELU: xavier для проекций, kaiming для линейных.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # GEGLU.proj мы не трогаем здесь (он внутри отдельного модуля),
                # а для остальных линеек:
                nn.init.kaiming_uniform_(m.weight, a=0.0, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Для GEGLU — xavier на общей проекции
        if isinstance(self.ff[0], GEGLU):
            proj = self.ff[0].proj
            nn.init.xavier_uniform_(proj.weight)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ff(self.norm(x))
        h = self.drop(h)
        x = self.skip(x)
        return x + h


class AdvancedRanker(nn.Module):
    """
    Продвинутый скорер для ранжирования пилотов внутри гонки.

    Архитектура:
      - (опц.) LayerNorm на входе + input dropout
      - стек ResidualBlock с GEGLU, LayerNorm внутри блока и DropPath
      - финальный LayerNorm + Linear(->1) с "мелкой" инициализацией

    Args:
        in_dim: размерность входа
        hidden: список размеров скрытых блоков, напр. [512, 256, 128]
        dropout: dropout внутри блоков
        input_dropout: dropout прямо на входе (по фичам)
        use_layernorm: вкл/выкл LayerNorm (внутри блоков и финальный)
        droppath: вероятность DropPath (стохастической глубины) равномерно по блокам
        small_last_init: если True, последний слой инициализируется малыми весами
    """
    def __init__(
        self,
        in_dim: int,
        hidden: Iterable[int] = (512, 256, 128),
        dropout: float = 0.10,
        input_dropout: float = 0.00,
        use_layernorm: bool = True,
        droppath: float = 0.00,
        small_last_init: bool = True,
    ):
        super().__init__()
        dims: List[int] = [int(in_dim)] + [int(h) for h in (list(hidden) if hidden is not None else [])]
        self.in_norm = nn.LayerNorm(in_dim) if use_layernorm else nn.Identity()
        self.in_drop = nn.Dropout(input_dropout) if (input_dropout and input_dropout > 0.0) else nn.Identity()

        blocks: List[nn.Module] = []
        L = len(dims) - 1
        for i in range(L):
            p = droppath * float(i + 1) / float(L) if droppath and L > 0 else 0.0  # линейный градиент DropPath
            blocks.append(
                ResidualBlock(
                    in_dim=dims[i],
                    out_dim=dims[i + 1],
                    dropout=dropout,
                    use_layernorm=use_layernorm,
                    droppath=p,
                )
            )
        self.backbone = nn.Sequential(*blocks)

        self.out_norm = nn.LayerNorm(dims[-1]) if use_layernorm else nn.Identity()
        self.head = nn.Linear(dims[-1], 1)
        if small_last_init:
            nn.init.uniform_(self.head.weight, -1e-3, 1e-3)
            nn.init.zeros_(self.head.bias)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: [N, F] → scores: [N]
        """
        X = self.in_norm(X)
        X = self.in_drop(X)
        H = self.backbone(X) if len(self.backbone) > 0 else X
        H = self.out_norm(H)
        s = self.head(H).squeeze(-1)
        return s


def make_model(
    in_dim: int,
    hidden: Iterable[int] = (512, 256, 128),
    dropout: float = 0.10,
    input_dropout: float = 0.00,
    use_layernorm: bool = True,
    droppath: float = 0.00,
    small_last_init: bool = True,
) -> AdvancedRanker:
    """
    Фабрика под ваши конфиги/скрипты. Совместима с существующим пайплайном.
    Пример:
        model = make_model(in_dim=features, hidden=[256,128], dropout=0.1, droppath=0.05)
    """
    return AdvancedRanker(
        in_dim=in_dim,
        hidden=list(hidden) if hidden is not None else [],
        dropout=dropout,
        input_dropout=input_dropout,
        use_layernorm=use_layernorm,
        droppath=droppath,
        small_last_init=small_last_init,
    )


__all__ = ["AdvancedRanker", "make_model"]
