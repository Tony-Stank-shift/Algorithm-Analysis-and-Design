"""纯 torch 消息传递：sum / mean（黑盒臂）与 min-sum（结构臂）。

三种聚合只差一个参数，这是实验对照干净的前提。

记号
----
对节点 v，第 k 层：
    m_v = AGG_{u in N(v)} psi(h_u, w_uv)
    h_v = h_v + phi([h_v, m_v])          # 残差，避免深层退化
min-sum 臂则退化为标量状态（就是 Bellman-Ford 的松弛）：
    d_v <- min(d_v, min_{u in N(v)} [d_u + g(w_uv)])
其中 g 是可学习的、保证单调递增的边权变换（softplus）。
`identity=True` 时 g(w)=w，此时该模型**逐字就是 Bellman-Ford**——
这条等价关系是报告第 4 章的桥，也是 3.1/3.5 的验收工具。
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

W_SCALE = 10.0  # 边权在 [1,10]，统一除以 10 便于训练
LARGE = 1e4


# ---------------------------------------------------------------- 批处理
def collate(samples: List[Tuple[dict, int]]) -> Dict[str, torch.Tensor]:
    """把若干 (图记录, 源点) 拼成一个**不相交并集**，一次前向算完。

    不相交并集上做消息传递，等价于每张图各算各的——没有跨图边。
    """
    xs, eis, ews, gids, ys, n_nodes = [], [], [], [], [], 0
    for i, (rec, s) in enumerate(samples):
        n = rec["n"]
        ei = torch.from_numpy(rec["ei"]).long()
        w = torch.from_numpy(rec["w"]).float() / W_SCALE

        deg = torch.zeros(n)
        deg.index_add_(0, ei[:, 0], torch.ones(ei.shape[0]))
        deg.index_add_(0, ei[:, 1], torch.ones(ei.shape[0]))

        # 无向图必须**双向**建边：只按存储方向建，消息会变成单向流动
        # （词例子恰好所有边都从小号指向大号，所以这个 bug 一开始没暴露）
        ei_sym = torch.cat([ei, ei.flip(1)], dim=0)
        w_sym = torch.cat([w, w], dim=0)

        feats = torch.zeros(n, 3)
        feats[:, 0] = (torch.arange(n) == s).float()      # 是否源点
        # 度数必须用**逐点局部量**：早先用 deg/deg.max()，那是个全局归一化，
        # 会让整个模型的感受野超出 L 跳（在远处加一个叶子就能改变根节点的输出）。
        feats[:, 1] = torch.log1p(deg)
        feats[:, 2] = 1.0                                  # 常数项（给 Readout 偏置）

        xs.append(feats)
        eis.append(ei_sym.T + n_nodes)
        ews.append(w_sym)
        gids.append(torch.full((n,), i, dtype=torch.long))
        ys.append(torch.from_numpy(rec["d_true"]).float() / W_SCALE)
        n_nodes += n

    return {
        "x": torch.cat(xs),
        "edge_index": torch.cat(eis, dim=1),
        "edge_w": torch.cat(ews),
        "graph_id": torch.cat(gids),
        "y": torch.cat(ys),
        "n_total": n_nodes,
    }


# ---------------------------------------------------------------- 黑盒臂
class SumMPNN(nn.Module):
    """黑盒臂：sum / mean 聚合 + MLP 更新。学的是"平均"，没有 min 的选择语义。"""

    def __init__(self, layers: int = 4, hidden: int = 64, agg: str = "sum"):
        super().__init__()
        assert agg in ("sum", "mean")
        self.agg = agg
        self.layers = layers
        self.hidden = hidden
        self.inp = nn.Linear(3, hidden)
        self.msg = nn.ModuleList(
            [nn.Sequential(nn.Linear(hidden + 1, hidden), nn.ReLU()) for _ in range(layers)]
        )
        self.upd = nn.ModuleList(
            [nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU()) for _ in range(layers)]
        )
        self.out = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        h = self.inp(batch["x"])
        u, v = batch["edge_index"]
        ew = batch["edge_w"].unsqueeze(-1)
        n = batch["n_total"]
        for k in range(self.layers):
            m = self.msg[k](torch.cat([h[u], ew], dim=-1))       # (E, hidden)
            agg = torch.zeros(n, self.hidden)
            agg.index_add_(0, v, m)
            if self.agg == "mean":
                cnt = torch.zeros(n).index_add_(0, v, torch.ones(v.shape[0]))
                agg = agg / cnt.clamp(min=1.0).unsqueeze(-1)
            h = h + self.upd[k](torch.cat([h, agg], dim=-1))
        return self.out(h).squeeze(-1)


# ---------------------------------------------------------------- 结构臂
class MinSumMPNN(nn.Module):
    """结构臂：把 Bellman-Ford 的松弛写成一层网络。

    identity=True 时 g(w)=w，此时前向与 n 轮同步 Bellman-Ford 完全一致。
    """

    def __init__(self, layers: int = 4, identity: bool = False):
        super().__init__()
        self.layers = layers
        self.identity = identity
        if not identity:
            # g(w) = softplus(a)*w + b，再夹到正数。
            # 这个参数化里 a=log(e-1)、b=0 恰好就是 g(w)=w（恒等），即"精确 Bellman-Ford"，
            # 所以假设类**确实包含**目标算法，训练能不能找到它是实验要回答的问题。
            self.a = nn.Parameter(torch.tensor(0.0))
            self.b = nn.Parameter(torch.tensor(0.0))

    def edge_g(self, w: torch.Tensor) -> torch.Tensor:
        if self.identity:
            return w
        return (F.softplus(self.a) * w + self.b).clamp(min=1e-3)

    def forward(self, batch: Dict[str, torch.Tensor],
                layers: int | None = None) -> torch.Tensor:
        """layers=None 用训练深度；给了更大的值就是「推理时多迭代几轮」。

        因为每层共享同一个 g，这个模型可以在测试时跑 K' > K 轮——这正是
        白盒算法（迭代到收敛）有、而逐层独立参数的黑盒没有的能力。
        """
        n = batch["n_total"]
        u, v = batch["edge_index"]
        k_layers = self.layers if layers is None else int(layers)
        d = torch.where(batch["x"][:, 0] > 0.5,
                        torch.zeros(n), torch.full((n,), LARGE))
        g = self.edge_g(batch["edge_w"])
        for _ in range(k_layers):
            cand = d[u] + g
            agg = torch.full((n,), LARGE).scatter_reduce_(0, v, cand, reduce="amin",
                                                          include_self=True)
            d = torch.minimum(d, agg)
        return d

    @torch.no_grad()
    def explain(self, batch: Dict[str, torch.Tensor], s: int, t: int,
                layers: int | None = None):
        """返回 (每层后的距离, 每层的 argmin 入边, 路径节点, 路径边号, 是否到达源点)。

        解释不是事后猜的：每层 argmin 就是这次前向实际走到的那条边。
        回溯必须用**第 k 层当时**的距离，不能用最终距离——否则层对不上。

        `layers` 允许用比训练深度更多的轮数（推理时迭代到收敛），
        否则 t 往往在 K 跳之外，解出来的"路径"根本到不了 s。
        返回的边号是**原始图**的边号（batch 里每条无向边出现两次，取 e % E）。
        """
        u, v = batch["edge_index"]
        g = self.edge_g(batch["edge_w"])
        n_all = batch["n_total"]
        k_layers = self.layers if layers is None else int(layers)
        e_orig = (torch.arange(u.shape[0]) % (u.shape[0] // 2)
                  if u.shape[0] % 2 == 0 else torch.arange(u.shape[0]))
        d = torch.where(batch["x"][:, 0] > 0.5, torch.zeros(n_all),
                        torch.full((n_all,), LARGE))
        d_at: List[torch.Tensor] = [d.clone()]
        arg_at: List[torch.Tensor] = []
        for _ in range(k_layers):
            cand = d[u] + g
            best = torch.full((n_all,), LARGE).scatter_reduce_(
                0, v, cand, reduce="amin", include_self=True)
            idx = torch.arange(cand.shape[0])
            masked = torch.where(cand <= best[v] + 1e-9, idx,
                                 torch.full_like(idx, cand.shape[0]))
            arg = torch.full((n_all,), cand.shape[0], dtype=torch.long).scatter_reduce_(
                0, v, masked, reduce="amin")
            d = torch.minimum(d, best)
            d_at.append(d.clone())
            arg_at.append(arg)

        path, edges = [t], []
        cur = t
        for k in range(k_layers, 0, -1):
            if float(d_at[k][cur]) >= float(d_at[k - 1][cur]) - 1e-9:
                continue
            e = int(arg_at[k - 1][cur])
            if e >= u.shape[0]:
                break
            edges.append(int(e_orig[e]))
            cur = int(u[e])
            path.append(cur)
            if cur == s:
                break
        return d_at, arg_at, path[::-1], edges[::-1], (cur == s)


# ---------------------------------------------------------------- 便捷推理
@torch.no_grad()
def predict(model: nn.Module, rec: dict, s: int) -> torch.Tensor:
    model.eval()
    batch = collate([(rec, s)])
    return model(batch) * W_SCALE
