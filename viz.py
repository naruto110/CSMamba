"""Quick visualization helper for demo graphs."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import networkx as nx

from demo_dataset import load_demo_graph


def main():
    parser = argparse.ArgumentParser(description="Visualize demo graph communities.")
    parser.add_argument("--graph", default="toy", choices=["toy", "lfr"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--with-labels", action="store_true", help="Show node labels.")
    args = parser.parse_args()

    G, node_to_comm = load_demo_graph(args.graph, seed=args.seed, cache_dir=args.data_dir)
    pos = nx.spring_layout(G, seed=args.seed)
    colors = [node_to_comm.get(n, 0) for n in G.nodes()]

    nx.draw_networkx(
        G,
        pos=pos,
        node_color=colors,
        cmap="tab10",
        with_labels=args.with_labels,
        node_size=200,
        edge_color="#888",
        linewidths=0.5,
        font_size=8,
    )
    plt.title(f"{args.graph} graph (seed={args.seed})")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
