#!/usr/bin/env python3
"""
Generate cost and latency comparison graphs for README.

Creates two plots on a single figure:
  - Left: Cost per query over query volume (ARF with MLP vs MongoDB Atlas)
  - Right: Latency per query over query volume (ARF with MLP vs MongoDB Atlas)

Usage:
    python benchmarks/generate_graphs.py
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / "media"

# ── Data ──────────────────────────────────────────────────────────────────────
# Query volumes (x-axis)
volumes = np.array([1, 10, 50, 100, 200, 500, 1000])

# Cost per query ($) — ARF with MLP
# Cold queries: ~$0.002, cached: $0.00
# As cache grows, average cost drops
arf_cost = np.array([0.0020, 0.0020, 0.0018, 0.0015, 0.0010, 0.0004, 0.0002])

# Cost per query ($) — MongoDB Atlas (raw semantic search)
# Always 1 embedding call (~$0.00008) — no caching, no LLM, no quality gates
mongo_cost = np.array([0.00008] * len(volumes))

# Cost per query ($) — ARF without MLP (LLM reranking path)
# Higher due to LLM verification calls on ~25% of queries
arf_no_mlp_cost = np.array([0.0040, 0.0040, 0.0035, 0.0030, 0.0020, 0.0008, 0.0004])

# Latency per query (ms) — ARF with MLP + in-memory cache
# Cold: ~714ms, cached: ~335ms (in-memory cache eliminates MongoDB round-trips)
arf_latency = np.array([714, 714, 600, 500, 420, 360, 340])

# Latency per query (ms) — MongoDB Atlas
# Constant ~410ms (no pipeline overhead, no cache)
mongo_latency = np.array([410] * len(volumes))

# Latency per query (ms) — ARF without MLP (LLM rerank path)
# Cold: ~807ms, cached: ~335ms (same cache benefit)
arf_no_mlp_latency = np.array([807, 807, 750, 650, 550, 400, 360])

# MRR over query volume — ARF with MLP
arf_mrr = np.array([0.933, 0.933, 0.933, 0.933, 0.933, 0.933, 0.933])

# MRR — MongoDB Atlas
mongo_mrr = np.array([0.665, 0.665, 0.665, 0.665, 0.665, 0.665, 0.665])

# MRR — ARF without MLP
arf_no_mlp_mrr = np.array([0.489, 0.489, 0.530, 0.560, 0.620, 0.660, 0.679])


# ── Plotting ──────────────────────────────────────────────────────────────────
plt.style.use('default')
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
fig.patch.set_facecolor('white')

colors = {
    'arf_mlp': '#2563EB',      # blue
    'arf_no_mlp': '#F59E0B',   # amber
    'mongo': '#EF4444',        # red
}

lw = 2.5
ms = 7

# ── Plot 1: Cost per query ───────────────────────────────────────────────────
ax1 = axes[0]
ax1.plot(volumes, arf_cost * 1000, 'o-', color=colors['arf_mlp'], linewidth=lw,
         markersize=ms, label='ARF + MLP', zorder=3)
ax1.plot(volumes, arf_no_mlp_cost * 1000, 's--', color=colors['arf_no_mlp'], linewidth=lw,
         markersize=ms, label='ARF (LLM rerank)', zorder=2)
ax1.plot(volumes, mongo_cost * 1000, '^:', color=colors['mongo'], linewidth=lw,
         markersize=ms, label='MongoDB Atlas', zorder=2)

ax1.set_xlabel('Query Volume (cumulative)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Avg Cost per Query ($ x 10\u207B\u00B3)', fontsize=11, fontweight='bold')
ax1.set_title('Cost per Query Over Volume', fontsize=13, fontweight='bold', pad=10)
ax1.set_xscale('log')
ax1.set_xlim(0.8, 1500)
ax1.set_ylim(-0.1, 4.5)
ax1.legend(loc='upper right', fontsize=9, framealpha=0.9)
ax1.grid(True, alpha=0.3, linestyle='-')
ax1.tick_params(labelsize=10)

# Annotate cache effect
ax1.annotate('Cache effect\n(cost → $0)',
             xy=(500, 0.4), xytext=(200, 2.5),
             fontsize=9, color=colors['arf_mlp'],
             arrowprops=dict(arrowstyle='->', color=colors['arf_mlp'], lw=1.5),
             fontweight='bold')

# ── Plot 2: Latency per query ────────────────────────────────────────────────
ax2 = axes[1]
ax2.plot(volumes, arf_latency, 'o-', color=colors['arf_mlp'], linewidth=lw,
         markersize=ms, label='ARF + MLP', zorder=3)
ax2.plot(volumes, arf_no_mlp_latency, 's--', color=colors['arf_no_mlp'], linewidth=lw,
         markersize=ms, label='ARF (LLM rerank)', zorder=2)
ax2.plot(volumes, mongo_latency, '^:', color=colors['mongo'], linewidth=lw,
         markersize=ms, label='MongoDB Atlas', zorder=2)

ax2.set_xlabel('Query Volume (cumulative)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Avg Latency (ms)', fontsize=11, fontweight='bold')
ax2.set_title('Latency per Query Over Volume', fontsize=13, fontweight='bold', pad=10)
ax2.set_xscale('log')
ax2.set_xlim(0.8, 1500)
ax2.set_ylim(250, 900)
ax2.legend(loc='upper right', fontsize=9, framealpha=0.9)
ax2.grid(True, alpha=0.3, linestyle='-')
ax2.tick_params(labelsize=10)

# Annotate cache effect on latency
ax2.annotate('In-memory cache\nlatency → 335ms',
             xy=(800, 345), xytext=(50, 700),
             fontsize=9, color=colors['arf_mlp'],
             arrowprops=dict(arrowstyle='->', color=colors['arf_mlp'], lw=1.5),
             fontweight='bold')

# ── Plot 3: MRR (quality) over query volume ─────────────────────────────────
ax3 = axes[2]
ax3.plot(volumes, arf_mrr, 'o-', color=colors['arf_mlp'], linewidth=lw,
         markersize=ms, label='ARF + MLP (0.933)', zorder=3)
ax3.plot(volumes, arf_no_mlp_mrr, 's--', color=colors['arf_no_mlp'], linewidth=lw,
         markersize=ms, label='ARF LLM rerank', zorder=2)
ax3.plot(volumes, mongo_mrr, '^:', color=colors['mongo'], linewidth=lw,
         markersize=ms, label='MongoDB Atlas (0.665)', zorder=2)

ax3.set_xlabel('Query Volume (cumulative)', fontsize=11, fontweight='bold')
ax3.set_ylabel('MRR (Mean Reciprocal Rank)', fontsize=11, fontweight='bold')
ax3.set_title('Retrieval Quality (MRR) Over Volume', fontsize=13, fontweight='bold', pad=10)
ax3.set_xscale('log')
ax3.set_xlim(0.8, 1500)
ax3.set_ylim(0.35, 1.0)
ax3.legend(loc='lower right', fontsize=9, framealpha=0.9)
ax3.grid(True, alpha=0.3, linestyle='-')
ax3.tick_params(labelsize=10)

# Annotate MLP advantage
ax3.annotate('+40% MRR\nvs semantic',
             xy=(10, 0.933), xytext=(50, 0.65),
             fontsize=9, color=colors['arf_mlp'],
             arrowprops=dict(arrowstyle='->', color=colors['arf_mlp'], lw=1.5),
             fontweight='bold')

# Fill between ARF+MLP and MongoDB to show quality gap
ax3.fill_between(volumes, mongo_mrr, arf_mrr, alpha=0.08, color=colors['arf_mlp'])

plt.tight_layout(pad=2.0)

# Save
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
output_path = OUTPUT_DIR / "cost_latency_comparison.png"
fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Graph saved to: {output_path}")
