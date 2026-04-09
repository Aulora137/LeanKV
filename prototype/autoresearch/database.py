"""SQLite results database for autoresearch sweep."""

import sqlite3
import json
import time
from typing import List, Dict


class ResultsDB:
    def __init__(self, db_path: str = "prototype/autoresearch/results/sweep.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_table()

    def _create_table(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                config_name TEXT,
                k_bits REAL,
                v_bits REAL,
                rotation TEXT,
                group_size INTEGER,
                layer_policy TEXT,
                use_qjl INTEGER,
                k_cos_mean REAL,
                k_cos_min REAL,
                v_cos_mean REAL,
                v_cos_min REAL,
                attn_cos_mean REAL,
                attn_kl_mean REAL,
                effective_k_bits REAL,
                effective_v_bits REAL,
                total_bits_per_kv_pair REAL,
                compression_ratio REAL,
                raw_json TEXT
            )
        """)
        self.conn.commit()

    def insert_result(self, result: Dict):
        self.conn.execute("""
            INSERT INTO results (
                timestamp, config_name, k_bits, v_bits, rotation,
                group_size, layer_policy, use_qjl,
                k_cos_mean, k_cos_min, v_cos_mean, v_cos_min,
                attn_cos_mean, attn_kl_mean,
                effective_k_bits, effective_v_bits, total_bits_per_kv_pair,
                compression_ratio, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            time.time(),
            result.get("config_name", ""),
            result.get("k_bits"),
            result.get("v_bits"),
            result.get("rotation", ""),
            result.get("group_size"),
            result.get("layer_policy", ""),
            int(result.get("use_qjl", False)),
            result.get("k_cosine_sim_mean"),
            result.get("k_cosine_sim_min"),
            result.get("v_cosine_sim_mean"),
            result.get("v_cosine_sim_min"),
            result.get("attn_cosine_sim_mean"),
            result.get("attn_kl_divergence_mean"),
            result.get("effective_k_bits"),
            result.get("effective_v_bits"),
            result.get("total_bits_per_kv_pair"),
            result.get("compression_ratio"),
            json.dumps(result),
        ))
        self.conn.commit()

    def get_all_results(self) -> List[Dict]:
        rows = self.conn.execute("SELECT * FROM results ORDER BY total_bits_per_kv_pair").fetchall()
        return [dict(r) for r in rows]

    def get_pareto_frontier(self) -> List[Dict]:
        """Find Pareto-optimal configs: no other config is better on BOTH quality AND memory.

        Quality = min(k_cos_mean, v_cos_mean) — worst-case quality metric.
        Memory = total_bits_per_kv_pair — lower is better.
        """
        all_results = self.get_all_results()
        if not all_results:
            return []

        sorted_results = sorted(all_results, key=lambda r: r["total_bits_per_kv_pair"])

        pareto = []
        best_quality = -1.0

        for r in sorted_results:
            quality = min(r["k_cos_mean"] or 0, r["v_cos_mean"] or 0)
            if quality > best_quality:
                best_quality = quality
                pareto.append(r)

        return pareto

    def close(self):
        self.conn.close()
