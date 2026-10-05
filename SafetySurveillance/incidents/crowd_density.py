"""
Crowd Density Surge & Stampede Hazard Detection Module
Monitors localized spatial crowd density, interpersonal proximity collapse,
and high-velocity divergence / panic acceleration signatures.
"""

from typing import List, Dict, Any, Tuple
import numpy as np


class CrowdDensityDetector:
    """Detects dangerous crowd congestion, sudden density surges, and stampede risks."""

    def __init__(
        self,
        max_density_threshold: int = 5,       # >5 persons in proximity cluster indicates congestion
        cluster_dist_px: float = 110.0,       # Interpersonal distance cutoff for crowd cluster
        stampede_velocity_thresh: float = 65.0 # Sudden high-velocity collective movement
    ):
        self.max_density_threshold = max_density_threshold
        self.cluster_dist_px = cluster_dist_px
        self.stampede_velocity_thresh = stampede_velocity_thresh

    def evaluate(self, active_tracks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyzes active person tracks for spatial density bottlenecks or panic dispersion.

        Returns:
            List of crowd safety incident alerts.
        """
        person_tracks = [t for t in active_tracks if t["class"] == "person"]
        if len(person_tracks) < 3:
            return []

        alerts = []
        positions = []
        track_map = []

        for p in person_tracks:
            bbox = p["bbox"]
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            positions.append([cx, cy])
            track_map.append(p)

        pos_arr = np.array(positions, dtype=np.float32)
        n = len(pos_arr)

        # Pairwise Euclidean distance matrix
        diff = pos_arr[:, np.newaxis, :] - pos_arr[np.newaxis, :, :]
        dist_matrix = np.linalg.norm(diff, axis=-1)

        # Identify connected crowd clusters (adjacency graph)
        adj_matrix = (dist_matrix < self.cluster_dist_px) & ~np.eye(n, dtype=bool)

        visited = set()
        clusters: List[List[int]] = []

        for i in range(n):
            if i in visited:
                continue
            # BFS to find connected component
            component = []
            queue = [i]
            visited.add(i)

            while queue:
                curr = queue.pop(0)
                component.append(curr)
                neighbors = np.where(adj_matrix[curr])[0]
                for nbr in neighbors:
                    if nbr not in visited:
                        visited.add(nbr)
                        queue.append(nbr)

            if len(component) >= 3:
                clusters.append(component)

        for cluster in clusters:
            cluster_size = len(cluster)
            cluster_tracks = [track_map[idx] for idx in cluster]
            involved_ids = [t["id"] for t in cluster_tracks]

            # Bounding box of the entire cluster
            all_bboxes = np.array([t["bbox"] for t in cluster_tracks])
            x1 = int(np.min(all_bboxes[:, 0]))
            y1 = int(np.min(all_bboxes[:, 1]))
            x2 = int(np.max(all_bboxes[:, 2]))
            y2 = int(np.max(all_bboxes[:, 3]))

            # Check velocity of cluster members
            velocities = []
            for t in cluster_tracks:
                hist = t.get("history")
                if hist and len(hist.velocities) > 0:
                    v = hist.velocities[-1]
                    speed = np.linalg.norm(v)
                    velocities.append(speed)

            avg_cluster_speed = float(np.mean(velocities)) if velocities else 0.0

            # Condition 1: Stampede / Rapid Panic Dispersion
            is_stampede = avg_cluster_speed > self.stampede_velocity_thresh and cluster_size >= 3
            # Condition 2: Overcrowding / Density Surge
            is_overcrowding = cluster_size >= self.max_density_threshold

            if is_stampede:
                conf = min(0.97, round(0.78 + (avg_cluster_speed / 150.0) * 0.19, 3))
                alerts.append({
                    "incident_type": "Stampede / Panic Surge Hazard",
                    "severity": "CRITICAL",
                    "confidence": conf,
                    "involved_tracks": involved_ids,
                    "bbox": [x1, y1, x2, y2],
                    "reasons": [
                        f"Rapid collective crowd velocity detected ({avg_cluster_speed:.1f} px/s avg)",
                        f"Co-located cluster of {cluster_size} subjects in rapid motion",
                        "High risk of pedestrian crushing or uncontrolled evacuation surge",
                    ],
                    "evidence_metrics": {
                        "cluster_occupancy": cluster_size,
                        "collective_velocity": round(avg_cluster_speed, 1),
                        "cluster_span_px": [x2 - x1, y2 - y1],
                    },
                })
            elif is_overcrowding:
                conf = min(0.92, round(0.70 + (cluster_size / 10.0) * 0.22, 3))
                alerts.append({
                    "incident_type": "Overcrowding / Density Hazard",
                    "severity": "HIGH",
                    "confidence": conf,
                    "involved_tracks": involved_ids,
                    "bbox": [x1, y1, x2, y2],
                    "reasons": [
                        f"Crowd density threshold exceeded ({cluster_size} persons clustered)",
                        f"Interpersonal proximity < {self.cluster_dist_px}px across cluster",
                        "Potential bottleneck at exit corridor or restricted passageway",
                    ],
                    "evidence_metrics": {
                        "cluster_occupancy": cluster_size,
                        "mean_proximity_px": round(float(np.mean(dist_matrix[np.ix_(cluster, cluster)])), 1),
                    },
                })

        return alerts
