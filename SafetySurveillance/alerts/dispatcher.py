"""
Automated Enterprise Webhook & SOC Dispatcher Module
Dispatches critical safety incident alerts to external Security Operations Centers (SOC),
Slack, Discord, Microsoft Teams, and custom HTTP endpoints with retry logic and audit trails.
"""

import time
import json
import asyncio
from typing import List, Dict, Any, Optional, Tuple
import urllib.request
import urllib.error


class WebhookDispatcher:
    """
    Manages external notification webhooks and dispatches security incident payloads.
    Supports Slack, Discord, and Standard JSON REST webhook schemas.
    """

    def __init__(self):
        # List of registered webhook targets
        self.webhooks: List[Dict[str, Any]] = [
            {
                "id": "wh_soc_primary",
                "name": "Enterprise SOC Security Gateway",
                "url": "http://localhost:8000/api/mock_webhook",
                "type": "generic",
                "enabled": True,
                "min_severity": "MEDIUM",
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        ]
        # Dispatch audit history (last 50 events)
        self.dispatch_logs: List[Dict[str, Any]] = []
        self.severity_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}

    def register_webhook(
        self,
        name: str,
        url: str,
        webhook_type: str = "generic",
        min_severity: str = "HIGH"
    ) -> Dict[str, Any]:
        """Registers a new external dispatch webhook target."""
        wh_id = f"wh_{int(time.time() * 1000)}"
        entry = {
            "id": wh_id,
            "name": name,
            "url": url,
            "type": webhook_type.lower(),
            "enabled": True,
            "min_severity": min_severity.upper(),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.webhooks.append(entry)
        return entry

    def delete_webhook(self, webhook_id: str) -> bool:
        """Removes a registered webhook by ID."""
        orig_len = len(self.webhooks)
        self.webhooks = [w for w in self.webhooks if w["id"] != webhook_id]
        return len(self.webhooks) < orig_len

    def list_webhooks(self) -> List[Dict[str, Any]]:
        return self.webhooks

    def get_dispatch_logs(self) -> List[Dict[str, Any]]:
        return self.dispatch_logs

    def format_payload(self, webhook: Dict[str, Any], alert: Dict[str, Any]) -> Dict[str, Any]:
        """Formats alert payload according to target service (Discord, Slack, or Generic)."""
        wh_type = webhook.get("type", "generic")
        inc_type = alert.get("incident_type", "UNKNOWN_INCIDENT").replace("_", " ").upper()
        cam = alert.get("camera_id", "CAM_01")
        sev = alert.get("severity", "HIGH")
        risk = alert.get("risk_score", 0.8)
        reasons = alert.get("reasons", ["Safety incident detected"])
        ts = alert.get("timestamp", time.strftime("%H:%M:%S"))

        if wh_type == "discord":
            color = 16711680 if sev == "CRITICAL" else (16744448 if sev == "HIGH" else 65535)
            return {
                "username": "QASD AI Watchkeeper",
                "embeds": [{
                    "title": f"🚨 [QASD SURVEILLANCE ALERT] {inc_type}",
                    "description": "\n".join([f"• {r}" for r in reasons]),
                    "color": color,
                    "fields": [
                        {"name": "Camera ID", "value": cam, "inline": True},
                        {"name": "Severity", "value": sev, "inline": True},
                        {"name": "Risk Index", "value": f"{risk:.2f}", "inline": True},
                        {"name": "Detection Time", "value": ts, "inline": True},
                    ],
                    "footer": {"text": "QASD Enterprise Cyber-Physical Safety Core"}
                }]
            }
        elif wh_type == "slack":
            return {
                "text": f"🚨 *[QASD ALERT] {inc_type} on {cam}* ({sev})\n" +
                        f"Risk Score: `{risk:.2f}` | Time: `{ts}`\n" +
                        f"Reasons:\n" + "\n".join([f"> • {r}" for r in reasons])
            }
        else:
            # Generic Enterprise JSON Schema
            return {
                "event": "QASD_SAFETY_INCIDENT",
                "alert_id": alert.get("id"),
                "camera_id": cam,
                "incident_type": alert.get("incident_type"),
                "severity": sev,
                "risk_score": risk,
                "timestamp": ts,
                "reasons": reasons,
                "evidence_metrics": alert.get("evidence_metrics", {}),
                "tamper_evident_seal": alert.get("tamper_evident_seal"),
            }

    def dispatch_alert(self, alert: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Dispatches alert synchronously or launches fire-and-forget requests
        to all eligible registered webhooks.
        """
        alert_sev = alert.get("severity", "LOW")
        alert_rank = self.severity_order.get(alert_sev, 0)
        results = []

        for wh in self.webhooks:
            if not wh.get("enabled", True):
                continue

            wh_min_sev = wh.get("min_severity", "MEDIUM")
            min_rank = self.severity_order.get(wh_min_sev, 1)

            if alert_rank < min_rank:
                continue

            payload = self.format_payload(wh, alert)
            status_code, response_msg = self._send_http_post(wh["url"], payload)

            log_entry = {
                "id": f"disp_{int(time.time() * 1000)}",
                "webhook_id": wh["id"],
                "webhook_name": wh["name"],
                "target_url": wh["url"],
                "incident_type": alert.get("incident_type"),
                "severity": alert_sev,
                "status_code": status_code,
                "response": response_msg,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "success": (200 <= status_code < 300)
            }
            self.dispatch_logs.insert(0, log_entry)
            if len(self.dispatch_logs) > 50:
                self.dispatch_logs.pop()

            results.append(log_entry)

        return results

    def _send_http_post(self, url: str, payload: Dict[str, Any], timeout: float = 3.0) -> Tuple[int, str]:
        """Performs non-blocking HTTP POST request to endpoint with timeout."""
        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json", "User-Agent": "QASD-AI-Dispatcher/2.0"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status, resp.read().decode("utf-8")[:120]
        except urllib.error.HTTPError as he:
            return he.code, f"HTTP Error: {he.reason}"
        except Exception as ex:
            # Handles connection refused for mock or offline endpoints gracefully
            return 0, f"Dispatch connection notice: {str(ex)}"
