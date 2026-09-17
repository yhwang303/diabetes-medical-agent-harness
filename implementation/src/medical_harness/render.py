"""Trusted, non-evolvable report renderer. No free text from agents is rendered."""

from .contracts import Proposal

TEMPLATE_VERSION = "fixture-template-v1"
RULE_VERSION = "fixture-engineering-rules-v1"
KB_VERSION = "technical-notice-v1"
WARNING = "工程测试替身：非真实预测、非真实RL、非医学审核，不用于用药。"
TECHNICAL_NOTICE = {
    "doc_id": "fixture-limitations", "version": KB_VERSION,
    "scope": "engineering", "section": "limitations",
    "text": "本报告只验证产物来源、绑定、审核与发布流程。所有数值均为测试夹具输出，未执行任何给药动作。",
}


def render(proposal: Proposal, prediction: dict, policy: dict, evidence_hash: str, *, profile=None) -> dict:
    slots = {
        "forecast": {"kind": "forecast", "label": "预测接口占位输出",
                     "artifact_id": prediction["id"], "data": prediction["payload"]},
        "policy": {"kind": "policy", "label": "RL接口占位输出（未执行）",
                   "artifact_id": policy["id"], "data": policy["payload"]},
        "limitations": {"kind": "limitations", "text": TECHNICAL_NOTICE["text"],
                        "citation": {"doc_id": TECHNICAL_NOTICE["doc_id"], "version": KB_VERSION}},
    }
    report = {"report_type": "EngineeringFixtureReport", "origin": "fixture", "clinical_use": False,
              "warning": WARNING, "template_version": TEMPLATE_VERSION, "evidence_hash": evidence_hash,
              "sections": [slots[key] for key in proposal.sections]}
    if profile:
        from .basal_actions import PROFILE_VERSION, display_action, display_rate_action
        display = display_rate_action if profile['version'] == PROFILE_VERSION else display_action
        slots['policy']['action_display'] = display(policy['payload']['action'])
        report.update(template_version=profile['template_version'], use_profile=profile['use_profile'],
                      time_basis=profile['time_basis'], valid_until=profile['binding']['expires_at'])
    return report
