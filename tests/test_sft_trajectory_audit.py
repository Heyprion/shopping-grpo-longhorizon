import json

from scripts.audit_sft_trajectories import behavioral_metrics, structure_issues
from shopping_grpo.collection.sft import build_sft_row
from shopping_grpo.environment.actions import action_reject_reason
from shopping_grpo.environment.observation import render_structured_observation
from shopping_grpo.environment.reward import REWARD_VERSION


def test_action_audit_catches_space_mismatch_hidden_by_rendered_guard():
    state = {
        "observation_version": "shopping-observation-v2",
        "page_type": "product_detail",
        "search_available": False,
        "actions": ["规格  黑色", "Buy Now"],
        "product": {"asin": "100000000001", "title": "保护套", "price": 20},
        "available_options": {"颜色": ["规格  黑色"]},
        "selected_options": {},
    }
    observation = render_structured_observation(state)
    assert action_reject_reason("select_option", {"value": "规格 黑色"}, observation) is None
    t = {
        "task_id": 1,
        "trajectory_id": "test",
        "attempt_index": 0,
        "done": True,
        "status": "done",
        "final_reward": 1,
        "initial_result": {"observation_state": state},
        "terminal_result": {
            "done": True,
            "over": True,
            "reward_detail": {
                "reward_version": REWARD_VERSION,
                "reward_valid": True,
                "terminal_utility": 1,
                "reward_components": {"r_type": 1, "r_att": 1, "r_option": 1, "r_price": 1},
            },
        },
        "messages": [
            {"role": "system", "content": "shop"},
            {"role": "user", "content": "买保护套"},
        ],
        "steps": [],
    }
    for i, (name, arguments, action, done) in enumerate(
        [
            ("select_option", {"value": "规格 黑色"}, "click[规格 黑色]", False),
            ("buy_now", {}, "click[Buy Now]", True),
        ]
    ):
        call = {
            "id": str(i),
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(arguments, ensure_ascii=False),
            },
        }
        t["messages"].extend(
            [
                {"role": "assistant", "content": None, "tool_calls": [call]},
                {"role": "tool", "tool_call_id": str(i), "name": name, "content": observation},
            ]
        )
        t["steps"].append(
            {
                "tool_call": call,
                "tool_name": name,
                "env_action": action,
                "observation": observation,
                "done": done,
                "result": {"observation_state": state},
            }
        )
    assert structure_issues(t, build_sft_row(t)) == ["action_not_in_structured_observation"]


def test_repeated_information_read_is_distinct_from_returning_to_candidate():
    product = {"page_type": "product_detail", "selected_options": {}}
    info = {"page_type": "information_subpage", "content": "same information"}
    t = {
        "initial_result": {"observation_state": product},
        "messages": [],
        "terminal_result": {"reward_detail": {}},
        "steps": [],
    }
    for name, state in [("view_features", info), ("prev_page", product), ("view_features", info)]:
        t["steps"].append(
            {"tool_name": name, "parameters": {}, "result": {"observation_state": state}}
        )
    assert behavioral_metrics(t)["redundant_info_roundtrip_steps"] == [2]
    t["steps"][2]["tool_name"] = "view_description"
    assert behavioral_metrics(t)["redundant_info_roundtrip_steps"] == []
