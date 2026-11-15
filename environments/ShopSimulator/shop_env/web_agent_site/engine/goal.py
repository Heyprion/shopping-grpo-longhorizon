"""Build goals from the frozen product/task annotations."""

import os
from collections import defaultdict

from rich import print

from web_agent_site.engine.constraints import explicit_budget_from_instruction
from web_agent_site.engine.reward_features import compile_reward_features
from web_agent_site.engine.task_budget import (
    budget_contract,
    budget_upper,
    load_budget_index,
    query_hash,
)


def get_goals(all_products, product_prices, if_persona=False):
    return get_existed_goals(all_products, product_prices, if_persona)


def get_existed_goals(all_products, product_prices, if_persona=False):
    budget_path = os.environ.get("SHOPPING_TASK_BUDGETS")
    budget_index = load_budget_index(budget_path) if budget_path else None
    goals = []
    cnt_atts = defaultdict(int)
    cnt_1, cnt_2, cnt_3 = 0, 0, 0
    goal_instructions = []
    for item in all_products:
        if "instructions" not in item:
            cnt_1 += 1
            continue
        asin = item["asin"]
        for product in item["instructions"]:
            if product["instruction"] in goal_instructions:
                cnt_2 += 1
                # continue
            else:
                goal_instructions.append(product["instruction"])
            attributes = product.get("attributes", [])
            if len(attributes) == 0:
                cnt_3 += 1
                continue

            if product_prices is not None:
                # Strict Reward must not invent an unstated budget from the Gold
                # product. Price availability remains a hard verifiability
                # requirement, but an upper bound exists only when the user
                # instruction states one.
                price_upper = explicit_budget_from_instruction(product["instruction"])
            else:
                price_upper = 10000000

            # Process user_persona, place __reasoning__ field in the first position
            if not isinstance(item["user_persona"], dict):
                item["user_persona"] = {}
            user_persona = item["user_persona"].copy()
            reason_key = item["reason_key"]
            if user_persona and "__reasoning__" in user_persona:
                reasoning_value = user_persona.pop("__reasoning__")
                # Create a new ordered dictionary with __reasoning__ in the first position
                ordered_persona = {"__reasoning__": reasoning_value}
                ordered_persona.update(user_persona)
                user_persona = ordered_persona

            if if_persona:
                instruction_text = product["instruction_sample"]
            else:
                instruction_text = product["instruction"]
            goal = {
                "asin": asin,
                "category": item["category"],
                "query": item["query"],
                "name": item["title"],
                "instruction_text": instruction_text,
                "instruction_simple": product["instruction_simple"],
                "attributes": attributes,
                "price_upper": price_upper,
                "goal_options": product["instruction_options"],
                "user_persona": user_persona,
                "reason_key": reason_key,
            }
            goal.update(compile_reward_features(product, item))
            if budget_index is not None:
                canonical = budget_index.get((asin, query_hash(product["instruction"])))
                if canonical is None:
                    canonical = budget_contract(product["instruction"], [], "unresolved")
                goal["task_budget"] = canonical
                goal["price_upper"] = (
                    budget_upper(canonical) if canonical["state"] != "unresolved" else None
                )
            goals.append(goal)
            for att in attributes:
                cnt_atts[att] += 1
            # goals += product_goals
    for goal in goals:
        goal["weight"] = 1
    print("skipped")
    print(cnt_1, cnt_2, cnt_3)
    return goals
